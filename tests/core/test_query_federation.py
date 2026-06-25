from pathlib import Path

from dryml.core2 import Definition, Object, Repo, SKIP_ARGS
from dryml.core2.query.federation import CACHE_SOURCE_KEY, RepoGenerationVector, StoreIndexBinding
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig
from dryml.core2.store.dir import DirStore


class FederationLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class FederationParent(Object):
    def __init__(self, child=None, *, name="parent"):
        super().__init__()
        self.child = child
        self.name = name


class RecordingIndex:
    def __init__(self, store):
        self.store = store
        self.registered = []
        self.closed = False

    def register_stored_roots(self, graph, roots):
        assert all(self.store.has(root) for root in roots)
        self.registered.append((graph, tuple(roots)))

    def status(self):
        from dryml.core2.query.model import QueryIndexStatus

        return QueryIndexStatus(
            backend="recording",
            store_key=self.store.catalog_key(),
            generation=len(self.registered),
            schema_version=None,
            semantic_versions={},
            state="ready",
        )

    def close(self):
        self.closed = True


class FailingIndex:
    def register_stored_roots(self, graph, roots):
        raise RuntimeError("index failed after object publish")


def test_repo_federation_bindings_follow_store_priority(tmp_path):
    store1 = DirStore(tmp_path / "store1", query_index="memory")
    store2 = DirStore(tmp_path / "store2", query_index="none")
    repo = Repo(stores=[store1, store2])

    bindings = repo._query_index.store_bindings

    assert tuple(type(binding) for binding in bindings) == (StoreIndexBinding, StoreIndexBinding)
    assert [binding.store for binding in bindings] == [store1, store2]
    assert [binding.priority for binding in bindings] == [0, 1]

    repo.set_default_store(store2)

    bindings = repo._query_index.store_bindings
    assert [binding.store for binding in bindings] == [store2, store1]
    assert [binding.priority for binding in bindings] == [0, 1]


def test_repo_federation_add_store_updates_bindings(tmp_path):
    repo = Repo()
    store1 = DirStore(tmp_path / "store1", query_index="memory")
    store2 = DirStore(tmp_path / "store2", query_index="memory")

    repo.add_store(store1)
    repo.add_store(store2, make_default=True)

    assert [binding.store for binding in repo._query_index.store_bindings] == [store2, store1]


def test_sources_for_domain(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)

    assert repo._query_index.sources_for_domain("cached") == (CACHE_SOURCE_KEY,)
    assert repo._query_index.sources_for_domain("stored") == repo._query_index.store_bindings
    assert repo._query_index.sources_for_domain("nested") == repo._query_index.store_bindings
    known_sources = repo._query_index.sources_for_domain("known")
    assert known_sources[:-1] == repo._query_index.store_bindings
    assert known_sources[-1] == CACHE_SOURCE_KEY


def test_generation_vector_includes_cache_generation(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store", query_index="memory"))
    obj = FederationLeaf("cached", repo=repo)
    repo.add_objects(obj)

    vector = repo._query_index.generation_vector()

    assert isinstance(vector, RepoGenerationVector)
    assert vector.generations[CACHE_SOURCE_KEY] == repo._query_catalog.generation


def test_index_status_reports_mixed_store_policies(tmp_path):
    memory_store = DirStore(tmp_path / "memory", query_index="memory")
    none_store = DirStore(tmp_path / "none", query_index="none")
    sqlite_store = DirStore(tmp_path / "sqlite", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[memory_store, none_store, sqlite_store])

    statuses = {status.store_key: status for status in repo.index_status()}

    assert statuses[memory_store.catalog_key()].backend == "memory"
    assert statuses[memory_store.catalog_key()].state == "ready"
    assert statuses[none_store.catalog_key()].backend == "none"
    assert statuses[none_store.catalog_key()].state == "disabled"
    assert statuses[sqlite_store.catalog_key()].backend == "sqlite"
    assert statuses[sqlite_store.catalog_key()].state == "missing"
    assert not Path(sqlite_store.query_index_path).exists()


def test_index_status_can_filter_one_store(tmp_path):
    store1 = DirStore(tmp_path / "store1", query_index="memory")
    store2 = DirStore(tmp_path / "store2", query_index="none")
    repo = Repo(stores=[store1, store2])

    statuses = repo.index_status(store=store2)

    assert len(statuses) == 1
    assert statuses[0].store_key == store2.catalog_key()


def test_save_registration_fans_out_to_custom_store_index_after_publish(tmp_path):
    opened = []

    def factory(store):
        index = RecordingIndex(store)
        opened.append(index)
        return index

    store = DirStore(tmp_path / "store", query_index=factory)
    repo = Repo(stores=store)
    obj = FederationLeaf("saved", repo=repo)

    repo.save_object(obj)

    assert len(opened) == 1
    assert len(opened[0].registered) == 1
    _, roots = opened[0].registered[0]
    assert obj.definition in roots


def test_save_registration_failure_marks_query_index_dirty(tmp_path):
    store = DirStore(tmp_path / "store", query_index=lambda store: FailingIndex())
    repo = Repo(stores=store)
    obj = FederationLeaf("dirty", repo=repo)

    import pytest

    with pytest.raises(RuntimeError, match="index failed"):
        repo.save_object(obj)

    assert store.has(obj.definition)
    assert store.query_index_is_dirty()


def test_save_registration_updates_sqlite_store_index(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = FederationLeaf("sqlite-save", repo=repo)

    repo.save_object(obj)

    statuses = repo.index_status(store=store)
    assert statuses[0].backend == "sqlite"
    assert statuses[0].state == "ready"
    assert statuses[0].generation == 1
    assert Path(store.query_index_path).exists()
    index = store.open_query_index()
    with index.read_view() as view:
        assert view.cdef_id(obj.definition) in view.all_stored_ids()


def test_repo_close_closes_opened_store_indexes(tmp_path):
    opened = []

    def factory(store):
        index = RecordingIndex(store)
        opened.append(index)
        return index

    store = DirStore(tmp_path / "store", query_index=factory)
    repo = Repo(stores=store)

    repo.index_status()
    repo.close(flush=False)

    assert opened and opened[0].closed


def test_sqlite_federated_stored_query_uses_sidecar_without_hydration(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    wanted = FederationParent(child=FederationLeaf(name="wanted", repo=repo), name="root", repo=repo)
    other = FederationParent(child=FederationLeaf(name="other", repo=repo), name="root", repo=repo)
    repo.save_object(wanted)
    repo.save_object(other)

    reopened_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo2 = Repo(stores=reopened_store)

    def fail_hydrate():
        raise AssertionError("federated SQLite query should not hydrate the Store")

    reopened_store.hydrate_index = fail_hydrate
    selector = Definition(FederationParent, SKIP_ARGS, child=Definition(FederationLeaf, SKIP_ARGS, name="wanted"))

    results = repo2.query(selector).stored().defs()

    assert list(results) == [wanted.definition]
    assert results.replicas(wanted.definition) == (reopened_store,)
    assert results.explanation.refresh_action == "federated"


def test_sqlite_federated_multistore_dedup_and_replica_priority(tmp_path):
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    obj = FederationLeaf(name="shared", repo=repo)
    repo.save_object(obj, store=store1)
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[
        DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
    ])
    selector = Definition(FederationLeaf, SKIP_ARGS, name="shared")

    results = repo2.query(selector).stored().defs()

    assert list(results) == [obj.definition]
    assert tuple(store.base_dir for store in results.replicas(obj.definition)) == (store2.base_dir, store1.base_dir)


def test_sqlite_federated_nested_definitions_owners_and_occurrences(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    leaf = FederationLeaf(name="nested", repo=repo)
    owner = FederationParent(child=leaf, name="owner", repo=repo)
    repo.save_object(owner)

    repo2_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo2 = Repo(stores=repo2_store)
    selector = Definition(FederationLeaf, SKIP_ARGS, name="nested")

    definitions = repo2.query(selector).nested().definitions().defs()
    owners = repo2.query(selector).nested().owners().defs()
    occurrences = tuple(repo2.query(selector).nested().max_occurrences(10).execute())

    assert list(definitions) == [leaf.definition]
    assert list(owners) == [owner.definition]
    assert owners.replicas(owner.definition) == (repo2_store,)
    assert len(occurrences) == 1
    assert occurrences[0].owner == owner.definition
    assert occurrences[0].definition == leaf.definition
    assert str(occurrences[0].path) == "$.child"
