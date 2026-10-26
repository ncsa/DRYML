import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Definition, Object, Repo, SKIP_ARGS
from dryml.core.store.dir import DirStore


class QueryRoot(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class CountingStore(DirStore):
    def __init__(self, store):
        super().__init__(store.base_dir, query_index=store.query_index)
        self.definition_read_count = 0
        self.restore_count = 0

    def read_node(self, digest, *, kind="recipe"):
        self.definition_read_count += 1
        return super().read_node(digest, kind=kind)

    def catalog_key(self):
        return f"{DirStore.__module__}.{DirStore.__qualname__}:{self.base_dir}"

    def read_states(self, *args, **kwargs):
        self.restore_count += 1
        return super().read_states(*args, **kwargs)


def test_identity_query_does_not_materialize_until_explicit_repo_load(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = QueryRoot("saved", repo=repo)
    repo.save_object(obj)

    counting_store = CountingStore(DirStore(store.base_dir, query_index="memory"))
    repo2 = Repo(stores=counting_store)

    defs = repo2.query().sel(Definition(QueryRoot, SKIP_ARGS)).cdefs().stored().collect()
    assert defs.count() == 1
    assert repo2._num_constructions == 0
    assert counting_store.restore_count == 0

    loaded = repo2.load_object(defs.one())
    assert loaded.name == "saved"
    assert repo2._num_constructions == 1


def test_exact_root_query_reads_definition_record_without_full_hydration(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = QueryRoot("saved", repo=repo)
    repo.save_object(obj)

    counting_store = CountingStore(DirStore(store.base_dir, query_index="memory"))
    repo2 = Repo(stores=counting_store)

    defs = repo2.query().sel(obj.definition).cdefs().stored().collect()

    assert len(defs) == 1 and next(iter(defs)).graph_equal(obj.definition)
    stats = repo2.query().sel(obj.definition).cdefs().stored().explain(analyze=True)
    assert stats.cdef_blobs_decoded == 1
    assert counting_store.definition_read_count == 0
    assert stats.store_scan_count == 1


def test_query_has_a_producer_knowledge_domain_without_domain_selection():
    repo = Repo()
    assert repo.query().cdefs().count() == 0


def test_query_identities_are_loaded_explicitly_by_the_repo(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = QueryRoot("saved", repo=repo)
    repo.save_object(obj)

    repo2 = Repo(stores=DirStore(store.base_dir))
    result = repo2.query().sel(Definition(QueryRoot, SKIP_ARGS)).cdefs().stored().collect()
    loaded = repo2.load_object(result.one())

    assert loaded.definition.graph_equal(obj.definition)
    assert loaded.name == "saved"
