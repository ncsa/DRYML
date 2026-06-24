import pytest
import shutil

from dryml.core2 import Definition, Object, Repo, Serializable, SKIP_ARGS
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.query import QueryIndexError, SetMember
from dryml.core2.query.path import get_subtree
from dryml.core2.store.dir import DirStore
from dryml.core2.store.store import Store
from dryml.core2.utils.general import pickle_load, pickle_save
from dryml.core2.utils.stable_hash import stable_hash_function


class IndexLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class IndexPersistent(Serializable):
    def __init__(self, child, *, state=0):
        super().__init__()
        self.child = child
        self.state = state

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        pickle_save(self.state, f"{dest_dir}/state.pkl")

    def restore_state_from_dir_imp(self, src_dir, revision=None):
        self.state = pickle_load(f"{src_dir}/state.pkl")


class BadIndexStore(Store):
    def __init__(self, values):
        self.values = values

    @property
    def base_dir(self):
        return "bad-index-store"

    @property
    def object_root_dir(self):
        return "bad-index-store/objects"

    def has(self, cdef):
        return False

    def hydrate_index(self):
        return tuple(self.values)

    def _object_dir(self, cdef):
        return "bad-index-store/objects/missing"

    def commit(self):
        pass


def test_catalog_registering_same_definition_is_idempotent():
    repo = Repo()
    cdef = IndexLeaf("x").definition

    first = repo._query_catalog.register_cached(cdef)
    second = repo._query_catalog.register_cached(cdef)

    assert first == second
    assert len(repo._query_catalog.definitions_by_id) == 1
    assert all(list(posting).count(first) == 1 for posting in repo._query_catalog.postings.values())


def test_cached_only_definition_is_known_not_stored_then_save_promotes(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = IndexLeaf("cached", repo=repo)
    repo.add_objects(obj)

    assert list(repo.find_defs(None, scope="cached", refresh=False)) == [obj.definition]
    assert len(repo.find_defs(None, scope="stored", refresh=False)) == 0

    repo.save_object(obj)

    assert list(repo.find_defs(None, scope="stored", refresh=False)) == [obj.definition]


def test_same_cdef_in_two_stores_has_one_record_and_two_replicas(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    obj = IndexPersistent(IndexLeaf("x", repo=repo), repo=repo)
    repo.save_object(obj, store=store1)
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[DirStore(store1.base_dir), DirStore(store2.base_dir)])
    results = repo2.find_defs(None)

    assert list(results) == [obj.definition]
    assert len(results.replicas(obj.definition)) == 2


def test_graph_registration_records_direct_edges_once():
    repo = Repo()
    child = IndexLeaf("child", repo=repo)
    parent = IndexPersistent(child, repo=repo)

    repo._query_catalog.register_cached(parent.definition)
    repo._query_catalog.register_cached(parent.definition)

    catalog = repo._query_catalog
    parent_id = catalog.cdef_id(parent.definition)
    child_id = catalog.cdef_id(child.definition)

    assert len(catalog.edge_by_key) == 1
    edge = next(iter(catalog.edge_by_key.values()))
    assert edge.parent_id == parent_id
    assert edge.child_id == child_id
    assert str(edge.path) == "$.args[0]"


def test_shared_child_local_fingerprint_compiled_once(monkeypatch):
    from dryml.core2.query import index as index_mod

    repo = Repo()
    child_cdef = ConcreteDefinition(IndexLeaf, FrozenTuple(("shared",)), FrozenDict({}))
    left_cdef = ConcreteDefinition(IndexPersistent, FrozenTuple((child_cdef,)), FrozenDict({"state": 1}))
    right_cdef = ConcreteDefinition(IndexPersistent, FrozenTuple((child_cdef,)), FrozenDict({"state": 2}))
    calls = []
    original = index_mod.target_local_fingerprint

    def spy(cdef):
        calls.append(cdef)
        return original(cdef)

    monkeypatch.setattr(index_mod, "target_local_fingerprint", spy)

    repo._query_catalog.register_cached(left_cdef)
    repo._query_catalog.register_cached(right_cdef)

    assert calls.count(child_cdef) == 1


def test_parent_local_postings_do_not_contain_child_interior_features():
    repo = Repo()
    child = IndexLeaf("needle", repo=repo)
    parent = IndexPersistent(child, repo=repo)

    repo._query_catalog.register_cached(parent.definition)
    parent_id = repo._query_catalog.cdef_id(parent.definition)
    parent_record = repo._query_catalog.definitions_by_id[parent_id]
    child_name_hash = stable_hash_function(child.definition.args[0])

    assert all(
        not (token.kind == "SCALAR_VALUE" and token.payload == child_name_hash)
        for token in parent_record.local_fingerprint.counts
    )


def test_repeated_nested_cdef_keeps_every_owner_path_occurrence(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("shared", repo=repo)
    parent = IndexPersistent([child, child], repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    occurrences = repo2.find_occurrences(Definition(IndexLeaf, SKIP_ARGS))

    assert occurrences.count() == 2
    assert {str(occ.path) for occ in occurrences} == {"$.args[0][0]", "$.args[0][1]"}
    assert occurrences.definitions().count() == 1


def test_refresh_failure_rolls_back_catalog_atomically(tmp_path):
    good_store = DirStore(tmp_path / "good")
    repo = Repo(stores=good_store)
    good = IndexLeaf("good", repo=repo)
    repo.save_object(good)

    old_results = repo.find_defs(None)
    assert list(old_results) == [good.definition]

    repo.add_store(BadIndexStore([good.definition, "not-a-cdef"]))
    before = catalog_state(repo)

    with pytest.raises(QueryIndexError):
        repo.find_defs(None, refresh=True)

    assert catalog_state(repo) == before
    assert list(repo.find_defs(None, refresh=False)) == [good.definition]

    repo.stores = [good_store]
    assert list(repo.find_defs(None, refresh=True)) == [good.definition]


def test_nested_definition_inside_set_is_indexed_with_defined_path(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("set-child", repo=repo)
    parent = IndexPersistent({child}, repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    occurrences = repo2.find_occurrences(Definition(IndexLeaf, SKIP_ARGS))

    assert occurrences.count() == 1
    assert occurrences.one().definition == child.definition
    assert isinstance(occurrences.one().path[-1], SetMember)
    assert get_subtree(parent.definition, occurrences.one().path) == child.definition


def test_forced_refresh_removes_deleted_root_and_occurrences(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("child", repo=repo)
    parent = IndexPersistent(child, repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    assert repo2.find_defs(None).count() == 1
    assert repo2.find_occurrences(Definition(IndexLeaf, SKIP_ARGS)).count() == 1

    shutil.rmtree(store.object_dir(parent.definition))

    assert repo2.find_defs(None, refresh=True).count() == 0
    assert repo2.find_occurrences(Definition(IndexLeaf, SKIP_ARGS), refresh=False).count() == 0


def test_exact_store_probe_confirms_persisted_definition_after_hash_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(ConcreteDefinition, "stable_hash", lambda self: "collision")
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    stored = IndexLeaf("stored", repo=repo)
    queried = IndexLeaf("queried", repo=repo)
    repo.save_object(stored)

    repo2 = Repo(stores=DirStore(store.base_dir))

    assert repo2.query(queried.definition).stored().count() == 0
    assert repo2.query(stored.definition).stored().count() == 1


def catalog_state(repo):
    catalog = repo._query_catalog
    return {
        "definitions": tuple(sorted(catalog.definitions_by_id.keys())),
        "replicas": tuple(sorted((k, tuple(sorted(v))) for k, v in catalog.replicas_by_definition.items())),
        "stored_by_store": tuple(sorted((k, tuple(sorted(v))) for k, v in catalog.stored_definitions_by_store.items())),
        "occurrences": tuple(sorted((k[0], str(k[1]), k[2]) for k in catalog.occurrence_by_key.keys())),
        "postings": tuple(sorted((repr(k), tuple(sorted(v.items()))) for k, v in catalog.postings.items())),
        "hydrated": tuple(sorted(catalog.hydrated_stores)),
        "store_by_id": tuple(sorted(catalog.store_by_id.keys())),
        "light_index": tuple(sorted(cdef.stable_hash() for cdef in repo.light_index)),
        "generation": catalog.generation,
    }
