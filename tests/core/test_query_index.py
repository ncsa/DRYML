import pytest

from dryml.core2 import Definition, Object, Repo, Serializable, SKIP_ARGS
from dryml.core2.query import QueryIndexError
from dryml.core2.store.dir import DirStore
from dryml.core2.store.store import Store
from dryml.core2.utils.general import pickle_load, pickle_save


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

    with pytest.raises(QueryIndexError):
        repo.find_defs(None, refresh=True)

    assert list(repo.find_defs(None, refresh=False)) == [good.definition]


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
    assert str(occurrences.one().path) == "$.args[0][0]"
