from dryml.core import Object, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.utils.general import pickle_load, pickle_save


class MultiLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class MultiState(Serializable):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.value = 0

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        pickle_save(self.value, f"{dest_dir}/value.pkl")

    def restore_state_from_dir_imp(self, src_dir, revision=None):
        self.value = pickle_load(f"{src_dir}/value.pkl")


class CountingDirStore(DirStore):
    calls = 0

    def hydrate_index(self):
        type(self).calls += 1
        return super().hydrate_index()


def test_distinct_cdefs_in_two_stores_produce_two_logical_results(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    first = MultiLeaf("first", repo=repo)
    second = MultiLeaf("second", repo=repo)
    repo.save_object(first, store=store1)
    repo.save_object(second, store=store2)

    repo2 = Repo(stores=[DirStore(store1.base_dir), DirStore(store2.base_dir)])

    assert repo2.find_defs(None).count() == 2


def test_same_cdef_in_two_stores_deduplicates_and_tracks_replicas(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    obj = MultiLeaf("same", repo=repo)
    repo.save_object(obj, store=store1)
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[DirStore(store1.base_dir), DirStore(store2.base_dir)])
    results = repo2.find_defs(None)

    assert list(results) == [obj.definition]
    assert len(results.replicas(obj.definition)) == 2


def test_equivalent_store_instances_share_physical_identity(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = MultiLeaf("same", repo=repo)
    repo.save_object(obj)

    repo2 = Repo(stores=[DirStore(store.base_dir), DirStore(store.base_dir)])
    results = repo2.find_defs(None)

    assert list(results) == [obj.definition]
    assert len(results.replicas(obj.definition)) == 1


def test_forced_refresh_deduplicates_equivalent_store_instances(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = MultiLeaf("same", repo=repo)
    repo.save_object(obj)

    CountingDirStore.calls = 0
    repo2 = Repo(stores=[CountingDirStore(store.base_dir), CountingDirStore(store.base_dir)])

    assert repo2.find_defs(None, refresh=True).count() == 1
    assert CountingDirStore.calls == 1


def test_forced_refresh_replaces_stale_store_reference(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = MultiLeaf("same", repo=repo)
    repo.save_object(obj)

    old_store = DirStore(store.base_dir)
    repo2 = Repo(stores=old_store)
    assert repo2.find_defs(None).count() == 1

    new_store = DirStore(store.base_dir)
    repo2.stores = [new_store]
    refreshed = repo2.find_defs(None, refresh=True)

    assert refreshed.replicas(obj.definition) == (new_store,)


def test_materialization_uses_store_priority_for_replicated_cdef(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    obj = MultiState("same", repo=repo)
    obj.value = 1
    repo.save_object(obj, store=store1)
    obj.value = 2
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[DirStore(store2.base_dir), DirStore(store1.base_dir)])
    loaded = repo2.query(obj.definition).stored().objects().one()

    assert loaded.value == 2


def test_result_order_is_independent_of_store_order(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    first = MultiLeaf("first", repo=repo)
    second = MultiLeaf("second", repo=repo)
    repo.save_object(first, store=store1)
    repo.save_object(second, store=store2)

    order_a = list(Repo(stores=[DirStore(store1.base_dir), DirStore(store2.base_dir)]).find_defs(None))
    order_b = list(Repo(stores=[DirStore(store2.base_dir), DirStore(store1.base_dir)]).find_defs(None))

    assert order_a == order_b
