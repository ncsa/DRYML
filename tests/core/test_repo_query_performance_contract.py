from dryml.core2 import Definition, Object, Repo, Serializable, SKIP_ARGS
from dryml.core2.store.dir import DirStore
from dryml.core2.store.store import Store


class PerfLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class PerfParent(Serializable):
    def __init__(self, child):
        super().__init__()
        self.child = child


class CountingStore(Store):
    def __init__(self, store):
        self.store = store
        self.has_count = 0
        self.hydrate_count = 0
        self.restore_count = 0

    @property
    def base_dir(self):
        return self.store.base_dir

    @property
    def object_root_dir(self):
        return self.store.object_root_dir

    def has(self, cdef):
        self.has_count += 1
        return self.store.has(cdef)

    def hydrate_index(self):
        self.hydrate_count += 1
        return self.store.hydrate_index()

    def _object_dir(self, cdef):
        return self.store.object_dir(cdef)

    def save_object(self, obj, *, revision=None):
        return self.store.save_object(obj, revision=revision)

    def restore_object(self, obj, *, revision=None):
        self.restore_count += 1
        return self.store.restore_object(obj, revision=revision)

    def read_main_def(self):
        return self.store.read_main_def()

    def set_main_def(self, main_def):
        return self.store.set_main_def(main_def)

    def write_main_def(self, main_def):
        return self.store.write_main_def(main_def)

    def read_aliases(self):
        return self.store.read_aliases()

    def write_aliases(self, aliases):
        return self.store.write_aliases(aliases)

    def commit(self):
        return self.store.commit()


def test_first_auto_broad_query_hydrates_once_and_second_reuses(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    repo.save_object(PerfLeaf("x", repo=repo))

    counting = CountingStore(DirStore(store.base_dir))
    repo2 = Repo(stores=counting)

    assert repo2.find_defs(None).count() == 1
    assert counting.hydrate_count == 1
    assert repo2.find_defs(None).count() == 1
    assert counting.hydrate_count == 1


def test_refresh_false_never_hydrates(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    repo.save_object(PerfLeaf("x", repo=repo))

    counting = CountingStore(DirStore(store.base_dir))
    repo2 = Repo(stores=counting)

    assert repo2.find_defs(None, refresh=False).count() == 0
    assert counting.hydrate_count == 0


def test_definition_count_and_explain_do_not_materialize(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    repo.save_object(PerfLeaf("x", repo=repo))

    counting = CountingStore(DirStore(store.base_dir))
    repo2 = Repo(stores=counting)

    assert repo2.query(None).stored().count() == 1
    repo2.query(None).stored().explain()

    assert repo2._num_constructions == 0
    assert counting.restore_count == 0


def test_exact_root_lookup_avoids_broad_hydration(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = PerfLeaf("x", repo=repo)
    repo.save_object(obj)

    counting = CountingStore(DirStore(store.base_dir))
    repo2 = Repo(stores=counting)

    assert repo2.query(obj.definition).stored().count() == 1
    assert counting.has_count >= 1
    assert counting.hydrate_count == 0


def test_nested_owner_query_does_not_materialize(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = PerfLeaf("child", repo=repo)
    parent = PerfParent(child, repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    owners = repo2.query(Definition(PerfLeaf, SKIP_ARGS)).nested().owners().defs()

    assert owners.count() == 1
    assert repo2._num_constructions == 0


def test_successful_save_incrementally_registers_without_rehydrate(tmp_path):
    counting = CountingStore(DirStore(tmp_path / "store"))
    repo = Repo(stores=counting)
    obj = PerfLeaf("x", repo=repo)
    repo.save_object(obj)

    assert repo.find_defs(None, scope="stored", refresh=False).count() == 1
    assert counting.hydrate_count == 0
