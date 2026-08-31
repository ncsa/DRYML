from dryml.core import Definition, Object, Repo, Serializable, SKIP_ARGS
from dryml.core.store.dir import DirStore


class PerfLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class PerfParent(Serializable):
    def __init__(self, child):
        super().__init__()
        self.child = child


class CountingStore(DirStore):
    def __init__(self, store):
        super().__init__(store.base_dir, query_index=store.query_index)
        self.definition_read_count = 0
        self.hydrate_count = 0
        self.restore_count = 0

    def read_definition_record(self, digest):
        self.definition_read_count += 1
        return super().read_definition_record(digest)

    def catalog_key(self):
        return f"{DirStore.__module__}.{DirStore.__qualname__}:{self.base_dir}"

    def iter_definition_records(self):
        self.hydrate_count += 1
        return super().iter_definition_records()

    def open_local_state(self, graph_hash, state_hash):
        self.restore_count += 1
        return super().open_local_state(graph_hash, state_hash)


def test_first_auto_broad_query_hydrates_once_and_second_reuses(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    repo.save_object(PerfLeaf("x", repo=repo))

    counting = CountingStore(DirStore(store.base_dir, query_index="memory"))
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

    assert repo2.find_defs(None, refresh=False).count() == 1
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
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = PerfLeaf("x", repo=repo)
    repo.save_object(obj)

    counting = CountingStore(DirStore(store.base_dir, query_index="memory"))
    repo2 = Repo(stores=counting)

    assert repo2.query(obj.definition).stored().count() == 1
    assert counting.definition_read_count >= 1
    assert counting.hydrate_count == 0


def test_exact_query_refresh_false_uses_ready_sidecar_without_store_io(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = PerfLeaf("x", repo=repo)
    repo.save_object(obj)

    counting = CountingStore(DirStore(store.base_dir))
    repo2 = Repo(stores=counting)

    assert repo2.query(obj.definition).stored(refresh=False).count() == 1
    assert counting.definition_read_count == 0
    assert counting.hydrate_count == 0
    assert counting.restore_count == 0


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
