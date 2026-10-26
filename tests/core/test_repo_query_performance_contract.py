import pytest

from dryml.core import Definition, Object, Repo, Serializable, SKIP_ARGS
from dryml.core.query import EdgePolicy
from dryml.core.store.dir import DirStore

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


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
        self.restore_count = 0

    def read_node(self, digest, *, kind="recipe"):
        self.definition_read_count += 1
        return super().read_node(digest, kind=kind)

    def catalog_key(self):
        return f"{DirStore.__module__}.{DirStore.__qualname__}:{self.base_dir}"

    def read_states(self, *args, **kwargs):
        self.restore_count += 1
        return super().read_states(*args, **kwargs)


def test_equivalent_broad_v3_workload_captures_each_query_cut_once(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    repo.save_object(PerfLeaf("x", repo=repo))

    counting = CountingStore(DirStore(store.base_dir, query_index="memory"))
    repo2 = Repo(stores=counting)

    for _ in range(2):
        stats = repo2.query().cdefs().stored().explain(analyze=True)
        assert stats.result_count == stats.store_scan_count == 1


def test_u2_completed_exact_positive_has_no_maintenance_inventory(tmp_path, monkeypatch):
    from dryml.core.query.sqlite import SQLiteQueryIndexConfig

    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    root = Definition(PerfLeaf, "root").concretize()
    store.publish_node(root)
    store.rebuild_query_index()
    counting = CountingStore(store)
    query = counting.query().refresh(True)
    query.count()
    counting.definition_read_count = 0
    monkeypatch.setattr(counting.open_query_index(), "refresh", lambda *a, **kw: pytest.fail("completed request refreshed"))
    stats = query.sel(root).cdefs().stored().explain(analyze=True)
    assert stats.result_count == 1 and stats.store_scan_count == 0
    assert stats.cdef_blobs_decoded == 1 and counting.definition_read_count == 0


def test_v3_broad_query_does_not_restore_payloads(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    repo.save_object(PerfLeaf("x", repo=repo))

    counting = CountingStore(DirStore(store.base_dir))
    repo2 = Repo(stores=counting)

    assert repo2.query().cdefs().stored().count() == 1
    assert counting.restore_count == 0


def test_definition_count_and_explain_do_not_materialize(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    repo.save_object(PerfLeaf("x", repo=repo))

    counting = CountingStore(DirStore(store.base_dir))
    repo2 = Repo(stores=counting)

    assert repo2.query().cdefs().stored().count() == 1
    repo2.query().cdefs().stored().explain()

    assert repo2._num_constructions == 0
    assert counting.restore_count == 0


def test_indexed_concrete_recipe_lookup_avoids_broad_hydration(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(stores=store)
    obj = PerfLeaf("x", repo=repo)
    repo.save_object(obj)
    for i in range(4):
        store.publish_node(Definition(PerfLeaf, "unrelated-" + str(i)).concretize())

    counting = CountingStore(DirStore(store.base_dir, query_index="sqlite"))
    repo2 = Repo(stores=counting)

    stats = repo2.query().sel(obj.definition).cdefs().stored().refresh(False).require_indexed().explain(analyze=True)
    assert stats.result_count == 1 and stats.cdef_blobs_decoded == 1 and stats.store_scan_count == 0
    assert stats.candidate_rows_read == 1
    assert counting.definition_read_count == 0


def test_exact_v3_query_reads_only_direct_definition_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = PerfLeaf("x", repo=repo)
    repo.save_object(obj)

    counting = CountingStore(DirStore(store.base_dir))
    repo2 = Repo(stores=counting)

    stats = repo2.query().sel(obj.definition).cdefs().stored().explain(analyze=True)
    assert stats.result_count == 1 and stats.cdef_blobs_decoded == 1 and stats.store_scan_count == 0
    assert counting.definition_read_count == 0
    assert counting.restore_count == 0


def test_nested_owner_query_does_not_materialize(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = PerfLeaf("child", repo=repo)
    parent = PerfParent(child, repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    owners = (
        repo2.query()
        .cdefs()
        .stored()
        .nested(Definition(PerfLeaf, SKIP_ARGS), edges=EdgePolicy.ALL)
        .owners()
        .cdefs()
    )

    assert owners.count() == 1
    assert repo2._num_constructions == 0


def test_v3_broad_save_query_records_its_expanded_inventory_cost(tmp_path):
    counting = CountingStore(DirStore(tmp_path / "store"))
    repo = Repo(stores=counting)
    obj = PerfLeaf("x", repo=repo)
    repo.save_object(obj)

    stats = repo.query().cdefs().stored().explain(analyze=True)
    assert stats.result_count == 1 and stats.store_scan_count == 0


def test_v3_multistore_take_uses_only_canonical_index_frontier_pages(
        tmp_path, monkeypatch):
    """Duplicate source frontiers merge without broad authority enumeration."""

    first = DirStore(tmp_path / "first", query_index="sqlite")
    second = DirStore(tmp_path / "second", query_index="sqlite")
    values = sorted(
        (Definition(PerfLeaf, f"value-{index}").concretize() for index in range(6)),
        key=lambda value: value.exact_digest(),
    )
    for value in values[:4]:
        first.publish_node(value)
    for value in (*values[:3], values[-1]):
        second.publish_node(value)
    first.rebuild_query_index()
    second.rebuild_query_index()
    monkeypatch.setattr(
        "dryml.core.query.authority.capture_store_projection",
        lambda *a, **kw: pytest.fail("bounded query scanned Store authority"),
    )
    result = Repo((first, second)).query().cdefs().stored().require_indexed().max_verify(3).take(3)

    assert all(a.graph_equal(b) for a, b in zip(result, values[:3]))
    assert result.bounded and result.requested_limit == 3
    assert result.diagnostic().pages_fetched == 2
    assert result.diagnostic().candidate_rows_read == 8
    assert result.diagnostic().cdef_blobs_decoded == 8
    assert all(len(result.sources(value)) == 2 for value in values[:3])

    refined = result.query().sel(values[0]).collect()
    combined = result.union(refined)
    assert refined.diagnostic().candidate_rows_read == 8
    assert refined.diagnostic().cdef_blobs_decoded == 8
    assert refined.diagnostic().pages_fetched == 2
    assert combined.diagnostic().candidate_rows_read == 8
    assert combined.diagnostic().cdef_blobs_decoded == 8
    assert combined.diagnostic().pages_fetched == 2


@pytest.mark.parametrize("root_count", (10, 30, 100))
def test_reference_validation_visits_roots_linearly_per_selected_cut(
        tmp_path, monkeypatch, root_count):
    """Candidate filtering does not repeatedly validate the selected root set."""

    from dryml.core import ObjectId
    from dryml.core.query import source

    store = DirStore(tmp_path / "store", query_index="sqlite")
    for index in range(root_count):
        store.publish_node(Definition(PerfParent, index).concretize().with_identity(ObjectId(("linear",))))
    visits = 0
    from dryml.core.store.records import DeclarationAnchorRecord
    original_read = source._NodePage.read
    def read(page, selected, path, reader, *args, **kwargs):
        nonlocal visits
        if reader is DeclarationAnchorRecord:
            visits += 1
        return original_read(page, selected, path, reader, *args, **kwargs)
    monkeypatch.setattr(source._NodePage, "read", read)
    query = Repo(store).query().object_refs().stored()
    assert query.count() == root_count
    assert visits == root_count
    assert query.count() == root_count
    assert visits == 2 * root_count
