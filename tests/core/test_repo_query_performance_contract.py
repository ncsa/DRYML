import pytest

from dryml.core import Definition, Object, Repo, Serializable, SKIP_ARGS
from dryml.core.query import EdgePolicy
from dryml.core.query.sqlite.index import SQLiteQueryIndexReadView
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord

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


def test_equivalent_broad_v3_workload_captures_each_query_cut_once(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    repo.save_object(PerfLeaf("x", repo=repo))

    counting = CountingStore(DirStore(store.base_dir, query_index="memory"))
    repo2 = Repo(stores=counting)

    assert repo2.query().cdefs().stored().count() == 1
    assert counting.hydrate_count == 1
    assert repo2.query().cdefs().stored().count() == 1
    assert counting.hydrate_count == 2


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


def test_exact_root_lookup_avoids_broad_hydration(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = PerfLeaf("x", repo=repo)
    repo.save_object(obj)

    counting = CountingStore(DirStore(store.base_dir, query_index="memory"))
    repo2 = Repo(stores=counting)

    assert repo2.query().sel(obj.definition).cdefs().stored().count() == 1
    assert counting.definition_read_count >= 1
    assert counting.hydrate_count == 0


def test_exact_v3_query_reads_only_direct_definition_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = PerfLeaf("x", repo=repo)
    repo.save_object(obj)

    counting = CountingStore(DirStore(store.base_dir))
    repo2 = Repo(stores=counting)

    assert repo2.query().sel(obj.definition).cdefs().stored().count() == 1
    assert counting.definition_read_count >= 1
    assert counting.hydrate_count == 0
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

    assert repo.query().cdefs().stored().count() == 1
    assert counting.hydrate_count == 1


def test_v3_multistore_take_uses_only_canonical_index_frontier_pages(
        tmp_path, monkeypatch):
    """Duplicate source frontiers merge without broad authority enumeration."""

    first = DirStore(tmp_path / "first", query_index="sqlite")
    second = DirStore(tmp_path / "second", query_index="sqlite")
    values = sorted(
        (Definition(PerfLeaf, f"value-{index}").concretize() for index in range(6)),
        key=lambda value: value.graph_hash(),
    )
    for value in values[:4]:
        first.write_definition_record(DefinitionRecord(value))
    for value in (*values[:3], values[-1]):
        second.write_definition_record(DefinitionRecord(value))
    first.rebuild_query_index()
    second.rebuild_query_index()
    monkeypatch.setattr(
        first,
        "iter_definition_records",
        lambda: pytest.fail("bounded query scanned first Store authority"),
    )
    monkeypatch.setattr(
        second,
        "iter_definition_records",
        lambda: pytest.fail("bounded query scanned second Store authority"),
    )
    pages = 0
    rows = 0
    original = SQLiteQueryIndexReadView.iter_stored_identity_cdef_batches

    def counted(self, **kwargs):
        nonlocal pages, rows
        for batch in original(self, **kwargs):
            pages += 1
            rows += len(batch.cdefs)
            yield batch

    monkeypatch.setattr(
        SQLiteQueryIndexReadView,
        "iter_stored_identity_cdef_batches",
        counted,
    )

    result = Repo((first, second)).query().cdefs().stored().require_indexed().take(3)

    assert tuple(result) == tuple(values[:3])
    assert result.bounded and result.requested_limit == 3
    assert result.diagnostic().pages_fetched == 2
    assert result.diagnostic().candidate_rows_read == 8
    assert result.diagnostic().cdef_blobs_decoded == 8
    assert pages == 2
    assert rows == 8

    refined = result.query().sel(values[0]).collect()
    combined = result.union(refined)
    assert refined.diagnostic().candidate_rows_read == 8
    assert refined.diagnostic().cdef_blobs_decoded == 8
    assert refined.diagnostic().pages_fetched == 2
    assert combined.diagnostic().candidate_rows_read == 8
    assert combined.diagnostic().cdef_blobs_decoded == 8
    assert combined.diagnostic().pages_fetched == 2
