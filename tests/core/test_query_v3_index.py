"""Current Store query coverage and retained standalone SQLite contracts.

The former Store-3 tests exercised DefinitionRecord/snapshot-JSON, alias-only
root promotion and a fence spanning all pages. Those layouts/protocols are no
longer supported. These ports retain their authority, membership, metadata,
recovery, admission, interruption and continuation assertions against Store 4.
Standalone legacy SQLite delta/cursor tests remain live at the bottom.
"""

from pathlib import Path
import sqlite3

import pytest

from dryml.core import Definition, Object, Repo, SaveAnnotations, Serializable
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.query import field
from dryml.core.query.model import (
    QueryError, QueryIndexDirty, QueryIndexIncompatible, QueryIndexUnavailable,
    QueryVerifyBudgetExceeded, QueryWouldScanError, QueryStats,
)
from dryml.core.query.sqlite import SQLiteQueryIndexConfig, sqlite_available
from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core.store.dir import DirStore
from dryml.core.store.records import NodeNameRecord
from dryml.core.store.store import StoreAuthorityError
from dryml.core.store.zip import ZipStore

pytestmark = [pytest.mark.usefixtures("fixed_snapshot_environment"),
              pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")]


class ProjectionLeaf(Object):
    def __init__(self, name="leaf"):
        self.name = name


class ProjectionPair(Object):
    def __init__(self, left, right):
        self.left, self.right = left, right


class ProjectionStateful(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="utf-8")


def _authority_bytes(store):
    return {str(path.relative_to(store.base_dir)): path.read_bytes()
            for path in Path(store.base_dir).rglob("*")
            if path.is_file() and ".dryml" not in path.parts and path.name.endswith((".record", ".pkl"))}


def _ready(tmp_path, count=3):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    values = sorted((Definition(ProjectionLeaf, str(i)).concretize() for i in range(count)),
                    key=lambda value: value.exact_digest())
    records = [store.publish_node(value) for value in values]
    return store, values, records


def test_explicit_cycle_preserves_graph_distinct_roots_and_current_main(tmp_path):
    leaf = Definition(ProjectionLeaf, "same").concretize()
    shared = Definition(ProjectionPair, leaf, leaf).concretize()
    separate = Definition(ProjectionPair, leaf, leaf.copy_graph()).concretize()
    assert not shared.graph_equal(separate)
    store = DirStore(tmp_path / "store", query_index="sqlite")
    roots = [store.publish_node(value) for value in (shared, separate)]
    store.compare_and_set_node_name(NodeNameRecord("main", "recipe", roots[1].digest), expected=None)
    index = store.open_query_index()
    generation, before = index.status().generation, _authority_bytes(store)
    stats = QueryStats()
    index.refresh(True, stats=stats)
    assert stats.store_scan_count == 0 and stats.refresh_action == "sqlite-validate"
    assert index.status().generation == generation
    results = store.query().cdefs().stored().refresh(False).require_indexed().collect()
    assert len(results) == 2 and all(any(value.graph_equal(expected) for value in results) for expected in (shared, separate))
    assert _authority_bytes(store) == before


def test_immutable_root_collision_refuses_without_replacing_authority(tmp_path, monkeypatch):
    from dryml.core.store.records import NodeRootRecord
    from dryml.core.store.authority import StoreMutationError
    monkeypatch.setattr(NodeRootRecord, "digest", property(lambda self: "c" * 64))
    store = DirStore(tmp_path / "store", query_index="none")
    store.publish_node(Definition(ProjectionLeaf, "first").concretize())
    before = _authority_bytes(store)
    with pytest.raises((StoreAuthorityError, StoreMutationError), match="collision|conflict"):
        store.publish_node(Definition(ProjectionLeaf, "different").concretize())
    assert _authority_bytes(store) == before


@pytest.mark.parametrize("family", ["node", "edge", "facet", "discovery", "name", "requested_metadata", "retention"])
def test_missing_projection_family_guard_refuses_empty_and_count(tmp_path, family):
    store, _, _ = _ready(tmp_path)
    index = store.open_query_index()
    with sqlite3.connect(index.path) as con:
        con.execute("DELETE FROM store_projection_coverage WHERE family=?", (family,))
    for query in (store.query().cdefs().stored(), store.query().sel(Definition(ProjectionLeaf, "absent"))):
        with pytest.raises(QueryIndexDirty):
            query.refresh(False).require_indexed().count()


@pytest.mark.parametrize("family", ["recipe_root", "discovery", "node", "edge"])
def test_missing_projection_row_is_not_a_certified_negative(tmp_path, family):
    store, _, _ = _ready(tmp_path)
    index = store.open_query_index()
    if family == "edge":
        child = Definition(ProjectionLeaf, "child").concretize()
        store.publish_node(Definition(ProjectionPair, child, child).concretize())
    with sqlite3.connect(index.path) as con:
        con.execute("DELETE FROM store_projection_rows WHERE family=?", (family,))
    with pytest.raises(QueryIndexDirty):
        store.query().cdefs().stored().refresh(False).require_indexed().count()
    before = _authority_bytes(store)
    assert store.query().cdefs().stored().count() == (4 if family == "edge" else 3)
    assert _authority_bytes(store) == before


@pytest.mark.parametrize("damage", ["missing", "corrupt", "schema7", "schema9", "coverage", "dirty"])
def test_sidecar_policy_and_derived_only_recovery(tmp_path, monkeypatch, damage):
    import dryml.core.query.authority as authority
    store, _, _ = _ready(tmp_path)
    index, before = store.open_query_index(), _authority_bytes(store)
    index._connections.close_all_current_process()
    if damage == "missing":
        index.path.unlink()
    elif damage == "corrupt":
        index.path.write_bytes(b"broken SQLite sidecar")
    elif damage == "dirty":
        store.mark_query_index_dirty()
    else:
        with sqlite3.connect(index.path) as con:
            if damage in {"schema7", "schema9"}:
                con.execute("PRAGMA user_version=" + damage.removeprefix("schema"))
            else:
                con.execute("DELETE FROM store_projection_coverage WHERE family='edge'")
    with monkeypatch.context() as scoped:
        scoped.setattr(authority, "capture_store_projection", lambda *a, **k: pytest.fail("forbidden recovery inventoried Store"))
        assert store.query().refresh(True).take(0).count() == 0
        for query in (store.query().refresh(False).require_indexed(), store.query().scan_policy("forbid")):
            with pytest.raises((QueryIndexDirty, QueryIndexUnavailable, QueryWouldScanError)):
                query.cdefs().stored().count()
    stats = store.query().cdefs().stored().explain(analyze=True)
    assert stats.result_count == 3 and stats.refresh_action == "rebuild" and stats.store_scan_count == 1
    assert index.status().state == "ready" and _authority_bytes(store) == before


@pytest.mark.parametrize("version", [11, 100])
def test_future_sidecar_is_refused_intact(tmp_path, version):
    store, _, _ = _ready(tmp_path)
    index = store.open_query_index()
    index._connections.close_all_current_process()
    with sqlite3.connect(index.path) as con:
        con.execute(f"PRAGMA user_version={version}")
    before = index.path.read_bytes(), _authority_bytes(store)
    with pytest.raises(QueryIndexIncompatible):
        store.query().cdefs().stored().count()
    assert (index.path.read_bytes(), _authority_bytes(store)) == before


@pytest.mark.parametrize("policy", ["none", "memory", "zip"])
def test_nonpersistent_sources_observe_current_names_and_membership(tmp_path, policy):
    store = ZipStore(tmp_path / "store.zip") if policy == "zip" else DirStore(tmp_path / "store", query_index=policy)
    try:
        root = store.publish_node(Definition(ProjectionLeaf, "first").concretize())
        first = NodeNameRecord("main", "recipe", root.digest)
        store.compare_and_set_node_name(first, expected=None)
        root = store.publish_node(Definition(ProjectionLeaf, "second").concretize())
        store.compare_and_set_node_name(NodeNameRecord("main", "recipe", root.digest), expected=first)
        result = store.query().cdefs().alias("main").refresh(False).collect()
        assert result.one().parameters["name"] == "second"
        assert not Path(store.query_index_path).exists()
        if policy == "zip":
            store.commit()
    finally:
        store.close()


def test_header_and_names_cannot_promote_dependency_roots(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    child = Definition(ProjectionLeaf, "child").concretize()
    store.install_creation(child)
    before = store.authority_cut()
    with pytest.raises(StoreAuthorityError, match="membership"):
        store.compare_and_set_node_name(NodeNameRecord("main", "recipe", child.creation_digest()), expected=None)
    assert store.authority_cut() == before
    store.publish_node(Definition(ProjectionPair, child, child).concretize())
    store.rebuild_query_index()
    assert store.query().cdefs().stored().refresh(False).count() == 1
    assert store.query().cdefs().refresh(False).count() == 2
    assert store.query().local_states().refresh(False).count() == 0


@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_name_authority_errors_precede_empty_alias_pages(tmp_path, damage):
    store, _, records = _ready(tmp_path)
    store.compare_and_set_node_name(NodeNameRecord("main", "recipe", records[0].digest), expected=None)
    if damage == "malformed":
        Path(store._name_path("main")).write_bytes(b"malformed")
    else:
        Path(store._node_path("membership/recipe", records[0].digest)).unlink()
    before = _authority_bytes(store)
    with pytest.raises((StoreAuthorityError, ValueError, QueryError)):
        store.query().alias("main").sel(Definition(ProjectionLeaf, "absent")).refresh(False).count()
    with pytest.raises((StoreAuthorityError, ValueError)):
        store.rebuild_query_index()
    assert _authority_bytes(store) == before


def test_metadata_rebuild_retains_sources_without_reviving_captured_values(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    obj = ProjectionStateful("first", repo=repo)
    first = repo.save(obj, annotations=SaveAnnotations(object={"project": "captured"}))
    obj.value = "second"
    second = repo.save(obj)
    repo.set_metadata(first.object_view(), {"project": "current"}, scope="object")
    monkeypatch.setattr(store, "_authenticate_state_data", lambda *a: pytest.fail("rebuild read a payload"))
    store.rebuild_query_index()
    rows = store.open_query_index().projection_rows("requested_metadata", refresh=False)
    assert sum("/capture/" in row[0] for row in rows) == 2
    assert sum("/lineage/" in row[0] for row in rows) == 1
    assert repo.query().object_refs().where(field("object", "project").eq("current")).refresh(False).count() == 1
    assert repo.delete_metadata(first.object_view(), scope="object")
    store.rebuild_query_index()
    assert repo.query().object_refs().where(field("object", "project").eq("captured")).refresh(False).count() == 0
    assert repo.query().object_refs().where(field("object").missing()).refresh(False).count() == 1
    assert repo.get_snapshot_metadata(first).captured_object_annotations == {"project": "captured"}
    assert repo.get_snapshot_metadata(second).captured_object_annotations == {"project": "captured"}


def test_clean_incremental_save_and_metadata_do_not_inventory_history(tmp_path, monkeypatch):
    import dryml.core.query.authority as authority
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    repo.save(ProjectionStateful("historical", repo=repo))
    original = authority.capture_store_projection
    def capture(*args, **kwargs):
        assert not kwargs.get("rebuild"), "clean mutation inventoried history"
        return original(*args, **kwargs)
    monkeypatch.setattr(authority, "capture_store_projection", capture)
    monkeypatch.setattr(store, "iter_state_ref_records", lambda: pytest.fail("legacy snapshots enumerated"))
    saved = repo.save(ProjectionStateful("new", repo=repo), alias="latest", main=True,
                      annotations=SaveAnnotations(state={"visible": True}))
    assert repo.query().saved().alias("latest").where(field("state", "visible").eq(True)).refresh(False).require_indexed().one().graph_equal(saved)
    assert repo.query().local_states(origin=saved.state.creation_cdef).refresh(False).count() == 1
    repo.set_metadata(saved, {"visible": False})
    assert repo.query().saved().where(field("state", "visible").eq(False)).refresh(False).count() == 1


def test_interrupted_staged_rebuild_preserves_authority_and_releases_resources(tmp_path, monkeypatch):
    store, _, _ = _ready(tmp_path)
    index, before = store.open_query_index(), _authority_bytes(store)
    with monkeypatch.context() as scoped:
        scoped.setattr(index, "_activate_replacement", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            index.rebuild()
    assert _authority_bytes(store) == before
    assert not index._connections._leases and not index._build_claim_path().exists()
    index.rebuild()
    assert store.query().cdefs().stored().refresh(False).count() == 3


def test_indexed_admissions_share_replicas_and_do_not_charge_ordering_lookahead(tmp_path):
    first = DirStore(tmp_path / "first", query_index="sqlite")
    second = DirStore(tmp_path / "second", query_index="sqlite")
    values = sorted((Definition(ProjectionLeaf, str(i)).concretize() for i in range(3)), key=lambda node: node.exact_digest())
    for store in (first, second):
        for node in values:
            store.publish_node(node)
    query = Repo((first, second)).query().cdefs().stored().refresh(False).require_indexed()
    assert query.max_verify(0).take(0).count() == 0
    with pytest.raises(QueryVerifyBudgetExceeded):
        query.max_verify(0).take(1)
    result = query.max_verify(1).take(1)
    assert result.one().graph_equal(values[0]) and len(result.sources(result.one())) == 2
    assert result.diagnostic().candidate_rows_read == 4


def test_schema10_current_consumer_does_not_admit_legacy_read_views_or_lifecycle(tmp_path):
    store, _, _ = _ready(tmp_path, 1)
    assert store.query().cdefs().stored().refresh(False).require_indexed().count() == 1
    index = store.open_query_index()
    with pytest.raises(QueryIndexUnavailable, match="Legacy"):
        with index.read_view():
            pytest.fail("legacy graph rows admitted Store-4 authority")
    with pytest.raises(QueryIndexUnavailable, match="lifecycle"):
        index.projection_rows("lifecycle", refresh=False)
    assert index.projection_rows("requested_metadata", refresh=False) == ()


def test_foreign_sidecar_query_is_refused_intact(tmp_path):
    """A foreign-owned sidecar cannot be repurposed by query recovery."""
    store, _, _ = _ready(tmp_path)
    index = store.open_query_index()
    index._connections.close_all_current_process()
    with sqlite3.connect(index.path) as con:
        con.execute("UPDATE catalog_state SET store_key='foreign-owner'")
    before = index.path.read_bytes(), _authority_bytes(store)
    with pytest.raises(QueryIndexIncompatible):
        store.query().cdefs().stored().count()
    assert (index.path.read_bytes(), _authority_bytes(store)) == before


def test_metadata_rebuild_keeps_overlapping_dirty_token_without_payload_access(tmp_path, monkeypatch):
    """A later metadata writer invalidates detached rebuild work before activation."""
    import threading
    import dryml.core.query.authority as authority
    from dryml.core.store.authority import StoreCutChanged
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    saved = repo.save(ProjectionStateful(1, repo=repo), annotations=SaveAnnotations(object={"project": "before"}))
    index = store.open_query_index()
    original = authority.capture_store_projection
    waiting, resume = threading.Event(), threading.Event()
    errors = []

    def capture(*args, **kwargs):
        prepared = original(*args, **kwargs)
        if threading.current_thread().name == "old-metadata-rebuild":
            waiting.set()
            assert resume.wait(30)
        return prepared

    def rebuild():
        try:
            index.rebuild()
        except BaseException as error:
            errors.append(error)
        finally:
            index._connections.close_current()

    monkeypatch.setattr(authority, "capture_store_projection", capture)
    monkeypatch.setattr(store, "_authenticate_state_data", lambda *a: pytest.fail("metadata rebuild read payload"))
    worker = threading.Thread(target=rebuild, name="old-metadata-rebuild")
    worker.start()
    try:
        assert waiting.wait(30)
        disabled = DirStore.open_existing(store.base_dir, query_index="none")
        Repo(disabled).set_metadata(saved.object_view(), {"project": "after"}, scope="object")
        before = _authority_bytes(store)
    finally:
        resume.set()
        worker.join(30)
    assert not worker.is_alive() and len(errors) == 1 and isinstance(errors[0], StoreCutChanged)
    assert _authority_bytes(store) == before and index.status().state == "dirty"
    assert repo.query().object_refs().where(field("object", "project").eq("after")).one().graph_equal(saved.object_view())
    assert _authority_bytes(store) == before and index.status().state == "ready"


def _sqlite_index(tmp_path):
    return SQLiteStoreQueryIndex(source_key="v3-projection-store", path=tmp_path / "v3-projection.sqlite",
                                 config=SQLiteQueryIndexConfig(journal_mode="delete"))


def test_incremental_v3_publication_does_not_rescan_historical_projection_rows(tmp_path, monkeypatch):
    from dryml.core.query.sqlite import index as sqlite_index
    index = _sqlite_index(tmp_path)
    first = Definition(ProjectionLeaf, "first").concretize()
    second = Definition(ProjectionLeaf, "second").concretize()
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(first), (first,))
    monkeypatch.setattr(sqlite_index, "_v3_projection_digest", lambda *a, **k: pytest.fail("incremental save scanned historical projection rows"))
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(second), (second,))
    index.remove_stored_roots((first,))


def test_stored_identity_cursor_rejects_a_later_generation(tmp_path):
    index = _sqlite_index(tmp_path)
    first = Definition(ProjectionLeaf, "first").concretize()
    second = Definition(ProjectionLeaf, "second").concretize()
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(first), (first,))
    with index.read_view(include_cached=False) as view:
        cursor = next(view.iter_stored_identity_cdef_batches(batch_size=1)).next_cursor
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(second), (second,))
    with index.read_view(include_cached=False) as view:
        with pytest.raises(QueryIndexDirty, match="generation change"):
            next(view.iter_stored_identity_cdef_batches(after=cursor, batch_size=1))
