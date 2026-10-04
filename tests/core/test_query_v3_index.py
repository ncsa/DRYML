"""Focused U6 contracts for derived Query V3 index projections.

The projection ledger is deliberately advisory.  These tests assert its useful
integrity boundary without treating a ready sidecar as complete Store authority.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
import threading

import pytest

from dryml.core import (
    Definition,
    Object,
    ObjectRef,
    Repo,
    SaveAnnotations,
    Serializable,
)
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.query import field
from dryml.core.query.memory import MemoryStoreQueryIndex
from dryml.core.query.model import QueryIndexBusy, QueryIndexDirty, QueryIndexUnavailable
from dryml.core.query.sqlite import SQLiteQueryIndexConfig, sqlite_available
from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core.query.sqlite.schema import SQLITE_QUERY_INDEX_SCHEMA_VERSION
from dryml.core.query.source import SourceCapture
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord, MainRefRecord, ObjectAliasRecord
from dryml.core.store.zip import ZipStore


class ProjectionLeaf(Object):
    """Small graph leaf used to retain sharing topology in a V3 projection."""

    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class ProjectionPair(Object):
    """Small graph root with two independently inspectable edges."""

    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right


class ProjectionStateful(Serializable):
    """Stateful fixture for reference and metadata projection rebuilds."""

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Persist distinct payload bytes without affecting query inspection."""

        Path(dest_dir, "value").write_text(str(self.value), encoding="utf-8")


@pytest.fixture
def unavailable_environment(monkeypatch):
    """Keep projection tests independent of installed-package inspection."""

    def unavailable():
        raise OSError("synthetic unavailable environment")

    monkeypatch.setattr("dryml.environments.introspection.inspect_current", unavailable)


def _graph_variants():
    shared_leaf = ProjectionLeaf("same")
    shared = ProjectionPair(shared_leaf, shared_leaf).definition
    separate = ProjectionPair(ProjectionLeaf("same"), ProjectionLeaf("same")).definition
    assert shared == separate
    assert not shared.graph_equal(separate)
    return shared, separate


def _sqlite_index(tmp_path, *, store=None):
    return SQLiteStoreQueryIndex(
        source_key="v3-projection-store",
        path=tmp_path / "v3-projection.sqlite",
        config=SQLiteQueryIndexConfig(journal_mode="delete"),
        store=store,
    )


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_memory_and_sqlite_retain_graph_distinct_projection_roots(tmp_path):
    """Projection rows retain graph topology even when legacy CDefs compare equal."""

    shared, separate = _graph_variants()
    graph = ConcreteDefinitionGraph.from_roots((shared, separate))
    store = DirStore(tmp_path / "memory", query_index="memory")
    memory = MemoryStoreQueryIndex(Repo(store)._query_catalog, store)
    memory.register_stored_roots(graph, (shared, separate))

    sqlite = _sqlite_index(tmp_path)
    sqlite.register_stored_roots(graph, (shared, separate))

    memory_coverage = memory.v3_projection_coverage()
    sqlite_coverage = sqlite.v3_projection_coverage()
    assert memory_coverage.source_roots == 2
    assert sqlite_coverage.source_roots == 2
    assert sqlite_coverage.identities >= memory_coverage.identities
    assert memory_coverage.projection_complete
    assert sqlite_coverage.projection_complete
    assert not memory_coverage.indexed_only
    assert not sqlite_coverage.indexed_only
    with sqlite.read_view(include_cached=False) as view:
        page = next(view.iter_stored_identity_cdef_batches(batch_size=3))
    assert len(page.cdefs) == 2
    assert any(value.graph_equal(shared) for value in page.cdefs)
    assert any(value.graph_equal(separate) for value in page.cdefs)


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
@pytest.mark.parametrize("family", ("identity", "relationship"))
def test_missing_v3_projection_row_disables_indexed_only_claim(tmp_path, family):
    """A ready legacy sidecar cannot turn a deleted candidate row into absence."""

    shared, separate = _graph_variants()
    index = _sqlite_index(tmp_path)
    index.register_stored_roots(
        ConcreteDefinitionGraph.from_roots((shared, separate)), (shared, separate),
    )
    assert index.v3_projection_coverage().projection_complete

    with sqlite3.connect(index.path) as con:
        table = "v3_identity_projection" if family == "identity" else "v3_relationship_projection"
        con.execute(
            f"DELETE FROM {table} WHERE root_graph_hash = "
            f"(SELECT root_graph_hash FROM {table} LIMIT 1)"
        )

    coverage = index.v3_projection_coverage()
    assert not coverage.projection_complete
    assert not coverage.authority_complete
    assert not coverage.indexed_only


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_missing_stored_root_page_falls_back_or_rejects_indexed_only(tmp_path):
    """Unmanaged candidate deletion cannot become a false bounded absence."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    values = sorted(
        (
            Definition(ProjectionLeaf, f"value-{index}").concretize()
            for index in range(4)
        ),
        key=lambda value: value.graph_hash(),
    )
    for value in values:
        store.write_definition_record(DefinitionRecord(value))
    store.rebuild_query_index()
    index = store.open_query_index()
    with sqlite3.connect(index.path) as con:
        con.execute(
            "DELETE FROM stored_roots WHERE root_graph_hash = ?",
            (values[0].graph_hash(),),
        )

    query = Repo(store).query().cdefs().stored().refresh(False)
    assert tuple(query.take(3)) == tuple(values[:3])
    with pytest.raises(Exception, match="outside managed index publication"):
        query.require_indexed().take(3)


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_v3_schema_mismatch_rebuilds_only_the_sidecar(tmp_path):
    """Version-7 sidecars rebuild derived rows without touching Store records."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    root = Definition(ProjectionLeaf, "stored").concretize()
    record = store.write_definition_record(DefinitionRecord(root))
    index = store.open_query_index()
    index.rebuild()
    before = store.read_definition_record(record.digest).to_bytes()
    with sqlite3.connect(index.path) as con:
        con.execute("PRAGMA user_version = 7")

    index.refresh("auto")

    with sqlite3.connect(index.path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == SQLITE_QUERY_INDEX_SCHEMA_VERSION
    assert store.read_definition_record(record.digest).to_bytes() == before
    assert index.v3_projection_coverage().projection_complete


def test_no_index_and_zip_capture_follow_current_main_authority(tmp_path):
    """Main and alias movement are observed from authority, not index flags."""

    first = Definition(ProjectionLeaf, "first").concretize()
    second = Definition(ProjectionLeaf, "second").concretize()
    directory = DirStore(tmp_path / "directory", query_index="none")
    directory.write_definition_record(DefinitionRecord(first))
    directory.write_definition_record(DefinitionRecord(second))
    directory.write_main_ref(MainRefRecord(DefinitionRecord(first).digest))
    first_facts = SourceCapture().capture_store(directory)
    directory.write_main_ref(MainRefRecord(DefinitionRecord(second).digest))
    second_facts = SourceCapture().capture_store(directory)
    assert first_facts.main_definition == first
    assert second_facts.main_definition == second

    repo = Repo(directory)
    first_state = repo.save_object(ProjectionLeaf("alias-first", repo=repo))
    second_state = repo.save_object(ProjectionLeaf("alias-second", repo=repo))
    directory.write_object_alias(ObjectAliasRecord("current", first_state.object))
    first_alias_facts = SourceCapture().capture_store(directory)
    directory.write_object_alias(ObjectAliasRecord("current", second_state.object))
    second_alias_facts = SourceCapture().capture_store(directory)
    assert first_alias_facts.object_aliases[0].object_ref == first_state.object
    assert second_alias_facts.object_aliases[0].object_ref == second_state.object

    archive = tmp_path / "authority.zip"
    seed = ZipStore(archive)
    seed.write_definition_record(DefinitionRecord(second))
    seed.write_main_ref(MainRefRecord(DefinitionRecord(second).digest))
    seed.commit()
    seed.close()
    zipped = ZipStore.open_existing(archive)
    try:
        assert second in SourceCapture().capture_store(zipped).knowledge()
    finally:
        zipped.close()


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_incremental_v3_publication_does_not_rescan_historical_projection_rows(tmp_path, monkeypatch):
    """A clean save updates coverage with its delta, not the entire sidecar."""

    from dryml.core.query.sqlite import index as sqlite_index

    index = _sqlite_index(tmp_path)
    first = Definition(ProjectionLeaf, "first").concretize()
    second = Definition(ProjectionLeaf, "second").concretize()
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(first), (first,))
    monkeypatch.setattr(
        sqlite_index, "_v3_projection_digest",
        lambda *_args, **_kwargs: pytest.fail("incremental save scanned historical projection rows"),
    )

    index.register_stored_roots(ConcreteDefinitionGraph.from_root(second), (second,))
    index.remove_stored_roots((first,))


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_alias_only_authority_survives_query_index_rebuild(tmp_path):
    """An alias remains structural authority without a DefinitionRecord."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    reference = ObjectRef(Definition(ProjectionLeaf, "only").concretize(), {})
    store.write_object_alias(ObjectAliasRecord("only", reference))

    metadata = store.query_index_record_metadata(reference.definition)
    assert metadata[:2] == (reference.digest(), "refs/objects/only.record")
    store.rebuild_query_index()

    assert not store.query_index_is_dirty()
    assert Repo(store).query().stored().cdefs().one() == reference.definition


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_bounded_alias_authority_falls_back_instead_of_becoming_absent(tmp_path):
    """A paged alias candidate requires complete authoritative fallback."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    reference = ObjectRef(Definition(ProjectionLeaf, "only").concretize(), {})
    store.write_object_alias(ObjectAliasRecord("only", reference))
    store.rebuild_query_index()

    query = Repo(store).query().stored().cdefs().refresh(False)
    assert query.take(1).one() == reference.definition
    with pytest.raises(QueryIndexUnavailable, match="lacks direct stored-root authority"):
        query.require_indexed().take(1)


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_refresh_false_does_not_initialize_a_missing_sidecar(tmp_path):
    """A disabled refresh falls back without creating derived state."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    reference = ObjectRef(Definition(ProjectionLeaf, "only").concretize(), {})
    store.write_object_alias(ObjectAliasRecord("only", reference))
    store.rebuild_query_index()
    index = store.open_query_index()
    index.close()
    index.path.unlink()

    query = Repo(store).query().stored().cdefs().refresh(False)
    assert query.take(1).one() == reference.definition
    assert not index.path.exists()
    with pytest.raises(QueryIndexUnavailable, match="not ready"):
        query.require_indexed().take(1)
    assert not index.path.exists()


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_stored_identity_cursor_rejects_a_later_generation(tmp_path):
    """Every stored-root page is bound to one sidecar generation."""

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


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_sqlite_reference_rows_rebuild_from_unchanged_authority(
        tmp_path, unavailable_environment):
    """Reference projections can be deleted and rebuilt without authority drift."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    state = repo.save_object(ProjectionStateful(1, repo=repo))

    assert repo.query().object_id(state.object_id).state_refs().one() == state
    index = store.open_query_index()
    index.rebuild()
    with closing(sqlite3.connect(index.path)) as con:
        before = tuple(con.execute(
            "SELECT reference_kind, reference_digest "
            "FROM reference_records ORDER BY 1, 2"
        ))
        assert before
        assert con.execute(
            "SELECT COUNT(*) FROM reference_object_ids"
        ).fetchone()[0] == 2

    index.close()
    index.path.unlink()
    store.reconcile_query_index()
    assert repo.query().object_id(state.object_id).state_refs().one() == state
    with closing(sqlite3.connect(index.path)) as con:
        after = tuple(con.execute(
            "SELECT reference_kind, reference_digest "
            "FROM reference_records ORDER BY 1, 2"
        ))
    assert after == before


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_metadata_rebuild_retains_sources_without_reviving_captured_values(
        tmp_path, unavailable_environment):
    """Current metadata deletion cannot revive snapshot-captured annotations."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    value = ProjectionStateful("state", repo=repo)
    state = repo.save_object(
        value, annotations=SaveAnnotations(object={"project": "captured"}),
    )
    captured = repo.get_snapshot_metadata(state)
    value.value = "second"
    second_state = repo.save_object(value)
    repo.set_metadata(state.object, {"project": "current"})
    index = store.open_query_index()
    index.rebuild(force=True)

    with closing(sqlite3.connect(index.path)) as con:
        snapshot_rows = con.execute(
            "SELECT source_record_id, scope, reference_digest "
            "FROM metadata_records WHERE source_kind = 'snapshot'"
        ).fetchall()
        current = con.execute(
            "SELECT source_record_id, present FROM metadata_records "
            "WHERE source_kind = 'current' AND scope = 'object' "
            "AND reference_digest = ?",
            (state.object.digest(),),
        ).fetchone()
    assert {row[1] for row in snapshot_rows} == {"lineage", "snapshot"}
    assert any(row[1:] == ("snapshot", state.digest()) for row in snapshot_rows)
    assert any(row[1:] == ("snapshot", second_state.digest()) for row in snapshot_rows)
    state_source_ids = {row[0] for row in snapshot_rows if row[1] == "snapshot"}
    lineage_source_ids = {row[0] for row in snapshot_rows if row[1] == "lineage"}
    assert len(state_source_ids) == 2
    assert lineage_source_ids == state_source_ids
    assert current is not None and current[0] and current[1] == 1
    assert repo.query().where(
        field("object", "project").eq("current")
    ).object_refs().one() == state.object

    assert repo.delete_metadata(state.object)
    index.rebuild(force=True)
    assert (
        repo.get_snapshot_metadata(state).captured_object_annotations
        == captured.captured_object_annotations
    )
    assert repo.query().where(
        field("object", "project").eq("captured")
    ).object_refs().count() == 0
    assert repo.query().where(field("object").missing()).object_refs().one() == state.object
    with closing(sqlite3.connect(index.path)) as con:
        current = con.execute(
            "SELECT source_record_id, present FROM metadata_records "
            "WHERE source_kind = 'current' AND scope = 'object' "
            "AND reference_digest = ?",
            (state.object.digest(),),
        ).fetchone()
    assert current == ("", 0)


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_metadata_rebuild_keeps_overlapping_dirty_token_without_payload_access(
        tmp_path, monkeypatch, unavailable_environment):
    """A mutation after a rebuild cut remains dirty and recovers from records."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    state = repo.save_object(
        ProjectionStateful("state", repo=repo),
        annotations=SaveAnnotations(object={"project": "before"}),
    )
    index = store.open_query_index()
    index.rebuild(force=True)
    repo.set_metadata(state.object, {"project": "cut"})
    original_activate = index._activate_replacement

    def publish_after_cut(path, *, quarantine_existing):
        store.write_metadata(state.object, {"project": "after"})
        original_activate(path, quarantine_existing=quarantine_existing)

    monkeypatch.setattr(index, "_activate_replacement", publish_after_cut)
    index.rebuild(force=True)
    assert index.status().state == "dirty"
    monkeypatch.setattr(index, "_activate_replacement", original_activate)
    monkeypatch.setattr(
        store,
        "open_local_state",
        lambda *_args: pytest.fail("metadata rebuild opened a payload"),
    )
    assert repo.query().where(
        field("object", "project").eq("after")
    ).object_refs().one() == state.object
    assert index.status().state == "ready"


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_incremental_metadata_save_preserves_unrelated_dirty_tokens(
        tmp_path, monkeypatch, unavailable_environment):
    """Incremental registration cannot acknowledge unrelated authority changes."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    first = repo.save_object(
        ProjectionStateful("first", repo=repo),
        annotations=SaveAnnotations(object={"project": "before"}),
    )
    index = store.open_query_index()
    index.rebuild(force=True)
    store.write_metadata(first.object, {"project": "after"})
    unrelated = set(store._query_index_dirty_markers())
    unscoped = store.mark_query_index_dirty()

    def no_scan():
        pytest.fail("incremental save scanned unrelated authority")

    with monkeypatch.context() as scoped:
        scoped.setattr(index, "rebuild", lambda: None)
        scoped.setattr(store, "iter_state_ref_records", no_scan)
        scoped.setattr(store, "iter_declaration_records", no_scan)
        with pytest.raises(RepoSaveError) as raised:
            repo.save_object(
                ProjectionStateful("second", repo=repo),
                annotations=SaveAnnotations(object={"project": "second"}),
            )
        second = raised.value.report.snapshots[0].state_ref
    assert set(store._query_index_dirty_markers()) == unrelated | {unscoped}
    with closing(sqlite3.connect(index.path)) as con:
        assert con.execute(
            "SELECT count(*) FROM metadata_records WHERE scope = 'snapshot'"
        ).fetchone() == (2,)
    assert repo.get_snapshot_metadata(second).captured_object_annotations == {
        "project": "second",
    }
    index.rebuild(force=True)
    assert not store.query_index_is_dirty()
    assert repo.query().where(
        field("object", "project").eq("after")
    ).object_refs().one() == first.object


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_clean_incremental_save_does_not_enumerate_historical_references(
        tmp_path, monkeypatch):
    """A successful clean save publishes only its new graph and StateRef rows."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    historical = tuple(
        repo.save_object(ProjectionStateful(f"old-{index}", repo=repo))
        for index in range(3)
    )
    index = store.open_query_index()
    index.rebuild(force=True)
    monkeypatch.setattr(
        store,
        "iter_state_ref_records",
        lambda: pytest.fail("clean save enumerated historical StateRefs"),
    )
    monkeypatch.setattr(
        store,
        "iter_declaration_records",
        lambda: pytest.fail("clean save enumerated historical declarations"),
    )

    saved = repo.save_object(ProjectionStateful("new", repo=repo))

    assert saved not in historical
    assert index.status().state == "ready"
    with closing(sqlite3.connect(index.path)) as con:
        assert con.execute(
            "SELECT count(*) FROM reference_records "
            "WHERE source_kind = 'state-ref' AND reference_kind = 'state'"
        ).fetchone() == (4,)


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
@pytest.mark.parametrize("direct_child", (False, True))
def test_incremental_graph_registration_does_not_rescan_historical_snapshots(
        tmp_path, monkeypatch, direct_child):
    """Clear covered definition tokens without scanning snapshot authority."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    Repo(store).save_object(ProjectionStateful("historical"))
    child = Definition(ProjectionLeaf, "child").concretize()
    parent = Definition(ProjectionPair, child, child).concretize()
    if direct_child:
        store.write_definition_record(DefinitionRecord(child), stored_root=False)
    store.write_definition_record(DefinitionRecord(parent))

    with monkeypatch.context() as scoped:
        scoped.setattr(
            store,
            "iter_state_ref_records",
            lambda: pytest.fail("incremental registration scanned historical snapshots"),
        )
        store.open_query_index().register_stored_roots(
            ConcreteDefinitionGraph.from_root(parent), (parent,),
        )

    assert store.query_index_status().state == "ready"
    query = Repo(store).query()
    assert query.sel(parent).cdefs().stored().count() == 1
    assert query.sel(child).cdefs().stored().count() == 0


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_incremental_registration_checks_rebuild_claim_inside_authority_fence(
        tmp_path, monkeypatch, unavailable_environment):
    """An older rebuild cannot replace projection rows newer registration owns."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    state = repo.save_object(
        ProjectionStateful("state", repo=repo),
        annotations=SaveAnnotations(object={"project": "before"}),
    )
    index = store.open_query_index()
    waiting, resume = threading.Event(), threading.Event()
    errors = []
    original_fence = store.authority_read_fence

    @contextmanager
    def pause_registration():
        if threading.current_thread() is worker:
            waiting.set()
            assert resume.wait(10)
        with original_fence():
            yield

    def register():
        try:
            index.register_saved_graph(
                ConcreteDefinitionGraph.from_root(state.definition),
                (state.definition,),
                (state,),
            )
        except BaseException as error:
            errors.append(error)
        finally:
            index._connections.close_current()

    worker = threading.Thread(target=register)
    monkeypatch.setattr(store, "authority_read_fence", pause_registration)
    worker.start()
    try:
        assert waiting.wait(10)
        with index._build_claim(force=True) as acquired:
            assert acquired
            cut = index._capture_authority_cut()
            store.write_metadata(state.object, {"project": "after"})
            resume.set()
            worker.join(10)
            assert not worker.is_alive()
            assert len(errors) == 1 and isinstance(errors[0], QueryIndexBusy)
            with monkeypatch.context() as scoped:
                scoped.setattr(index, "_capture_authority_cut", lambda: cut)
                index._rebuild_owned()
        assert store.query_index_is_dirty()
    finally:
        resume.set()
        worker.join(10)
    index.rebuild(force=True)
    assert not store.query_index_is_dirty()
    assert repo.query().where(
        field("object", "project").eq("after")
    ).object_refs().one() == state.object


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_corrupt_sidecar_rebuild_preserves_authoritative_record_bytes(tmp_path):
    """Missing and corrupt SQLite files recover without rewriting authority."""

    store = DirStore(
        tmp_path / "store",
        query_index=SQLiteQueryIndexConfig(journal_mode="delete"),
    )
    root = Definition(ProjectionLeaf, "authority").concretize()
    record = store.write_definition_record(DefinitionRecord(root))
    store.rebuild_query_index()
    authority = store.read_definition_record(record.digest).to_bytes()
    index = store.open_query_index()

    index.close()
    index.path.unlink()
    assert store.reconcile_query_index().action == "rebuild"
    index.close()
    index.path.write_bytes(b"not a sqlite database")
    assert store.query_index_status().state == "corrupt"
    assert store.reconcile_query_index().action == "rebuild"

    assert store.read_definition_record(record.digest).to_bytes() == authority
    assert store.validate_query_index(thorough=True).ok
    assert list(index.path.parent.glob(f"{index.path.name}.quarantine-*"))


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_interrupted_staged_rebuild_preserves_ready_sidecar_and_authority(
        tmp_path, monkeypatch):
    """Interrupted validation cannot activate partial derived index state."""

    store = DirStore(
        tmp_path / "store",
        query_index=SQLiteQueryIndexConfig(journal_mode="delete"),
    )
    first = store.write_definition_record(DefinitionRecord(
        Definition(ProjectionLeaf, "first").concretize()
    ))
    store.rebuild_query_index()
    index = store.open_query_index()
    before = index.path.read_bytes()
    second = store.write_definition_record(DefinitionRecord(
        Definition(ProjectionLeaf, "second").concretize()
    ))

    def interrupt_validation(self, *, roots):
        raise KeyboardInterrupt("injected staged rebuild interruption")

    monkeypatch.setattr(
        SQLiteStoreQueryIndex,
        "_validate_rebuild_before_ready",
        interrupt_validation,
    )
    with pytest.raises(KeyboardInterrupt, match="staged rebuild interruption"):
        store.rebuild_query_index()

    assert index.path.read_bytes() == before
    assert store.read_definition_record(first.digest) == first
    assert store.read_definition_record(second.digest) == second
    assert store.query_index_status().state == "dirty"
    assert not list(index.path.parent.glob(f"{index.path.name}.rebuild-*.tmp*"))
