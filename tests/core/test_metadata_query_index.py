"""U6 parity tests for advisory SQLite metadata projections."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
import threading

import pytest

from dryml.core import MetadataConflictError, Object, Repo, SaveAnnotations, Serializable
from dryml.core.query import QueryError, field
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.query.model import QueryIndexBusy
from dryml.core.repo import RepoSaveError
from dryml.core.query.sqlite import sqlite_available
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter


class MetadataIndexedValue(Serializable):
    """Small stateful fixture whose snapshots need no payload inspection."""

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Write state bytes so mutations create distinct exact StateRefs."""

        Path(dest_dir, "value").write_text(str(self.value), encoding="utf-8")


class MetadataIndexedContainer(Object):
    """Stateless root retaining one materializing metadata holder."""

    def __init__(self, child):
        self.child = child


@pytest.fixture(autouse=True)
def _unavailable_environment(monkeypatch):
    """Keep index tests independent of installed-distribution inspection."""

    def unavailable():
        raise OSError("synthetic unavailable environment")

    monkeypatch.setattr("dryml.environments.introspection.inspect_current", unavailable)


def _save(repo, value, *, object_values=None, state_values=None):
    """Publish one small snapshot with optional current metadata mappings."""

    return repo.save_object(
        MetadataIndexedValue(value, repo=repo),
        annotations=SaveAnnotations(object=object_values, state=state_values),
    )


def _answers(repo, predicate):
    """Return a stable result/error signature for metadata-parity assertions."""

    try:
        return ("result", tuple(item.digest() for item in repo.references().where(predicate).state_refs()))
    except Exception as error:
        return ("error", type(error), str(error))


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_metadata_predicates_match_authority_across_index_states(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    selected = _save(
        writer, "selected", object_values={"team": "vision", "tags": ["baseline"], "score": 7},
        state_values={"score": 9},
    )
    _save(
        writer, "other", object_values={"team": "platform", "tags": ["other"], "score": 3},
        state_values={"score": 2},
    )
    predicates = (
        field("object", "team").eq("vision"),
        field("object", "tags").contains("baseline"),
        field("object", "score").ge(7),
        field("state", "score").gt(5),
        field("snapshot", "saved_at").ge(0),
        field("object", "team").lt(2),
    )
    expected = [_answers(Repo(DirStore(store.base_dir, query_index="none")), predicate) for predicate in predicates]

    sqlite_store = DirStore(store.base_dir, query_index="sqlite")
    sqlite_repo = Repo(sqlite_store)
    assert [_answers(sqlite_repo, predicate) for predicate in predicates] == expected
    index = sqlite_store.open_query_index()
    assert index is not None and index.status().state == "ready"

    index.path.unlink()
    assert [_answers(sqlite_repo, predicate) for predicate in predicates] == expected

    sqlite_store.mark_query_index_dirty()
    assert [_answers(sqlite_repo, predicate) for predicate in predicates] == expected

    index.close()
    index.path.write_bytes(b"not sqlite")
    assert [_answers(sqlite_repo, predicate) for predicate in predicates] == expected

    index.close()
    con = sqlite3.connect(index.path)
    try:
        con.execute("PRAGMA user_version = 999")
    finally:
        con.close()
    assert [_answers(sqlite_repo, predicate) for predicate in predicates] == expected
    assert selected in tuple(sqlite_repo.references().where(field("state", "score").eq(9)).state_refs())


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_metadata_rebuild_retains_source_ids_and_never_revives_captured_current_values(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    value = MetadataIndexedValue("state", repo=repo)
    state = repo.save_object(
        value, annotations=SaveAnnotations(object={"project": "captured"}),
    )
    captured = repo.get_snapshot_metadata(state)
    value.value = "second"
    second_state = repo.save_object(value)
    repo.set_metadata(state.object, {"project": "current"})
    index = store.open_query_index()
    assert index is not None
    index.rebuild(force=True)

    con = sqlite3.connect(index.path)
    try:
        snapshot_rows = con.execute(
            "SELECT source_record_id, scope, reference_digest FROM metadata_records WHERE source_kind = 'snapshot'"
        ).fetchall()
        current = con.execute(
            "SELECT source_record_id, present FROM metadata_records WHERE source_kind = 'current' AND scope = 'object' AND reference_digest = ?",
            (state.object.digest(),),
        ).fetchone()
    finally:
        con.close()
    assert {row[1] for row in snapshot_rows} == {"lineage", "snapshot"}
    assert any(row[1:] == ("snapshot", state.digest()) for row in snapshot_rows)
    assert any(row[1:] == ("snapshot", second_state.digest()) for row in snapshot_rows)
    state_source_ids = {row[0] for row in snapshot_rows if row[1] == "snapshot"}
    lineage_source_ids = {row[0] for row in snapshot_rows if row[1] == "lineage"}
    assert len(state_source_ids) == 2
    assert lineage_source_ids == state_source_ids
    assert current is not None and current[0] and current[1] == 1
    assert repo.references().where(field("object", "project").eq("current")).object_refs().one() == state.object

    assert repo.delete_metadata(state.object)
    index.rebuild(force=True)
    assert repo.get_snapshot_metadata(state).captured_object_annotations == captured.captured_object_annotations
    assert repo.references().where(field("object", "project").eq("captured")).object_refs().count() == 0
    assert repo.references().where(field("object").missing()).object_refs().one() == state.object
    con = sqlite3.connect(index.path)
    try:
        current = con.execute(
            "SELECT source_record_id, present FROM metadata_records WHERE source_kind = 'current' AND scope = 'object' AND reference_digest = ?",
            (state.object.digest(),),
        ).fetchone()
    finally:
        con.close()
    assert current == ("", 0)


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_metadata_rebuild_projects_only_authoritative_nested_object_holders(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    state = repo.save_object(
        MetadataIndexedContainer(MetadataIndexedValue("child", repo=repo), repo=repo),
    )
    child = state.object.at(GraphPath((Parameter("child"),)))
    repo.set_metadata(child, {"scope": "nested"})
    index = store.open_query_index()
    assert index is not None
    index.rebuild(force=True)

    con = sqlite3.connect(index.path)
    try:
        row = con.execute(
            "SELECT present FROM metadata_records WHERE source_kind = 'current' AND scope = 'object' AND reference_digest = ?",
            (child.digest(),),
        ).fetchone()
    finally:
        con.close()
    assert row == (1,)
    assert repo.references().exact(child).where(
        field("object", "scope").eq("nested")
    ).object_refs().one() == child


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_metadata_rebuild_keeps_overlapping_dirty_token_and_uses_no_payload_or_materialization(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    state = _save(repo, "state", object_values={"project": "before"})
    index = store.open_query_index()
    assert index is not None
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
    monkeypatch.setattr(store, "open_local_state", lambda *_args: pytest.fail("metadata rebuild opened a payload"))
    monkeypatch.setattr(repo, "_materialize_cdef", lambda *_args: pytest.fail("metadata rebuild materialized an Object"))
    assert repo.references().where(field("object", "project").eq("after")).object_refs().one() == state.object
    assert index.status().state == "ready"


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_metadata_sqlite_refresh_cannot_hide_conflicts_or_typed_leaf_errors(tmp_path):
    first = DirStore(tmp_path / "first", query_index="sqlite")
    second = DirStore(tmp_path / "second", query_index="none")
    writer = Repo(first)
    value = MetadataIndexedValue("state", repo=writer)
    state = writer.save_object(value, annotations=SaveAnnotations(object={"value": "text"}))
    combined = Repo([first, second])
    combined.save_object(value, store=second, source_store=first)
    combined.set_metadata(state.object, {"value": "different"}, store=second)

    with pytest.raises(MetadataConflictError):
        Repo([first, second]).references().where(field("object", "value").eq("text")).object_refs()
    with pytest.raises(QueryError, match="ordering"):
        Repo(first).references().where(field("object", "value").lt(2)).object_refs()


def test_incremental_metadata_save_does_not_scan_and_preserves_unrelated_tokens(tmp_path, monkeypatch):
    """A save projects only its targets and cannot acknowledge another mutation."""
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    first = _save(repo, "first", object_values={"project": "before"})
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
            _save(repo, "second", object_values={"project": "second"})
        second = raised.value.report.snapshots[0].state_ref
    assert set(store._query_index_dirty_markers()) == unrelated | {unscoped}
    with sqlite3.connect(index.path) as con:
        assert con.execute(
            "SELECT count(*) FROM metadata_records WHERE scope = 'snapshot'"
        ).fetchone() == (2,)
    assert repo.get_snapshot_metadata(second).captured_object_annotations == {"project": "second"}
    index.rebuild(force=True)
    assert not store.query_index_is_dirty()
    assert repo.references().where(field("object", "project").eq("after")).object_refs().one() == first.object


def test_incremental_metadata_replaces_current_rows_without_losing_snapshots(tmp_path):
    """Updated, absent, and stateless-root mappings have one current projection."""
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    value = MetadataIndexedContainer(MetadataIndexedValue("child", repo=repo), repo=repo)
    state = repo.save_object(value, annotations=SaveAnnotations(object={"version": 1}))
    assert not store.query_index_is_dirty()
    repo.save_object(value, annotations=SaveAnnotations(object={"version": 2}))
    assert not store.query_index_is_dirty()
    repo.delete_metadata(state.object)
    repo.save_object(value)
    assert not store.query_index_is_dirty()
    with sqlite3.connect(store.open_query_index().path) as con:
        assert con.execute(
            "SELECT source_record_id, present FROM metadata_records "
            "WHERE source_kind = 'current' AND scope = 'object' AND reference_digest = ?",
            (state.object.digest(),),
        ).fetchall() == [("", 0)]
        assert con.execute(
            "SELECT count(*) FROM metadata_records WHERE scope = 'snapshot'"
        ).fetchone() == (1,)


def test_incremental_metadata_keeps_token_published_after_capture(tmp_path, monkeypatch):
    """A mutation after projection cannot be cleared by the older token set."""
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    state = _save(repo, "state", object_values={"project": "before"})
    index = store.open_query_index()
    original = index._register_metadata_rows

    def publish_after_capture(*args, **kwargs):
        original(*args, **kwargs)
        store.write_metadata(state.object, {"project": "after"})

    with monkeypatch.context() as scoped:
        scoped.setattr(index, "_register_metadata_rows", publish_after_capture)
        scoped.setattr(index, "rebuild", lambda: None)
        with pytest.raises(RepoSaveError):
            _save(repo, "other")
    assert store.query_index_is_dirty()
    index.rebuild(force=True)
    assert not store.query_index_is_dirty()
    assert repo.references().where(field("object", "project").eq("after")).object_refs().one() == state.object


def test_incremental_registration_checks_rebuild_claim_inside_authority_fence(tmp_path, monkeypatch):
    """An older rebuild cannot replace a newer projection after its token clears."""
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    state = _save(repo, "state", object_values={"project": "before"})
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
                (state.definition,), (state,),
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
    assert repo.references().where(field("object", "project").eq("after")).object_refs().one() == state.object
