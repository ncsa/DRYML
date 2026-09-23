"""Focused U5 contracts for typed, authoritative metadata reference queries."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Definition, MetadataConflictError, Object, Repo, SaveAnnotations, Serializable
from dryml.core.query import MetadataField, MetadataPredicate, QueryError, field
from dryml.core.query.codecs import QueryCodecError, decode_metadata_predicate, encode_metadata_predicate
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore


class QueryMetadataValue(Serializable):
    """Small stateful value used to create complete metadata snapshots."""

    def __init__(self, name):
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Write the name so one fixture can publish distinct StateRefs."""

        Path(dest_dir, "name").write_text(self.name, encoding="utf-8")


class QueryMetadataRoot(Object):
    """Stateless root used to exercise unknown root lineage evidence."""

    def __init__(self, child):
        self.child = child


def _saved(repo, *, name="first", object_values=None, state_values=None):
    return repo.save_object(
        QueryMetadataValue(name, repo=repo),
        annotations=SaveAnnotations(object=object_values, state=state_values),
    )


def test_field_validates_paths_and_all_predicates_are_inert_and_round_trip():
    selector = field("object", "nested", 0)
    predicate = (
        selector.exists()
        & field("object", "missing").missing()
        & selector.eq({"names": ("vision",), "enabled": True})
        & field("object", "number").lt(8)
        & field("object", "number").le(7)
        & field("object", "number").gt(3)
        & field("object", "number").ge(4)
        & field("object", "text").contains("is")
    ) | ~field("object", "tags").contains("other")

    decoded = decode_metadata_predicate(encode_metadata_predicate(predicate))

    assert decoded.to_data() == predicate.to_data()
    assert decoded.fingerprint() == predicate.fingerprint()
    assert encode_metadata_predicate(decoded) == encode_metadata_predicate(predicate)
    with pytest.raises(ValueError):
        field("captured", "value")
    with pytest.raises(ValueError):
        field("object", True)
    with pytest.raises(ValueError):
        field("object", -1)
    with pytest.raises(ValueError):
        field("object", *("x",) * 33)
    assert field("object", "x" * 4096).path == ("x" * 4096,)
    with pytest.raises(ValueError, match="4096"):
        field("object", "x" * 4097)
    with pytest.raises(ValueError, match="4096"):
        MetadataField("object", ("x" * 4097,))
    with pytest.raises(TypeError, match="tuple"):
        MetadataField("object", ["value"])
    with pytest.raises(QueryCodecError, match="field data"):
        MetadataPredicate.from_data({
            "kind": "exists",
            "field": {"scope": "object", "path": ["x" * 4097]},
        })
    with pytest.raises(TypeError):
        bool(predicate)


def test_predicate_bounds_and_codec_reject_malformed_noncanonical_input():
    leaf = field("object", "value").exists()
    leaves = [leaf for _ in range(256)]
    while len(leaves) > 1:
        leaves = [left | right for left, right in zip(leaves[::2], leaves[1::2])]
    predicate = leaves[0]
    blob = encode_metadata_predicate(predicate)
    decoded = decode_metadata_predicate(blob)
    assert decoded.to_data() == predicate.to_data()
    assert decoded.fingerprint() == predicate.fingerprint()
    with pytest.raises(ValueError, match="256"):
        predicate | leaf
    predicate = leaf
    for _ in range(32):
        predicate = ~predicate
    with pytest.raises(ValueError, match="32"):
        ~predicate
    nested = 0
    for _ in range(9):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting"):
        field("object", "value").eq(nested)
    blob = encode_metadata_predicate(leaf)
    with pytest.raises(QueryCodecError, match="noncanonical"):
        decode_metadata_predicate(blob + b" ")
    with pytest.raises(QueryCodecError):
        decode_metadata_predicate(b'{"kind":"metadata-predicate"}')
    deeply_nested = leaf.to_data()
    for _ in range(33):
        deeply_nested = {"kind": "not", "operand": deeply_nested}
    with pytest.raises(QueryCodecError, match="bounds"):
        MetadataPredicate.from_data(deeply_nested)


def test_current_metadata_paths_types_and_empty_roots_are_scope_local(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    absent = _saved(repo, name="absent")
    empty = _saved(repo, name="empty", object_values={}, state_values={})
    state = _saved(
        repo,
        name="values",
        object_values={
            "nested": [{"items": ("vision", 2)}], "null": None,
            "bool": True, "int": 1, "float": 1.0, "text": "vision baseline",
            "tags": ["vision", "baseline"],
        },
        state_values={"score": 7},
    )

    assert repo.references().exact(absent.object).where(field("object").missing()).object_refs().one() == absent.object
    assert repo.references().exact(empty.object).where(field("object").exists()).object_refs().one() == empty.object
    assert repo.references().exact(empty.object).where(field("object").eq({})).object_refs().one() == empty.object
    assert repo.references().state_hash(next(iter(empty.states.values()))).where(field("state").eq({})).state_refs().one() == empty
    assert repo.references().where(field("object", "nested", 0, "items", 0).eq("vision")).object_refs().one() == state.object
    assert repo.references().where(field("object", "nested", 0, "items", 1).eq(2)).object_refs().one() == state.object
    assert repo.references().where(field("object", "null").eq(None)).object_refs().one() == state.object
    assert repo.references().where(field("object", "bool").eq(1)).object_refs().count() == 0
    assert repo.references().where(field("object", "int").eq(1.0)).object_refs().count() == 0
    assert repo.references().where(field("object", "tags").eq(("vision", "baseline"))).object_refs().count() == 0
    assert repo.references().where(field("object", "int").lt(2)).object_refs().one() == state.object
    assert repo.references().where(field("object", "text").contains("vision")).object_refs().one() == state.object
    assert repo.references().where(field("object", "tags").contains("vision")).object_refs().one() == state.object
    assert repo.references().where(field("state", "score").eq(7)).state_refs().one() == state
    assert repo.references().where(field("state", "score").eq(7)).object_refs().one() == state.object
    assert [item.value for item in repo.references().where(field("state", "score").eq(7)).occurrences()] == [state]
    assert repo.references().where(field("object", "score").missing()).object_refs().count() == 3
    assert repo.references().where(field("state", "team").missing()).state_refs().count() == 3


def test_populated_invalid_order_or_contains_cannot_be_short_circuited(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    state = _saved(repo, object_values={"value": "not-number", "items": {"not": "a sequence"}})

    with pytest.raises(QueryError, match="ordering"):
        repo.references().exact(state.object).where(
            field("object", "value").lt(2) | field("object", "missing").eq("x")
        ).object_refs()
    with pytest.raises(QueryError, match="containment"):
        repo.references().exact(state.object).where(
            field("object", "items").contains("x") & field("object", "missing").eq("x")
        ).object_refs()


def test_predicate_construction_rejects_invalid_input_before_store_scanning(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    monkeypatch.setattr(store, "iter_declaration_records", lambda: pytest.fail("invalid predicate scanned Store"))
    monkeypatch.setattr(store, "iter_state_ref_records", lambda: pytest.fail("invalid predicate scanned Store"))

    with pytest.raises(TypeError, match="ordering"):
        field("object", "value").lt("not-a-number")
    with pytest.raises(ValueError, match="timezone"):
        field("lineage", "created_at").ge(datetime(1970, 1, 1))
    with pytest.raises(ValueError, match="non-negative"):
        field("object", -1)
    with pytest.raises(ValueError, match="4096"):
        field("object", "x" * 4097)
    with pytest.raises(ValueError, match="string"):
        field("object", "value").eq("x" * 4097)

    leaves = [field("object", "value").exists() for _ in range(256)]
    while len(leaves) > 1:
        leaves = [left | right for left, right in zip(leaves[::2], leaves[1::2])]
    predicate = leaves[0]
    with pytest.raises(ValueError, match="256"):
        predicate | field("object", "value").exists()
    assert repo.references().where(field("object", "value").exists())


def test_metadata_composition_and_boolean_order_are_semantically_invariant(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    selected = _saved(repo, name="selected", object_values={"team": "vision", "score": 7})
    _saved(repo, name="other", object_values={"team": "platform", "score": 7})
    team = field("object", "team").eq("vision")
    score = field("object", "score").eq(7)
    definition = Definition(QueryMetadataValue, name="selected")

    repeated = repo.references().where(team).where(score).object_refs()
    combined = repo.references().where(team & score).object_refs()
    metadata_first = repo.references().where(team).definition(definition).object_refs()
    structure_first = repo.references().definition(definition).where(team).object_refs()

    assert list(repeated) == list(combined) == [selected.object]
    assert list(metadata_first) == list(structure_first) == [selected.object]

    invalid = field("object", "team").lt(2)
    missing = field("object", "missing").eq("value")
    for predicate in (invalid | missing, missing | invalid):
        with pytest.raises(QueryError, match="ordering"):
            repo.references().exact(selected.object).where(predicate).object_refs()


def test_state_metadata_projects_unique_objects_and_keeps_structure_constraints(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    value = QueryMetadataValue("selected", repo=repo)
    first = repo.save_object(value, annotations=SaveAnnotations(state={"score": 7}))
    value.name = "updated"
    second = repo.save_object(value, annotations=SaveAnnotations(state={"score": 7}))
    predicate = field("state", "score").eq(7)

    assert first != second
    query = repo.query(Definition(QueryMetadataValue, name="selected")).where(predicate)
    assert set(query.state_refs()) == {first, second}
    assert list(query.object_refs()) == [first.object]


def test_u4_prepared_selector_composes_with_metadata_sidecars(tmp_path):
    """Metadata filtering verifies prepared semantic selectors against authority."""
    from dryml.core import categorical_definition

    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    selected = _saved(repo, name="selected", object_values={"team": "vision"})
    _saved(repo, name="other", object_values={"team": "vision"})
    selector = categorical_definition(Definition(QueryMetadataValue, "selected"))

    assert repo.query(selector).where(field("object", "team").eq("vision")).object_refs().one() == selected.object


def test_unknown_and_epoch_lineage_timestamps_have_distinct_query_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dryml.core.snapshot_capture.current_utc_time",
        lambda: datetime(1970, 1, 1, tzinfo=timezone.utc),
    )
    repo = Repo(DirStore(tmp_path / "store", query_index="none"), clock=lambda: 0)
    unknown = repo.save_object(QueryMetadataRoot(QueryMetadataValue("child", repo=repo), repo=repo))
    known = _saved(repo, name="known")

    assert repo.get_lineage_metadata(unknown.object).creation_status == "unknown"
    assert repo.references().exact(unknown.object).where(
        field("lineage", "creation_status").eq("unknown")
    ).object_refs().one() == unknown.object
    assert repo.references().exact(unknown.object).where(
        field("lineage", "created_at").ge(0)
    ).object_refs().count() == 0
    assert repo.get_lineage_metadata(known.object).created_at == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert repo.references().exact(known.object).where(
        field("lineage", "created_at").eq(0)
    ).object_refs().one() == known.object


def test_store_order_preserves_answers_and_exposes_current_or_snapshot_conflicts(tmp_path):
    first = DirStore(tmp_path / "first", query_index="none")
    second = DirStore(tmp_path / "second", query_index="none")
    writer = Repo([first, second])
    value = QueryMetadataValue("replica", repo=writer)
    state = writer.save_object(value, store=first, annotations=SaveAnnotations(object={"team": "vision"}))
    writer.save_object(value, store=second, source_store=first)
    writer.set_metadata(state.object, {"team": "vision"}, store=second)
    predicate = field("object", "team").eq("vision")

    assert list(Repo([first, second]).references().where(predicate).object_refs()) == [state.object]
    assert list(Repo([second, first]).references().where(predicate).object_refs()) == [state.object]
    writer.set_metadata(state.object, {"team": "platform"}, store=second)
    for stores in ((first, second), (second, first)):
        with pytest.raises(MetadataConflictError, match="current metadata"):
            Repo(stores).references().where(predicate).object_refs()

    captured_first = DirStore(tmp_path / "captured-first", query_index="none")
    captured_second = DirStore(tmp_path / "captured-second", query_index="none")
    first_writer = Repo(captured_first, clock=lambda: 0)
    captured_value = QueryMetadataValue("captured", repo=first_writer)
    captured = first_writer.save_object(captured_value)
    second_writer = Repo(captured_second, clock=lambda: 1)
    assert second_writer.save_object(captured_value) == captured
    first_evidence = first_writer.get_snapshot_metadata(captured)

    for stores in ((captured_first, captured_second), (captured_second, captured_first)):
        with pytest.raises(MetadataConflictError, match="snapshot metadata"):
            Repo(stores).references().exact(captured.object).where(
                field("snapshot", "saved_at").eq(first_evidence.saved_at)
            ).state_refs()


def test_lineage_and_snapshot_projections_use_timestamps_and_exclude_captured_annotations(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="none"), clock=lambda: 0)
    state = _saved(repo, object_values={"before": "capture"}, state_values={"score": 1})
    repo.set_metadata(state.object, {"after": "current"})
    lineage = repo.get_lineage_metadata(state.object)
    snapshot = repo.get_snapshot_metadata(state)

    assert repo.references().where(field("lineage", "creation_status").eq(lineage.creation_status)).object_refs().one() == state.object
    assert repo.references().where(field("lineage", "created_at").eq(lineage.created_at)).object_refs().one() == state.object
    assert repo.references().where(field("lineage", "created_at").ge(0)).object_refs().one() == state.object
    assert repo.references().where(field("snapshot", "saved_at").eq(snapshot.saved_at)).state_refs().one() == state
    assert repo.references().where(field("snapshot", "environment_status").eq(snapshot.environment_status)).state_refs().one() == state
    assert repo.references().where(field("snapshot", "requirements_status").eq(snapshot.requirements_status)).state_refs().one() == state
    assert repo.references().where(field("snapshot", "requirements_coverage").eq(snapshot.requirements_coverage)).state_refs().one() == state
    assert repo.references().where(field("snapshot").exists()).state_refs().one() == state
    assert repo.references().where(field("object", "before").missing()).object_refs().one() == state.object
    assert repo.references().where(field("snapshot", "captured_object_annotations").missing()).state_refs().one() == state
    if snapshot.environment is not None:
        assert repo.references().where(field("snapshot", "environment", "kind").eq(snapshot.environment.kind)).state_refs().one() == state
    if snapshot.requirements is not None:
        assert repo.references().where(field("snapshot", "requirements", "tags").eq(list(snapshot.requirements.tags))).state_refs().one() == state
    with pytest.raises(ValueError, match="timezone"):
        field("lineage", "created_at").eq(datetime(1970, 1, 1))
    with pytest.raises(TypeError, match="Datetime"):
        field("object", "value").eq(datetime.now(timezone.utc))


def test_definition_where_composition_store_authority_and_zip_scan(tmp_path, monkeypatch):
    first = DirStore(tmp_path / "first", query_index="sqlite")
    second = DirStore(tmp_path / "second", query_index="memory")
    repo = Repo([first, second])
    value = QueryMetadataValue("first", repo=repo)
    state = repo.save_object(value, store=first, annotations=SaveAnnotations(object={"team": "vision"}))
    repo.save_object(value, store=second, source_store=first)
    repo.set_metadata(state.object, {"team": "platform"}, store=second)
    predicate = field("object", "team").eq("vision")

    with pytest.raises(MetadataConflictError):
        repo.references().where(predicate).object_refs()
    assert repo.references().where(predicate).in_store(first).object_refs().one() == state.object
    assert repo.references().in_store(first).where(predicate).object_refs().one() == state.object
    assert repo.references().in_store(first).where(
        field("object", "team").eq("vision") & field("object", "missing").missing()
    ).object_refs().one() == state.object
    with pytest.raises(MetadataConflictError):
        repo.query(Definition(QueryMetadataValue, name="first")).where(predicate).object_refs()
    assert repo.query(Definition(QueryMetadataValue, name="first")).references().in_store(first).where(predicate).object_refs().one() == state.object
    with pytest.raises(ValueError, match="connected"):
        repo.references().in_store(DirStore(tmp_path / "other"))

    archive = tmp_path / "metadata.zip"
    zip_store = ZipStore(archive)
    zip_repo = Repo(zip_store)
    zip_state = _saved(zip_repo, object_values={"team": "vision"})
    zip_repo.close(flush=True)
    reopened_store = ZipStore.open_existing(archive)
    reopened = Repo(reopened_store)
    monkeypatch.setattr(reopened_store, "open_query_index", lambda: pytest.fail("metadata query used a derived index"))
    monkeypatch.setattr(reopened_store, "open_local_state", lambda *_args: pytest.fail("metadata query opened a payload"))
    monkeypatch.setattr(reopened, "_materialize_cdef", lambda *_args: pytest.fail("metadata query materialized an Object"))
    assert reopened.references().where(predicate).object_refs().one() == zip_state.object
