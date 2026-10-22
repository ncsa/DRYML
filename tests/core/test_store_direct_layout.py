from pathlib import Path

import pytest

from dryml.core import Object
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord


class DirectLayoutObject(Object):
    pass


def test_definition_records_use_direct_digest_paths_without_object_generations(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(DirectLayoutObject().definition)

    store.write_definition_record(record)

    path = Path(store.base_dir) / "definitions" / record.digest[:2] / f"{record.digest}.record"
    assert path.is_file()
    decoded = store.read_definition_record(record.digest)
    assert decoded.digest == record.digest
    assert decoded.definition.graph_equal(record.definition)
    assert not list(Path(store.base_dir).rglob(".state-generations"))
    assert not list(Path(store.base_dir).rglob(".state-current.pkl"))


@pytest.mark.parametrize("consumer", ("query", "rebuild"))
def test_definition_root_inventory_decodes_authority_once(tmp_path, monkeypatch, consumer):
    """Query capture and SQLite rebuild share already decoded root targets."""

    from dryml.core import Definition
    from dryml.core.query.source import SourceCapture

    store = DirStore(tmp_path / "store", query_index="sqlite")
    records = sorted(
        (DefinitionRecord(Definition(cls).concretize()) for cls in (DirectLayoutObject, Object)),
        key=lambda record: record.digest,
    )
    for record in records:
        store.write_definition_record(record)
    decoded = []
    original = store._read_file

    def read(path, record_type):
        value = original(path, record_type)
        if record_type is DefinitionRecord and value is not None:
            decoded.append(value.digest)
        return value

    monkeypatch.setattr(store, "_read_file", read)
    monkeypatch.setattr(store, "open_local_state", lambda *_a, **_kw: pytest.fail("inventory read payloads"))
    if consumer == "query":
        assert SourceCapture().capture_store(store).is_stored(records[0].definition)
    else:
        roots, _, _, _ = store.open_query_index()._capture_authority_cut()
        assert len(roots) == len(records)
        assert all(sum(root.graph_equal(record.definition) for root in roots) == 1
                   for record in records)
    assert decoded == [record.digest for record in records]


@pytest.mark.parametrize("damage", ("definition", "marker", "marker-path", "target", "main", "alias"))
def test_validated_inventory_preserves_malformed_authority_diagnostics(tmp_path, monkeypatch, damage):
    """Capture/rebuild refuse corrupt authority without changing it or reading payloads."""

    from dryml.core import Definition, ObjectRef
    from dryml.core.query.source import SourceCapture
    from dryml.core.store.records import MainRefRecord, StateAliasRecord
    from dryml.core.store.store import StoreAuthorityError, StoreCapabilityError

    store = DirStore(tmp_path / "store", query_index="sqlite")
    record = DefinitionRecord(Definition(DirectLayoutObject).concretize())
    store.write_definition_record(record)
    definition_path = Path(store._definition_path(record.digest))
    marker_path = Path(store._stored_root_path(record.digest))
    if damage == "definition":
        definition_path.write_bytes(b"incomplete")
    elif damage == "marker":
        marker_path.write_bytes(b"incomplete")
    elif damage == "marker-path":
        marker_path.rename(marker_path.with_name("0" * 64 + ".record"))
    elif damage == "target":
        definition_path.unlink()
    elif damage == "main":
        store.write_main_ref(MainRefRecord("0" * 64))
    else:
        reference = ObjectRef(record.definition, {})
        store.write_state_alias(StateAliasRecord("missing", reference, "0" * 64))
    files = {path.relative_to(store.base_dir): path.read_bytes()
             for path in Path(store.base_dir).rglob("*.record")}
    monkeypatch.setattr(store, "open_local_state", lambda *_a, **_kw: pytest.fail("inventory read payloads"))
    for reader in (lambda: SourceCapture().capture_store(store),
                   store.open_query_index()._capture_authority_cut):
        with pytest.raises((StoreAuthorityError, StoreCapabilityError)):
            reader()
    assert files == {path.relative_to(store.base_dir): path.read_bytes()
                     for path in Path(store.base_dir).rglob("*.record")}
    if damage in {"definition", "marker", "marker-path", "target"}:
        with pytest.raises(StoreAuthorityError):
            tuple(store.iter_stored_root_records())


def test_missing_marker_does_not_promote_closure_definition_and_fallback_validates(tmp_path):
    """Absent membership remains distinct from knowledge; legacy seam stays compatible."""

    from dryml.core.query.source import SourceCapture
    from dryml.core.store.store import Store

    store = DirStore(tmp_path / "store", query_index="memory")
    record = DefinitionRecord(DirectLayoutObject().definition)
    store.write_definition_record(record, stored_root=False)
    definitions, roots = Store._validated_definition_root_inventory(store)
    assert roots == () and len(definitions) == 1
    assert definitions[0].digest == record.digest
    assert definitions[0].definition.graph_equal(record.definition)
    facts = SourceCapture().capture_store(store)
    assert any(record.definition.graph_equal(known) for known in facts.knowledge())
    assert not facts.is_stored(record.definition)
