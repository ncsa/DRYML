import pytest

from dryml.formats.ids import content_id
from dryml.formats.refs import format_cdef_id
from dryml.records import (
    AdapterRecord,
    DataRecord,
    ProgramRecord,
    RecordValidationError,
    StorageRef,
    StoredStateRecord,
    default_object_state_representation_spec,
    typed_record_from_envelope,
)


def _cdef(char="a"):
    return format_cdef_id(char * 64)


def _repr(payload=None):
    return default_object_state_representation_spec()["id"] if payload is None else content_id("repr", 1, payload)


def _record(char="r"):
    return content_id("record", 1, {"record": char})


def test_stored_state_round_trip_preserves_descriptive_save_extra():
    record = StoredStateRecord(
        subject_cdef_id=_cdef(),
        representation_id=_repr(),
        storage=(StorageRef.object_dir(_cdef()),),
        extra={"save": {"reason": "explicit-root"}},
    )
    wrapped = StoredStateRecord.from_envelope(record.to_envelope())

    assert wrapped.to_payload() == record.to_payload()
    assert wrapped.extra["save"]["reason"] == "explicit-root"


def test_data_program_and_adapter_round_trips():
    data = DataRecord(representation_id=_repr({"data": 1}), storage=(StorageRef.self_product(role="data"),), manifest={"entries": []}, preview={"rows": 3})
    assert DataRecord.from_envelope(data.to_envelope()).to_payload() == data.to_payload()

    program = ProgramRecord(
        representation_id=_repr({"program": 1}),
        storage=(StorageRef.self_product(role="program"),),
        target={"platform": "fake"},
        entrypoints={"main": "program.fake"},
        provider={"name": "fake.compiler"},
        toolchain={"name": "fake-toolchain"},
        manifest={"entries": []},
    )
    assert ProgramRecord.from_envelope(program.to_envelope()).to_payload() == program.to_payload()

    source = _record("source")
    target = _record("target")
    adapter = AdapterRecord(
        adapter={"name": "fake.normalize"},
        source_record_id=source,
        source_representation_id=_repr({"source": 1}),
        target_record_id=target,
        target_representation_id=_repr({"target": 1}),
        produced_records=(target,),
        derived_from=(source,),
    )
    assert AdapterRecord.from_envelope(adapter.to_envelope()).to_payload() == adapter.to_payload()


def test_typed_record_dispatch_and_invalid_payloads():
    stored = StoredStateRecord(_cdef(), _repr(), (StorageRef.object_dir(_cdef()),))
    assert isinstance(typed_record_from_envelope(stored.to_envelope()), StoredStateRecord)
    with pytest.raises(RecordValidationError):
        StoredStateRecord(_cdef(), _repr(), ())
    with pytest.raises(RecordValidationError):
        AdapterRecord(adapter={}, source_record_id=_record(), source_representation_id=_repr({"a": 1}), target_representation_id=_repr({"b": 1}))


def test_typed_record_from_envelope_rejects_string_sequences():
    stored = StoredStateRecord(_cdef(), _repr(), (StorageRef.object_dir(_cdef()),)).to_envelope()
    stored["payload"]["storage"] = "not-a-list"
    with pytest.raises(RecordValidationError):
        StoredStateRecord.from_envelope(stored)

    source = _record("source")
    target = _record("target")
    adapter = AdapterRecord(
        adapter={"name": "fake.normalize"},
        source_record_id=source,
        source_representation_id=_repr({"source": 1}),
        target_record_id=target,
        target_representation_id=_repr({"target": 1}),
        produced_records=(target,),
        derived_from=(source,),
    ).to_envelope()
    adapter["payload"]["produced_records"] = "not-a-list"
    with pytest.raises(RecordValidationError):
        AdapterRecord.from_envelope(adapter)
