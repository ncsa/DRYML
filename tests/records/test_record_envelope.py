import pytest

from dryml.records import (
    RecordValidationError,
    attach_record_id,
    compute_record_id,
    make_record,
    record_payload_for_id,
    validate_record,
)


def test_make_record_builds_json_ready_record():
    record = make_record(kind="stored_state", payload={"b": 2, "a": 1}, metadata={"writer": "test"})

    assert record["schema"] == "dryml.record.v1"
    assert record["schema_version"] == 1
    assert record["payload"] == {"a": 1, "b": 2}
    assert validate_record(record) is record


def test_record_id_stable_and_payload_sensitive_but_metadata_ignored():
    left = attach_record_id(make_record(kind="stored_state", payload={"items": [{"b": 2, "a": 1}]}, metadata={"writer": "a"}))
    right = attach_record_id(make_record(kind="stored_state", payload={"items": [{"a": 1, "b": 2}]}, metadata={"writer": "b"}))
    changed = attach_record_id(make_record(kind="stored_state", payload={"items": [{"a": 1, "b": 3}]}, metadata={"writer": "a"}))

    assert left["id"] == right["id"]
    assert left["id"] != changed["id"]
    assert "metadata" not in record_payload_for_id(left)


@pytest.mark.parametrize(
    "record, match",
    [
        ({"kind": "stored_state", "schema_version": 1, "payload": {}}, "schema"),
        ({"schema": "dryml.record.v1", "schema_version": 1, "payload": {}}, "kind"),
        ({"schema": "dryml.record.v1", "schema_version": 1, "kind": "stored_state"}, "payload"),
        ({"schema": "wrong", "schema_version": 1, "kind": "stored_state", "payload": {}}, "envelope"),
        ({"schema": "dryml.record.v1", "schema_version": 2, "kind": "stored_state", "payload": {}}, "schema_version"),
        ({"schema": "dryml.record.v1", "schema_version": 1, "kind": "stored_state", "payload": []}, "payload"),
        ({"schema": "dryml.record.v1", "schema_version": 1, "kind": "stored_state", "payload": {}, "metadata": []}, "metadata"),
        ({"schema": "dryml.record.v1", "schema_version": 1, "kind": "unknown", "payload": {}}, "kind"),
        ({"schema": "dryml.record.v1", "schema_version": 1, "kind": "stored_state", "payload": {}, "semantic": "top"}, "unknown top-level"),
    ],
)
def test_validate_record_rejects_malformed_records(record, match):
    with pytest.raises(RecordValidationError, match=match):
        validate_record(record)


def test_record_rejects_mismatched_existing_id():
    record = make_record(kind="stored_state", payload={"x": 1})
    wrong = dict(record, id=compute_record_id(make_record(kind="stored_state", payload={"x": 2})))

    with pytest.raises(RecordValidationError, match="does not match"):
        attach_record_id(wrong)

    with pytest.raises(RecordValidationError, match="does not match"):
        validate_record(wrong)


def test_make_record_rejects_non_mapping_payload_and_metadata():
    with pytest.raises(RecordValidationError):
        make_record(kind="stored_state", payload=[])
    with pytest.raises(RecordValidationError):
        make_record(kind="stored_state", metadata=[])
