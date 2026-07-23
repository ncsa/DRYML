import pytest

from dryml.formats.canonical import canonical_json_dumps
from dryml.formats.envelope import envelope_payload_for_id, make_envelope, validate_envelope
from dryml.formats.errors import EnvelopeError


class FalseyMapping(dict):
    def __bool__(self):
        return False


def test_make_envelope_returns_canonical_json_compatible_data():
    envelope = make_envelope(
        schema="dryml.example.v1",
        kind="example",
        schema_version=1,
        payload={"b": 2, "a": [1]},
    )

    assert canonical_json_dumps(envelope) == '{"kind":"example","payload":{"a":[1],"b":2},"schema":"dryml.example.v1","schema_version":1}'


def test_payload_defaults_to_empty_mapping():
    assert make_envelope(schema="dryml.example.v1", kind="example")["payload"] == {}


def test_make_envelope_preserves_falsey_mapping_payload():
    envelope = make_envelope(
        schema="dryml.example.v1",
        kind="example",
        payload=FalseyMapping({"kept": True}),
    )

    assert envelope["payload"] == {"kept": True}


def test_matching_schema_and_kind_validates():
    envelope = make_envelope(schema="dryml.example.v1", kind="example")
    assert validate_envelope(envelope, schema="dryml.example.v1", kind="example") is envelope


def test_mismatched_schema_or_kind_rejects():
    envelope = make_envelope(schema="dryml.example.v1", kind="example")
    with pytest.raises(EnvelopeError, match="schema mismatch"):
        validate_envelope(envelope, schema="dryml.other.v1")
    with pytest.raises(EnvelopeError, match="kind mismatch"):
        validate_envelope(envelope, kind="other")


def test_missing_required_fields_reject():
    with pytest.raises(EnvelopeError, match="missing schema"):
        validate_envelope({"kind": "example", "payload": {}})
    with pytest.raises(EnvelopeError, match="missing kind"):
        validate_envelope({"schema": "dryml.example.v1", "payload": {}})


def test_non_mapping_payload_rejects():
    with pytest.raises(EnvelopeError, match="payload must be a mapping"):
        make_envelope(schema="dryml.example.v1", kind="example", payload=["not", "mapping"])
    with pytest.raises(EnvelopeError, match="payload must be a mapping"):
        validate_envelope({"schema": "dryml.example.v1", "kind": "example", "payload": []})


def test_validate_envelope_rejects_non_json_ready_payload_by_default():
    with pytest.raises(EnvelopeError, match="payload is not JSON serializable"):
        validate_envelope({"schema": "dryml.example.v1", "kind": "example", "payload": {"bad": object()}})


def test_validate_envelope_rejects_non_json_ready_metadata_by_default():
    with pytest.raises(EnvelopeError, match="metadata is not JSON serializable"):
        validate_envelope(
            {
                "schema": "dryml.example.v1",
                "kind": "example",
                "payload": {},
                "metadata": {"value": float("nan")},
            }
        )


def test_validate_envelope_can_skip_json_ready_validation_explicitly():
    envelope = {"schema": "dryml.example.v1", "kind": "example", "payload": {"bad": object()}}

    assert validate_envelope(envelope, require_json_ready=False) is envelope


@pytest.mark.parametrize("schema_version", [0, -1, True, "1"])
def test_envelope_schema_version_rejects_invalid_values(schema_version):
    with pytest.raises(EnvelopeError, match="schema_version is invalid"):
        validate_envelope(
            {
                "schema": "dryml.example.v1",
                "kind": "example",
                "schema_version": schema_version,
                "payload": {},
            }
        )


def test_id_payload_excludes_id_by_default():
    envelope = make_envelope(
        schema="dryml.example.v1",
        kind="example",
        schema_version=1,
        id="record-v1-" + "0" * 64,
        payload={"x": True},
    )

    assert "id" not in envelope_payload_for_id(envelope)
    assert envelope_payload_for_id(envelope, include_id=True)["id"] == envelope["id"]


def test_metadata_is_deterministic_but_excluded_from_id_payload():
    envelope = make_envelope(
        schema="dryml.example.v1",
        kind="example",
        payload={"x": 1},
        metadata={"created_at": "volatile", "source": "test"},
    )

    assert canonical_json_dumps(envelope["metadata"]) == '{"created_at":"volatile","source":"test"}'
    assert "metadata" not in envelope_payload_for_id(envelope)
