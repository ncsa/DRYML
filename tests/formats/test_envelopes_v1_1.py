import pytest

from dryml.formats import ContentIDError, EnvelopeError, make_envelope, semantic_id, validate_envelope


def test_envelopes_are_closed_versioned_and_bound():
    value_id = semantic_id("sample", "dryml.sample.v1.1", "sample", {"value": 1})
    data = make_envelope(schema="dryml.sample.v1.1", kind="sample", prefix="sample", payload={"value": 1}, semantic_id=value_id)
    with pytest.raises(EnvelopeError, match="unsupported metadata contract version"):
        validate_envelope({**data, "contract_version": "1"}, schema="dryml.sample.v1.1", kind="sample", prefix="sample", identifying_payload={"value": 1})
    with pytest.raises(EnvelopeError, match="fields are closed"):
        validate_envelope({**data, "future": True}, schema="dryml.sample.v1.1", kind="sample", prefix="sample", identifying_payload={"value": 1})
    with pytest.raises(EnvelopeError, match="exceeds byte bound"):
        make_envelope(
            schema="dryml.sample.v1.1",
            kind="sample",
            prefix="sample",
            payload={"value": "x" * 10},
            semantic_id=semantic_id("sample", "dryml.sample.v1.1", "sample", {"value": "x" * 10}),
            max_bytes=10,
        )


def test_envelope_output_validates_its_id_and_old_versions_report_context():
    with pytest.raises(ContentIDError, match="does not match"):
        make_envelope(
            schema="dryml.sample.v1.1",
            kind="sample",
            prefix="sample",
            payload={"value": 1},
            semantic_id="sample-v1.1-" + "0" * 64,
        )

    with pytest.raises(EnvelopeError) as excinfo:
        validate_envelope(
            {"schema_version": 1, "payload": {}},
            schema="dryml.sample.v1.1",
            kind="sample",
            prefix="sample",
            identifying_payload={},
        )
    assert excinfo.value.context == {"observed_version": None, "supported_version": "1.1"}
