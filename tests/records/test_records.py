import pytest


def test_generic_records_are_closed_versioned_and_detached():
    from dryml.records import GenericRecord, decode_record, encode_record

    source = {"nested": [None, {"label": "cafe"}]}
    encoded = encode_record(GenericRecord("dryml.test", 1, source))
    source["nested"][1]["label"] = "changed"

    decoded = decode_record(encoded)

    assert decoded == GenericRecord("dryml.test", 1, {"nested": [None, {"label": "cafe"}]})
    first = decoded.data
    first["nested"][1]["label"] = "mutated"
    assert decode_record(encoded).data["nested"][1]["label"] == "cafe"


def test_generic_records_reject_wrong_schema_and_oversized_envelopes():
    from dryml.formats import EnvelopeError
    from dryml.records import GenericRecord, decode_record, encode_record

    encoded = encode_record(GenericRecord("dryml.test", 1, {"value": 1}))
    encoded["schema"] = "dryml.other.v1.1"
    with pytest.raises(EnvelopeError, match="schema mismatch"):
        decode_record(encoded)

    with pytest.raises(EnvelopeError, match="exceeds byte bound"):
        encode_record(GenericRecord("dryml.test", 1, {"value": "x" * 100}), max_bytes=32)


def test_generic_records_reject_cyclic_data_without_mutating_input():
    from dryml.formats import CanonicalJSONError
    from dryml.records import GenericRecord

    source = {"items": []}
    source["items"].append(source)

    with pytest.raises(CanonicalJSONError, match="cyclic"):
        GenericRecord("dryml.test", 1, source)
    assert source["items"][0] is source


def test_records_do_not_import_core_or_environment_consumers():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dryml.records; "
            "assert 'dryml.core' not in sys.modules; "
            "assert 'dryml.environments' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
