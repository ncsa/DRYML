from datetime import datetime, timedelta, timezone

import pytest

def test_tagged_metadata_mapping_preserves_types_and_detaches_values():
    from dryml.core import decode_metadata_mapping, encode_metadata_mapping

    source = {
        "text": "naive caf\u00e9",
        "list": [1, 1.0, None],
        "tuple": (True, {"nested": ("ok",)}),
    }
    encoded = encode_metadata_mapping(source)
    source["list"].append("changed")

    decoded = decode_metadata_mapping(encoded)

    assert decoded == {
        "list": [1, 1.0, None],
        "text": "naive caf\u00e9",
        "tuple": (True, {"nested": ("ok",)}),
    }
    decoded["list"].append("changed")
    assert decode_metadata_mapping(encoded)["list"] == [1, 1.0, None]


def test_tagged_metadata_mapping_rejects_noncanonical_and_overbound_values():
    from dryml.formats import CanonicalJSONError
    from dryml.core import decode_metadata_mapping, encode_metadata_mapping

    with pytest.raises(CanonicalJSONError, match="finite"):
        encode_metadata_mapping({"bad": float("nan")})
    with pytest.raises(CanonicalJSONError, match="depth"):
        encode_metadata_mapping({"value": [[[[[[[[[None]]]]]]]]]})
    with pytest.raises(CanonicalJSONError, match="canonical"):
        decode_metadata_mapping(["map", [["number", ["int", "01"]]]])


def test_timestamp_codec_handles_utc_fractional_and_unknown_values():
    from dryml.core import timestamp_from_seconds, timestamp_to_seconds

    instant = datetime(1970, 1, 1, 0, 0, 0, 250000, tzinfo=timezone.utc)
    assert timestamp_to_seconds(instant) == 0.25
    assert timestamp_from_seconds(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert timestamp_from_seconds(None) is None

    with pytest.raises(ValueError, match="UTC"):
        timestamp_to_seconds(datetime(1970, 1, 1))
    with pytest.raises(ValueError, match="UTC"):
        timestamp_to_seconds(datetime(1970, 1, 1, tzinfo=timezone(timedelta(hours=1))))
    with pytest.raises(ValueError, match="timestamp"):
        timestamp_from_seconds(True)


def test_capture_status_and_coverage_combinations_are_validated():
    from dryml.core import SnapshotCapture

    capture = SnapshotCapture(
        lineages={},
        saved_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
        environment=None,
        environment_status="unavailable",
        requirements=None,
        requirements_status="empty",
        requirements_coverage="complete",
    )
    assert capture.requirements_status == "empty"

    with pytest.raises(ValueError, match="unavailable"):
        SnapshotCapture(
            lineages={},
            saved_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
            environment=None,
            environment_status="unavailable",
            requirements=None,
            requirements_status="unavailable",
            requirements_coverage="complete",
        )

    with pytest.raises(ValueError, match="valued"):
        SnapshotCapture(
            lineages={},
            saved_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
            environment=None,
            environment_status="unavailable",
            requirements=None,
            requirements_status="value",
            requirements_coverage="complete",
        )
