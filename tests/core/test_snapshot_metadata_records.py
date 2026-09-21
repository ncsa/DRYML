from datetime import datetime, timedelta, timezone

import pytest

from dryml.core.definition import Definition
from dryml.core.object import Object
from dryml.core.reference_values import ObjectRef, StateRef
from dryml.core.utils.graph.path import GraphPath
from dryml.environments import EnvironmentRecord, PlatformRecord, PythonRecord


class _MetadataFixture(Object):
    pass


class _OtherMetadataFixture(Object):
    pass


def _snapshot_target(cls=_MetadataFixture):
    definition = Definition(cls).concretize()
    object_ref = ObjectRef(definition, {})
    return StateRef(object_ref, {})


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


def test_snapshot_metadata_round_trips_absent_and_empty_captured_mappings():
    from dryml.core import LineageMetadata, SnapshotMetadata, decode_snapshot_metadata, encode_snapshot_metadata
    from dryml.formats import EnvelopeError
    from dryml.records import GenericRecord, encode_record

    target = _snapshot_target()
    environment = EnvironmentRecord(
        python=PythonRecord("3.12", "CPython", executable="/python"),
        platform=PlatformRecord("Linux", "release", "version", "machine", "platform"),
        details={"preserved": "full domain payload"},
    )
    value = SnapshotMetadata(
        state_ref=target,
        lineages={GraphPath(): LineageMetadata(target.object, "unknown", None)},
        saved_at=datetime(1970, 1, 1, 0, 0, 0, 250000, tzinfo=timezone.utc),
        environment=environment,
        environment_status="known",
        requirements=None,
        requirements_status="empty",
        requirements_coverage="complete",
        captured_object_annotations=None,
        captured_state_annotations={},
    )

    encoded = encode_snapshot_metadata(value)
    decoded = decode_snapshot_metadata(encoded, target)

    assert decoded.saved_at == value.saved_at
    assert decoded.captured_object_annotations is None
    assert decoded.captured_state_annotations == {}
    assert decoded.lineages[GraphPath()] == value.lineages[GraphPath()]
    assert decoded.environment == environment

    wrong_target = _snapshot_target(_OtherMetadataFixture)
    with pytest.raises(Exception, match="target"):
        decode_snapshot_metadata(encoded, wrong_target)

    malformed_data = encoded["payload"]["data"]
    malformed_data["environment"] = {"not": "an environment envelope"}
    malformed = encode_record(GenericRecord("dryml.core.snapshot_metadata", 1, malformed_data))
    with pytest.raises(EnvelopeError, match="owner validation"):
        decode_snapshot_metadata(malformed, target)
