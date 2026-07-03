import pytest

from dryml.formats.ids import content_id
from dryml.records import LocatedRecordRef, LocatedSpecRef, RecordRef, RecordValidationError, SpecRef, SpecValidationError


def test_record_ref_json_round_trip():
    record_id = content_id("record", 1, {"x": 1})
    ref = RecordRef(record_id)

    assert str(ref) == record_id
    assert ref.to_json() == record_id
    assert RecordRef.from_json(record_id) == ref


def test_record_ref_rejects_wrong_prefix_and_malformed_ids():
    with pytest.raises(RecordValidationError):
        RecordRef(content_id("spec", 1, {}))
    with pytest.raises(RecordValidationError):
        RecordRef("record-v1-nothex")


def test_located_record_ref_json_round_trip_and_store_ref_required():
    record_id = content_id("record", 1, {"x": 1})
    ref = LocatedRecordRef("store://local", record_id)

    assert LocatedRecordRef.from_json(ref.to_json()) == ref
    with pytest.raises(RecordValidationError):
        LocatedRecordRef("", record_id)


def test_spec_ref_json_round_trip_and_kind_prefix_validation():
    spec_id = content_id("repr", 1, {"x": 1})
    ref = SpecRef(spec_id, kind="representation")

    assert SpecRef.from_json(ref.to_json()) == ref
    assert SpecRef.from_json(spec_id) == SpecRef(spec_id)
    with pytest.raises(SpecValidationError):
        SpecRef(spec_id, kind="operation")


def test_located_spec_ref_json_round_trip_and_store_ref_required():
    spec_id = content_id("op", 1, {"x": 1})
    ref = LocatedSpecRef("store://local", spec_id, kind="operation")

    assert LocatedSpecRef.from_json(ref.to_json()) == ref
    with pytest.raises(SpecValidationError):
        LocatedSpecRef("", spec_id)
