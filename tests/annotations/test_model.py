"""Closed annotation-model and semantic-identity coverage."""

import pytest

from dryml.annotations import AnnotationFragment, AnnotationTarget, AnnotationValidationError, SourceTrace


def _fragment(**changes):
    values = {"target": AnnotationTarget("function", "test_mod", "f"), "namespace": "runtime", "kind": "default", "fragment": {"limits": {"threads": 1}}, "source": SourceTrace("synthetic", label="first", namespace="runtime"), "priority": 1, "merge_policy": "merge"}
    values.update(changes)
    return AnnotationFragment(**values)


def test_annotation_fragment_round_trip_and_identity_includes_all_payload_fields():
    fragment = _fragment()
    assert AnnotationFragment.from_data(fragment.to_data()) == fragment
    assert fragment.id.startswith("annotation-v1.1-")
    for field, value in (("target", AnnotationTarget("function", "test_mod", "other")), ("namespace", "other"), ("kind", "requirement"), ("fragment", {"limits": {"threads": 2}}), ("source", SourceTrace("synthetic", label="other", namespace="runtime")), ("priority", 2), ("merge_policy", "replace")):
        assert _fragment(**{field: value}).id != fragment.id


def test_annotation_fragment_rejects_closed_malformed_and_version_mismatch_data():
    payload = _fragment().to_data()
    payload["payload"]["extra"] = True
    with pytest.raises(AnnotationValidationError):
        AnnotationFragment.from_data(payload)
    payload = _fragment().to_data()
    payload["contract_version"] = "1.0"
    with pytest.raises(AnnotationValidationError):
        AnnotationFragment.from_data(payload)
    with pytest.raises(AnnotationValidationError):
        _fragment(fragment={"not-json": object()})


def test_annotation_fragment_accepts_an_unattached_id_and_validates_source_lines():
    """IDs are optional envelope attachments and source lines are one-based."""

    data = _fragment().to_data()
    data.pop("id")
    assert AnnotationFragment.from_data(data) == _fragment()
    with pytest.raises(AnnotationValidationError, match="positive"):
        SourceTrace("synthetic", line=0)
