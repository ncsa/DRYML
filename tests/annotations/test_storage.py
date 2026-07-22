import pytest

import dryml.annotations as ann


def _fragment(value="dryml"):
    return ann.AnnotationFragment("environment", "requirement", {"requirements": [value]}, ann.SourceTrace("synthetic"))


def test_annotation_id_stability_and_semantic_changes():
    left = ann.attach_annotation_id(ann.make_annotation_spec(_fragment()))
    right = ann.attach_annotation_id(ann.make_annotation_spec(ann.AnnotationFragment.from_data(_fragment().to_data())))
    changed = ann.attach_annotation_id(ann.make_annotation_spec(_fragment("dryml>=1")))

    assert left["schema"] == "dryml.annotation.v1"
    assert left["id"].startswith("annotation-v1-")
    assert left["id"] == right["id"]
    assert left["id"] != changed["id"]
    assert ann.compute_annotation_id(left) == left["id"]
    assert "dryml_object" not in left


def test_annotation_spec_validation_rejects_bad_payload():
    spec = ann.make_annotation_spec(_fragment())
    spec["payload"] = {"kind": "requirement"}
    with pytest.raises(ann.AnnotationValidationError):
        ann.validate_annotation_spec(spec)
