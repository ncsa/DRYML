import pytest

import dryml.annotations as ann


def test_model_round_trips():
    target = ann.AnnotationTarget("function", "mod", "fn", metadata={"a": [1]})
    source = ann.SourceTrace("decorator", target, label="label", namespace="environment")
    fragment = ann.AnnotationFragment("environment", "requirement", {"requirements": ["dryml"]}, source)

    assert ann.AnnotationTarget.from_data(target.to_data()).to_data() == target.to_data()
    assert ann.SourceTrace.from_data(source.to_data()).to_data() == source.to_data()
    assert ann.AnnotationFragment.from_data(fragment.to_data()).to_data() == fragment.to_data()


def test_invalid_model_data_rejected():
    with pytest.raises(ann.AnnotationValidationError):
        ann.AnnotationFragment("bad namespace", "requirement", {}, ann.SourceTrace("synthetic"))
    with pytest.raises(ann.AnnotationValidationError):
        ann.AnnotationFragment("environment", "bad", {}, ann.SourceTrace("synthetic"))
    with pytest.raises(ann.AnnotationValidationError):
        ann.AnnotationFragment("environment", "requirement", {"x": object()}, ann.SourceTrace("synthetic"))
