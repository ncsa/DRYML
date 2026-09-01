"""Tests for the passive annotation carrier."""

import pytest

from dryml.annotations import Annotation, AnnotationValidationError


class OpaqueValue:
    """Value that fails if the kernel tries structural comparison or hashing."""

    def __eq__(self, other):
        raise AssertionError("annotation values must remain opaque")

    def __hash__(self):
        raise AssertionError("annotation values must remain opaque")


def test_annotation_is_a_shallow_frozen_identity_carrier():
    """Carriers retain values by identity without structural equality or hashing."""

    value = OpaqueValue()
    first = Annotation("method.trait", value)
    second = Annotation("method.trait", value)

    assert first.key == "method.trait"
    assert first.value is value
    assert first != second
    assert hash(first) != hash(second)
    assert not hasattr(first, "__dict__")
    assert not hasattr(first, "id")
    assert not hasattr(first, "to_data")
    with pytest.raises(AttributeError):
        first.key = "other"


@pytest.mark.parametrize(
    "key",
    [None, "", "1invalid", "has space", "slash/name", "x" * 129, "naive\u0308"],
)
def test_annotation_rejects_invalid_keys(key):
    """Keys are bounded ASCII identifiers before any target can be mutated."""

    with pytest.raises(AnnotationValidationError, match="key"):
        Annotation(key, object())


def test_annotation_accepts_owner_qualified_key_and_opaque_mutable_value():
    """The kernel preserves opaque value identity without copying or deep freezing."""

    value = {"consumer": []}
    annotation = Annotation("requirements.owner-v2", value)

    assert annotation.value is value
    value["consumer"].append("later")
    assert annotation.value["consumer"] == ["later"]
