"""Tests for deterministic static annotation collection."""

from dataclasses import dataclass

import pytest

from dryml.annotations import (
    Annotation,
    AnnotationValidationError,
    UnsupportedAnnotationTargetError,
    annotations_for_class,
    attach_annotation,
    collect_annotations,
)


def _attach(key, value):
    """Return a consumer-owned decorator using the passive attachment seam."""

    def decorate(target):
        return attach_annotation(target, Annotation(key, value))

    return decorate


def test_class_collection_uses_reversed_c3_and_direct_declaration_order():
    """Diamond declarations are ordered base-to-subclass through reversed C3."""

    @_attach("requirements.entry", "root")
    class Root:
        pass

    @_attach("requirements.entry", "left")
    class Left(Root):
        pass

    @_attach("requirements.entry", "right")
    class Right(Root):
        pass

    @_attach("requirements.entry", "leaf-first")
    @_attach("requirements.entry", "leaf-second")
    class Leaf(Left, Right):
        pass

    assert [item.value for item in annotations_for_class(Leaf)] == [
        "root",
        "right",
        "left",
        "leaf-second",
        "leaf-first",
    ]
    assert [item.value for item in collect_annotations(Leaf, key="requirements.entry")] == [
        "root",
        "right",
        "left",
        "leaf-second",
        "leaf-first",
    ]


def test_collection_filters_by_exact_key_without_interpreting_consumer_values():
    """Independent consumer types select their own annotations by owner key."""

    @dataclass(frozen=True)
    class MethodTrait:
        label: str

    @dataclass(frozen=True)
    class Requirement:
        capability: str

    class Subject:
        pass

    method_trait = MethodTrait("vectorized")
    requirement = Requirement("gpu")
    attach_annotation(Subject, Annotation("method.trait", method_trait))
    attach_annotation(Subject, Annotation("requirements.capability", requirement))

    assert collect_annotations(Subject, key="method.trait") == (Subject.__dict__["__dryml_annotations__"][0],)
    assert [item.value.capability for item in collect_annotations(Subject, key="requirements.capability")] == ["gpu"]


@pytest.mark.parametrize("key", [0, "", "1invalid", "x" * 129, "not-ascii-\u00f1"])
def test_collection_rejects_invalid_filter_keys(key):
    """Filtering validates only the generic annotation-key grammar."""

    class Subject:
        pass

    with pytest.raises(AnnotationValidationError, match="key"):
        collect_annotations(Subject, key=key)


def test_collection_rejects_unsupported_targets_and_corrupted_metadata():
    """Collection fails rather than returning incomplete generic declarations."""

    class DefinitionLike:
        pass

    with pytest.raises(UnsupportedAnnotationTargetError):
        collect_annotations(DefinitionLike())
    with pytest.raises(AnnotationValidationError):
        annotations_for_class(DefinitionLike())

    class Corrupt:
        pass

    Corrupt.__dryml_annotations__ = ("bad",)
    with pytest.raises(AnnotationValidationError, match="malformed"):
        annotations_for_class(Corrupt)
