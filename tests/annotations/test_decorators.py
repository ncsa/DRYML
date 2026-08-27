"""Identity and unsupported-target tests for direct annotation decorators."""

import pytest

from dryml.annotations import AnnotationTarget, SourceTrace, UnsupportedAnnotationTargetError, collect_fragments, require


def _decorator():
    return require(namespace="runtime", fragment={"limits": {"threads": 1}})


def test_supported_targets_preserve_identity_and_ordinary_behavior():
    def function(value):
        return value + 1
    assert _decorator()(function) is function
    static = staticmethod(function)
    class_method = classmethod(function)
    assert _decorator()(static) is static
    assert _decorator()(class_method) is class_method

    class Descriptor:
        def __get__(self, instance, owner):
            return function
    descriptor = Descriptor()
    assert _decorator()(descriptor) is descriptor
    assert function(1) == 2
    assert len(collect_fragments(function)) == 1


def test_non_extensible_property_and_descriptor_fail_without_partial_mutation():
    property_target = property(lambda self: 1)
    with pytest.raises(UnsupportedAnnotationTargetError):
        _decorator()(property_target)
    with pytest.raises(UnsupportedAnnotationTargetError):
        _decorator()(int.__add__)
    assert not hasattr(property_target, "__dryml_annotation_fragments__")


def test_declared_target_identity_is_independent_from_a_supplied_source_target():
    """Source provenance cannot replace the object receiving direct metadata."""

    source = SourceTrace("override", target=AnnotationTarget("synthetic", "source", "other"), namespace="runtime")
    descriptor = staticmethod(lambda: None)
    require(namespace="runtime", fragment={"value": 1}, source=source)(descriptor)
    fragment = descriptor.__dryml_annotation_fragments__[0]

    assert fragment.source is source
    assert fragment.target.kind == "descriptor"
    assert fragment.target.descriptor_kind == "staticmethod"
    assert fragment.target != source.target
