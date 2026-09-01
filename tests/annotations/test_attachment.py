"""Tests for direct process-local annotation attachment."""

from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import pytest

from dryml.annotations import (
    ANNOTATION_ATTR,
    Annotation,
    AnnotationValidationError,
    UnsupportedAnnotationTargetError,
    attach_annotation,
    own_annotations,
)


def test_attachment_preserves_supported_target_identity_and_behavior():
    """Functions, classes, and safe descriptors retain their ordinary behavior."""

    def function(value):
        return value + 1

    def class_body(cls, value):
        return cls.__name__, value

    class Descriptor:
        def __get__(self, instance, owner):
            return function

    static_descriptor = staticmethod(function)
    class_descriptor = classmethod(class_body)
    descriptor = Descriptor()
    targets = (function, type("Subject", (), {}), static_descriptor, class_descriptor, descriptor)

    for index, target in enumerate(targets):
        annotation = Annotation(f"consumer.entry-{index}", index)
        assert attach_annotation(target, annotation) is target
        assert own_annotations(target) == (annotation,)

    class Subject:
        static = static_descriptor
        method = class_descriptor
        custom = descriptor

    assert function(1) == 2
    assert Subject.static(1) == 2
    assert Subject.method(1) == ("Subject", 1)
    assert Subject().custom(1) == 2


def test_attachment_rejects_unsafe_targets_without_partial_mutation():
    """Unsupported targets fail before the kernel installs an attachment tuple."""

    class UnsafeDescriptor:
        def __get__(self, instance, owner):
            return None

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

    property_target = property(lambda self: 1)
    unsafe = UnsafeDescriptor()
    targets = (property_target, int.__add__, object(), unsafe)

    for target in targets:
        with pytest.raises(UnsupportedAnnotationTargetError):
            attach_annotation(target, Annotation("consumer.unsupported", object()))
        if hasattr(target, "__dict__"):
            assert ANNOTATION_ATTR not in vars(target)

    with pytest.raises(AnnotationValidationError, match="requires an Annotation"):
        attach_annotation(lambda: None, object())


def test_attachment_rejects_class_with_mutation_intercepting_metaclass():
    """Class attachment never invokes a custom metaclass mutation hook."""

    class InterceptingMeta(type):
        def __setattr__(cls, name, value):
            raise AssertionError("metaclass mutation hook must not run")

    class Subject(metaclass=InterceptingMeta):
        pass

    with pytest.raises(UnsupportedAnnotationTargetError):
        attach_annotation(Subject, Annotation("consumer.unsupported", object()))
    assert ANNOTATION_ATTR not in type.__getattribute__(Subject, "__dict__")


def test_attachment_rejects_reserved_data_descriptor_without_invoking_it():
    """A target cannot intercept the kernel's reserved attachment attribute."""

    class CollisionDescriptor:
        def __get__(self, instance, owner):
            return self

        def get_annotations(self):
            raise AssertionError("reserved attachment getter must not run")

        def set_annotations(self, value):
            raise AssertionError("reserved attachment setter must not run")

        __dryml_annotations__ = property(get_annotations, set_annotations)

    target = CollisionDescriptor()
    with pytest.raises(UnsupportedAnnotationTargetError):
        attach_annotation(target, Annotation("consumer.collision", object()))
    assert object.__getattribute__(target, "__dict__") == {}


def test_unsupported_error_does_not_invoke_a_hostile_type_name_hook():
    """Bounded target diagnostics use the native type name."""

    class HostileMeta(type):
        @property
        def __name__(cls):
            raise AssertionError("type name hook must not run")

    class Unsupported(metaclass=HostileMeta):
        pass

    with pytest.raises(UnsupportedAnnotationTargetError):
        attach_annotation(Unsupported(), Annotation("consumer.unsupported", object()))


def test_direct_lookup_is_exact_ordered_and_rejects_corruption():
    """Direct lookup neither inherits nor repairs malformed target metadata."""

    class Base:
        pass

    class Child(Base):
        pass

    first = Annotation("consumer.first", 1)
    second = Annotation("consumer.second", 2)
    assert ANNOTATION_ATTR == "__dryml_annotations__"
    attach_annotation(Base, first)
    attach_annotation(Base, second)

    assert own_annotations(Base) == (first, second)
    assert own_annotations(Child) == ()

    def non_tuple():
        return None

    setattr(non_tuple, ANNOTATION_ATTR, "not-a-tuple")
    with pytest.raises(AnnotationValidationError, match="malformed"):
        own_annotations(non_tuple)

    def function():
        return None

    setattr(function, ANNOTATION_ATTR, (first, "not-an-annotation"))
    with pytest.raises(AnnotationValidationError, match="malformed"):
        own_annotations(function)
    with pytest.raises(AnnotationValidationError, match="malformed"):
        attach_annotation(function, second)
    assert getattr(function, ANNOTATION_ATTR) == (first, "not-an-annotation")


def test_direct_lookup_rejects_hostile_tuple_and_annotation_subclasses_without_hooks():
    """Metadata validation rejects subclasses before iteration or field access."""

    class HostileTuple(tuple):
        def __iter__(self):
            raise AssertionError("metadata iteration hook must not run")

    class HostileAnnotation(Annotation):
        hooks_enabled = False

        def __getattribute__(self, name):
            if name == "key" and HostileAnnotation.hooks_enabled:
                raise AssertionError("annotation field hook must not run")
            return super().__getattribute__(name)

    entry = Annotation("consumer.entry", object())
    hostile_entry = HostileAnnotation("consumer.hostile", object())
    HostileAnnotation.hooks_enabled = True

    def tuple_target():
        return None

    setattr(tuple_target, ANNOTATION_ATTR, HostileTuple((entry,)))
    with pytest.raises(AnnotationValidationError, match="malformed"):
        own_annotations(tuple_target)

    def annotation_target():
        return None

    setattr(annotation_target, ANNOTATION_ATTR, (hostile_entry,))
    with pytest.raises(AnnotationValidationError, match="malformed"):
        own_annotations(annotation_target)
    with pytest.raises(AnnotationValidationError, match="requires"):
        attach_annotation(annotation_target, hostile_entry)


def test_attachment_maps_immutable_class_assignment_failure_without_mutation():
    """Immutable built-in classes expose the bounded target error with its cause."""

    with pytest.raises(UnsupportedAnnotationTargetError) as error:
        attach_annotation(int, Annotation("consumer.immutable", object()))

    assert isinstance(error.value.__cause__, TypeError)
    assert ANNOTATION_ATTR not in int.__dict__


def test_concurrent_read_only_collection_observes_completed_attachment():
    """Readers may collect a stable target only after its setup is complete."""

    class Subject:
        pass

    annotation = Annotation("consumer.readers", object())
    attach_annotation(Subject, annotation)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(own_annotations, (Subject,) * 12))

    assert results == [(annotation,)] * 12


def test_fresh_interpreter_reconstructs_its_own_annotations():
    """Process-local attachments are not expected to transfer to a fresh worker."""

    script = """
from dryml.annotations import Annotation, attach_annotation, own_annotations
def target():
    return None
attach_annotation(target, Annotation('consumer.worker', 1))
assert [entry.key for entry in own_annotations(target)] == ['consumer.worker']
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
