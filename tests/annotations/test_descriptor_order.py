"""Tests for descriptor-first collection and identity deduplication."""

import pytest

from dryml.annotations import (
    Annotation,
    UnsupportedAnnotationTargetError,
    annotations_for_members,
    annotations_for_method,
    attach_annotation,
    collect_annotations,
)


def test_shared_descriptor_and_function_annotation_is_emitted_once():
    """A shared carrier keeps its first descriptor position while equal peers remain."""

    def function():
        return "ok"

    descriptor = staticmethod(function)
    shared = Annotation("consumer.shared", "same")
    distinct_equal = Annotation("consumer.shared", "same")
    attach_annotation(descriptor, shared)
    attach_annotation(descriptor, distinct_equal)
    attach_annotation(function, shared)

    class Subject:
        method = descriptor

    assert annotations_for_method(Subject, "method") == (shared, distinct_equal)
    assert collect_annotations(descriptor) == (shared, distinct_equal)


def test_known_static_and_class_descriptors_are_unwrapped_without_binding():
    """Known descriptor metadata precedes their statically obtained functions."""

    def static_body():
        return "static"

    def class_body(cls):
        return cls.__name__

    static_descriptor = staticmethod(static_body)
    class_descriptor = classmethod(class_body)
    static_entry = Annotation("consumer.static", "descriptor")
    static_function_entry = Annotation("consumer.static", "function")
    class_entry = Annotation("consumer.class", "descriptor")
    class_function_entry = Annotation("consumer.class", "function")
    attach_annotation(static_descriptor, static_entry)
    attach_annotation(static_body, static_function_entry)
    attach_annotation(class_descriptor, class_entry)
    attach_annotation(class_body, class_function_entry)

    class Subject:
        static = static_descriptor
        method = class_descriptor

    assert annotations_for_method(Subject, "static") == (static_entry, static_function_entry)
    assert annotations_for_method(Subject, "method") == (class_entry, class_function_entry)
    assert [member.annotations for member in annotations_for_members(Subject)] == [
        (static_entry, static_function_entry),
        (class_entry, class_function_entry),
    ]


def test_known_descriptor_subclass_bypasses_hostile_func_hook():
    """Native descriptor storage wins over a subclass's dynamic ``__func__``."""

    class HostileStaticMethod(staticmethod):
        @property
        def __func__(self):
            raise AssertionError("descriptor function hook must not run")

    def function():
        return "ok"

    descriptor = HostileStaticMethod(function)
    descriptor_entry = Annotation("consumer.static", "descriptor")
    function_entry = Annotation("consumer.static", "function")
    attach_annotation(descriptor, descriptor_entry)
    attach_annotation(function, function_entry)

    assert collect_annotations(descriptor) == (descriptor_entry, function_entry)


def test_member_collection_does_not_unwrap_safe_custom_descriptors():
    """Custom descriptor annotations remain direct even when an inner function exists."""

    class Descriptor:
        def __init__(self, function):
            self.function = function

        def __get__(self, instance, owner):
            raise AssertionError("binding must not run")

    def function():
        return "ok"

    descriptor = Descriptor(function)
    descriptor_entry = Annotation("consumer.custom", "descriptor")
    function_entry = Annotation("consumer.custom", "function")
    attach_annotation(descriptor, descriptor_entry)
    attach_annotation(function, function_entry)

    class Subject:
        method = descriptor

    assert annotations_for_members(Subject)[0].annotations == (descriptor_entry,)


def test_member_collection_rejects_unsupported_descriptor_before_returning_partial_results():
    """Unsupported descriptor declarations use the existing bounded error path."""

    class Subject:
        def annotated(self):
            return "annotated"

        unsupported = property(lambda self: "unsupported")

    attach_annotation(Subject.__dict__["annotated"], Annotation("consumer.member", "annotated"))

    with pytest.raises(UnsupportedAnnotationTargetError):
        annotations_for_members(Subject)
