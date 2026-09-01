"""Tests for descriptor-first collection and identity deduplication."""

from dryml.annotations import Annotation, annotations_for_method, attach_annotation, collect_annotations


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
