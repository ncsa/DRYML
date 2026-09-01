"""Tests for static selected-method annotation collection."""

import pytest

from dryml.annotations import (
    Annotation,
    AnnotationValidationError,
    annotations_for_method,
    attach_annotation,
)


def test_method_collection_selects_only_normal_mro_implementation():
    """Class metadata precedes the selected override without inheriting its body."""

    class Base:
        def method(self):
            return "base"

    class Leaf(Base):
        def method(self):
            return "leaf"

    root = Annotation("consumer.class", "base")
    leaf = Annotation("consumer.class", "leaf")
    base_method = Annotation("consumer.method", "base-method")
    leaf_method = Annotation("consumer.method", "leaf-method")
    attach_annotation(Base, root)
    attach_annotation(Leaf, leaf)
    attach_annotation(Base.__dict__["method"], base_method)
    attach_annotation(Leaf.__dict__["method"], leaf_method)

    assert annotations_for_method(Leaf, "method") == (root, leaf, leaf_method)


def test_method_collection_bypasses_hostile_metaclass_lookup_and_selects_override():
    """Raw class namespaces select the override without metaclass attribute lookup."""

    class HostileMeta(type):
        def __getattribute__(cls, name):
            raise AssertionError("metaclass attribute lookup must not run")

    def base_body(self):
        return "base"

    def leaf_body(self):
        return "leaf"

    class Base(metaclass=HostileMeta):
        method = base_body

    class Leaf(Base):
        method = leaf_body

    root = Annotation("consumer.class", "base")
    leaf = Annotation("consumer.class", "leaf")
    base_method = Annotation("consumer.method", "base-method")
    leaf_method = Annotation("consumer.method", "leaf-method")
    attach_annotation(Base, root)
    attach_annotation(Leaf, leaf)
    attach_annotation(base_body, base_method)
    attach_annotation(leaf_body, leaf_method)

    assert annotations_for_method(Leaf, "method") == (root, leaf, leaf_method)


def test_hostile_descriptors_are_inspected_without_binding_or_dynamic_lookup():
    """Static collection reads direct metadata without executing descriptor hooks."""

    class Hostile:
        def __getattribute__(self, name):
            raise AssertionError("dynamic lookup must not run")

        def __get__(self, instance, owner):
            raise AssertionError("binding must not run")

    descriptor = Hostile()
    entry = Annotation("consumer.hostile", object())
    attach_annotation(descriptor, entry)

    class Subject:
        method = descriptor

    assert annotations_for_method(Subject, "method") == (entry,)


def test_method_collection_rejects_missing_and_malformed_member_names():
    """Method selection reports bounded validation errors before returning values."""

    class Subject:
        pass

    with pytest.raises(AnnotationValidationError, match="requires a class"):
        annotations_for_method(Subject(), "missing")
    with pytest.raises(AnnotationValidationError, match="method name"):
        annotations_for_method(Subject, 1)
    with pytest.raises(AnnotationValidationError, match="not declared"):
        annotations_for_method(Subject, "missing")


def test_method_collection_rejects_hostile_string_subclasses_before_validation():
    """Method names must be exact strings before hashing or later validation."""

    class HostileString(str):
        def __hash__(self):
            raise AssertionError("method name hash hook must not run")

        def __eq__(self, other):
            raise AssertionError("method name equality hook must not run")

    class Subject:
        def method(self):
            return None

    with pytest.raises(AnnotationValidationError, match="method name"):
        annotations_for_method(Subject, HostileString("method"), key=0)
    with pytest.raises(AnnotationValidationError, match="key"):
        annotations_for_method(Subject, "missing", key=0)
