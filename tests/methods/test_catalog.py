"""Tests for deterministic Method implementation catalog construction."""

import pytest

from dryml.annotations import Annotation, attach_annotation
from dryml.methods import ImplementationDeclarationError, Method, Traits, traits


def test_catalog_preserves_raw_targets_and_normal_descriptor_binding():
    """Instance, static, and class declarations stay raw until carrier invocation."""

    class Variants(Method):
        @traits(backend="numpy")
        def instance_call(self, value):
            return ("instance", value)

        @traits(backend="numpy")
        @staticmethod
        def static_call(value):
            return ("static", value)

        @traits(backend="numpy")
        @classmethod
        def class_call(cls, value):
            return (cls.__name__, value)

    method = object.__new__(Variants)
    catalog = method.implementations()

    assert [implementation.name for implementation in catalog] == [
        "instance_call", "static_call", "class_call",
    ]
    assert [implementation.target for implementation in catalog] == [
        Variants.__dict__["instance_call"],
        Variants.__dict__["static_call"],
        Variants.__dict__["class_call"],
    ]
    assert catalog[0](1) == ("instance", 1)
    assert catalog[1](2) == ("static", 2)
    assert catalog[2](3) == ("Variants", 3)


def test_catalog_replaces_annotated_overrides_in_the_inherited_slot():
    """Annotated subclasses replace inherited names without changing slot order."""

    class Base(Method):
        @traits(backend="numpy")
        def alpha(self, value):
            return ("base-alpha", value)

        @traits(batch_mode="element")
        def beta(self, value):
            return ("base-beta", value)

    class Leaf(Base):
        @traits(backend="torch")
        def alpha(self, value):
            return ("leaf-alpha", value)

        @traits(backend="numpy")
        def gamma(self, value):
            return ("leaf-gamma", value)

    catalog = object.__new__(Leaf).implementations()

    assert [implementation.name for implementation in catalog] == ["alpha", "beta", "gamma"]
    assert catalog[0].target is Leaf.__dict__["alpha"]
    assert catalog[1].target is Base.__dict__["beta"]


def test_catalog_rejects_unannotated_shadow_before_any_target_runs():
    """A visible shadow cannot silently discard an inherited implementation."""

    class Base(Method):
        @traits(backend="numpy")
        def implementation(self, value):
            raise AssertionError("catalog inspection must not invoke targets")

    class Shadow(Base):
        def implementation(self, value):
            raise AssertionError("catalog inspection must not invoke targets")

    with pytest.raises(ImplementationDeclarationError, match="shadow"):
        object.__new__(Shadow).implementations()


def test_catalog_rejects_multiple_method_annotations_and_malformed_values():
    """Ambiguous or foreign annotation payloads fail before catalog publication."""

    with pytest.raises(ImplementationDeclarationError):

        class Multiple(Method):
            @traits(backend="numpy")
            @traits(batch_mode="element")
            def implementation(self, value):
                return value

    class Malformed(Method):
        def implementation(self, value):
            return value

    attach_annotation(
        Malformed.__dict__["implementation"],
        Annotation("dryml.methods.traits", object()),
    )

    with pytest.raises(ImplementationDeclarationError, match="Traits"):
        object.__new__(Malformed).implementations()


def test_catalog_rejects_unrelated_inherited_name_conflicts():
    """Two independent inherited declarations cannot silently choose an owner."""

    class Left(Method):
        @traits(backend="numpy")
        def implementation(self, value):
            return value

    class Right(Method):
        @traits(backend="torch")
        def implementation(self, value):
            return value

    class Conflict(Left, Right):
        pass

    with pytest.raises(ImplementationDeclarationError, match="conflict"):
        object.__new__(Conflict).implementations()


def test_catalog_keeps_equal_trait_alternatives_and_does_not_invoke_them():
    """Distinct equal-trait declarations remain ordered candidates without execution."""

    calls = []

    class Alternatives(Method):
        @traits(backend="numpy")
        def first(self, value):
            calls.append("first")
            return value

        @traits(backend="numpy")
        def second(self, value):
            calls.append("second")
            return value

    catalog = object.__new__(Alternatives).implementations()

    assert [implementation.name for implementation in catalog] == ["first", "second"]
    assert [implementation.traits for implementation in catalog] == [
        Traits(backend="numpy"),
        Traits(backend="numpy"),
    ]
    assert calls == []


def test_catalog_rejects_unsupported_descriptors_and_ignores_overwritten_sources():
    """Only completed class evidence is catalogued, and custom descriptors are refused."""

    class Descriptor:
        def __get__(self, instance, owner):
            raise AssertionError("catalog inspection must not bind descriptors")

    descriptor = Descriptor()
    traits(backend="numpy")(descriptor)

    class Unsupported(Method):
        implementation = descriptor

    with pytest.raises(ImplementationDeclarationError, match="descriptor"):
        object.__new__(Unsupported).implementations()

    class Overwritten(Method):
        @traits(backend="numpy")
        def implementation(self, value):
            return "discarded"

        def implementation(self, value):
            return "kept"

    assert object.__new__(Overwritten).implementations() == ()
