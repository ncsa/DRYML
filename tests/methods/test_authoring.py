"""Tests for Method authoring and the simple central call gateway."""

import pytest

from dryml.core.backend import Backend
from dryml.core.tensor_spec import BatchMode
from dryml.methods import (
    ImplementationDeclarationError,
    Method,
    Traits,
    traits,
)


class CooperativeBase(Method):
    """Module-scoped Object fixture with a direct base implementation."""

    def __call__(self, value):
        return ["base", value]


class CooperativeZeroArg(CooperativeBase):
    """Module-scoped fixture using zero-argument cooperative super."""

    def __call__(self, value):
        return ["zero", *super().__call__(value)]


class CooperativeExplicit(CooperativeBase):
    """Module-scoped fixture using explicit cooperative super."""

    def __call__(self, value):
        return ["explicit", *super(CooperativeExplicit, self).__call__(value)]


def test_traits_are_closed_immutable_and_normalized():
    """Traits normalize supported strings while leaving omitted dimensions unset."""

    value = Traits(backend="numpy", batch_mode="element")

    assert value.backend is Backend.numpy
    assert value.batch_mode is BatchMode.element
    assert Traits().backend is None
    assert Traits().batch_mode is None
    with pytest.raises((TypeError, ValueError, ImplementationDeclarationError)):
        Traits(backend="unsupported")


def test_decorator_preserves_the_exact_target_and_attaches_one_annotation():
    """Trait declaration transports passive metadata without replacing its target."""

    def target(value):
        return value

    decorated = traits(backend="numpy")(target)

    assert decorated is target


def test_simple_direct_call_is_catalogued_and_forwards_through_gateway():
    """A direct implementation remains inspectable and executes through Method."""

    class Increment(Method):
        def __call__(self, value):
            return value + 1

    method = Increment()
    implementation, = method.implementations()

    assert implementation.name == "__call__"
    assert implementation.target is not Increment.__dict__["__call__"]
    assert implementation(2) == 3
    assert method(2) == 3


def test_cooperative_direct_calls_reach_each_captured_owner_once():
    """Zero-argument and explicit super calls bypass reselection of the leaf target."""

    assert CooperativeZeroArg()(1) == ["zero", "base", 1]
    assert CooperativeExplicit()(2) == ["explicit", "base", 2]


def test_mixed_direct_and_alternative_authoring_is_rejected():
    """One class cannot declare both direct and trait-selected implementations."""

    with pytest.raises(ImplementationDeclarationError):

        class Mixed(Method):
            def __call__(self, value):
                return value

            @traits(backend="numpy")
            def numpy_call(self, value):
                return value


def test_mixed_direct_and_alternative_inheritance_is_rejected():
    """Normal Python overrides cannot silently coexist with inherited alternatives."""

    class AlternativeBase(Method):
        @traits(backend="numpy")
        def numpy_call(self, value):
            return value

    class DirectLeaf(AlternativeBase):
        def __call__(self, value):
            return value

    class DirectBase(Method):
        def __call__(self, value):
            return value

    class AlternativeLeaf(DirectBase):
        @traits(backend="numpy")
        def numpy_call(self, value):
            return value

    with pytest.raises(ImplementationDeclarationError, match="hierarchy"):
        object.__new__(DirectLeaf).implementations()
    with pytest.raises(ImplementationDeclarationError, match="hierarchy"):
        object.__new__(AlternativeLeaf).implementations()
