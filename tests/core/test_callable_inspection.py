"""Passive core callable-owner inspection contracts."""

from __future__ import annotations

import functools

from dryml.core import function
from dryml.core._callable_inspection import (
    _FunctionInvocationOwner,
    describe_callable,
)


def _generator():
    """Provide an unsupported native root for owner-modality coverage."""

    yield 1


def test_function_owner_exposes_its_exact_raw_target_and_native_modality():
    """A recognized owner reports the target Execute actually invokes."""

    target = _generator

    def wrapped():
        return target()

    owner = _FunctionInvocationOwner(target)
    owner.wrapper = wrapped
    wrapped.__dryml_function_invocation_owner__ = owner

    description = describe_callable(wrapped)

    assert description.owner == "function"
    assert description.raw_target is target
    assert description.declaration_carriers == (wrapped, target)
    assert description.native_modality == "generator"


def test_ordinary_wrapper_is_not_unwrapped_from_copied_owner_metadata():
    """Copied metadata cannot replace an actual wrapper body."""

    def target():
        return 1

    wrapped = function(target)

    @functools.wraps(wrapped)
    def ordinary(*args, **kwargs):
        return wrapped(*args, **kwargs)

    description = describe_callable(ordinary)

    assert description.owner == "ordinary"
    assert description.raw_target is ordinary
    assert description.declaration_carriers == (ordinary,)
    assert description.native_modality == "sync"


def test_bound_function_owner_keeps_the_original_receiver():
    """A bound owner projects a bound raw target without descriptor binding."""

    class Receiver:
        @function
        def call(self):
            return 1

    receiver = Receiver()
    description = describe_callable(receiver.call)

    assert description.owner == "function"
    assert description.raw_target.__self__ is receiver
    assert description.raw_target.__func__.__name__ == "call"


def test_sync_wrapper_does_not_inherit_deferred_modality_from_metadata():
    """Native wrapper code, rather than metadata, determines modality."""

    @functools.wraps(_generator)
    def wrapper():
        return _generator

    assert describe_callable(wrapper).native_modality == "sync"


def test_callable_inspection_never_uses_metaclass_lookup_or_marker_equality():
    """
    Unknown callable instances stay ordinary without activating class hooks.
    """

    class Marker:
        def __eq__(self, other):
            raise AssertionError("marker equality must not run")

    class Metaclass(type):
        @property
        def __mro__(cls):
            raise AssertionError("metaclass MRO lookup must not run")

        @property
        def __dict__(cls):
            raise AssertionError("metaclass dictionary lookup must not run")

    class Target(metaclass=Metaclass):
        __dryml_execute_owner__ = Marker()

        def __call__(self):
            return None

    description = describe_callable(Target())

    assert description.owner == "ordinary"
    assert description.native_modality == "sync"


def test_function_owner_requires_the_wrapper_closure_relationship():
    """
    Copied owner metadata cannot replace the callable body Execute invokes.
    """

    def target():
        return None

    wrapped = function(target)

    def impostor():
        return None

    impostor.__dryml_function_invocation_owner__ = (
        wrapped.__dryml_function_invocation_owner__
    )

    description = describe_callable(impostor)

    assert description.owner == "ordinary"
    assert description.raw_target is impostor


def test_nested_function_owners_retain_each_declaration_layer_once():
    """
    Nested established wrappers expose the actual deferred root without
    duplicates.
    """

    def target():
        return None

    inner = function(target)
    outer = function(inner)

    description = describe_callable(outer)

    assert description.owner == "function"
    assert description.raw_target is target
    assert description.declaration_carriers == (outer, inner, target)
    assert description.native_modality == "sync"
