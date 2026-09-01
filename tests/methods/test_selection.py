"""Contract tests for Method candidate inspection and direct selection."""

import numpy as np
import pytest

from dryml.core.tensor_spec import TensorSpec
from dryml.methods import (
    ImplementationSelectionError,
    Method,
    Traits,
    traits,
)


class Variants(Method):
    """Method fixture with generic, backend, and batch-specific alternatives."""

    @traits()
    def generic(self, value, *extra, **options):
        """Return the generic branch and forwarded arguments."""

        return ("generic", value, extra, options)

    @traits(backend="numpy")
    def numpy_element(self, value, *extra, **options):
        """Return the NumPy element branch and forwarded arguments."""

        return ("numpy-element", value, extra, options)

    @traits(backend="numpy", batch_mode="batched")
    def numpy_batched(self, value, *extra, **options):
        """Return the NumPy batched branch and forwarded arguments."""

        return ("numpy-batched", value, extra, options)


class BatchVariants(Method):
    """Method fixture with concrete element and batched alternatives."""

    @traits(backend="numpy", batch_mode="element")
    def element(self, value):
        """Identify element selection."""

        return "element", value

    @traits(backend="numpy", batch_mode="batched")
    def batched(self, value):
        """Identify batched selection."""

        return "batched", value


def test_candidate_inspection_returns_catalog_order_without_selecting_or_invoking():
    """Compatible inspection filters only known constraints and has no call side effects."""

    method = Variants()

    assert [candidate.name for candidate in method.compatible_implementations()] == [
        "generic", "numpy_element", "numpy_batched",
    ]
    assert [candidate.name for candidate in method.compatible_implementations(backend="numpy")] == [
        "generic", "numpy_element", "numpy_batched",
    ]
    assert method.compatible_implementations(backend="torch") == (method.implementations()[0],)


def test_find_implementation_selects_one_specific_callable_and_forwards_only_logical_args():
    """Selection returns the raw target carrier without passing control keywords through."""

    method = Variants()
    implementation = method.find_implementation(backend="numpy", batch_mode="batched")

    assert implementation.name == "numpy_batched"
    assert implementation.traits == Traits(backend="numpy", batch_mode="batched")
    assert implementation.target is Variants.__dict__["numpy_batched"]
    assert implementation(np.ones((2, 3)), "later", named="value")[0] == "numpy-batched"


def test_unknown_or_equal_best_traits_fail_before_any_target_runs():
    """Direct selection does not guess concrete traits or silently resolve ties."""

    class ConcreteOnly(Method):
        @traits(backend="numpy")
        def numpy(self, value):
            raise AssertionError("selection must fail before invocation")

    class Equal(Method):
        @traits(backend="numpy")
        def first(self, value):
            raise AssertionError("selection must fail before invocation")

        @traits(backend="numpy")
        def second(self, value):
            raise AssertionError("selection must fail before invocation")

    with pytest.raises(ImplementationSelectionError) as unknown:
        object.__new__(ConcreteOnly)(object())
    assert unknown.value.reason == "unknown_traits"
    assert unknown.value.unknown_traits == ("backend",)

    with pytest.raises(ImplementationSelectionError) as ambiguous:
        object.__new__(Equal).find_implementation(backend="numpy")
    assert ambiguous.value.reason == "ambiguous"


def test_default_batched_selects_only_when_runtime_batch_intent_is_unobservable():
    """The eager default fills unknown NumPy-array intent but never overrides known facts."""

    method = BatchVariants()
    value = np.ones((2, 3), dtype=np.float32)

    with pytest.raises(ImplementationSelectionError) as unknown:
        method(value)
    assert unknown.value.reason == "unknown_traits"
    assert unknown.value.unknown_traits == ("batch_mode",)

    method.default_batched = False
    assert method(value)[0] == "element"
    method.default_batched = True
    assert method(value)[0] == "batched"

    explicit_element = TensorSpec("float32", shape=(2, 3), backend="numpy")
    assert method(explicit_element)[0] == "element"
    explicit_batch = TensorSpec("float32", shape=(3,), batch=2, backend="numpy")
    method.default_batched = False
    assert method(explicit_batch)[0] == "batched"


def test_explicit_candidate_apis_ignore_default_batched():
    """Inspection and direct selection use only their explicit normalized constraints."""

    method = BatchVariants()
    method.default_batched = True

    assert [candidate.name for candidate in method.compatible_implementations(backend="numpy")] == [
        "element",
        "batched",
    ]
    with pytest.raises(ImplementationSelectionError) as unknown:
        method.find_implementation(backend="numpy")
    assert unknown.value.reason == "unknown_traits"
