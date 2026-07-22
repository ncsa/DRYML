from __future__ import annotations

import inspect

import pytest

from dryml.core import Object
from dryml.managed import (
    ManagedLifecycleUnavailableError,
    ManagedOutput,
    UnknownOutputError,
    managed,
)


class Calculator(Object):
    @managed(outputs=(
        ManagedOutput("value", primary=True, kind="data"),
        ManagedOutput("details", kind="data"),
    ))
    def compute(self, value=1):
        """Compute a value."""
        return value + 1


class ChildCalculator(Calculator):
    pass


class OverrideCalculator(Calculator):
    @managed(outputs=(ManagedOutput("replacement", primary=True),))
    def compute(self, value=1):
        return value + 2


def test_bound_managed_method_is_callable_and_exposes_logical_outputs():
    calculator = Calculator()
    bound = calculator.compute

    assert bound(3) == 4
    assert bound.__func__ is Calculator.__dict__["compute"].__func__
    assert bound.__self__ is calculator
    assert bound.__name__ == "compute"
    assert bound.__doc__ == "Compute a value."
    assert str(inspect.signature(bound)) == "(value=1)"
    assert tuple(bound.outputs) == ("value", "details")
    assert bound.result is bound.outputs["value"]
    assert bound.result.producer == calculator.definition
    assert bound.result.method == "compute"
    assert bound.result.slot == "value"
    with pytest.raises(UnknownOutputError):
        bound.output("missing")
    with pytest.raises(ManagedLifecycleUnavailableError):
        bound.status()


def test_descriptor_metadata_is_static_and_inheritance_obeys_python_lookup():
    raw = inspect.getattr_static(Calculator, "compute")

    assert raw is Calculator.__dict__["compute"]
    assert raw.__func__.__name__ == "compute"
    assert raw.outputs.slots == ("value", "details")
    assert ChildCalculator().compute.result.slot == "value"
    assert OverrideCalculator().compute.result.slot == "replacement"


def test_descriptor_rejects_unknown_declared_output_selection():
    with pytest.raises(UnknownOutputError):
        Calculator.__dict__["compute"].output_ref(Calculator(), "missing")
