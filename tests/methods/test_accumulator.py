"""Contracts for explicit observation/carry accumulator Methods."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.methods import Accumulator, AccumulatorGroup, ImplementationSelectionError, traits


OBSERVATION_SPEC = TensorSpec("float32", shape=(2,), batch=Dynamic, backend="numpy")
CARRY_SPEC = TensorSpec("float32", shape=(2,), backend="numpy")


class Adding(Accumulator):
    """Add each batched observation total to an explicit vector carry."""

    def __init__(self):
        self.selections = 0

    def find_implementation(self, *args, **kwargs):
        """Count static selection without changing the inherited behavior."""

        self.selections += 1
        return super().find_implementation(*args, **kwargs)

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Return a fresh carry after reducing the observation batch."""

        return state + observation.sum(axis=0)

    def infer_output_spec(self, observation_spec, state_spec):
        """Preserve the carry slot without allocating or selecting a target."""

        return state_spec


class WrongDtype(Adding):
    """Return an invalid carry so group branch validation is observable."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Violate the declared float32 carry contract after target invocation."""

        return (state + observation.sum(axis=0)).astype(np.float64)


def test_accumulator_is_abstract_and_trait_implementations_discharge_only_call():
    """The accumulator interface requires pure inference as well as a logical call."""

    assert inspect.isabstract(Accumulator)
    assert not inspect.isabstract(Adding)


def test_accumulator_validates_fixed_carry_independently_of_short_dynamic_batches():
    """A fixed unbatched carry coexists with varying observation batch lengths."""

    accumulator = Adding()
    selected = accumulator.find_implementation(OBSERVATION_SPEC, CARRY_SPEC, output_spec=CARRY_SPEC)
    state = np.zeros((2,), dtype=np.float32)

    state = selected(np.ones((3, 2), dtype=np.float32), state)
    state = selected(np.ones((1, 2), dtype=np.float32), state)
    np.testing.assert_equal(state, np.array([4, 4], dtype=np.float32))
    assert accumulator.selections == 1

    with pytest.raises(ImplementationSelectionError):
        selected(np.ones((1, 2), dtype=np.float32), np.ones((2,), dtype=np.float64))
    with pytest.raises(ImplementationSelectionError):
        selected(
            np.ones((1, 2), dtype=np.float32),
            TensorSpec("float32", shape=(2,), backend="torch"),
        )


@pytest.mark.parametrize("structure", ("list", "tuple", "mapping"))
def test_accumulator_group_preserves_declared_state_structure_and_child_selection(structure):
    """Selected groups retain one child carrier per declared state branch."""

    child = Adding()
    if structure == "list":
        accumulators, state_spec, state = [child, child], [CARRY_SPEC, CARRY_SPEC], [
            np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32),
        ]
    elif structure == "tuple":
        accumulators, state_spec, state = (child, child), (CARRY_SPEC, CARRY_SPEC), (
            np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32),
        )
    else:
        accumulators, state_spec, state = {"left": child, "right": child}, {
            "left": CARRY_SPEC, "right": CARRY_SPEC,
        }, {
            "left": np.zeros((2,), dtype=np.float32), "right": np.zeros((2,), dtype=np.float32),
        }
    group = AccumulatorGroup(accumulators)
    selected = group.find_implementation(OBSERVATION_SPEC, state_spec, output_spec=state_spec)

    result = selected(np.ones((1, 2), dtype=np.float32), state)
    assert type(result) is type(state)
    if isinstance(result, dict):
        assert tuple(result) == ("left", "right")
        values = tuple(result.values())
    else:
        values = tuple(result)
    assert all(value.tolist() == [1.0, 1.0] for value in values)
    assert values[0] is not values[1]
    assert child.selections == 2
    assert child.call_mode == "eager"

    selected(np.ones((1, 2), dtype=np.float32), result)
    assert child.selections == 2


def test_accumulator_group_inference_selects_no_child_or_runtime_target():
    """Pure group inference composes child specs without touching selection state."""

    child = Adding()
    group = AccumulatorGroup({"total": child})

    assert group.infer_output_spec(OBSERVATION_SPEC, {"total": CARRY_SPEC}) == {"total": CARRY_SPEC}
    assert child.selections == 0
    assert child.call_mode == "eager"


def test_accumulator_group_stops_on_an_invalid_branch_output():
    """A failed child carry contract cannot produce a completed group state."""

    group = AccumulatorGroup((Adding(), WrongDtype()))
    selected = group.find_implementation(
        OBSERVATION_SPEC,
        (CARRY_SPEC, CARRY_SPEC),
        output_spec=(CARRY_SPEC, CARRY_SPEC),
    )

    with pytest.raises(ImplementationSelectionError):
        selected(
            np.ones((1, 2), dtype=np.float32),
            (np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)),
        )
