"""Contracts for explicit observation/carry accumulator Methods."""

from __future__ import annotations

import inspect
import gc
import weakref

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


def test_prepared_nested_groups_release_weak_state_owners():
    """Global preparation facts must not close over group owners, even indirectly."""
    child = Adding()
    inner = AccumulatorGroup((child,))
    outer = AccumulatorGroup({"nested": inner, "other": child})
    outer.learn(OBSERVATION_SPEC, {"nested": (CARRY_SPEC,), "other": CARRY_SPEC})
    refs = weakref.ref(outer), weakref.ref(inner)
    del outer, inner
    gc.collect()
    assert all(reference() is None for reference in refs)


def test_retained_group_compiles_only_during_preparation(monkeypatch):
    """Five and one hundred transitions reuse each finalized child signature."""
    from dryml.core import signatures
    compiled = []
    original = signatures.compile_signature
    monkeypatch.setattr(signatures, "compile_signature",
                        lambda target, **kw: (compiled.append(target), original(target, **kw))[1])
    child = Adding()
    group = AccumulatorGroup((child, child))
    compiled.clear()
    selected = group.find_implementation(OBSERVATION_SPEC, (CARRY_SPEC, CARRY_SPEC))
    assert compiled == []
    prepared = selected.prepared_invoker()
    assert len(compiled) == 3
    state = (np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    for length in (5, 100):
        for _ in range(length):
            state = prepared(np.ones((1, 2), dtype=np.float32), state)
        assert len(compiled) == 3
        assert child.selections == 2


def test_nested_group_occurrences_retain_distinct_carry_constraints_and_invokers():
    """Repeated leaves keep separate carry contracts in inspection and execution."""
    class BroadcastAdding(Adding):
        @traits(backend="numpy", batch_mode="batched")
        def numpy(self, observation, state):
            return state + observation.sum()

    child = BroadcastAdding()
    child.learn()
    small = TensorSpec("float32", shape=(), backend="numpy")
    large = TensorSpec("float32", shape=(3,), backend="numpy")
    inner = AccumulatorGroup((child,))
    group = AccumulatorGroup({"small": inner, "large": child})
    graph = group.method_graph()
    graph.learn(OBSERVATION_SPEC, {"small": (small,), "large": large})
    nodes = [node for node in graph.method_nodes if node.method_type is BroadcastAdding]
    assert [node.inputs[1].spec.value.shape for node in nodes] == [(), (3,)]
    root = graph.method_nodes[0].selected
    assert root.children[0].children[0] is nodes[0].selected
    assert root.children[1] is nodes[1].selected
    result = group(np.ones((1, 2), dtype=np.float32),
                   {"small": (np.array(0, dtype=np.float32),), "large": np.zeros(3, dtype=np.float32)})
    assert result["small"][0] == 2
    np.testing.assert_equal(result["large"], np.full(3, 2, dtype=np.float32))
    assert child.call_mode == "learning"
    assert child.cached_signature is None
