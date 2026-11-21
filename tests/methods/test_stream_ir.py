"""Contracts for explicit iterator-port stream IR declarations."""

import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.methods.stream import IteratorPort, StreamNode


def test_custom_two_input_variable_output_node_declares_bounded_pull_contract():
    """Custom stream declarations retain specs, ordered inputs, and bounded state."""
    spec = TensorSpec("int64", shape=(), backend="numpy")
    node = StreamNode(
        name="paired",
        inputs=(IteratorPort(spec), IteratorPort(spec)),
        output=IteratorPort(spec),
        element_spec_transform=lambda specs: specs[0],
        cardinality_transform=lambda cardinalities: Cardinality.UNKNOWN,
        pull_policy=("left_then_right", "shortest"),
        max_buffered_items=2,
        implementation=lambda left, right: (value for pair in zip(left, right) for value in pair),
    )

    assert node.output_spec((spec, spec)) == spec
    assert node.output_cardinality((Cardinality.finite(2), Cardinality.finite(3))) == Cardinality.UNKNOWN
    assert node.max_buffered_items == 2
    assert tuple(node.open(iter((1, 2)), iter((3, 4)))) == (1, 3, 2, 4)


@pytest.mark.parametrize("buffer", (-1, True, 1.5))
def test_custom_stream_node_rejects_invalid_buffer_bounds(buffer):
    """Bounds are explicit exact nonnegative declarations, not inferred at runtime."""
    spec = TensorSpec("int64", shape=(), backend="numpy")

    with pytest.raises((TypeError, ValueError)):
        StreamNode(
            name="invalid",
            inputs=(IteratorPort(spec),),
            output=IteratorPort(spec),
            element_spec_transform=lambda specs: specs[0],
            cardinality_transform=lambda cardinalities: cardinalities[0],
            pull_policy=("one_to_one",),
            max_buffered_items=buffer,
            implementation=lambda source: source,
        )


def test_custom_stream_node_requires_explicit_iterator_ports():
    """Reject ambiguous stream declarations before planning or execution."""

    spec = TensorSpec("int64", shape=(), backend="numpy")
    with pytest.raises(TypeError, match="inputs"):
        StreamNode(
            name="invalid",
            inputs=(spec,),
            output=IteratorPort(spec),
            element_spec_transform=lambda specs: specs[0],
            cardinality_transform=lambda cardinalities: cardinalities[0],
            pull_policy=("one_to_one",),
            max_buffered_items=0,
            implementation=lambda source: source,
        )


def test_prepared_occurrences_detach_specs_and_preserve_reused_source_identity():
    """Occurrence paths distinguish opens while immutable evidence retains sharing."""
    from dataclasses import FrozenInstanceError
    from dryml.methods.stream import StreamPlan

    class Binding:
        def open(self, epoch):
            return iter(({"x": 1},))

    class Source:
        _stream_operator = "source"
        spec = {"x": TensorSpec("int64", shape=(), backend="numpy")}

        def yield_cardinality(self):
            return Cardinality.finite(1)

        def _prepare_stream_source(self, spec):
            return Binding()

    class Zip:
        _stream_operator = "zip"

        def __init__(self, source):
            self.sources = [source, source]
            self.spec = [source.spec, source.spec]

        def yield_cardinality(self):
            return Cardinality.finite(1)

    source = Source()
    plan = StreamPlan(Zip(source))
    plan.learn()
    root = plan._prepared.root
    assert [child.occurrence for child in root.inputs] == [(0,), (1,)]
    assert root.inputs[0].dataset is root.inputs[1].dataset is source
    source.spec["new"] = source.spec.pop("x")
    assert tuple(root.inputs[0].spec) == ("x",)
    decoded = root.spec
    decoded[0].clear()
    assert tuple(root.spec[0]) == ("x",)
    with pytest.raises(FrozenInstanceError):
        root.controls = (1,)
    assert list(plan.iterator()) == [[{"x": 1}, {"x": 1}]]
