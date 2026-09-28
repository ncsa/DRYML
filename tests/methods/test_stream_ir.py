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
