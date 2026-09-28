"""Focused graph-iteration contracts for qualified Dataset stream operators."""

import pytest
import numpy as np

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Chain, Dataset, Map, StreamDataset, Zip
from dryml.methods import IteratorPort, Method, StreamNode, traits


class TrackingDataset(Dataset):
    """Re-iterable source recording independently acquired generator resources."""

    def __init__(self, values, *, spec=None, close_error=None):
        self.values = tuple(values)
        self.opens = 0
        self.yields = []
        self.closes = []
        self.close_error = close_error
        super().__init__(spec=spec or TensorSpec("int64", shape=(), backend="numpy"))

    def __iter__(self):
        identifier = self.opens
        self.opens += 1
        try:
            for value in self.values:
                self.yields.append((identifier, value))
                yield value
        finally:
            self.closes.append(identifier)
            if self.close_error is not None:
                raise self.close_error

    def __len__(self):
        return Cardinality.finite(len(self.values))


class FailingOpenDataset(Dataset):
    """Source fixture whose iterator acquisition fails before yielding a value."""

    def __init__(self):
        self.opens = 0
        super().__init__(spec=TensorSpec("int64", shape=(), backend="numpy"))

    def __iter__(self):
        self.opens += 1
        raise RuntimeError("acquisition failure")
        yield  # pragma: no cover - makes this a generator for DatasetCursor.

    def __len__(self):
        return Cardinality.UNKNOWN


class NumpyIdentity(Method):
    """Select one NumPy implementation without changing an element spec."""

    @traits(backend="numpy")
    def numpy(self, value):
        return value

    def infer_output_spec(self, input_spec):
        return input_spec


def _paired_spec(specs):
    return specs[0]


def _unknown_cardinality(cardinalities):
    return Cardinality.UNKNOWN


def _interleave(left, right):
    return (value for pair in zip(left, right) for value in pair)


class PairedStreamDataset(StreamDataset):
    """Module-level custom stream fixture with a process-local declaration."""

    node = StreamNode(
        name="emit-pairs",
        inputs=(
            IteratorPort(TensorSpec("int64", shape=(), backend="numpy")),
            IteratorPort(TensorSpec("int64", shape=(), backend="numpy")),
        ),
        output=IteratorPort(TensorSpec("int64", shape=(), backend="numpy")),
        element_spec_transform=_paired_spec,
        cardinality_transform=_unknown_cardinality,
        pull_policy=("left_then_right", "shortest"),
        max_buffered_items=2,
        implementation=_interleave,
    )


def test_dataset_graph_is_inert_then_reuses_independent_source_occurrences():
    """Graph construction/planning does not open inputs, while Zip opens each occurrence."""
    source = TrackingDataset((1, 2))
    dataset = Zip(source, source)
    graph = dataset.method_graph()

    assert source.opens == 0
    graph.learn(source.spec, source.spec, output_spec=dataset.spec)
    assert source.opens == 0
    node = graph.method_nodes[-1]
    assert [port.kind for port in node.inputs] == ["iterator", "iterator"]
    assert node.stream_node.output_spec((source.spec, source.spec)) == dataset.spec
    assert node.stream_node.max_buffered_items == 0
    assert [tuple(value) for value in graph.iterator()] == [(1, 1), (2, 2)]
    assert source.opens == 2
    assert sorted(source.closes) == [0, 1]


def test_graph_zip_preserves_shortest_exhaustion_and_earlier_positional_overpull():
    """A later exhausted input leaves the earlier positional read consumed."""
    left = TrackingDataset((1, 2))
    right = TrackingDataset((10,))
    graph = Zip(left, right).method_graph()
    graph.learn()
    cursor = graph.iterator()

    assert next(cursor) == (1, 10)
    with pytest.raises(StopIteration):
        next(cursor)

    assert left.yields == [(0, 1), (0, 2)]
    assert right.yields == [(0, 10)]
    assert left.closes == [0]
    assert right.closes == [0]


def test_graph_chain_opens_and_advances_sources_in_declaration_order():
    """Chain defers later acquisition until the preceding source is exhausted."""
    left = TrackingDataset((1,))
    right = TrackingDataset((2,))
    graph = Chain(left, right).method_graph()
    graph.learn()
    cursor = graph.iterator()

    assert (left.opens, right.opens) == (0, 0)
    assert next(cursor) == 1
    assert (left.opens, right.opens) == (1, 0)
    assert next(cursor) == 2
    assert (left.opens, right.opens) == (1, 1)
    cursor.close()
    assert left.closes == [0]
    assert right.closes == [0]


def test_graph_unknown_spec_discovery_buffers_each_reused_occurrence_independently():
    """Each Map branch retains only its own first discovery value."""
    source = TrackingDataset(
        (np.int64(1), np.int64(2)),
        spec=TensorSpec("int64", shape=(), backend=None),
    )
    graph = Zip(Map(source, NumpyIdentity()), Map(source, NumpyIdentity())).method_graph()

    graph.learn()
    assert list(graph.iterator()) == [(1, 1), (2, 2)]
    assert source.opens == 2
    assert source.yields == [(0, 1), (1, 1), (0, 2), (1, 2)]


def test_graph_cursor_closes_once_on_explicit_close_and_preserves_body_failure():
    """The graph owns acquired sources and does not mask a selected-body failure."""
    source = TrackingDataset((np.int64(1),))
    graph = Map(source, NumpyIdentity()).method_graph()
    graph.learn()
    cursor = graph.iterator()
    assert next(cursor) == 1
    cursor.close()
    cursor.close()
    assert source.closes == [0]

    class Failing(Method):
        @traits(backend="numpy")
        def numpy(self, value):
            raise ValueError("body failure")

        def infer_output_spec(self, input_spec):
            return input_spec

    failing_source = TrackingDataset((np.int64(1),))
    failing_source.close_error = RuntimeError("cleanup failure")
    failing = Map(failing_source, Failing()).method_graph()
    failing.learn()
    with pytest.raises(ValueError, match="body failure"):
        next(failing.iterator())
    assert failing_source.closes == [0]


def test_graph_cursor_closes_partial_acquisition_and_surfaces_cleanup_without_primary():
    """Partial opens close acquired inputs, while standalone cleanup failure is visible."""
    acquired = TrackingDataset((1,))
    graph = Zip(acquired, FailingOpenDataset()).method_graph()
    graph.learn()
    with pytest.raises(RuntimeError, match="acquisition failure"):
        next(graph.iterator())
    assert acquired.closes == [0]

    cleanup = TrackingDataset((1,))
    cleanup.close_error = RuntimeError("cleanup failure")
    graph = Zip(cleanup, TrackingDataset((2,))).method_graph()
    graph.learn()
    cursor = graph.iterator()
    assert next(cursor) == (1, 2)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        cursor.close()
    assert cleanup.closes == [0]


def test_graph_cursor_attempts_every_close_when_multiple_cleanups_fail():
    """Surface the first reverse-order failure without skipping other resources."""

    left = TrackingDataset((1,))
    right = TrackingDataset((2,))
    left.close_error = RuntimeError("left cleanup")
    right.close_error = RuntimeError("right cleanup")
    graph = Zip(left, right).method_graph()
    graph.learn()
    cursor = graph.iterator()
    assert next(cursor) == (1, 2)

    with pytest.raises(RuntimeError, match="right cleanup"):
        cursor.close()

    assert left.closes == [0]
    assert right.closes == [0]


def test_unsupported_operator_rejects_graph_planning_without_changing_eager_iteration():
    """Only the initial qualified stream subset receives graph execution semantics."""
    from dryml.data import Take

    source = TrackingDataset((1, 2))
    graph = Take(source, 1).method_graph()
    with pytest.raises(NotImplementedError, match="Take"):
        graph.learn()
    assert list(Take(source, 1)) == [1]


def test_custom_stream_dataset_composes_bounded_two_input_variable_output_node():
    """The custom declaration composes through the same graph-owned lifecycle."""
    dataset = PairedStreamDataset(TrackingDataset((1, 2)), TrackingDataset((3, 4)))
    graph = dataset.method_graph()

    graph.learn()
    assert list(graph.iterator()) == [1, 3, 2, 4]
