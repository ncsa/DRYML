"""Focused graph-iteration contracts for qualified Dataset stream operators."""

import pytest
import numpy as np

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Chain, Dataset, Map, Pipe, StreamDataset, Take, Zip
from dryml.methods import IteratorPort, Method, StreamNode, traits


def _cursor(dataset, prepared):
    if not prepared:
        return dataset.iterator()
    graph = dataset.method_graph()
    graph.learn()
    return graph.iterator()


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


class FailingIdentity(Method):
    @traits(backend="numpy")
    def numpy(self, value):
        raise self.failure

    def infer_output_spec(self, input_spec):
        return input_spec


class LifecycleResource:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.values = iter((1,))

    def __next__(self):
        return next(self.values)

    def close(self):
        self.events.append(self.name)
        raise RuntimeError(self.name + " cleanup")


class LifecycleSource(TrackingDataset):
    def iterator(self):
        from dryml.data.dataset import DatasetCursor
        if hasattr(self, "open_error"):
            raise self.open_error
        return DatasetCursor(LifecycleResource(self.values[0], self.events))


class BackendIdentity(Method):
    """Select a matching dense backend so eager reset can prove reselection."""

    @traits(backend="numpy")
    def numpy(self, value):
        return value

    @traits(backend="torch")
    def torch(self, value):
        return value

    def infer_output_spec(self, input_spec):
        return input_spec


@pytest.mark.parametrize("prepared", (False, True))
@pytest.mark.parametrize("converted", (False, True))
@pytest.mark.parametrize("empty", (False, True))
def test_unknown_backend_nested_pipe_and_successor_map_discover_locally(prepared, converted, empty):
    """Each occurrence discovers at most one value and never changes declarations."""
    from dryml.data import Cast

    class TorchIdentity(Method):
        @traits(backend="torch")
        def torch(self, value):
            return value

        def infer_output_spec(self, spec):
            return spec

    values = () if empty else (np.ones(2, dtype=np.float32), np.full(2, 2, dtype=np.float32))
    source = TrackingDataset(values, spec=TensorSpec("float32", shape=(2,), backend=None))
    first = Map(source, TorchIdentity() if converted else Pipe(Cast("float64"), Pipe(Cast("float32"))))
    successor = Map(first, Pipe(Pipe(Cast("float64")), Cast("float32")))
    assert source.opens == 0
    assert source.spec.backend is first.spec.backend is successor.spec.backend is None
    if converted:
        pytest.importorskip("torch")
    for _ in range(2):
        cursor = _cursor(successor, prepared)
        assert source.opens == _
        with cursor:
            if not empty:
                result = next(cursor)
                assert len(source.yields) == 2 * _ + 1
                np.testing.assert_equal(np.asarray(result) if not converted else result.numpy(), [1, 1])
                assert len(list(cursor)) == 1
            else:
                assert list(cursor) == []
    assert source.spec.backend is first.spec.backend is successor.spec.backend is None
    assert sorted(source.closes) == [0, 1]


@pytest.mark.parametrize("mode", ("eager", "learning", "first-call", "known-spec"))
def test_dataset_preparation_and_reset_preserve_shared_child_state(mode):
    """Dataset-owned selections cannot replace independent direct-call state."""
    child = NumpyIdentity()
    child.default_batched = False
    direct = np.ones((2,), dtype=np.float32)
    if mode == "learning":
        child.learn()
    elif mode == "first-call":
        child.learn()
        child(direct)
    elif mode == "known-spec":
        child.learn(TensorSpec("float32", shape=(2,), backend="numpy"))
    before = (child.call_mode, child.cached_signature, child.default_batched)
    source = TrackingDataset((np.int64(1), np.int64(2)))
    prepared = Map(source, child).prepare()
    graph = prepared.graph
    assert (child.call_mode, child.cached_signature, child.default_batched) == before
    cursor = prepared.iterator()
    assert next(cursor) == 1
    other_source = TrackingDataset((np.ones(3, dtype=np.float64),),
                                   spec=TensorSpec("float64", shape=(3,), backend="numpy"))
    other = Map(other_source, child).prepare()
    with other.iterator() as other_cursor:
        assert next(other_cursor).shape == (3,)
    other.graph.eager()
    graph.eager()
    assert (child.call_mode, child.cached_signature, child.default_batched) == before
    assert list(cursor) == [2]
    assert child(direct) is direct


def test_later_branch_failed_reprepare_preserves_complete_plan_and_cursors(monkeypatch):
    """A failed replacement cannot publish a prefix or disturb captured readers."""
    from dryml.methods import ImplementationSelectionError
    first, later = NumpyIdentity(), NumpyIdentity()
    first.learn()
    direct = np.ones(2, dtype=np.float32)
    first(direct)
    before = first.cached_signature
    graph = Zip(Map(TrackingDataset((np.int64(1), np.int64(2))), first),
                Map(TrackingDataset((np.int64(3), np.int64(4))), later)).method_graph()
    graph.learn()
    nodes = graph.nodes
    left, right = graph.iterator(), graph.iterator()
    assert next(left) == next(right) == (1, 3)

    def fail(*args, **kwargs):
        raise ImplementationSelectionError("no_candidate")

    monkeypatch.setattr(later, "find_implementation", fail)
    with pytest.raises(ImplementationSelectionError):
        graph.learn()
    assert graph.nodes is nodes
    assert list(left) == list(right) == [(2, 4)]
    assert list(graph.iterator()) == [(1, 3), (2, 4)]
    assert first.cached_signature == before
    assert first(direct) is direct
    assert later.call_mode == "eager"


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

    first, second = graph.iterator(), graph.iterator()
    assert next(first) == next(second) == (1, 1)
    first.close()
    assert next(second) == (2, 2)
    assert list(second) == []
    assert first.observed_eof is False
    assert second.observed_eof is True
    assert sorted(source.closes) == list(range(6))


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

    empty = TrackingDataset((), spec=source.spec)
    graph = Map(empty, NumpyIdentity()).method_graph()
    graph.learn()
    cursor = graph.iterator()
    assert list(cursor) == []
    assert cursor.observed_eof is True
    assert (empty.opens, empty.closes) == (1, [0])


def test_graph_eager_resets_nested_method_selection_for_a_new_backend():
    """Resetting a Dataset plan permits another backend without mutating its Pipe."""

    torch = pytest.importorskip("torch")
    import dryml.torch

    source = TrackingDataset((np.asarray(1),), spec=TensorSpec("int64", shape=(), backend="numpy"))
    selected = BackendIdentity()
    pipe = Pipe(selected)
    graph = Map(source, pipe).method_graph()
    graph.learn()
    assert pipe.call_mode == "eager"

    graph.eager()
    assert pipe.call_mode == "eager"
    torch_source = TrackingDataset((), spec=TensorSpec("int64", shape=(), backend="torch"))
    torch_source.values = (torch.tensor(1),)
    graph = Map(torch_source, pipe).method_graph()
    graph.learn()

    assert pipe.call_mode == "eager"
    assert next(graph.iterator()).device.type == "cpu"


def test_graph_eager_resets_named_nested_zip_map_methods_for_a_new_backend():
    """Mapping-backed plans reset their own facts without changing nested Methods."""

    torch = pytest.importorskip("torch")
    import dryml.torch

    first, second = BackendIdentity(), BackendIdentity()
    first_pipe, second_pipe = Pipe(first), Pipe(second)
    numpy = Zip({
        "first": Map(TrackingDataset((np.asarray(1),)), first_pipe),
        "nested": (Map(TrackingDataset((np.asarray(2),)), second_pipe),),
    })
    graph = numpy.method_graph()
    graph.learn()
    assert graph.nodes
    assert first_pipe.call_mode == second_pipe.call_mode == "eager"

    graph.eager()
    assert graph.nodes == ()
    assert first_pipe.call_mode == second_pipe.call_mode == "eager"

    torch_first = TrackingDataset((), spec=TensorSpec("int64", shape=(), backend="torch"))
    torch_second = TrackingDataset((), spec=TensorSpec("int64", shape=(), backend="torch"))
    torch_first.values, torch_second.values = (torch.tensor(1),), (torch.tensor(2),)
    torch_graph = Zip({
        "first": Map(torch_first, first_pipe),
        "nested": (Map(torch_second, second_pipe),),
    }).method_graph()
    torch_graph.learn()

    assert torch_graph.nodes
    assert next(torch_graph.iterator())["first"].device.type == "cpu"


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


def test_take_stream_node_preserves_strict_prefix_and_closes_source():
    """A planned Take delegates strict counting and source ownership to its cursor."""
    source = TrackingDataset((1, 2))
    graph = Take(source, 1).method_graph()
    graph.learn()
    assert list(graph.iterator()) == [1]
    assert source.closes == [0]


def test_custom_stream_dataset_composes_bounded_two_input_variable_output_node():
    """The custom declaration composes through the same graph-owned lifecycle."""
    dataset = PairedStreamDataset(TrackingDataset((1, 2)), TrackingDataset((3, 4)))
    graph = dataset.method_graph()

    graph.learn()
    assert list(graph.iterator()) == [1, 3, 2, 4]


@pytest.mark.parametrize("prepared", (False, True))
def test_take_shortfall_and_partial_zip_preserve_counts_and_primary(prepared):
    """Strict shortfall and later acquisition failure close every acquired input."""
    from dryml.data import DatasetExhaustedError
    events = []
    def source(name):
        result = LifecycleSource((name,))
        result.events = events
        return result

    cursor = _cursor(Take(source("short"), 2), prepared)
    assert next(cursor) == 1
    with pytest.raises(DatasetExhaustedError) as error:
        next(cursor)
    assert type(error.value) is DatasetExhaustedError
    assert (error.value.requested, error.value.yielded) == (2, 1)
    assert cursor.observed_eof is False
    assert events == ["short"]

    events.clear()
    later = source("later")
    later.open_error = ValueError("later acquisition")
    cursor = _cursor(Zip(source("left"), source("right"), later), prepared)
    with pytest.raises(ValueError, match="later acquisition"):
        next(cursor)
    assert events == ["right", "left"]
    assert cursor.observed_eof is False


@pytest.mark.parametrize("prepared", (False, True))
def test_zip_eof_closes_inputs_in_reverse_order_before_reporting_cleanup(prepared):
    """Child EOF must not close the left resource ahead of its later siblings."""
    events = []
    left, right = LifecycleSource(("left",)), LifecycleSource(("right",))
    left.events = right.events = events
    cursor = _cursor(Zip(left, right), prepared)
    assert next(cursor) == (1, 1)
    with pytest.raises(RuntimeError, match="right cleanup"):
        next(cursor)
    assert events == ["right", "left"]
    assert cursor.observed_eof is True
    cursor.close()
    assert events == ["right", "left"]


@pytest.mark.parametrize("prepared", (False, True))
@pytest.mark.parametrize("failure", (ValueError("body failed"), StopIteration("body stopped")))
def test_map_body_failure_closes_without_certifying_eof(prepared, failure):
    """Method StopIteration is causal failure, and cleanup cannot mask a body error."""
    method = FailingIdentity()
    method.failure = failure
    source = TrackingDataset((np.int64(1),))
    source.close_error = RuntimeError("cleanup failed")
    cursor = _cursor(Map(source, method), prepared)
    expected = RuntimeError if isinstance(failure, StopIteration) else ValueError
    with pytest.raises(expected) as error:
        next(cursor)
    if isinstance(failure, StopIteration):
        assert error.value.__cause__ is failure
    else:
        assert error.value is failure
    assert source.closes == [0]
    assert cursor.observed_eof is False


@pytest.mark.parametrize("prepared", (False, True))
def test_chain_acquisition_stopiteration_is_not_root_eof(prepared):
    """A later source acquisition cannot truncate a successfully consumed prefix."""
    stopped = StopIteration("acquisition stopped")

    later = LifecycleSource((2,))
    later.open_error = stopped
    cursor = _cursor(Chain(TrackingDataset((1,)), later), prepared)
    assert next(cursor) == 1
    with pytest.raises(RuntimeError) as error:
        next(cursor)
    assert error.value.__cause__ is stopped
    assert cursor.observed_eof is False


def test_root_eof_survives_cleanup_stopiteration_and_context_preserves_interruption():
    """Only real output exhaustion, recorded before cleanup, certifies EOF."""
    from dryml.data.dataset import DatasetCursor

    stopped = StopIteration("cleanup stopped")

    class Resource:
        def __next__(self):
            raise StopIteration

        def close(self):
            raise stopped

    cursor = DatasetCursor(Resource())
    with pytest.raises(RuntimeError) as error:
        next(cursor)
    assert error.value.__cause__ is stopped
    assert cursor.observed_eof is True
    cursor.close()

    source = TrackingDataset((np.int64(1),))
    source.close_error = RuntimeError("cleanup failed")
    cursor = _cursor(Map(source, NumpyIdentity()), True)
    interruption = KeyboardInterrupt("consumer interrupted")
    with pytest.raises(KeyboardInterrupt) as error:
        with cursor:
            next(cursor)
            raise interruption
    assert error.value is interruption
    assert source.closes == [0]
    assert cursor.observed_eof is False


def test_custom_open_stopiteration_after_chain_prefix_closes_borrowed_inputs():
    """Custom acquisition StopIteration is failure after a valid Chain prefix."""
    source = LifecycleSource(("borrowed",))
    source.events = []
    stopped = StopIteration("custom open stopped")

    def failing_open(iterator):
        raise stopped

    node = StreamNode(
        name="failing-open", inputs=(IteratorPort(source.spec),),
        output=IteratorPort(source.spec), element_spec_transform=_paired_spec,
        cardinality_transform=_unknown_cardinality, pull_policy=("one",),
        max_buffered_items=0,
        implementation=failing_open,
    )
    # Use the existing module-level custom type's instance declaration so no
    # closure-backed Dataset definition is needed for this lifecycle fixture.
    custom = PairedStreamDataset(source, TrackingDataset((3,)))
    custom.sources = (source,)
    custom.node = node
    cursor = _cursor(Chain(TrackingDataset((1,)), custom), True)
    assert next(cursor) == 1
    with pytest.raises(RuntimeError, match="Unexpected StopIteration") as error:
        next(cursor)
    assert error.value.__cause__ is stopped
    assert source.events == ["borrowed"]
    assert cursor.observed_eof is False
