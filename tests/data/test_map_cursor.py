import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArrayDataset, Cast, Dataset, DatasetExhaustedError, Map, Pipe, Project
from dryml.methods import Method, traits


class CountingMethod(Method):
    """Test Method with an optional explicit iteration-independence declaration."""

    def __init__(self, *, independent):
        self.independent = independent
        self.calls = []

    @property
    def iteration_independent(self):
        return self.independent

    def __call__(self, value):
        self.calls.append(int(value))
        return value + 10

    def infer_output_spec(self, input_spec):
        return input_spec


def _array_source():
    return ArrayDataset(
        np.arange(4, dtype=np.int64),
        spec=TensorSpec("int64", shape=(), backend="numpy"),
    )


@pytest.mark.parametrize(
    "build",
    (
        lambda methods: Pipe(*methods),
        lambda methods: Project(left=methods[0], right=methods[1]),
    ),
)
def test_qualified_map_pipe_and_project_delegate_skip_without_calling_prefix(build):
    first = CountingMethod(independent=True)
    second = CountingMethod(independent=True)
    mapped = Map(_array_source(), build((first, second)))
    cursor = mapped.iterator()

    cursor.skip(2)
    result = next(cursor)

    if isinstance(result, dict):
        assert {key: int(value) for key, value in result.items()} == {"left": 12, "right": 12}
        assert second.calls == [2]
    else:
        assert int(result) == 22
        assert second.calls == [12]
    assert first.calls == [2]
    assert cursor.position == 3


def test_qualified_map_consumes_a_nonindexed_prefix_without_transforming_it():
    class SequentialSource(Dataset):
        def __init__(self):
            self.opens = 0
            super().__init__(spec=TensorSpec("int64", shape=(), backend="numpy"))

        def __iter__(self):
            self.opens += 1
            yield from np.arange(4, dtype=np.int64)

        def __len__(self):
            return Cardinality.finite(4)

    source = SequentialSource()
    method = CountingMethod(independent=True)
    cursor = Map(source, method).iterator()

    cursor.skip(2)

    assert next(cursor) == 12
    assert method.calls == [2]
    assert source.opens == 1


def test_unqualified_map_executes_discarded_prefix_in_order():
    method = CountingMethod(independent=False)
    cursor = Map(_array_source(), method).iterator()

    cursor.skip(2)

    assert next(cursor) == 12
    assert method.calls == [0, 1, 2]


def test_map_fast_skip_reports_source_partial_progress_without_transforming_values():
    method = CountingMethod(independent=True)
    cursor = Map(_array_source(), method).iterator()

    with pytest.raises(DatasetExhaustedError) as error:
        cursor.skip(5)

    assert (error.value.requested, error.value.yielded) == (5, 4)
    assert cursor.position == 4
    assert method.calls == []


def test_unknown_backend_selection_is_not_started_by_zero_skip():
    class BackendCounting(Method):
        def __init__(self):
            self.calls = []

        @property
        def iteration_independent(self):
            return True

        @traits(backend="numpy")
        def numpy(self, value):
            self.calls.append(int(value))
            return value

        def infer_output_spec(self, input_spec):
            return input_spec

    class CountingSource(Dataset):
        def __init__(self):
            self.next_calls = 0
            super().__init__(spec=TensorSpec("int64", shape=(), backend=None))

        def __iter__(self):
            for value in np.arange(2, dtype=np.int64):
                self.next_calls += 1
                yield value

        def __len__(self):
            return Cardinality.finite(2)

    source = CountingSource()
    method = BackendCounting()
    cursor = Map(source, method).iterator()

    cursor.skip(0)
    assert source.next_calls == 0
    cursor.skip(1)

    assert source.next_calls == 1
    assert method.calls == []
    assert next(cursor) == 1
    assert method.calls == [1]


def test_prepared_map_graph_skip_and_replay_keep_iteration_dependent_calls():
    """Graph cursors replay fresh sources and do not omit dependent skipped calls."""
    method = CountingMethod(independent=False)
    graph = Map(_array_source(), method).method_graph()
    graph.learn()
    first = graph.iterator()
    first.skip(2)
    assert int(next(first)) == 12
    first.close()

    second = graph.iterator()
    assert int(next(second)) == 10
    assert method.calls == [0, 1, 2, 0]


@pytest.mark.parametrize("prepared", (False, True))
@pytest.mark.parametrize("unknown_backend", (False, True))
def test_retained_map_execution_compiles_and_selects_only_at_resolution(monkeypatch, prepared, unknown_backend):
    """Ordinary/prepared cursors retain invokers after known or cursor-local resolution."""
    from dryml.core import signatures
    compiled, selections = [], []
    original_compile = signatures.compile_signature
    original_select = Method.find_implementation

    def select(self, *args, **kwargs):
        selections.append(type(self))
        return original_select(self, *args, **kwargs)

    monkeypatch.setattr(signatures, "compile_signature",
                        lambda target, **kw: (compiled.append(target), original_compile(target, **kw))[1])
    monkeypatch.setattr(Method, "find_implementation", select)
    for length in (5, 100):
        source = ArrayDataset(np.arange(length, dtype=np.int64),
                              spec=TensorSpec("int64", shape=(), backend=None if unknown_backend else "numpy"))
        method = Cast("float64") if unknown_backend else Pipe(Cast("float64"), Cast("float32"))
        mapped = Map(source, method)
        if prepared:
            graph = mapped.method_graph()
            graph.learn()
            cursor = graph.iterator()
        else:
            cursor = mapped.iterator()
        assert next(cursor) == 0
        resolved = len(compiled), len(selections)
        assert list(cursor) == list(range(1, length))
        assert (len(compiled), len(selections)) == resolved
