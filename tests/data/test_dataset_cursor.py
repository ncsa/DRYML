import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArrayDataset, Dataset, DatasetExhaustedError, NpyFileDataset, Skip, Take, Zip


class TrackingDataset(Dataset):
    """Re-iterable test source that records iterator acquisition and closure."""

    def __init__(self, items, *, cardinality=None):
        self.items = tuple(items)
        self.opens = 0
        self.closes = []
        self.cardinality = cardinality or Cardinality.finite(len(self.items))
        super().__init__(spec=TensorSpec("int64", shape=(), backend="numpy"))

    def __iter__(self):
        iterator_id = self.opens
        self.opens += 1
        try:
            yield from self.items
        finally:
            self.closes.append(iterator_id)

    def __len__(self):
        return self.cardinality


def test_dataset_cursors_are_independent_and_close_their_own_sources():
    dataset = TrackingDataset([0, 1, 2])
    first = dataset.iterator()
    second = dataset.iterator()

    assert next(first) == 0
    second.skip(2)
    assert (first.position, second.position) == (1, 2)
    assert next(second) == 2

    first.close()
    assert 0 in dataset.closes
    with pytest.raises(StopIteration):
        next(first)
    assert list(second) == []
    assert sorted(dataset.closes) == [0, 1]
    assert first.observed_eof is False
    assert second.observed_eof is True
    with pytest.raises(DatasetExhaustedError) as error:
        second.skip(1)
    assert (error.value.requested, error.value.yielded) == (1, 0)
    assert second.observed_eof is True
    assert sorted(dataset.closes) == [0, 1]
    assert not hasattr(dataset, "_cursor")


def test_cleanup_diagnostic_survives_without_exception_add_note():
    """Python 3.10-style errors keep cleanup evidence and their primary identity."""
    from dryml.data import DatasetCursor

    class LegacyError(ValueError):
        add_note = None

    class Resource:
        def close(self):
            raise OSError("source close failed")

    primary = LegacyError("consumer failed")
    primary.__notes__ = ["prior diagnostic"]
    cursor = DatasetCursor(iter(()))
    cursor._own(Resource())
    with pytest.raises(LegacyError) as caught:
        with cursor:
            raise primary
    assert caught.value is primary
    assert primary.__notes__[0] == "prior diagnostic"
    assert "OSError: source close failed" in primary.__notes__[1]
    assert not cursor.observed_eof


@pytest.mark.parametrize("count", (-1, True, 1.0))
@pytest.mark.parametrize("prepared", (False, True))
def test_cursor_skip_rejects_invalid_counts_before_advancement(count, prepared):
    source = TrackingDataset([0, 1])
    if prepared:
        graph = source.method_graph()
        graph.learn()
        cursor = graph.iterator()
    else:
        cursor = source.iterator()

    with pytest.raises((TypeError, ValueError)):
        cursor.skip(count)

    assert cursor.position == 0
    cursor.skip(0)
    assert cursor.position == 0
    assert source.opens == 0
    assert source.closes == []
    assert cursor.observed_eof is False
    assert next(cursor) == 0
    cursor.close()


@pytest.mark.parametrize("kind", ("plain", "graph", "stateful_map", "independent_map", "indexed"))
def test_cursor_skip_reports_partial_advancement_on_exhaustion(kind):
    """Strict skip shortfall wins over cleanup without certifying root EOF."""
    from dryml.data import Map
    from dryml.methods import Method

    acquired = []
    closed = []

    class Resource:
        def __init__(self):
            self.values = iter(np.arange(2, dtype=np.int64))
            acquired.append(self)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.values)

        def close(self):
            closed.append(self)
            raise RuntimeError("source close failed")

    class Source(TrackingDataset):
        def __iter__(self):
            return self.resource_factory()

    class Identity(Method):
        def __init__(self, independent):
            self.independent = independent

        @property
        def iteration_independent(self):
            return self.independent

        def __call__(self, value):
            return value

        def infer_output_spec(self, input_spec):
            return input_spec

    source = Source([0, 1])
    source.resource_factory = Resource
    if kind == "graph":
        graph = Zip(source, source).method_graph()
        graph.learn()
        cursor = graph.iterator()
    elif kind in {"stateful_map", "independent_map"}:
        cursor = Map(source, Identity(kind == "independent_map")).iterator()
    elif kind == "indexed":
        cursor = ArrayDataset(np.arange(2)).iterator()
        cursor._own(Resource())
    else:
        cursor = source.iterator()

    with pytest.raises(DatasetExhaustedError) as error:
        cursor.skip(3)

    assert (error.value.requested, error.value.yielded) == (3, 2)
    assert type(error.value) is DatasetExhaustedError
    assert cursor.position == 2
    assert cursor.observed_eof is False
    assert closed == acquired[::-1]
    assert len(acquired) == (2 if kind == "graph" else 1)
    assert any("source close failed" in note for note in error.value.__notes__)
    cursor.close()
    cursor.skip(0)
    with pytest.raises(DatasetExhaustedError) as closed_error:
        cursor.skip(1)
    assert (closed_error.value.requested, closed_error.value.yielded) == (1, 0)
    for invalid in (-1, True, 1.0):
        with pytest.raises((TypeError, ValueError)):
            cursor.skip(invalid)
    assert cursor.position == 2
    assert cursor.observed_eof is False
    assert closed == acquired[::-1]


def test_array_cursor_skip_uses_index_without_reading_discarded_elements(monkeypatch):
    reads = []
    values = np.arange(4, dtype=np.int64)
    dataset = ArrayDataset(values, spec=TensorSpec("int64", shape=(), backend="numpy"))
    from dryml.data.source import _tree_index

    def tracked_tree_index(arrays, index):
        reads.append(index)
        return _tree_index(arrays, index)

    monkeypatch.setattr("dryml.data.source._tree_index", tracked_tree_index)
    cursor = dataset.iterator()

    cursor.skip(2)

    assert int(next(cursor)) == 2
    assert reads == [2]


def test_npy_cursor_skip_uses_sorted_file_index_without_loading_discarded_files(tmp_path, monkeypatch):
    for name, value in (("c", 2), ("a", 0), ("b", 1)):
        np.save(tmp_path / f"{name}.npy", np.array([value], dtype=np.int64))
    dataset = NpyFileDataset(
        str(tmp_path),
        spec=TensorSpec("int64", shape=(1,), backend="numpy"),
    )
    loads = []
    original_load = np.load

    def tracked_load(path, *args, **kwargs):
        loads.append(path.name)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr("dryml.data.source.np.load", tracked_load)
    cursor = dataset.iterator()
    cursor.skip(2)

    assert next(cursor).tolist() == [2]
    assert loads == ["c.npy"]


@pytest.mark.parametrize("count", (-1, True, 1.0))
def test_take_rejects_invalid_exact_counts(count):
    with pytest.raises((TypeError, ValueError)):
        Take(TrackingDataset([0]), count)


def test_take_zero_is_finite_and_does_not_open_its_source():
    source = TrackingDataset([0])
    dataset = Take(source, 0)

    assert dataset.__len__() == Cardinality.finite(0)
    assert list(dataset) == []
    assert source.opens == 0


@pytest.mark.parametrize("cardinality", (Cardinality.finite(2), Cardinality.UNKNOWN))
def test_take_raises_after_its_short_prefix(cardinality):
    source = TrackingDataset([0, 1], cardinality=cardinality)
    cursor = Take(source, 3).iterator()

    assert [next(cursor), next(cursor)] == [0, 1]
    with pytest.raises(DatasetExhaustedError) as error:
        next(cursor)

    assert (error.value.requested, error.value.yielded) == (3, 2)
    assert source.closes == [0]
    assert cursor.observed_eof is False


def test_private_range_count_clips_only_declared_finite_sources_not_strict_take():
    source = TrackingDataset([0, 1], cardinality=Cardinality.finite(2))

    assert source._range_yield_cardinality(0, 3) == Cardinality.finite(2)
    assert Take(source, 3).yield_cardinality() == Cardinality.finite(3)


def test_take_closes_an_open_source_cursor_explicitly_and_skip_remains_forgiving():
    source = TrackingDataset([0, 1])
    cursor = Take(source, 2).iterator()

    assert next(cursor) == 0
    cursor.close()

    assert source.closes == [0]
    assert list(Skip(source, 10)) == []


def test_prepared_graph_cursor_skip_preserves_exact_position_and_exhaustion_counts():
    """Graph cursors retain the DatasetCursor exact skip contract independently."""
    left = TrackingDataset([0, 1])
    right = TrackingDataset([10, 11])
    graph = Zip(left, right).method_graph()
    graph.learn()
    cursor = graph.iterator()

    cursor.skip(1)
    assert cursor.position == 1
    assert next(cursor) == (1, 11)
    with pytest.raises(DatasetExhaustedError) as error:
        cursor.skip(1)
    assert (error.value.requested, error.value.yielded) == (1, 0)


@pytest.mark.parametrize("skip", (False, True))
def test_source_read_failure_preserves_primary_when_owned_cleanup_fails(skip):
    """A failed source read closes immediately without turning closure into EOF."""
    from dryml.data import DatasetCursor

    primary = OSError("source read failed")

    class Resource:
        closes = 0

        def __next__(self):
            raise primary

        def close(self):
            self.closes += 1
            raise RuntimeError("source close failed")

    resource = Resource()
    cursor = DatasetCursor(resource)
    with pytest.raises(OSError) as error:
        if skip:
            cursor.skip(1)
        else:
            next(cursor)
    assert error.value is primary
    assert resource.closes == 1
    assert cursor.observed_eof is False
    assert list(cursor) == []
    cursor.close()
    assert resource.closes == 1
