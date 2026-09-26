import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArrayDataset, Dataset, DatasetExhaustedError, NpyFileDataset, Skip, Take


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
    assert not hasattr(dataset, "_cursor")


@pytest.mark.parametrize("count", (-1, True, 1.0))
def test_cursor_skip_rejects_invalid_counts_before_advancement(count):
    cursor = TrackingDataset([0, 1]).iterator()

    with pytest.raises((TypeError, ValueError)):
        cursor.skip(count)

    assert cursor.position == 0
    cursor.skip(0)
    assert cursor.position == 0


def test_cursor_skip_reports_partial_advancement_on_exhaustion():
    cursor = TrackingDataset([0, 1]).iterator()

    with pytest.raises(DatasetExhaustedError) as error:
        cursor.skip(3)

    assert (error.value.requested, error.value.yielded) == (3, 2)
    assert cursor.position == 2


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


def test_take_closes_an_open_source_cursor_explicitly_and_skip_remains_forgiving():
    source = TrackingDataset([0, 1])
    cursor = Take(source, 2).iterator()

    assert next(cursor) == 0
    cursor.close()

    assert source.closes == [0]
    assert list(Skip(source, 10)) == []
