import sys

import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import ArrayDataset, Batch, Cast, Chain, GeneratorDataset, Map, NpyFileDataset, Repeat, Shuffle, Skip, Take, Unbatch, Zip
from dryml.data.dataset import Dataset


def _canonical(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in value}
    if isinstance(value, tuple):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


class _CardinalityDataset(Dataset):
    def __init__(self, items, cardinality, *, spec=None):
        self.items = list(items)
        self.cardinality = cardinality
        super().__init__(spec=spec or TensorSpec("int32", shape=(1,), backend="numpy"))

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        if self.cardinality == "unavailable":
            raise NotImplementedError("cardinality is not available")
        return self.cardinality


def _base_items(count=5):
    return [np.array([i], dtype=np.int32) for i in range(count)]


def _iter_base_items():
    return iter(_base_items())


def _finite_dataset(count=5):
    return _CardinalityDataset(_base_items(count), Cardinality.finite(count))


def _unknown_dataset(count=5):
    return _CardinalityDataset(_base_items(count), Cardinality.UNKNOWN)


def _infinite_dataset(count=5):
    return _CardinalityDataset(_base_items(count), Cardinality.INFINITE)


def _unavailable_dataset(count=5):
    return _CardinalityDataset(_base_items(count), "unavailable")


def _dataset_cases(tmp_path):
    np.save(tmp_path / "a.npy", np.array([1], dtype=np.int32))
    np.save(tmp_path / "b.npy", np.array([2], dtype=np.int32))

    base = _finite_dataset()
    left = _finite_dataset(2)
    right = _CardinalityDataset(
        [np.array([10], dtype=np.int32), np.array([11], dtype=np.int32)],
        Cardinality.finite(2),
    )

    return [
        ("array", ArrayDataset(np.arange(5, dtype=np.int32)[:, None])),
        (
            "generator",
            GeneratorDataset(
                _iter_base_items,
                cardinality=Cardinality.finite(5),
                spec=TensorSpec("int32", shape=(1,), backend="numpy"),
            ),
        ),
        ("npy-file", NpyFileDataset(str(tmp_path))),
        ("map", Map(base, Cast("float32"))),
        ("batch", Batch(base, 2)),
        ("unbatch", Unbatch(Batch(base, 2))),
        ("take", Take(base, 3)),
        ("skip", Skip(base, 2)),
        ("repeat", Repeat(Take(base, 2), 2)),
        ("shuffle", Shuffle(base, 3, seed=11)),
        ("zip", Zip(left, right)),
        ("chain", Chain(left, right)),
    ]


def test_finite_datasets_have_repeatable_non_consuming_peek(tmp_path):
    for case_id, ds in _dataset_cases(tmp_path):

        first_peek = _canonical(ds.peek())
        second_peek = _canonical(ds.peek())
        first_iteration = _canonical(list(ds))
        second_iteration = _canonical(list(ds))

        assert first_peek == second_peek, case_id
        assert first_peek == first_iteration[0], case_id
        assert first_iteration == second_iteration, case_id


def test_empty_dataset_peek_raises_value_error():
    ds = ArrayDataset(np.empty((0, 1), dtype=np.int32))

    with pytest.raises(ValueError, match="empty dataset"):
        ds.peek()


@pytest.mark.parametrize(
    ("source_factory", "drop_remainder", "expected"),
    [
        (_finite_dataset, False, Cardinality.finite(3)),
        (_finite_dataset, True, Cardinality.finite(2)),
        (_unknown_dataset, False, Cardinality.UNKNOWN),
        (_infinite_dataset, False, Cardinality.INFINITE),
        (_unavailable_dataset, False, Cardinality.UNKNOWN),
    ],
)
def test_batch_cardinality_handles_source_cardinality_modes(source_factory, drop_remainder, expected):
    assert Batch(source_factory(), 2, drop_remainder=drop_remainder).__len__() == expected


def test_batch_unbatch_roundtrip_preserves_finite_values():
    ds = _finite_dataset(5)

    out = [_canonical(item) for item in Unbatch(Batch(ds, 2))]

    assert out == [[0], [1], [2], [3], [4]]
    assert Unbatch(Batch(ds, 2)).__len__() == Cardinality.finite(5)


def test_numpy_batch_does_not_import_optional_frameworks(monkeypatch):
    requested = []
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.setattr(
        "dryml.runtime.import_configured_framework",
        lambda name: requested.append(name),
    )

    batch = next(iter(Batch(_finite_dataset(2), 2)))

    assert batch.tolist() == [[0], [1]]
    assert requested == []
    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules


def test_batch_unbatch_drop_remainder_cardinality_discards_partial_batch():
    ds = _finite_dataset(5)

    out = [_canonical(item) for item in Unbatch(Batch(ds, 2, drop_remainder=True))]

    assert out == [[0], [1], [2], [3]]
    assert Unbatch(Batch(ds, 2, drop_remainder=True)).__len__() == Cardinality.finite(4)


@pytest.mark.parametrize(
    ("source_factory", "expected"),
    [
        (_unknown_dataset, Cardinality.UNKNOWN),
        (_infinite_dataset, Cardinality.INFINITE),
        (_unavailable_dataset, Cardinality.UNKNOWN),
    ],
)
def test_unbatch_of_batch_handles_source_cardinality_modes(source_factory, expected):
    assert Unbatch(Batch(source_factory(), 2)).__len__() == expected


def test_unbatch_generic_finite_batched_source_is_conservative():
    ds = _CardinalityDataset(
        [np.array([[0], [1]], dtype=np.int32)],
        Cardinality.finite(1),
        spec=TensorSpec("int32", shape=(1,), batch=2, backend="numpy"),
    )

    assert Unbatch(ds).__len__() == Cardinality.UNKNOWN


def test_unbatch_generic_infinite_fixed_batch_source_is_infinite():
    ds = _CardinalityDataset(
        [],
        Cardinality.INFINITE,
        spec=TensorSpec("int32", shape=(1,), batch=2, backend="numpy"),
    )

    assert Unbatch(ds).__len__() == Cardinality.INFINITE


def test_unbatch_generic_infinite_dynamic_batch_source_is_conservative():
    ds = _CardinalityDataset(
        [],
        Cardinality.INFINITE,
        spec=TensorSpec("int32", shape=(1,), batch=Dynamic, backend="numpy"),
    )

    assert Unbatch(ds).__len__() == Cardinality.UNKNOWN


@pytest.mark.parametrize(
    ("source_factory", "expected"),
    [
        (_finite_dataset, Cardinality.finite(5)),
        (_unknown_dataset, Cardinality.UNKNOWN),
        (_infinite_dataset, Cardinality.INFINITE),
        (_unavailable_dataset, Cardinality.UNKNOWN),
    ],
)
def test_shuffle_preserves_source_cardinality_modes(source_factory, expected):
    assert Shuffle(source_factory(), 3, seed=17).__len__() == expected


@pytest.mark.parametrize(
    ("source_factory", "expected"),
    [
        (_finite_dataset, Cardinality.finite(3)),
        (_unknown_dataset, Cardinality.UNKNOWN),
        (_infinite_dataset, Cardinality.finite(3)),
        (_unavailable_dataset, Cardinality.UNKNOWN),
    ],
)
def test_take_cardinality_handles_source_cardinality_modes(source_factory, expected):
    assert Take(source_factory(), 3).__len__() == expected


def test_map_preserves_unknown_for_unavailable_source_cardinality():
    assert Map(_unavailable_dataset(), Cast("float32")).__len__() == Cardinality.UNKNOWN


def test_zip_preserves_unknown_for_unavailable_source_cardinality():
    assert Zip(_finite_dataset(), _unavailable_dataset()).__len__() == Cardinality.UNKNOWN


def test_chain_preserves_unknown_for_unavailable_source_cardinality():
    assert Chain(_finite_dataset(), _unavailable_dataset()).__len__() == Cardinality.UNKNOWN


def test_shuffle_preserves_count_and_multiset_for_finite_sources():
    out = [_canonical(item) for item in Shuffle(_finite_dataset(7), 3, seed=23)]

    assert len(out) == 7
    assert sorted(out) == [[0], [1], [2], [3], [4], [5], [6]]
