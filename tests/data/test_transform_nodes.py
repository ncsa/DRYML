import numpy as np
import pytest

from dryml.core2.cardinality import Cardinality
from dryml.core2.backend import Backend
from dryml.core2.tensor_spec import TensorSpec
from dryml.data.dataset import Dataset, Map
from dryml.data.transforms import Batch, Cast, Flatten, Pack, Pipe, Repeat, Scale, Select, Shuffle, Skip, Take, Unbatch


class ListDataset(Dataset):
    def __init__(self, items, spec):
        self.items = list(items)
        super().__init__(spec=spec)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return Cardinality.finite(len(self.items))


class CountingCast(Cast):
    def __init__(self, dtype):
        super().__init__(dtype)
        self.dispatch_count = 0

    def resolve_impl_for(self, *args, **kwargs):
        self.dispatch_count += 1
        return super().resolve_impl_for(*args, **kwargs)


def test_cast_infer_output_spec_and_iteration():
    src = ListDataset(
        [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)],
        TensorSpec("int32", shape=(2,), backend="numpy"),
    )

    ds = Map(src, Cast("float32"))
    out = list(ds)

    assert ds.spec == TensorSpec("float32", shape=(2,), backend="numpy")
    assert ds.spec.backend is Backend.numpy
    assert [item.dtype for item in out] == [np.dtype("float32"), np.dtype("float32")]


def test_select_infer_output_spec_and_iteration():
    src = ListDataset(
        [
            {"x": np.array([1, 2], dtype=np.int32), "y": np.array([3], dtype=np.int32)},
            {"x": np.array([4, 5], dtype=np.int32), "y": np.array([6], dtype=np.int32)},
        ],
        {
            "x": TensorSpec("int32", shape=(2,), backend="numpy"),
            "y": TensorSpec("int32", shape=(1,), backend="numpy"),
        },
    )

    ds = Map(src, Select("x"))

    assert ds.spec == TensorSpec("int32", shape=(2,), backend="numpy")
    assert [item.tolist() for item in ds] == [[1, 2], [4, 5]]


def test_select_accepts_multiple_indices():
    src = ListDataset(
        [
            {"x": (np.array([1, 2], dtype=np.int32),)},
            {"x": (np.array([3, 4], dtype=np.int32),)},
        ],
        {"x": (TensorSpec("int32", shape=(2,), backend="numpy"),)},
    )

    ds = Map(src, Select("x", 0))

    assert ds.spec == TensorSpec("int32", shape=(2,), backend="numpy")
    assert [item.tolist() for item in ds] == [[1, 2], [3, 4]]


def test_elementwise_dataset_resolves_dispatch_once_per_iterator():
    transform = CountingCast("float32")
    src = ListDataset(
        [
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
            np.array([5, 6], dtype=np.int32),
        ],
        TensorSpec("int32", shape=(2,), backend="numpy"),
    )

    ds = Map(src, transform)

    list(ds)
    list(ds)

    assert transform.dispatch_count == 2


def test_pipe_infer_output_spec_and_call():
    pipe = Pipe(Select("x"), Cast("float32"))
    spec = {
        "x": TensorSpec("int32", shape=(2,), backend="numpy"),
        "y": TensorSpec("int32", shape=(1,), backend="numpy"),
    }
    x = {"x": np.array([1, 2], dtype=np.int32), "y": np.array([3], dtype=np.int32)}

    out = pipe(x)

    assert pipe.infer_output_spec(spec) == TensorSpec("float32", shape=(2,), backend="numpy")
    assert out.dtype == np.dtype("float32")
    assert out.tolist() == [1.0, 2.0]


def test_map_accepts_multiple_transforms_as_pipe():
    transform = CountingCast("float32")
    src = ListDataset(
        [
            {"x": np.array([1, 2], dtype=np.int32)},
            {"x": np.array([3, 4], dtype=np.int32)},
        ],
        {"x": TensorSpec("int32", shape=(2,), backend="numpy")},
    )

    ds = Map(src, Select("x"), transform)
    out = list(ds)

    assert ds.spec == TensorSpec("float32", shape=(2,), backend="numpy")
    assert [item.dtype for item in out] == [np.dtype("float32"), np.dtype("float32")]
    assert transform.dispatch_count == 1


def test_flatten_and_scale_infer_output_spec_and_iteration():
    src = ListDataset(
        [np.array([[0, 255]], dtype=np.uint8)],
        TensorSpec("uint8", shape=(1, 2), backend="numpy"),
    )

    ds = Map(src, Scale.from_range(0, 255), Flatten())
    out = list(ds)

    assert ds.spec == TensorSpec("float32", shape=(2,), backend="numpy")
    np.testing.assert_allclose(out[0], np.array([0.0, 1.0], dtype=np.float32))


def test_batch_infer_output_spec_and_iteration():
    src = ListDataset(
        [
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
            np.array([5, 6], dtype=np.int32),
        ],
        TensorSpec("int32", shape=(2,), backend="numpy"),
    )

    ds = Batch(src, 2)
    out = list(ds)

    assert ds.spec == TensorSpec("int32", shape=(2,), batch=2, backend="numpy")
    assert ds.spec.backend is Backend.numpy
    assert [item.shape for item in out] == [(2, 2), (1, 2)]
    assert out[0].tolist() == [[1, 2], [3, 4]]
    assert out[1].tolist() == [[5, 6]]


def test_unbatch_infer_output_spec_and_iteration():
    src = ListDataset(
        [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6]], dtype=np.int32),
        ],
        TensorSpec("int32", shape=(2,), batch=2, backend="numpy"),
    )

    ds = Unbatch(src)
    out = list(ds)

    assert ds.spec == TensorSpec("int32", shape=(2,), backend="numpy")
    assert [item.tolist() for item in out] == [[1, 2], [3, 4], [5, 6]]


def test_take_skip_and_repeat_cardinality_and_iteration():
    src = ListDataset(
        [np.array([i], dtype=np.int32) for i in range(5)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )

    taken = Take(src, 3)
    skipped = Skip(src, 2)
    repeated = Repeat(Take(src, 2), 3)

    assert taken.__len__() == Cardinality.finite(3)
    assert skipped.__len__() == Cardinality.finite(3)
    assert repeated.__len__() == Cardinality.finite(6)
    assert [item.item() for item in taken] == [0, 1, 2]
    assert [item.item() for item in skipped] == [2, 3, 4]
    assert [item.item() for item in repeated] == [0, 1, 0, 1, 0, 1]


def test_take_zero_has_zero_cardinality():
    src = ListDataset(
        [np.array([1], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )

    ds = Take(src, 0)

    assert ds.__len__() == Cardinality.finite(0)
    assert list(ds) == []


def test_shuffle_is_seeded_and_preserves_elements():
    src = ListDataset(
        [np.array([i], dtype=np.int32) for i in range(5)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )

    first = [item.item() for item in Shuffle(src, 5, seed=7)]
    second = [item.item() for item in Shuffle(src, 5, seed=7)]

    assert first == second
    assert sorted(first) == [0, 1, 2, 3, 4]


def test_pack_positional_infer_output_spec_and_iteration():
    left = ListDataset(
        [np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )
    right = ListDataset(
        [np.array([3, 4], dtype=np.float32), np.array([5, 6], dtype=np.float32)],
        TensorSpec("float32", shape=(2,), backend="numpy"),
    )

    ds = Pack(left, right)
    out = list(ds)

    assert ds.spec == (left.spec, right.spec)
    assert [(a.tolist(), b.tolist()) for a, b in out] == [([1], [3.0, 4.0]), ([2], [5.0, 6.0])]


def test_pack_nested_tree_with_int_key():
    left = ListDataset(
        [np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )
    right = ListDataset(
        [np.array([3], dtype=np.float32)],
        TensorSpec("float32", shape=(1,), backend="numpy"),
    )

    ds = Pack({1: left, "b": {"2": right}})
    out = list(ds)

    assert ds.spec == {1: left.spec, "b": {"2": right.spec}}
    assert len(out) == 1
    assert out[0][1].tolist() == [1]
    assert out[0]["b"]["2"].tolist() == [3.0]


def test_nested_pack_is_dataset_leaf_for_outer_pack():
    left = ListDataset(
        [np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )
    right = ListDataset(
        [np.array([3], dtype=np.float32), np.array([4], dtype=np.float32)],
        TensorSpec("float32", shape=(1,), backend="numpy"),
    )

    inner = Pack({1: left, "test": right})
    outer = Pack(inner, left)
    out = list(outer)

    assert outer.spec == (inner.spec, left.spec)
    assert out[0][0][1].tolist() == [1]
    assert out[0][0]["test"].tolist() == [3.0]
    assert out[0][1].tolist() == [1]


def test_pack_rejects_noncanonical_dict_key():
    src = ListDataset(
        [np.array([1], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )

    with pytest.raises(TypeError):
        Pack({object(): src})
