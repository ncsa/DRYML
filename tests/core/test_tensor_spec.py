import pickle

import pytest

from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import (
    Dynamic,
    Layout,
    TensorSpec,
    is_spec_tree,
    batch_spec_tree,
    unbatch_spec_tree,
)
from dryml.core2.utils.recurse import iter_leaves, map_leaves


def test_tensor_spec_normalizes_dtype_and_shape():
    spec = TensorSpec(dtype="float32", shape=[32, Dynamic])

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (32, Dynamic)
    assert spec.batch is None
    assert spec.layout is Layout.DENSE


def test_tensor_spec_is_hashable_and_pickleable():
    spec = TensorSpec(
        dtype="float32",
        shape=(32, Dynamic),
        batch=Dynamic,
        layout=Layout.DENSE,
    )

    spec2 = pickle.loads(pickle.dumps(spec))

    assert spec2 == spec
    assert hash(spec2) == hash(spec)


def test_tensor_spec_rejects_negative_dim():
    with pytest.raises(ValueError):
        TensorSpec(dtype="float32", shape=(32, -1))


def test_tensor_spec_rejects_bool_dim():
    with pytest.raises(TypeError):
        TensorSpec(dtype="float32", shape=(True, 32))


def test_tensor_spec_unknown_rank():
    spec = TensorSpec(dtype="float32", shape=None)

    assert spec.shape is None
    assert spec.rank is None
    assert spec.full_rank is None
    assert spec.full_shape is None


def test_tensor_spec_axis_names_require_known_rank():
    with pytest.raises(ValueError):
        TensorSpec(dtype="float32", shape=None, axis_names=("x",))


def test_tensor_spec_axis_names_length_must_match_rank():
    with pytest.raises(ValueError):
        TensorSpec(dtype="float32", shape=(32, 16), axis_names=("x",))


def test_tensor_spec_full_shape_unbatched():
    spec = TensorSpec(dtype="float32", shape=(32, 16))
    assert spec.full_shape == (32, 16)
    assert spec.full_rank == 2
    assert spec.batched is False


def test_tensor_spec_full_shape_batched_dynamic():
    spec = TensorSpec(dtype="float32", shape=(32, 16), batch=Dynamic)
    assert spec.full_shape == (Dynamic, 32, 16)
    assert spec.full_rank == 3
    assert spec.batched is True


def test_tensor_spec_full_shape_batched_fixed():
    spec = TensorSpec(dtype="float32", shape=(32, 16), batch=64)
    assert spec.full_shape == (64, 32, 16)
    assert spec.full_rank == 3
    assert spec.batched is True


def test_tensor_spec_with_batch_without_batch():
    spec = TensorSpec(dtype="float32", shape=(32,))
    batched = spec.with_batch()
    unbatched = batched.without_batch()

    assert spec.batch is None
    assert batched.batch is Dynamic
    assert batched.batch_axis_name == "batch"
    assert unbatched.batch is None


def test_tensor_spec_with_fixed_batch():
    spec = TensorSpec(dtype="float32", shape=(32,))
    batched = spec.with_batch(batch=128, axis_name="N")

    assert batched.batch == 128
    assert batched.batch_axis_name == "N"
    assert batched.full_shape == (128, 32)


def test_tensor_spec_with_shape_and_dtype():
    spec = TensorSpec(dtype="float32", shape=(32,))
    spec2 = spec.with_dtype("int64").with_shape((16, 8))

    assert spec2.dtype == DType("int", 64)
    assert spec2.shape == (16, 8)


def test_tensor_spec_compatible_with_shape_unbatched():
    spec = TensorSpec(dtype="float32", shape=(32, Dynamic, 8))

    assert spec.compatible_with_shape((32, 10, 8))
    assert spec.compatible_with_shape((32, 0, 8))
    assert not spec.compatible_with_shape((31, 10, 8))
    assert not spec.compatible_with_shape((32, 10))
    assert not spec.compatible_with_shape((32, 10, 7))


def test_tensor_spec_compatible_with_shape_batched_dynamic():
    spec = TensorSpec(dtype="float32", shape=(32,), batch=Dynamic)

    assert spec.compatible_with_shape((1, 32))
    assert spec.compatible_with_shape((64, 32))
    assert not spec.compatible_with_shape((64, 31))
    assert not spec.compatible_with_shape((32,))


def test_tensor_spec_compatible_with_shape_batched_fixed():
    spec = TensorSpec(dtype="float32", shape=(32,), batch=8)

    assert spec.compatible_with_shape((8, 32))
    assert not spec.compatible_with_shape((7, 32))
    assert not spec.compatible_with_shape((8, 31))


def test_tensor_spec_layout_metadata():
    spec = TensorSpec(
        dtype="float32",
        shape=(Dynamic, 16),
        layout=Layout.RAGGED,
        ragged_rank=1,
        row_splits_dtype="int64",
    )

    assert spec.layout is Layout.RAGGED
    assert spec.ragged_rank == 1
    assert spec.row_splits_dtype == DType("int", 64)


def test_is_spec_tree_simple_dict():
    spec = {
        "cart": TensorSpec(dtype="float32", shape=(3,)),
        "sph": TensorSpec(dtype="float32", shape=(2,)),
    }

    assert is_spec_tree(spec)


def test_is_spec_tree_nested():
    spec = {
        "x": TensorSpec(dtype="float32", shape=(3,)),
        "y": (
            TensorSpec(dtype="int32", shape=()),
            {"z": TensorSpec(dtype="float64", shape=(2, 2))},
        ),
    }

    assert is_spec_tree(spec)


def test_is_spec_tree_rejects_bad_dict_key():
    spec = {
        0: TensorSpec(dtype="float32", shape=(3,)),
    }

    assert not is_spec_tree(spec)


def test_is_spec_tree_rejects_bad_leaf():
    spec = {
        "cart": TensorSpec(dtype="float32", shape=(3,)),
        "bad": "not a tensor spec",
    }

    assert not is_spec_tree(spec)


def test_map_leaves_applies_to_all_leaves():
    spec = {
        "cart": TensorSpec(dtype="float32", shape=(3,)),
        "sph": (
            TensorSpec(dtype="float32", shape=(2,)),
            {"mask": TensorSpec(dtype="bool", shape=(3,))},
        ),
    }

    new_spec = map_leaves(spec, lambda s: s.with_batch())

    assert new_spec["cart"].batch is Dynamic
    assert new_spec["sph"][0].batch is Dynamic
    assert new_spec["sph"][1]["mask"].batch is Dynamic

    # original unchanged
    assert spec["cart"].batch is None
    assert spec["sph"][0].batch is None
    assert spec["sph"][1]["mask"].batch is None


def test_iter_tensor_specs_yields_all_leaves():
    a = TensorSpec(dtype="float32", shape=(3,))
    b = TensorSpec(dtype="float32", shape=(2,))
    c = TensorSpec(dtype="bool", shape=(Dynamic,))

    spec = {
        "cart": a,
        "nested": (b, {"mask": c}),
    }

    out = list(iter_leaves(spec))

    assert out == [a, b, c]


def test_batch_and_unbatch_spec_tree():
    spec = {
        "cart": TensorSpec(dtype="float32", shape=(3,)),
        "sph": TensorSpec(dtype="float32", shape=(2,)),
    }

    batched = batch_spec_tree(spec)
    assert batched["cart"].batch is Dynamic
    assert batched["sph"].batch is Dynamic

    unbatched = unbatch_spec_tree(batched)
    assert unbatched["cart"].batch is None
    assert unbatched["sph"].batch is None
