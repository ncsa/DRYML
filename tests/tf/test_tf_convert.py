import pytest

tf = pytest.importorskip("tensorflow")
import dryml.tf as dryml_tf

from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import Dynamic, Layout, TensorSpec


def test_tf_dtype_from_dtype_object():
    assert dryml_tf.dtype(tf.float32) == DType("float", 32)
    assert dryml_tf.dtype(tf.int64) == DType("int", 64)
    assert dryml_tf.dtype(tf.bool) == DType("bool")


def test_tf_dtype_from_spec():
    x = tf.TensorSpec(shape=(None, 32), dtype=tf.float32)
    assert dryml_tf.dtype(x) == DType("float", 32)


def test_tf_tensor_spec_dense_unbatched():
    x = tf.TensorSpec(shape=(None, 32), dtype=tf.float32)

    spec = dryml_tf.tensor_spec(x, assume_batched=False)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (Dynamic, 32)
    assert spec.batch is None
    assert spec.layout is Layout.DENSE


def test_tf_tensor_spec_dense_assume_batched():
    x = tf.TensorSpec(shape=(None, 32), dtype=tf.float32)

    spec = dryml_tf.tensor_spec(x, assume_batched=True)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (32,)
    assert spec.batch is Dynamic
    assert spec.layout is Layout.DENSE
    assert spec.batch_axis_name == "batch"


def test_tf_tensor_spec_from_value():
    x = tf.zeros((4, 32), dtype=tf.float32)

    spec = dryml_tf.tensor_spec(x, assume_batched=True)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (32,)
    assert spec.batch == 4
    assert spec.layout is Layout.DENSE


def test_tf_tensor_spec_ragged():
    x = tf.RaggedTensorSpec(
        shape=(None, None, 8),
        dtype=tf.float32,
        ragged_rank=1,
        row_splits_dtype=tf.int64,
    )

    spec = dryml_tf.tensor_spec(x, assume_batched=False)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (Dynamic, Dynamic, 8)
    assert spec.batch is None
    assert spec.layout is Layout.RAGGED
    assert spec.ragged_rank == 1
    assert spec.row_splits_dtype == DType("int", 64)


def test_tf_tensor_spec_sparse():
    x = tf.SparseTensorSpec(shape=(None, 16), dtype=tf.float32)

    spec = dryml_tf.tensor_spec(x, assume_batched=False)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (Dynamic, 16)
    assert spec.batch is None
    assert spec.layout is Layout.SPARSE
    assert spec.sparse_format == "tf_sparse"


def test_tf_roundtrip_dense_if_forward_methods_installed():
    spec = TensorSpec(dtype="float32", shape=(32,), batch=Dynamic)

    if not hasattr(spec, "tf"):
        pytest.skip("TensorSpec.tf() is not installed.")

    tf_spec = spec.tf(include_batch=True)

    assert isinstance(tf_spec, tf.TensorSpec)
    assert tuple(tf_spec.shape.as_list()) == (None, 32)
    assert tf_spec.dtype == tf.float32


def test_tf_roundtrip_ragged_if_forward_methods_installed():
    spec = TensorSpec(
        dtype="float32",
        shape=(Dynamic, 8),
        batch=None,
        layout=Layout.RAGGED,
        ragged_rank=1,
        row_splits_dtype="int64",
    )

    if not hasattr(spec, "tf"):
        pytest.skip("TensorSpec.tf() is not installed.")

    tf_spec = spec.tf(include_batch=True)

    assert isinstance(tf_spec, tf.RaggedTensorSpec)
    assert tuple(tf_spec.shape.as_list()) == (None, 8)
    assert tf_spec.dtype == tf.float32
    assert tf_spec.ragged_rank == 1
    assert tf_spec.row_splits_dtype == tf.int64
