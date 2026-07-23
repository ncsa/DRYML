import pytest

import dryml.runtime as runtime
from dryml.core.backend import discover_backend
import numpy as np

tf = pytest.importorskip("tensorflow")
import dryml.tf as dryml_tf

from dryml.core.dtype import DType
from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec, as_tensor_spec
from dryml.core.backend import discover_backend


def _runtime_bootstrap(framework_name):
    spec = runtime.RuntimeContextSpec.from_data({"mode": "probe", "frameworks": {framework_name: {}}, "device_visibility": {"policy": "none"}})
    plan = runtime.build_runtime_bootstrap_plan(spec, runtime.NoAllocation)
    return runtime.enter_runtime(runtime.RuntimeMode.PROBE, runtime.NoAllocation, spec), runtime.activate_runtime_bootstrap(plan)


def test_tf_dtype_from_dtype_object():
    assert dryml_tf.dtype(tf.float32) == DType("float", 32)
    assert dryml_tf.dtype(tf.int64) == DType("int", 64)
    assert dryml_tf.dtype(tf.bool) == DType("bool")


def test_tf_dtype_from_spec():
    x = tf.TensorSpec(shape=(None, 32), dtype=tf.float32)
    assert dryml_tf.dtype(x) == DType("float", 32)


def test_tf_tensor_spec_dense_unbatched():
    x = tf.TensorSpec(shape=(None, 32), dtype=tf.float32)

    spec = dryml_tf.as_tensor_spec(x, batched=False)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (Dynamic, 32)
    assert spec.batch is None
    assert spec.layout is Layout.DENSE


def test_tf_tensor_spec_dense_batched():
    x = tf.TensorSpec(shape=(None, 32), dtype=tf.float32)

    spec = dryml_tf.as_tensor_spec(x, batched=True)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (32,)
    assert spec.batch is Dynamic
    assert spec.layout is Layout.DENSE
    assert spec.batch_axis_name == "batch"


def test_tf_tensor_spec_from_value():
    x = tf.zeros((4, 32), dtype=tf.float32)

    spec = dryml_tf.as_tensor_spec(x, batched=True)

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

    spec = dryml_tf.as_tensor_spec(x, batched=False)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (Dynamic, Dynamic, 8)
    assert spec.batch is None
    assert spec.layout is Layout.RAGGED
    assert spec.ragged_rank == 1
    assert spec.row_splits_dtype == DType("int", 64)


def test_tf_tensor_spec_sparse():
    x = tf.SparseTensorSpec(shape=(None, 16), dtype=tf.float32)

    spec = dryml_tf.as_tensor_spec(x, batched=False)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (Dynamic, 16)
    assert spec.batch is None
    assert spec.layout is Layout.SPARSE
    assert spec.sparse_format == "tf_sparse"


def test_tf_roundtrip_dense_if_forward_methods_installed():
    spec = TensorSpec(dtype="float32", shape=(32,), batch=Dynamic)

    if not hasattr(spec, "tf"):
        pytest.skip("TensorSpec.tf() is not installed.")

    runtime_scope, bootstrap_scope = _runtime_bootstrap("tensorflow")
    with runtime_scope:
        with bootstrap_scope:
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

    runtime_scope, bootstrap_scope = _runtime_bootstrap("tensorflow")
    with runtime_scope:
        with bootstrap_scope:
            tf_spec = spec.tf(include_batch=True)

    assert isinstance(tf_spec, tf.RaggedTensorSpec)
    assert tuple(tf_spec.shape.as_list()) == (None, 8)
    assert tf_spec.dtype == tf.float32
    assert tf_spec.ragged_rank == 1
    assert tf_spec.row_splits_dtype == tf.int64


def test_tf_backend_detectors():
    assert discover_backend(tf.constant(1)) == "tf"
    assert discover_backend(tf.constant(1.5, dtype=tf.float32)) == "tf"
    assert discover_backend(np.uint8(1)) == "numpy"
    assert discover_backend(np.float64(1.5)) == "numpy"


def test_tf_tensor_spec_auto_ingest():
    x = tf.TensorSpec((4, 32), dtype=tf.float32)
    spec = TensorSpec(dtype="float32", shape=(32,), batch=4)
    assert spec == as_tensor_spec(x, batched=True)

    x = tf.random.uniform(shape=(4, 32), dtype=tf.float32)
    spec = TensorSpec(dtype="float32", shape=(4, 32,))
    assert spec == as_tensor_spec(x)
