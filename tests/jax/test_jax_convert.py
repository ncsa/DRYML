import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
import dryml.jax as dryml_jax

from dryml.core2.dtype import DType
from dryml.core2.tensor_spec import Dynamic, Layout, TensorSpec


def test_jax_dtype_from_dtype_object():
    assert dryml_jax.dtype(jnp.float32) == DType("float", 32)
    assert dryml_jax.dtype(jnp.int64) == DType("int", 64)
    assert dryml_jax.dtype(jnp.bool_) == DType("bool")


def test_jax_dtype_from_shape_dtype_struct():
    x = jax.ShapeDtypeStruct((4, 32), jnp.float32)
    assert dryml_jax.dtype(x) == DType("float", 32)


def test_jax_tensor_spec_from_shape_dtype_struct_unbatched():
    x = jax.ShapeDtypeStruct((4, 32), jnp.float32)

    spec = dryml_jax.tensor_spec(x, assume_batched=False)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (4, 32)
    assert spec.batch is None
    assert spec.layout is Layout.DENSE


def test_jax_tensor_spec_from_shape_dtype_struct_assume_batched():
    x = jax.ShapeDtypeStruct((4, 32), jnp.float32)

    spec = dryml_jax.tensor_spec(x, assume_batched=True)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (32,)
    assert spec.batch == 4
    assert spec.layout is Layout.DENSE
    assert spec.batch_axis_name == "batch"


def test_jax_tensor_spec_from_array():
    x = jnp.zeros((3, 16), dtype=jnp.float32)

    spec = dryml_jax.tensor_spec(x, assume_batched=True)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (16,)
    assert spec.batch == 3
    assert spec.layout is Layout.DENSE


def test_jax_roundtrip_dense_if_forward_methods_installed():
    spec = TensorSpec(dtype="float32", shape=(32,), batch=8)

    if not hasattr(spec, "jax"):
        pytest.skip("TensorSpec.jax() is not installed.")

    jax_spec = spec.jax()

    assert isinstance(jax_spec, jax.ShapeDtypeStruct)
    assert jax_spec.shape == (8, 32)
    assert jax_spec.dtype == jnp.dtype("float32")


def test_jax_dynamic_dim_rejected_if_forward_methods_installed():
    spec = TensorSpec(dtype="float32", shape=(Dynamic, 32))

    if not hasattr(spec, "jax"):
        pytest.skip("TensorSpec.jax() is not installed.")

    with pytest.raises(ValueError):
        spec.jax()
