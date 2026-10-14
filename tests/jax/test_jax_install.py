import os
import subprocess
import sys

import pytest

jax = pytest.importorskip("jax")
if not hasattr(jax, "ShapeDtypeStruct"):
    sys.modules.pop("jax", None)
    pytest.skip("JAX is not installed.", allow_module_level=True)
jnp = pytest.importorskip("jax.numpy")
import dryml.jax as dryml_jax
import numpy as np

from dryml.core.dtype import DType
from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec, as_tensor_spec
from dryml.core.backend import discover_backend


def test_jax_dtype_from_dtype_object():
    assert dryml_jax.dtype(jnp.float32) == DType("float", 32)
    assert dryml_jax.dtype(jnp.int64) == DType("int", 64)
    assert dryml_jax.dtype(jnp.bool_) == DType("bool")


def test_jax_dtype_from_shape_dtype_struct():
    x = jax.ShapeDtypeStruct((4, 32), jnp.float32)
    assert dryml_jax.dtype(x) == DType("float", 32)


def test_jax_tensor_spec_from_shape_dtype_struct_unbatched():
    x = jax.ShapeDtypeStruct((4, 32), jnp.float32)

    spec = dryml_jax.as_tensor_spec(x, batched=False)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (4, 32)
    assert spec.batch is None
    assert spec.layout is Layout.DENSE


def test_jax_tensor_spec_from_shape_dtype_struct_batched():
    x = jax.ShapeDtypeStruct((4, 32), jnp.float32)

    spec = dryml_jax.as_tensor_spec(x, batched=True)

    assert spec.dtype == DType("float", 32)
    assert spec.shape == (32,)
    assert spec.batch == 4
    assert spec.layout is Layout.DENSE
    assert spec.batch_axis_name == "batch"


def test_jax_tensor_spec_from_array():
    x = jnp.zeros((3, 16), dtype=jnp.float32)

    spec = dryml_jax.as_tensor_spec(x, batched=True)

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


def test_jax_backend_detectors():
    assert discover_backend(jnp.array(1)) == "jax"
    assert discover_backend(jnp.float32(1.5)) == "jax"
    assert discover_backend(np.uint8(1)) == "numpy"
    assert discover_backend(np.float64(1.5)) == "numpy"


def test_jax_tensor_spec_auto_ingest():
    x = jax.ShapeDtypeStruct((4, 32), jnp.float32)
    spec = TensorSpec(dtype="float32", shape=(32,), batch=4)
    assert spec == as_tensor_spec(x, batched=True)

    key = jax.random.key(0)
    x = jax.random.uniform(key, shape=(4, 32), dtype=jnp.float32)
    spec = TensorSpec(dtype="float32", shape=(4, 32,))
    assert spec == as_tensor_spec(x)


@pytest.mark.parametrize(("x64_enabled", "expect_success"), (("0", False), ("1", True)))
def test_jax_handoffs_preserve_float64_exactly_without_mutating_global_configuration(
    x64_enabled, expect_success,
):
    """Fresh JAX processes either retain float64 exactly or reject its narrowing."""

    environment = os.environ | {
        "JAX_ENABLE_X64": x64_enabled,
        "JAX_PLATFORMS": "cpu",
    }
    script = """
import numpy as np
from dryml.core import TensorSpec
from dryml.methods.conversion import convert, make_edge

value = np.array([1.5], dtype=np.float64)
edge = make_edge(TensorSpec("float64", shape=(1,), backend="numpy"), "jax")
try:
    result = convert(edge, value)
except TypeError:
    assert not EXPECT_SUCCESS
else:
    assert EXPECT_SUCCESS
    assert np.asarray(result).dtype == np.dtype("float64")
""".replace("EXPECT_SUCCESS", repr(expect_success))

    completed = subprocess.run(
        [sys.executable, "-c", script], env=environment, text=True,
        capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_generic_conversion_import_does_not_import_jax_in_a_fresh_process():
    """Planning support keeps the optional JAX endpoint lazy until selected."""

    completed = subprocess.run(
        [sys.executable, "-c", "import sys; import dryml.methods.conversion; assert 'jax' not in sys.modules"],
        env=os.environ | {"JAX_PLATFORMS": "cpu"}, text=True, capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
