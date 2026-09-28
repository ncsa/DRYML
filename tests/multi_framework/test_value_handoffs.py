"""Tiny installed-backend value handoff contracts with no network fixtures."""

import numpy as np
import pytest

from dryml.core import TensorSpec
from dryml.core.tensor_spec import Dynamic
from dryml.methods.conversion import convert, make_edge


def _value(backend, array):
    if backend == "numpy":
        return array
    if backend == "tf":
        return pytest.importorskip("tensorflow").convert_to_tensor(array)
    # Torch cannot construct negative-stride tensors; use a transposed dense
    # tensor here while NumPy-source directions exercise negative strides.
    return pytest.importorskip("torch").tensor(np.ascontiguousarray(array)).transpose(0, 1).transpose(0, 1)


def _array(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return value.numpy()
    return value.detach().cpu().numpy()


@pytest.mark.parametrize(
    ("source", "target"),
    (("numpy", "tf"), ("numpy", "torch"), ("tf", "numpy"),
     ("tf", "torch"), ("torch", "numpy"), ("torch", "tf")),
)
def test_dense_cpu_handoffs_preserve_values_dtype_shape_and_owned_storage(source, target):
    """Every supported direct CPU direction copies a strided dense value exactly."""

    source_array = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1]
    source_array.setflags(write=False)
    spec = TensorSpec("float32", shape=(4,), batch=3, backend=source)
    edge = make_edge(spec, target)
    result = convert(edge, _value(source, source_array))

    np.testing.assert_array_equal(_array(result), source_array)
    assert _array(result).dtype == np.dtype("float32")
    assert _array(result).shape == (3, 4)
    if target == "torch":
        assert result.device.type == "cpu"
    elif target == "tf":
        assert "CPU" in result.device
    source_array.setflags(write=True)
    source_array[:] = -1
    np.testing.assert_array_equal(_array(result), np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1])


def test_nested_handoff_preserves_container_types_and_short_dynamic_batch():
    """Nested trees retain list/tuple/mapping shape with a short dynamic batch."""

    spec = {"x": TensorSpec("int32", shape=(2,), batch=Dynamic, backend="numpy"),
            "y": (TensorSpec("bool", shape=(), batch=Dynamic, backend="numpy"),)}
    value = {"x": np.array([[1, 2]], dtype=np.int32), "y": (np.array([True]),)}
    result = convert(make_edge(spec, "torch"), value)

    assert isinstance(result, dict) and isinstance(result["y"], tuple)
    np.testing.assert_array_equal(_array(result["x"]), value["x"])
    np.testing.assert_array_equal(_array(result["y"][0]), value["y"][0])


def test_torch_requires_grad_and_object_values_fail_before_crossing():
    """Gradient-bearing and lossy values reject at the adapter boundary."""

    torch = pytest.importorskip("torch")
    spec = TensorSpec("float32", shape=(2,), backend="torch")
    with pytest.raises(RuntimeError, match="gradients"):
        convert(make_edge(spec, "numpy"), torch.ones(2, requires_grad=True))

    object_spec = TensorSpec("object", shape=(1,), backend="numpy")
    with pytest.raises(TypeError):
        make_edge(object_spec, "torch")

    for dtype in ("uint16", "uint32", "uint64"):
        with pytest.raises(TypeError):
            make_edge(TensorSpec(dtype, shape=(1,), backend="numpy"), "torch")
