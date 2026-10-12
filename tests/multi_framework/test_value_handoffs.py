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
        tf = pytest.importorskip("tensorflow")
        with tf.device("/CPU:0"):
            return tf.convert_to_tensor(array)
    if backend == "jax":
        jax = pytest.importorskip("jax")
        return jax.device_put(np.ascontiguousarray(array), jax.devices("cpu")[0])
    # Torch cannot construct negative-stride tensors; use a transposed dense
    # tensor here while NumPy-source directions exercise negative strides.
    tensor = pytest.importorskip("torch").tensor(np.ascontiguousarray(array), device="cpu")
    return tensor.transpose(0, 1).transpose(0, 1) if tensor.ndim > 1 else tensor


def _array(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return value.numpy()
    if type(value).__module__.startswith(("jax", "jaxlib")):
        return np.asarray(value)
    return value.detach().cpu().numpy()


@pytest.mark.parametrize(
    ("source", "target"),
    (("numpy", "tf"), ("numpy", "torch"), ("tf", "numpy"),
     ("tf", "torch"), ("torch", "numpy"), ("torch", "tf"),
     ("numpy", "jax"), ("jax", "numpy"), ("tf", "jax"),
     ("jax", "tf"), ("torch", "jax"), ("jax", "torch")),
)
def test_dense_cpu_handoffs_preserve_values_dtype_shape_and_owned_storage(source, target, monkeypatch):
    """Every supported direct CPU direction copies a strided dense value exactly."""

    source_array = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1]
    source_array.setflags(write=False)
    spec = TensorSpec("float32", shape=(4,), batch=3, backend=source)
    edge = make_edge(spec, target)
    value = _value(source, source_array)
    import dryml.methods.conversion as conversion
    extractions = []
    original_host = conversion._host_copy
    monkeypatch.setattr(conversion, "_host_copy",
                        lambda leaf, backend: (extractions.append(leaf), original_host(leaf, backend))[1])
    result = convert(edge, value)
    assert len(extractions) == 1 and extractions[0] is value

    np.testing.assert_array_equal(_array(result), source_array)
    assert _array(result).dtype == np.dtype("float32")
    assert _array(result).shape == (3, 4)
    if target == "torch":
        assert result.device.type == "cpu"
    elif target == "tf":
        assert "CPU" in result.device
    elif target == "jax":
        assert len(result.devices()) == 1
        assert next(iter(result.devices())).platform == "cpu"
        assert result.is_fully_addressable
    source_array.setflags(write=True)
    source_array[:] = -1
    np.testing.assert_array_equal(_array(result), np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1])


def test_nested_handoff_preserves_container_types_and_short_dynamic_batch(monkeypatch):
    """Nested trees retain list/tuple/mapping shape with a short dynamic batch."""

    spec = {"x": TensorSpec("int32", shape=(2,), batch=Dynamic, backend="numpy"),
            "y": (TensorSpec("bool", shape=(), batch=Dynamic, backend="numpy"),
                  [TensorSpec("float32", shape=(), batch=Dynamic, backend="numpy")])}
    value = {"x": np.array([[2, 1]], dtype=np.int32)[:, ::-1],
             "y": (np.array([True]), [np.array([1.25], dtype=np.float32)])}
    value["x"].setflags(write=False)
    import dryml.methods.conversion as conversion
    copies = []
    original = conversion._host_copy

    def extract(leaf, backend):
        copies.append(id(leaf))
        return original(leaf, backend)

    monkeypatch.setattr(conversion, "_host_copy", extract)
    result = convert(make_edge(spec, "torch"), value)
    assert copies == [id(value["x"]), id(value["y"][0]), id(value["y"][1][0])]

    assert isinstance(result, dict) and isinstance(result["y"], tuple) and isinstance(result["y"][1], list)
    np.testing.assert_array_equal(_array(result["x"]), value["x"])
    np.testing.assert_array_equal(_array(result["y"][0]), value["y"][0])
    np.testing.assert_array_equal(_array(result["y"][1][0]), value["y"][1][0])
    value["x"].setflags(write=True)
    value["x"][:] = -1
    np.testing.assert_array_equal(_array(result["x"]), [[1, 2]])


@pytest.mark.parametrize(
    ("source", "target"),
    (("numpy", "jax"), ("jax", "numpy"), ("tf", "jax"),
     ("jax", "tf"), ("torch", "jax"), ("jax", "torch")),
)
def test_nested_jax_handoffs_preserve_short_dynamic_batches(source, target):
    """Every JAX edge retains nested structure and one-element dynamic batches."""

    spec = {
        "x": TensorSpec("int32", shape=(2,), batch=Dynamic, backend=source),
        "y": (TensorSpec("bool", shape=(), batch=Dynamic, backend=source),),
    }
    value = {
        "x": _value(source, np.array([[1, 2]], dtype=np.int32)),
        "y": (_value(source, np.array([True])),),
    }

    result = convert(make_edge(spec, target), value)

    assert isinstance(result, dict) and isinstance(result["y"], tuple)
    np.testing.assert_array_equal(_array(result["x"]), [[1, 2]])
    np.testing.assert_array_equal(_array(result["y"][0]), [True])


def test_torch_requires_grad_and_object_values_fail_before_crossing(monkeypatch):
    """Gradient-bearing and lossy values reject at the adapter boundary."""

    torch = pytest.importorskip("torch")
    spec = TensorSpec("float32", shape=(2,), backend="torch")
    monkeypatch.setattr(torch.Tensor, "numpy", lambda self: pytest.fail("gradient host extraction"))
    with pytest.raises(RuntimeError, match="gradients"):
        convert(make_edge(spec, "numpy"), torch.ones(2, requires_grad=True))

    object_spec = TensorSpec("object", shape=(1,), backend="numpy")
    with pytest.raises(TypeError):
        make_edge(object_spec, "torch")

    for dtype in ("uint16", "uint32", "uint64"):
        with pytest.raises(TypeError):
            make_edge(TensorSpec(dtype, shape=(1,), backend="numpy"), "torch")


def test_jax_narrowing_rejects_before_target_construction(monkeypatch):
    """A runtime x64-disabled target cannot silently narrow an admitted host leaf."""
    jax = pytest.importorskip("jax")
    monkeypatch.setattr(jax, "device_put", lambda *a, **k: pytest.fail("narrowed target construction"))
    with jax.experimental.enable_x64(False):
        with pytest.raises(TypeError, match="cannot represent this dtype exactly"):
            convert(make_edge(TensorSpec("float64", shape=(2,), backend="numpy"), "jax"),
                    np.ones(2, dtype=np.float64))


def test_unknown_runtime_leaf_rejects_before_extraction_or_target_activation(monkeypatch):
    """A partially unsupported runtime tree is a typed backend conflict."""
    import dryml.methods.conversion as conversion
    spec = TensorSpec("float32", shape=(2,), backend="numpy")
    monkeypatch.setattr(conversion, "_host_copy", lambda *a: pytest.fail("unsupported host extraction"))
    monkeypatch.setattr(conversion, "_target_copy", lambda *a: pytest.fail("unsupported target construction"))
    with pytest.raises(TypeError, match="backend"):
        convert(make_edge((spec, spec), "torch"), (np.ones(2, dtype=np.float32), object()))


def test_nested_gradient_rejection_preflights_all_leaves_before_extraction(monkeypatch):
    """A later rejected leaf must not cause extraction of an earlier valid leaf."""
    torch = pytest.importorskip("torch")
    import dryml.methods.conversion as conversion
    spec = TensorSpec("float32", shape=(2,), backend="torch")
    value = (torch.ones(2, device="cpu"), torch.ones(2, device="cpu", requires_grad=True))
    extracted = []
    original = torch.Tensor.numpy

    def extract(leaf):
        extracted.append(id(leaf))
        return original(leaf)

    monkeypatch.setattr(torch.Tensor, "numpy", extract)
    monkeypatch.setattr(conversion, "_target_copy", lambda *a: pytest.fail("rejected target construction"))
    with pytest.raises(RuntimeError, match="gradients"):
        convert(make_edge((spec, spec), "numpy"), value)
    assert extracted == []


def test_raw_validation_precedes_callback_and_never_replaces_selected_inference():
    """Inferred native output and explicit raw validation remain independent."""
    from dryml.methods import Method, traits, ImplementationSelectionError
    pytest.importorskip("torch")

    class Target(Method):
        @traits(backend="torch")
        def torch(self, value):
            return value

        def infer_output_spec(self, spec):
            return spec

    producer = TensorSpec("float32", shape=(2,), backend="numpy")
    selected = Target().find_implementation(
        producer, output_spec=TensorSpec("float32", shape=(3,), backend="torch"),
    )
    assert selected.infer_output_spec().equal_exact(
        TensorSpec("float32", shape=(2,), backend="torch"),
    )
    callbacks = []
    with pytest.raises(ImplementationSelectionError, match="conflict"):
        selected.invoke_with_raw_result((np.ones(2, dtype=np.float32),), {}, callbacks.append)
    assert callbacks == []
