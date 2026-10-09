"""Focused local handoff-selection and graph-fact contracts."""

import numpy as np
import pytest

from dryml.core import TensorSpec
from dryml.methods import ImplementationSelectionError, Method, traits


class TorchOnly(Method):
    """Test target that declares a Torch-only data boundary."""

    calls = 0

    @traits(backend="torch")
    def torch(self, value):
        type(self).calls += 1
        return value

    def infer_output_spec(self, input_spec):
        return input_spec


class NumpyOrTorch(Method):
    """Test catalog proving direct candidates outrank conversion candidates."""

    @traits(backend="numpy")
    def numpy(self, value):
        return "direct"

    @traits(backend="torch")
    def torch(self, value):
        return "converted"

    def infer_output_spec(self, input_spec):
        return input_spec


class TwoConvertedTargets(Method):
    """Test catalog with incomparable converted alternatives."""

    @traits(backend="torch")
    def torch(self, value):
        return value

    @traits(backend="tf")
    def tf(self, value):
        return value

    def infer_output_spec(self, input_spec):
        return input_spec


def test_known_spec_handoff_records_one_immutable_graph_edge_without_body_execution():
    """Preparation selects a conversion once and exposes its precise graph fact."""

    method = TorchOnly()
    spec = TensorSpec("float32", shape=(2,), backend="numpy")
    TorchOnly.calls = 0

    method.learn(spec)

    edges = method.method_graph().conversion_edges
    assert len(edges) == 1
    assert edges[0].adapter == "numpy_to_torch"
    assert edges[0].producer_spec == spec
    assert next(iter(_spec_leaves(edges[0].consumer_spec))).backend.value == "torch"
    assert TorchOnly.calls == 0


def test_direct_candidate_wins_over_a_supported_converted_candidate():
    """A compatible direct target is selected before any adapter is considered."""

    selected = NumpyOrTorch().find_implementation(
        TensorSpec("float32", shape=(2,), backend="numpy")
    )

    assert selected.conversion_edge is None
    assert selected(np.ones(2, dtype=np.float32)) == "direct"


def test_equally_specific_converted_candidates_fail_before_target_execution():
    """Handoff planning never picks a framework preference for a conversion tie."""

    with pytest.raises(ImplementationSelectionError) as error:
        TwoConvertedTargets().find_implementation(
            TensorSpec("float32", shape=(2,), backend="numpy")
        )

    assert error.value.reason == "ambiguous"


def test_native_model_declares_a_handoff_edge_instead_of_using_its_eager_converter():
    """Prepared model selection exposes its Torch input boundary from a NumPy spec."""

    torch = pytest.importorskip("torch")
    from dryml.models.torch import Model

    model = Model(torch.nn.Identity, output_spec=TensorSpec("float32", shape=(2,)))
    selected = model.find_implementation(TensorSpec("float32", shape=(2,), backend="numpy"))

    assert selected.conversion_edge.adapter == "numpy_to_torch"


def test_autoencoder_rejects_mixed_native_components_and_keeps_torch_gradients():
    """Model composition is native-only while same-framework gradients remain connected."""

    torch = pytest.importorskip("torch")
    from dryml.models import AutoEncoder, Model as BaseModel
    from dryml.models.tf import Model as TFModel
    from dryml.models.torch import Model as TorchModel

    torch_model = TorchModel(torch.nn.Identity, output_spec=TensorSpec("float32", shape=(2,)))
    same = AutoEncoder(torch_model, TorchModel(torch.nn.Identity, output_spec=TensorSpec("float32", shape=(2,))))
    value = torch.ones(2, requires_grad=True)
    output = same.find_implementation(TensorSpec("float32", shape=(2,), backend="torch"))(value)
    output.sum().backward()
    assert value.grad is not None

    import tensorflow as tf

    mixed = AutoEncoder(torch_model, TFModel(tf.keras.layers.Activation, "linear",
                                              output_spec=TensorSpec("float32", shape=(2,))))
    with pytest.raises(ImplementationSelectionError) as error:
        mixed.find_implementation(TensorSpec("float32", shape=(2,), backend="torch"))
    assert error.value.reason == "conflict"

    class NumpyModel(BaseModel):
        @traits(backend="numpy")
        def numpy(self, value):
            return value

    hidden_bridge = AutoEncoder(
        NumpyModel(output_spec=TensorSpec("float32", shape=(2,), backend="numpy")),
        TorchModel(torch.nn.Identity, output_spec=TensorSpec("float32", shape=(2,))),
    )
    with pytest.raises(ImplementationSelectionError) as error:
        hidden_bridge.find_implementation(
            TensorSpec("float32", shape=(2,), backend="numpy")
        )
    assert error.value.reason == "conflict"


def test_tf_autoencoder_keeps_native_gradients_through_supported_composition():
    """Same-framework TensorFlow composition remains differentiable end to end."""
    tf = pytest.importorskip("tensorflow")
    import dryml.tf
    from dryml.models import AutoEncoder
    from dryml.models.tf import Model as TFModel

    same = AutoEncoder(
        TFModel(tf.keras.layers.Dense, 2, use_bias=False, output_spec=TensorSpec("float32", shape=(2,))),
        TFModel(tf.keras.layers.Dense, 1, use_bias=False, output_spec=TensorSpec("float32", shape=(1,))),
    )
    value = tf.constant([[1.0, 2.0]], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(value)
        result = same.find_implementation(TensorSpec("float32", shape=(2,), batch=1, backend="tf"))(value)
        loss = tf.reduce_sum(result)
    gradients = tape.gradient(loss, (*same.encoder.obj.trainable_variables, *same.decoder.obj.trainable_variables))
    assert all(gradient is not None for gradient in gradients)


@pytest.mark.parametrize("source", ("tf", "torch"))
def test_gpu_cross_framework_handoffs_reject_before_any_host_copy(monkeypatch, source):
    """GPU-like source tensors fail before detach, NumPy, DLPack, or host fallback."""
    import dryml.methods.conversion as conversion

    class Sentinel:
        __module__ = "tensorflow" if source == "tf" else "torch"
        layout = object()

        def __init__(self):
            self.device = "/device:GPU:0" if source == "tf" else type("Device", (), {"type": "cuda"})()

        def numpy(self):
            raise AssertionError("conversion attempted NumPy host copying")

        def detach(self):
            raise AssertionError("conversion attempted detaching")

        def cpu(self):
            raise AssertionError("conversion attempted CPU fallback")

    value = Sentinel()
    if source == "tf":
        fake_tf = type("TF", (), {"is_tensor": staticmethod(lambda candidate: candidate is value), "RaggedTensor": (), "SparseTensor": ()})()
        monkeypatch.setattr(conversion, "import_module", lambda _: fake_tf)
        edge = conversion.make_edge(TensorSpec("float32", shape=(2,), backend="tf"), "torch")
    else:
        Sentinel.layout = object()
        fake_torch = type("Torch", (), {"Tensor": Sentinel, "strided": Sentinel.layout})()
        monkeypatch.setattr(conversion, "import_module", lambda _: fake_torch)
        edge = conversion.make_edge(TensorSpec("float32", shape=(2,), backend="torch"), "tf")

    with pytest.raises(TypeError):
        conversion.convert(edge, value)


def test_jax_abstract_tracer_and_sparse_values_reject_before_target_execution():
    """JAX non-concrete values never reach a converted Torch target body."""

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import sparse

    class TorchTarget(Method):
        calls = 0

        @traits(backend="torch")
        def torch(self, value):
            type(self).calls += 1
            return value

        def infer_output_spec(self, input_spec):
            return input_spec

    selected = TorchTarget().find_implementation(
        TensorSpec("float32", shape=(2,), backend="jax")
    )
    TorchTarget.calls = 0
    with pytest.raises(ImplementationSelectionError, match="conflict"):
        selected(jax.ShapeDtypeStruct((2,), jnp.float32))
    assert TorchTarget.calls == 0

    @jax.jit
    def traced(value):
        return selected(value)

    with pytest.raises(ImplementationSelectionError, match="conflict"):
        traced(jnp.ones(2, dtype=jnp.float32))
    assert TorchTarget.calls == 0
    with pytest.raises((TypeError, ImplementationSelectionError)):
        selected(sparse.BCOO.fromdense(jnp.eye(2, dtype=jnp.float32)))
    assert TorchTarget.calls == 0


@pytest.mark.parametrize(
    ("addressable", "platforms"),
    ((False, ("cpu",)), (True, ("cpu", "cpu")), (True, ("cuda",))),
)
def test_jax_placement_rejection_happens_before_host_materialization(
    monkeypatch, addressable, platforms,
):
    """Non-addressable, multi-device, and non-CPU JAX sources never materialize."""

    import dryml.methods.conversion as conversion

    class SentinelArray:
        __module__ = "jax.fake"

        is_fully_addressable = addressable

        def devices(self):
            return {type("Device", (), {"platform": platform})() for platform in platforms}

        def __array__(self, dtype=None):
            raise AssertionError("conversion attempted host materialization")

    value = SentinelArray()
    fake_jax = type("Jax", (), {"Array": SentinelArray})()
    monkeypatch.setattr(conversion, "import_module", lambda _: fake_jax)
    edge = conversion.make_edge(TensorSpec("float32", shape=(2,), backend="jax"), "numpy")

    with pytest.raises(TypeError):
        conversion.convert(edge, value)

def _spec_leaves(tree):
    if isinstance(tree, TensorSpec):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _spec_leaves(value)
    else:
        for value in tree:
            yield from _spec_leaves(value)
