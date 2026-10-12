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


def test_selected_outputs_use_consumer_inputs_independently_of_raw_constraint():
    """Graph outputs describe selected inference, never the validation constraint."""
    from dryml.methods.signature import spec_from_node

    producer = TensorSpec("float32", shape=(2,), backend="numpy")
    raw = TensorSpec("float32", shape=(3,), backend="torch")
    method = TorchOnly()
    selected = method.find_implementation(producer, output_spec=raw)
    expected = TensorSpec("float32", shape=(2,), backend="torch")
    assert selected.infer_output_spec().equal_exact(expected)
    method.learn(producer, output_spec=raw)
    node = next(node for node in method.method_graph().nodes if node.selected is not None)
    assert spec_from_node(node.outputs[0].spec).equal_exact(expected)
    assert spec_from_node(node.selected.output_spec) == raw


def test_converted_pipe_project_and_map_share_selected_output_facts():
    """A shape-changing converted child drives every successor and graph port."""
    from dryml.data import Cast, Dataset, Map, Pipe, Project, Select
    from dryml.methods.signature import spec_from_node

    class ReshapeTorch(TorchOnly):
        @traits(backend="torch")
        def torch(self, value):
            return value.reshape(4)

        def infer_output_spec(self, input_spec):
            return input_spec.with_shape((4,))

    class Source(Dataset):
        def __iter__(self):
            return iter((np.arange(4, dtype=np.float32).reshape(2, 2),))

    source = Source(spec=TensorSpec("float32", shape=(2, 2), backend="numpy"))
    mapped = Map(source, Project({"x": [Pipe(ReshapeTorch(), Cast("float64"))]}))
    expected = {"x": [TensorSpec("float64", shape=(4,), backend="torch")]}
    assert mapped.spec == expected
    assert mapped.spec["x"][0].backend.value == "torch"
    successor = Map(mapped, Project(Pipe(Select("x", 0), Cast("float32"))))
    assert successor.spec == TensorSpec("float32", shape=(4,), backend="torch")
    assert successor.spec.backend.value == "torch"
    graph = successor.method_graph()
    graph.learn()
    assert spec_from_node(graph.nodes[-1].outputs[0].spec) == successor.spec
    assert spec_from_node(graph.nodes[-1].outputs[0].spec).backend.value == "torch"
    pytest.importorskip("torch")
    assert next(iter(successor)).tolist() == [0, 1, 2, 3]
    with graph.iterator() as cursor:
        assert next(cursor).tolist() == [0, 1, 2, 3]


def test_direct_candidate_wins_over_a_supported_converted_candidate():
    """A compatible direct target is selected before any adapter is considered."""

    selected = NumpyOrTorch().find_implementation(
        TensorSpec("float32", shape=(2,), backend="numpy")
    )

    assert selected.conversion_edge is None
    assert selected(np.ones(2, dtype=np.float32)) == "direct"


def test_missing_runtime_dependency_uses_backend_context_error(monkeypatch):
    """An absent registry entry activates the plugin and reports unavailable input."""
    from dryml.context import ContextError
    from dryml.core.backend import Backend, backend_existence_testers
    import importlib.util

    monkeypatch.delitem(backend_existence_testers, Backend.torch, raising=False)
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *args: None if name == "torch" else original(name, *args))
    # Import the lightweight plugin before simulating its dependency availability.
    import dryml.torch
    monkeypatch.setitem(backend_existence_testers, Backend.torch, lambda: False)
    with pytest.raises(ContextError, match="not available"):
        Backend.torch.module


def test_autoencoder_completes_outer_backend_hint_before_native_only_children():
    """Backend-only uncertainty defers while an admitted hint plans one outer edge."""
    from dryml.models import AutoEncoder, Model

    class NativeIdentity(Model):
        native_backend = "torch"

        @traits(backend="torch")
        def torch(self, value):
            return value

        def infer_output_spec(self, spec):
            return spec

    method = AutoEncoder(NativeIdentity(), NativeIdentity())
    unknown = TensorSpec("float32", shape=(2,), backend=None)
    with pytest.raises(ImplementationSelectionError) as error:
        method.find_implementation(unknown)
    assert error.value.unknown_traits == ("backend",)
    selected = method.find_implementation(unknown, backend="numpy")
    assert selected.conversion_edge.adapter == "numpy_to_torch"
    assert selected.infer_output_spec().backend.value == "torch"
    assert all(child.conversion_edge is None for child in selected._children)
    assert unknown.backend is None


def test_fold_roles_infer_selected_consumer_outputs_before_invocation():
    """A converted initializer's inferred carry remains native through followups."""
    from dryml.artifacts import Fold
    from dryml.data import ArrayDataset
    from dryml.methods import Accumulator
    TorchOnly.calls = 0

    class PreserveCarry(Accumulator):
        def __call__(self, observation, state):
            return state

        def infer_output_spec(self, observation, state):
            return state

    producer = TensorSpec("float32", shape=(2,), backend="numpy")
    fold = Fold(ArrayDataset(np.ones((1, 2), dtype=np.float32)),
                initial_state=TorchOnly(), accumulator=PreserveCarry(), finalize=TorchOnly())
    carry, initializer = fold._select_initializer(producer)
    assert carry.backend.value == "torch"
    assert initializer.conversion_edge.adapter == "numpy_to_torch"
    with pytest.raises(ImplementationSelectionError, match="conflict"):
        fold._select_followups(producer, carry)
    transition, finalizer = fold._select_followups(initializer.conversion_edge.consumer_spec, carry)
    assert transition.conversion_edge is None
    assert finalizer.conversion_edge is None
    assert TorchOnly.calls == 0


def test_autoencoder_explicit_output_does_not_require_decoder_inference():
    """Declared composite output remains usable with an opaque decoder body."""
    from dryml.models import AutoEncoder, Model
    from dryml.methods.signature import spec_from_node

    class Encoder(Model):
        def __call__(self, value):
            return value

        def infer_output_spec(self, spec):
            return spec

    class Decoder(Model):
        def __call__(self, value):
            return value

    expected = TensorSpec("float32", shape=(2,), backend="numpy")
    model = AutoEncoder(Encoder(), Decoder(), output_spec=expected)
    selected = model.find_implementation(expected)
    assert selected.infer_output_spec().equal_exact(expected)
    model.learn(expected)
    assert spec_from_node(model.method_graph().method_nodes[0].outputs[0].spec).equal_exact(expected)


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
    with pytest.raises(TypeError, match="dense spec tree"):
        selected(jax.ShapeDtypeStruct((2,), jnp.float32))
    assert TorchTarget.calls == 0

    @jax.jit
    def traced(value):
        return selected(value)

    with pytest.raises(TypeError, match="dense spec tree"):
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
