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


def _spec_leaves(tree):
    if isinstance(tree, TensorSpec):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _spec_leaves(value)
    else:
        for value in tree:
            yield from _spec_leaves(value)
