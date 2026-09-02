import numpy as np

from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArrayDataset, Batch, collate_xy
from dryml.models import AutoEncoder, Model
from dryml.models.utils import prepare_training_data


class CountingIdentityModel(Model):
    def __init__(self):
        self.selections = 0

    def __call__(self, value):
        return value

    def infer_output_spec(self, input_spec):
        return input_spec

    def find_implementation(self, *args, **kwargs):
        self.selections += 1
        return super().find_implementation(*args, **kwargs)


class BackendChangingModel(Model):
    def __init__(self, backend):
        self.backend = backend
        super().__init__()

    def __call__(self, value):
        return TensorSpec(value.dtype, shape=value.shape, backend=self.backend)

    def infer_output_spec(self, input_spec):
        return TensorSpec(input_spec.dtype, shape=input_spec.shape, backend=self.backend)


def test_prepare_training_data_unbatches_and_takes_examples():
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    ds = Batch(ArrayDataset((x, y)), 2)

    train_data = prepare_training_data(ds, num_examples=2)
    out_x, out_y, n = collate_xy(train_data)

    assert n == 2
    assert out_x.shape == (2, 1)
    assert out_y.shape == (2, 1)
    np.testing.assert_allclose(out_x[:, 0], np.array([0.0, 1.0], dtype=np.float32))


def test_backend_model_packages_import_without_backend_runtime():
    import dryml.models.tf as tf_models
    import dryml.models.torch as torch_models

    assert hasattr(tf_models, "BasicTraining")
    assert hasattr(tf_models, "Training")
    assert hasattr(tf_models, "Optimizer")
    assert hasattr(tf_models, "Wrapper")
    assert hasattr(torch_models, "Training")
    assert not hasattr(torch_models, "BasicTraining")
    assert hasattr(torch_models, "Optimizer")
    assert hasattr(torch_models, "Wrapper")


def test_tf_keras_package_only_exports_keras_specific_symbols():
    import dryml.models.tf.keras as keras_models

    assert hasattr(keras_models, "Sequential")
    assert not hasattr(keras_models, "BasicTraining")
    assert not hasattr(keras_models, "Optimizer")


def test_autoencoder_selected_invoker_selects_children_once_without_cache_mutation():
    encoder = CountingIdentityModel()
    decoder = CountingIdentityModel()
    model = AutoEncoder(encoder, decoder)
    spec = TensorSpec("float32", shape=(2,), backend="numpy")

    selected = model.find_implementation(input_spec=spec)
    assert selected(np.ones((2,), dtype=np.float32)).shape == (2,)

    assert encoder.selections == 1
    assert decoder.selections == 1
    assert encoder.call_mode == decoder.call_mode == "eager"


def test_autoencoder_derives_decoder_backend_from_encoded_spec():
    model = AutoEncoder(BackendChangingModel("torch"), BackendChangingModel("tf"))
    source = TensorSpec("float32", shape=(2,), backend="numpy")

    model.learn()
    assert model(source).backend.value == "tf"
    assert model.call_mode == "cached"
