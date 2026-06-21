import warnings

import numpy as np
import pytest

from dryml.core2.tensor_spec import TensorSpec
from dryml.data import ArrayDataset, Map
from dryml.models import AutoEncoder, Experiment


tf = pytest.importorskip("tensorflow")


class TinyKerasModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, x):
        return self.dense(x)


def test_tf_basic_training_updates_experiment_state():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    ds = ArrayDataset((x, y))

    model = Model(TinyKerasModel)
    optimizer = Optimizer(tf.keras.optimizers.SGD, learning_rate=0.01)
    loss = Loss(tf.keras.losses.MeanSquaredError)
    train_fn = BasicTraining(optimizer=optimizer, loss=loss, epochs=1, batch_size=2, verbose=0)
    exp = Experiment(model, train_fn, train_data=ds)

    history = exp.train()

    assert model.obj is model.mdl
    assert optimizer.obj is not None
    assert history is not None
    assert exp.state.epoch == 1
    assert exp.state.step == 2
    assert exp.state.phase == "trained"
    assert float(optimizer.obj.learning_rate.numpy()) == pytest.approx(0.01)


def test_tf_sequential_infers_output_spec_without_explicit_output_spec():
    from dryml.models.tf import Sequential

    x = np.zeros((4, 3), dtype=np.float32)
    ds = ArrayDataset(x)
    model = Sequential(layer_defs=(("Dense", {"units": 2}),))

    assert Map(ds, model).spec == TensorSpec("float32", shape=(2,), backend="tf")


def _autoencoder_data():
    x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return ArrayDataset((x, x.copy()))


def _autoencoder_model():
    from dryml.models.tf import Sequential

    encoder = Sequential(
        layer_defs=(
            ("Dense", {"units": 8, "activation": "relu"}),
            ("Dense", {"units": 2, "activation": "linear"}),
        )
    )
    decoder = Sequential(
        layer_defs=(
            ("Dense", {"units": 8, "activation": "relu"}),
            ("Dense", {"units": 3, "activation": "linear"}),
        )
    )
    return AutoEncoder(encoder=encoder, decoder=decoder)


def test_tf_basic_training_builds_keras_adapter_for_autoencoder():
    from dryml.models.tf import BasicTraining, Wrapper

    model = _autoencoder_model()
    train_fn = BasicTraining(epochs=1, batch_size=2, verbose=0)
    exp = Experiment(
        model,
        train_fn,
        train_data=_autoencoder_data(),
        optimizer=Wrapper(tf.keras.optimizers.SGD, learning_rate=0.01),
        loss=Wrapper(tf.keras.losses.MeanSquaredError),
    )

    history = exp.train()

    assert history is not None
    assert exp.state.epoch == 1
    assert exp.state.step == 2
    assert model.encoder.obj.trainable_variables
    assert model.decoder.obj.trainable_variables


def test_tf_basic_training_repeats_finite_dataset_for_multiple_epochs():
    from dryml.models.tf import BasicTraining, Wrapper

    model = _autoencoder_model()
    train_fn = BasicTraining(epochs=2, batch_size=2, verbose=0)
    exp = Experiment(
        model,
        train_fn,
        train_data=_autoencoder_data(),
        optimizer=Wrapper(tf.keras.optimizers.SGD, learning_rate=0.01),
        loss=Wrapper(tf.keras.losses.MeanSquaredError),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        history = exp.train()

    assert len(history.epoch) == 2
    assert exp.state.epoch == 2
    assert exp.state.step == 4
    assert not any("Your input ran out of data" in str(warning.message) for warning in caught)


def test_tf_training_gradient_tape_trains_autoencoder():
    from dryml.models.tf import Loss, Optimizer, Training

    model = _autoencoder_model()
    train_fn = Training(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.01),
        loss=Loss(tf.keras.losses.MeanSquaredError),
        epochs=1,
        batch_size=2,
        verbose=0,
    )
    exp = Experiment(model, train_fn, train_data=_autoencoder_data())

    losses = exp.train()

    assert len(losses) == 2
    assert exp.state.epoch == 1
    assert exp.state.step == 2
    assert model.encoder.obj.trainable_variables
    assert model.decoder.obj.trainable_variables
