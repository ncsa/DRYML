import numpy as np
import pytest

from dryml.data import ArrayDataset
from dryml.models import Experiment


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
    train_fn = BasicTraining(optimizer=optimizer, loss=loss, epochs=1, batch_size=2)
    exp = Experiment(model, train_fn, train_data=ds)

    history = exp.train()

    assert model.obj is model.mdl
    assert optimizer.obj is not None
    assert history is not None
    assert exp.state.epoch == 1
    assert exp.state.step == 2
    assert exp.state.phase == "trained"
    assert float(optimizer.obj.learning_rate.numpy()) == pytest.approx(0.01)
