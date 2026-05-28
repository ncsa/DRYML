import numpy as np

import dryml.numpy
from dryml.core2.tensor_spec import TensorSpec
from dryml.data import ArrayDataset
from dryml.metrics import categorical_accuracy, mean_squared_error
from dryml.models import Model


class AddOne(Model):
    def __init__(self):
        self.output_spec = TensorSpec("float32", shape=(), backend="numpy")

    def __call__(self, x):
        return x + 1


class ParityClassifier(Model):
    def __init__(self):
        self.output_spec = TensorSpec("float32", shape=(2,), backend="numpy")

    def __call__(self, x):
        labels = np.asarray(x, dtype=np.int64) % 2
        return np.eye(2, dtype=np.float32)[labels]


def test_mean_squared_error_uses_explicit_selectors_and_batching():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    y = np.array([1.0, 3.0, 3.0], dtype=np.float32)
    ds = ArrayDataset((x, y))

    assert mean_squared_error(AddOne(), ds, batch_size=2) == 1.0 / 3.0


def test_categorical_accuracy_uses_map_pack_predictions():
    x = np.array([0, 1, 2, 3], dtype=np.int64)
    y = np.array([0, 1, 0, 0], dtype=np.int64)
    ds = ArrayDataset((x, y))

    assert categorical_accuracy(ParityClassifier(), ds, batch_size=2) == 0.75
