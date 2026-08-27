import numpy as np
import pytest

from dryml.core import Repo
from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArrayDataset
from dryml.models import Experiment
from dryml.models.sklearn import BasicTraining, RegressionModel


linear_model = pytest.importorskip("sklearn.linear_model")


def _train_data():
    x = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return ArrayDataset((x, y))


def test_basic_sklearn_training_updates_model_and_experiment_state():
    model = RegressionModel(linear_model.LinearRegression)
    exp = Experiment(model, BasicTraining(), train_data=_train_data())

    result = exp.train()

    assert model.obj is model.estimator
    assert result is model.estimator
    assert exp.state.epoch == 1
    assert exp.state.step == 3
    assert exp.state.phase == "trained"
    assert model.infer_output_spec(TensorSpec("float32", shape=(2,), backend="numpy")) == TensorSpec(
        "float32",
        shape=(),
        backend="numpy",
    )
    np.testing.assert_allclose(model(np.array([[3.0, 13.0]], dtype=np.float32)), np.array([4.0]), atol=1e-6)


def test_experiment_save_load_restores_train_state_and_model(tmp_path):
    model = RegressionModel(linear_model.LinearRegression)
    exp = Experiment(model, BasicTraining(), train_data=_train_data())
    exp.train()

    repo = Repo(stores=tmp_path)
    repo.save_object(exp, alias="exp")
    repo.close(flush=True)

    loaded = Repo(stores=tmp_path).load_alias("exp")

    assert loaded.state.epoch == 1
    assert loaded.state.step == 3
    assert loaded.state.phase == "trained"
    np.testing.assert_allclose(loaded.model(np.array([[3.0, 13.0]], dtype=np.float32)), np.array([4.0]), atol=1e-6)
