import numpy as np
import pytest

from dryml.core2 import Repo
from dryml.core2.store.dir import DirStore
from dryml.core2.tensor_spec import TensorSpec
from dryml.artifacts import CachedDataset
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


def test_managed_experiment_reopens_definition_and_hydrates_trained_model(tmp_path):
    store = DirStore(tmp_path / "store")
    model = RegressionModel(linear_model.LinearRegression)
    cached = CachedDataset(_train_data())
    cached.compute(store=store, representation="numpy-sequence")
    exp = Experiment(model, BasicTraining(), train_data=cached)
    result = exp.train(store=store)

    Repo(store).save_definition(exp.definition)
    reopened_store = DirStore(store.base_dir)
    loaded = Repo(reopened_store).load(exp.definition, restore_state=False)
    trained = loaded.trained_model(store=reopened_store)

    assert result.action == "start"
    assert loaded.train.status(store=reopened_store).status == "completed"
    np.testing.assert_allclose(
        trained(np.array([[3.0, 13.0]], dtype=np.float32)),
        np.array([4.0]),
        atol=1e-6,
    )
