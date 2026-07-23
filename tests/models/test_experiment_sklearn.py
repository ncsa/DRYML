import numpy as np
import pytest

from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import TensorSpec
from dryml.artifacts import CachedDataset
from dryml.data import ArrayDataset
from dryml.managed import (
    ControlRequest,
    ManagedCallback,
    ManagedInterruptedError,
    ManagedRerunRequiredError,
)
from dryml.models import Experiment
from dryml.models import TrainResumeMode
from dryml.models.sklearn import BasicTraining, RegressionModel


linear_model = pytest.importorskip("sklearn.linear_model")


class CountingLinearRegression(linear_model.LinearRegression):
    fits = 0

    def fit(self, *args, **kwargs):
        type(self).fits += 1
        return super().fit(*args, **kwargs)


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
    reused = exp.train(store=store)

    Repo(store).save_definition(exp.definition)
    reopened_store = DirStore(store.base_dir)
    loaded = Repo(reopened_store).load(exp.definition, restore_state=False)
    trained = loaded.trained_model(store=reopened_store)

    assert result.action == "start"
    assert reused.action == "reuse"
    assert reused.realization_id == result.realization_id
    assert loaded.train.status(store=reopened_store).status == "completed"
    np.testing.assert_allclose(
        trained(np.array([[3.0, 13.0]], dtype=np.float32)),
        np.array([4.0]),
        atol=1e-6,
    )


def test_sklearn_opaque_fit_explicitly_declares_non_resumable():
    capability = BasicTraining.resume_capability(BasicTraining().definition)

    assert capability.mode is TrainResumeMode.NONE
    assert capability.checkpoint_schema is None
    assert "fit" in capability.diagnostic


def test_interrupted_sklearn_fit_never_silently_restarts_and_requires_rerun(
    tmp_path,
):
    store = DirStore(tmp_path / "store")
    cached = CachedDataset(_train_data())
    cached.compute(store=store, representation="numpy-sequence")
    exp = Experiment(
        RegressionModel(CountingLinearRegression),
        BasicTraining(),
        train_data=cached,
    )
    requested = False

    def interrupt(event):
        nonlocal requested
        if event.kind == "safe_point" and not requested:
            requested = True
            return ControlRequest.INTERRUPT

    callback = ManagedCallback(
        interrupt,
        controls={ControlRequest.INTERRUPT},
        fail_soft=True,
    )
    CountingLinearRegression.fits = 0

    with pytest.raises(ManagedInterruptedError):
        exp.train(store=store, callbacks=(callback,))

    assert CountingLinearRegression.fits == 1
    assert exp.train.status(store=store).checkpoint_head is None
    with pytest.raises(ManagedRerunRequiredError):
        exp.train(store=store)
    assert CountingLinearRegression.fits == 1

    rerun = exp.train.rerun(store=store)
    assert rerun.action == "rerun"
    assert CountingLinearRegression.fits == 2
