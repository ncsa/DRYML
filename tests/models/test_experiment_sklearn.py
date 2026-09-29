import numpy as np
import pytest

from dryml.core import Repo, StateRef
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArrayDataset
from dryml.models import Experiment
from dryml.models.sklearn import BasicTraining, ClassifierModel, Model, RegressionModel
from dryml.managed import ManagedConfig


linear_model = pytest.importorskip("sklearn.linear_model")


class OpaqueEstimator:
    def __init__(self):
        self.calls = 0

    def predict(self, value):
        self.calls += 1
        raise AssertionError("inference must not probe an estimator")


class ArrayAPIClassifier:
    """Expose array-API dtype metadata without permitting prediction probes."""

    classes_ = np.array([0, 1])

    def __init__(self):
        self.calls = 0

    def __sklearn_tags__(self):
        return type("Tags", (), {"array_api_support": True})()

    def predict_proba(self, value):
        self.calls += 1
        raise AssertionError("inference must not probe an estimator")


def _train_data():
    x = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return ArrayDataset((x, y))


def test_basic_sklearn_training_updates_model_and_experiment_state(tmp_path):
    model = RegressionModel(linear_model.LinearRegression)
    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(model, BasicTraining(), train_data=_train_data(), repo=repo)

    result = exp.train(managed=ManagedConfig(state_repo=repo))

    assert model.obj is model.estimator
    assert isinstance(result, StateRef)
    assert result == exp.train.status(state_repo=repo).final_state_ref
    assert exp.state.epoch == 1
    assert exp.state.step == 3
    assert exp.state.phase == "trained"
    assert model.infer_output_spec(TensorSpec("float32", shape=(2,), backend="numpy")) == TensorSpec(
        "float32",
        shape=(),
        backend="numpy",
    )
    np.testing.assert_allclose(model(np.array([[3.0, 13.0]], dtype=np.float32)), np.array([4.0]), atol=1e-6)


def test_sklearn_training_rejects_unsupported_intermediate_safe_points():
    model = RegressionModel(linear_model.LinearRegression)

    with pytest.raises(NotImplementedError, match="intermediate safe points"):
        BasicTraining()(Experiment(model, BasicTraining(), train_data=_train_data()), callbacks=(lambda: None,))


def test_experiment_save_load_restores_train_state_and_model(tmp_path):
    model = RegressionModel(linear_model.LinearRegression)
    repo = Repo(stores=tmp_path)
    exp = Experiment(model, BasicTraining(), train_data=_train_data(), repo=repo)
    exp.train(managed=ManagedConfig(state_repo=repo))
    state = repo.save_object(exp, alias="exp")
    repo.set_state_alias("checkpoint", state)
    repo.close(flush=True)

    reopened = Repo(stores=tmp_path)
    assert reopened.resolve_object_alias("exp") == state.object
    loaded = reopened.load_state_ref(
        reopened.resolve_state_alias(state.object, "checkpoint"), reuse_live="never"
    )

    assert loaded.state.epoch == 1
    assert loaded.state.step == 3
    assert loaded.state.phase == "trained"
    np.testing.assert_allclose(loaded.model(np.array([[3.0, 13.0]], dtype=np.float32)), np.array([4.0]), atol=1e-6)


def test_sklearn_inference_does_not_probe_estimators_and_pickle_state_stays_local(tmp_path):
    opaque = Model(OpaqueEstimator)
    with pytest.raises(NotImplementedError, match="pass output_spec explicitly"):
        opaque.infer_output_spec(TensorSpec("float32", shape=(2,), backend="numpy"))
    assert opaque.obj.calls == 0

    model = Model(OpaqueEstimator, output_spec=TensorSpec("float32", shape=(1,), backend="numpy"))
    assert model.infer_output_spec(TensorSpec("float32", shape=(2,), batch=2, backend="numpy")) == TensorSpec(
        "float32",
        shape=(1,),
        batch=2,
        backend="numpy",
    )
    assert model.obj.calls == 0

    model.default_batched = True
    model.learn()
    repo = Repo(stores=tmp_path)
    state = repo.save_object(model)
    assert model.call_mode == "learning"
    assert model.default_batched is True
    assert repo.load_state_ref(state, reuse_live="matching") is model

    fresh = repo.load_state_ref(state, reuse_live="never")
    assert fresh.call_mode == "eager"
    assert fresh.default_batched is None


def test_classifier_probability_spec_matches_fitted_predict_proba_output():
    model = ClassifierModel(linear_model.LogisticRegression)
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    model.fit(x, y)

    output = model(np.array([[1.5]], dtype=np.float32))
    spec = model.infer_output_spec(TensorSpec("float32", shape=(1,), backend="numpy"))

    assert spec == TensorSpec(output.dtype, shape=(2,), backend="numpy")
    assert output.shape == (1, 2)


def test_classifier_probability_spec_uses_array_api_input_dtype_without_probing():
    model = ClassifierModel(ArrayAPIClassifier)

    spec = model.infer_output_spec(TensorSpec("float32", shape=(1,), backend="numpy"))

    assert spec == TensorSpec("float32", shape=(2,), backend="numpy")
    assert model.estimator.calls == 0
