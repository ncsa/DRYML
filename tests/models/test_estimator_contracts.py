"""Private one-shot acceptance exercised through Models, Data, and Experiment."""

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from dryml.core import Definition, Par, Repo, StateRef
from dryml.core.object import Serializable
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter
from dryml.data import ArrayDataset, Batch, as_supervised
from dryml.managed import ManagedConfig
from dryml.models import Experiment, ExperimentData, Model, TrainFunction, TrainState
from dryml.models import utils

from .test_experiment_artifacts import SavedModelValue


@dataclass(frozen=True)
class FittedMean:
    """Small NumPy-native fitted candidate with required feature metadata."""

    mean: float
    features: int


@dataclass(frozen=True)
class AcceptedMean:
    """Native prediction facts and the exact encoded payload accepted together."""

    native: FittedMean
    payload: bytes
    metadata: bytes


class MeanModel(Model, Serializable):
    """Test-only Serializable model using a single private accepted-state slot."""

    def __init__(self):
        self._accepted_fit = None

    def __call__(self, x):
        """Predict with the accepted native candidate, preserving the batch axis."""
        fitted = self._accepted_fit.native
        if x.shape[1] != fitted.features:
            raise ValueError("feature width")
        return np.full(len(x), fitted.mean)

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Save the retained test payload without encoding the native state again."""
        Path(dest_dir, "accepted.pkl").write_bytes(pickle.dumps(self._accepted_fit))

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        """Restore the accepted test payload for real Experiment checkpoint loads."""
        self._accepted_fit = pickle.loads(Path(src_dir, "accepted.pkl").read_bytes())


class MeanTraining(TrainFunction):
    """Test-only opaque trainer consuming the shared private acceptance helper."""

    supports_safe_points = False
    supports_observers = False

    def __call__(self, exp, *, callbacks=(), observer_session=None):
        """Fit and encode a detached native mean, then return its accepted bundle."""
        def fit_candidate(x, y):
            native = FittedMean(float(np.mean(y)), x.shape[1])
            if not np.isfinite(native.mean):
                raise ValueError("nonfinite fitted mean")
            return native

        def encode_candidate(native):
            return AcceptedMean(
                native, pickle.dumps(native),
                json.dumps({"features": native.features}, sort_keys=True).encode(),
            )

        return utils._fit_one_shot(
            exp, fit_candidate=fit_candidate, encode_candidate=encode_candidate,
            callbacks=callbacks, observer_session=observer_session,
        )


def _data():
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([1.0, 2.0, 6.0], dtype=np.float32)
    return Batch(as_supervised(ArrayDataset((x, y)), 0, 1), 2)


def _previous_fit():
    exp = Experiment(MeanModel(), MeanTraining(), train_data=_data())
    exp.train_fn(exp)
    return exp


def test_one_shot_accepts_exact_collected_exposure_once_without_telemetry():
    exp = Experiment(MeanModel(), MeanTraining(), train_data=_data())
    prior_state = exp.state

    accepted = exp.train_fn(exp)

    assert accepted is exp.model._accepted_fit
    assert pickle.loads(accepted.payload) == accepted.native
    assert json.loads(accepted.metadata) == {"features": 1}
    np.testing.assert_array_equal(exp.model(np.zeros((2, 1))), [3.0, 3.0])
    assert (exp.state.step, exp.state.epoch, exp.state.examples_seen) == (1, 1, 3)
    assert exp.state.next_batch == 0
    assert exp.state.pending_epoch_postlude is None
    assert exp.state.pending_epoch_metrics is None
    assert exp.state.safe_point_sequence == 0
    assert exp.state.stage_observation().training_loss is None
    assert prior_state == TrainState()


def test_successful_refit_finishes_preparation_and_encoding_before_acceptance(monkeypatch):
    exp = _previous_fit()
    prior_fit, prior_state = exp.model._accepted_fit, exp.state
    events = []

    def boundary(name):
        events.append(name)
        assert exp.model._accepted_fit is prior_fit
        assert exp.state is prior_state
        assert (prior_state.step, prior_state.epoch, prior_state.examples_seen) == (1, 1, 3)

    monkeypatch.setattr(exp.model, "prep_train", lambda: boundary("prep_train"))
    monkeypatch.setattr(exp.model, "prep_eval", lambda: boundary("prep_eval"))

    def fit_candidate(x, y):
        boundary("fit_and_validate")
        assert (x.shape, y.shape, x.dtype, y.dtype) == ((3, 1), (3,), np.float32, np.float32)
        np.testing.assert_array_equal(y, [1.0, 2.0, 6.0])
        return FittedMean(42.0, x.shape[1])

    def encode_candidate(native):
        boundary("encode")
        return AcceptedMean(native, pickle.dumps(native), b'{"features": 1}')

    accepted = utils._fit_one_shot(exp, fit_candidate=fit_candidate, encode_candidate=encode_candidate)

    assert events == ["prep_train", "fit_and_validate", "prep_eval", "encode"]
    assert accepted is exp.model._accepted_fit
    assert exp.state is not prior_state
    assert (exp.state.step, exp.state.epoch, exp.state.examples_seen) == (2, 2, 6)
    np.testing.assert_array_equal(exp.model(np.zeros((1, 1))), [42.0])
    assert prior_fit.native.mean == 3.0


@pytest.mark.parametrize("stage", ["prep_train", "fit", "validate", "prep_eval", "encode"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_preacceptance_failure_preserves_prior_predictions_facts_and_success_counts(
        monkeypatch, stage, error_type):
    exp = _previous_fit()
    prior_fit, prior_state = exp.model._accepted_fit, exp.state
    before = prior_state.__getstate__()
    events = []
    primary = error_type(stage)

    def boundary(name):
        events.append(name)
        assert exp.model._accepted_fit is prior_fit
        assert exp.state is prior_state
        if name == stage:
            raise primary

    monkeypatch.setattr(exp.model, "prep_train", lambda: boundary("prep_train"))
    monkeypatch.setattr(exp.model, "prep_eval", lambda: boundary("prep_eval"))

    def fit_candidate(x, y):
        boundary("fit")
        native = FittedMean(99.0, x.shape[1])
        boundary("validate")
        return native

    def encode_candidate(native):
        boundary("encode")
        return AcceptedMean(native, b"encoded", b"metadata")

    with pytest.raises(error_type) as caught:
        utils._fit_one_shot(exp, fit_candidate=fit_candidate, encode_candidate=encode_candidate)

    assert caught.value is primary
    assert events.count("prep_eval") == 1
    if stage != "encode":
        assert "encode" not in events
    assert exp.model._accepted_fit is prior_fit
    assert exp.state is prior_state
    assert exp.state.__getstate__() == before
    np.testing.assert_array_equal(exp.model(np.zeros((2, 1))), [3.0, 3.0])


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_primary_failure_survives_preparation_cleanup_interruption(monkeypatch, error_type):
    exp = _previous_fit()
    prior_fit, prior_state = exp.model._accepted_fit, exp.state
    primary = error_type("native primary")
    cleanup_calls = []

    def fail(x, y):
        raise primary

    def cleanup():
        cleanup_calls.append(True)
        raise KeyboardInterrupt("secondary cleanup")

    monkeypatch.setattr(exp.model, "prep_eval", cleanup)
    with pytest.raises(error_type) as caught:
        utils._fit_one_shot(exp, fit_candidate=fail, encode_candidate=lambda native: native)

    assert caught.value is primary
    assert cleanup_calls == [True]
    assert exp.model._accepted_fit is prior_fit
    assert exp.state is prior_state


def test_source_failure_preserves_prior_fit_without_native_work(monkeypatch):
    exp = _previous_fit()
    prior_fit, prior_state = exp.model._accepted_fit, exp.state
    events = []
    primary = RuntimeError("source acquisition failed")

    def fail_source(*args, **kwargs):
        raise primary

    monkeypatch.setattr(exp.train_data, "iterator", fail_source)
    monkeypatch.setattr(exp.model, "prep_train", lambda: events.append("prep_train"))
    with pytest.raises(RuntimeError) as caught:
        utils._fit_one_shot(
            exp, fit_candidate=lambda x, y: events.append("fit"),
            encode_candidate=lambda native: events.append("encode"),
        )

    assert caught.value is primary
    assert events == []
    assert exp.model._accepted_fit is prior_fit
    assert exp.state is prior_state
    assert (exp.state.step, exp.state.epoch, exp.state.examples_seen) == (1, 1, 3)


def test_missing_encoded_bundle_cannot_accept_success():
    exp = _previous_fit()
    prior_fit, prior_state = exp.model._accepted_fit, exp.state

    with pytest.raises(ValueError, match="bundle"):
        utils._fit_one_shot(
            exp, fit_candidate=lambda x, y: FittedMean(42.0, x.shape[1]),
            encode_candidate=lambda native: None,
        )

    assert exp.model._accepted_fit is prior_fit
    assert exp.state is prior_state


@pytest.mark.parametrize("corrupt", [
    {"step": -1}, {"examples_seen": True}, {"loss_numerator": float("nan")},
    {"target_epoch": 1}, {"next_batch": 1},
    {"pending_epoch_postlude": 0, "pending_epoch_postlude_phase": "start"},
])
def test_invalid_accounting_fails_before_native_fit_or_preparation(monkeypatch, corrupt):
    exp = _previous_fit()
    for name, value in corrupt.items():
        setattr(exp.state, name, value)
    prior_fit, prior_state = exp.model._accepted_fit, exp.state
    events = []
    monkeypatch.setattr(exp.model, "prep_train", lambda: events.append("prep"))

    with pytest.raises((TypeError, ValueError)):
        utils._fit_one_shot(
            exp, fit_candidate=lambda x, y: events.append("fit"),
            encode_candidate=lambda native: events.append("encode"),
        )

    assert events == []
    assert exp.model._accepted_fit is prior_fit
    assert exp.state is prior_state


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_local_installation_failure_rolls_back_native_bundle_and_accounting(error_type):
    exp = _previous_fit()
    prior_fit, prior_state = exp.model._accepted_fit, exp.state
    prior_values = prior_state.__getstate__()
    injected = error_type("local installation interrupted")
    observed = []

    def interrupt_between_installations(frame, event, arg):
        if (event == "line" and frame.f_code.co_name == "_fit_one_shot"
                and exp.model._accepted_fit is not prior_fit and exp.state is prior_state):
            observed.append(True)
            raise injected
        return interrupt_between_installations

    old_trace = sys.gettrace()
    sys.settrace(interrupt_between_installations)
    try:
        with pytest.raises(error_type) as caught:
            exp.train_fn(exp)
    finally:
        sys.settrace(old_trace)

    assert observed == [True]
    assert caught.value is injected
    assert exp.model._accepted_fit is prior_fit
    assert exp.state is prior_state
    assert exp.state.__getstate__() == prior_values
    np.testing.assert_array_equal(exp.model(np.zeros((1, 1))), [3.0])


@pytest.mark.parametrize("controls", [
    {"callbacks": (lambda: None,)}, {"observer_session": object()},
    {"callbacks": (None,)}, {"callbacks": None},
])
def test_direct_unsupported_requests_fail_before_source_acquisition(monkeypatch, controls):
    exp = Experiment(MeanModel(), MeanTraining(), train_data=_data())
    acquired = []

    def reject_open(*args, **kwargs):
        acquired.append(True)
        raise AssertionError("source must not be acquired")

    monkeypatch.setattr(exp.train_data, "iterator", reject_open)
    with pytest.raises((TypeError, NotImplementedError)):
        exp.train_fn(exp, **controls)
    assert acquired == []
    assert exp.model._accepted_fit is None
    assert exp.state == TrainState()


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_no_observer_experiment_publishes_final_associated_state_and_terminal_history(tmp_path):
    import pandas as pd

    repo = Repo(DirStore(tmp_path / "store"))
    SavedModelValue.calls.clear()
    exp = Experiment(MeanModel(), MeanTraining(), train_data=_data(), repo=repo,
                     checkpoint_every_steps=1,
                     artifacts={"fit": Definition(SavedModelValue, Par("this.model"))})
    terminal_calls = []

    def terminal(obj, context):
        history = ExperimentData.find(context.checkpoint_state_ref.object_projection(), repo=repo)
        terminal_calls.append((obj.state.examples_seen, history.data.iloc[0].evaluation_status))

    final = exp.train(managed=ManagedConfig(state_repo=repo, callbacks=[terminal]))

    assert isinstance(final, StateRef)
    assert final == exp.train.status(state_repo=repo).final_state_ref
    assert terminal_calls == [(3, "completed")]
    history = ExperimentData.find(final.object_projection(), repo=repo)
    assert len(history.data) == 1
    row = history.data.iloc[0]
    assert row.state_ref == final
    assert row.examples_seen == 3
    assert pd.isna(row.training_loss)
    assert SavedModelValue.calls == [final.at(GraphPath((Parameter("model"),)))]
    assert isinstance(row.eval_artifacts["fit"], StateRef)
    loaded = repo.load_state_ref(final, reuse_live="never")
    assert (loaded.state.step, loaded.state.epoch, loaded.state.examples_seen) == (1, 1, 3)
    assert loaded.state.pending_epoch_postlude is None
    assert loaded.model._accepted_fit.payload == exp.model._accepted_fit.payload
    np.testing.assert_array_equal(loaded.model(np.zeros((1, 1))), [3.0])


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_experiment_preacceptance_failure_marks_phase_failed_without_erasing_prior_fit(tmp_path, monkeypatch):
    repo = Repo(DirStore(tmp_path / "store"))
    exp = _previous_fit()
    prior_fit = exp.model._accepted_fit

    def fail_cleanup():
        raise RuntimeError("prep_eval failed")

    monkeypatch.setattr(exp.model, "prep_eval", fail_cleanup)
    with pytest.raises(RuntimeError, match="prep_eval failed"):
        exp.train(managed=ManagedConfig(state_repo=repo))

    assert exp.state.phase == TrainState.failed
    assert (exp.state.step, exp.state.epoch, exp.state.examples_seen) == (1, 1, 3)
    assert exp.model._accepted_fit is prior_fit
    assert exp.train.status(state_repo=repo).final_state_ref is None


def test_xgb_recipe_selection_and_inference_are_native_free():
    """Passive recipes, prepared Map, and refused foreign handoffs need no extra."""
    script = """
import sys
class RejectNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'xgboost', 'sklearn', 'torch', 'tensorflow', 'jax'}:
            raise AssertionError('unexpected native import: ' + fullname)
sys.meta_path.insert(0, RejectNative())
import dryml
import dryml.models
from dryml.models.xgb import Model, ClassifierModel, RegressionModel, BasicTraining
from dryml.models.xgb import base
from dryml.core import TensorSpec
from dryml.data import Map
from tests.data.test_materialize_supervised_numpy import TrackingDataset, pair
from dryml.methods import ImplementationSelectionError
from dryml.methods import conversion
base.version = lambda name: (_ for _ in ()).throw(AssertionError('package inspection'))
def reject_conversion(*args, **kwargs):
    raise AssertionError('unexpected conversion planning/activation')
conversion.make_edge = conversion.convert = reject_conversion
reg = RegressionModel(n_estimators=2)
clf = ClassifierModel(output_spec=TensorSpec('float32', shape=(2,), backend='numpy'))
assert reg.mdl is clf.mdl is None
assert BasicTraining().supports_safe_points is False
assert BasicTraining().supports_observers is False
spec = TensorSpec('float32', shape=(2,), backend='numpy')
assert reg.infer_output_spec(spec).shape == ()
assert clf.infer_output_spec(spec).shape == (2,)
reg.find_implementation(spec).prepared_invoker()
reg.learn(spec)
for backend, shape in ((b, s) for b in ('torch', 'tf', 'jax', None) for s in ((2,), (3, 2))):
    foreign = TensorSpec('float32', shape=shape, backend=backend)
    source = TrackingDataset((pair(),), spec=foreign)
    for action in (lambda: reg.find_implementation(foreign), lambda: reg.learn(foreign),
                   lambda: reg._prepare_implementation(foreign, backend=foreign.backend, batch_mode=None),
                   lambda: Map(source, reg).prepare()):
        try:
            action()
        except ImplementationSelectionError:
            pass
        else:
            raise AssertionError('foreign selection admitted')
    assert source.opens == 0
for batch in (None, 3):
    invalid = TensorSpec('float32', shape=(3, 2), batch=batch, backend='numpy')
    for action in (lambda: reg.find_implementation(invalid), lambda: reg.learn(invalid)):
        try:
            action()
        except ImplementationSelectionError:
            pass
        else:
            raise AssertionError('declared rank-two sample admitted')
assert not {'xgboost', 'sklearn', 'torch', 'tensorflow', 'jax'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_xgb_pure_specs_and_closed_configuration():
    from dryml.core import TensorSpec
    from dryml.models.xgb import ClassifierModel, RegressionModel, BasicTraining

    input_spec = TensorSpec("float64", shape=(3,), batch=5, backend="numpy")
    model = ClassifierModel()
    assert model.definition.parameters["n_jobs"] == 1
    assert model.definition.parameters["random_state"] == 0
    assert model.definition.parameters["n_estimators"] == 100
    with pytest.raises(NotImplementedError, match="output_spec"):
        model.infer_output_spec(input_spec)
    explicit = TensorSpec("float32", shape=(3,), backend="numpy")
    selected = ClassifierModel(output_spec=explicit).find_implementation(input_spec)
    assert selected.conversion_edge is None
    assert selected.infer_output_spec().equal_exact(explicit.with_batch(5))
    assert RegressionModel().infer_output_spec(input_spec).equal_exact(
        TensorSpec("float32", shape=(), batch=5, backend="numpy"))
    for options in ({"device": "cpu"}, {"objective": "binary:logistic"}, {"booster": "gbtree"},
                    {"tree_method": "hist"}, {"callbacks": []}, {"n_jobs": True}, {"n_jobs": 0},
                    {"n_estimators": -1}, {"max_depth": 1.5}, {"random_state": None},
                    {"learning_rate": float("nan")}, {"subsample": 1.1}, {"reg_lambda": -1},
                    {"gamma": 1e100}, {"reg_alpha": 2**1024}):
        with pytest.raises((TypeError, ValueError)):
            ClassifierModel(**options)
    for options in ({"verbose": 1}, {"eval_set": ()}, {"early_stopping_rounds": 2},
                    {"sample_weight": None}, {"xgb_model": None}, {"callbacks": ()}):
        with pytest.raises(TypeError):
            BasicTraining(**options)
    for output in (TensorSpec("float64", shape=(2,), backend="numpy"),
                   TensorSpec("float32", shape=(2,), backend="torch"),
                   TensorSpec("float32", shape=(1,), backend="numpy")):
        with pytest.raises((TypeError, ValueError)):
            ClassifierModel(output_spec=output)


def test_xgb_raw_prediction_refuses_foreign_coercion_and_unfitted_state():
    from dryml.models.xgb import RegressionModel

    class Convertible:
        def __array__(self, *args, **kwargs):
            pytest.fail("foreign coercion attempted")

    model = RegressionModel()
    for action in (model, model.predict):
        with pytest.raises(TypeError, match="dense NumPy"):
            action(Convertible())
        with pytest.raises(ValueError, match="unfitted"):
            action(np.ones((2, 3), dtype=np.float32))


def test_xgb_missing_extra_and_reader_version_errors_are_actionable(monkeypatch):
    import builtins
    from dryml.models.xgb import RegressionModel, base

    original_import = builtins.__import__

    def missing_native(name, *args, **kwargs):
        if name == "xgboost":
            raise ModuleNotFoundError("xgboost")
        return original_import(name, *args, **kwargs)

    model = RegressionModel(n_estimators=2)
    x, y = np.ones((3, 1), dtype=np.float32), np.arange(3, dtype=np.float32)
    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", missing_native)
        with pytest.raises(ImportError, match=r"dryml\[xgboost\]"):
            model.fit(x, y)
    with monkeypatch.context() as guard:
        guard.setitem(sys.modules, "xgboost", object())
        guard.setattr(base, "version", lambda name: "unqualified")
        with pytest.raises(ValueError, match="reader/writer versions"):
            model.fit(x, y)
    assert model._accepted_fit is None


def test_xgb_unsupported_training_controls_refuse_source_acquisition(monkeypatch):
    from dryml.models.xgb import ClassifierModel, BasicTraining

    exp = Experiment(ClassifierModel(), BasicTraining(), train_data=_data())
    monkeypatch.setattr(exp.train_data, "iterator", lambda *a, **k: pytest.fail("opened source"))
    for controls in ({"callbacks": (lambda: None,)}, {"observer_session": object()},
                     {"callbacks": None}, {"sample_weight": np.ones(3)}, {"eval_set": ()}):
        with pytest.raises((TypeError, NotImplementedError)):
            exp.train_fn(exp, **controls)
    assert exp.model._accepted_fit is None
    assert exp.state == TrainState()


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_terminal_callback_failure_keeps_accepted_fit_and_success_counts(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(MeanModel(), MeanTraining(), train_data=_data(), repo=repo)

    def fail_terminal(obj, context):
        raise RuntimeError("terminal callback failed")

    with pytest.raises(RuntimeError, match="terminal callback failed"):
        exp.train(managed=ManagedConfig(state_repo=repo, callbacks=[fail_terminal]))

    assert (exp.state.step, exp.state.epoch, exp.state.examples_seen) == (1, 1, 3)
    np.testing.assert_array_equal(exp.model(np.zeros((1, 1))), [3.0])
    status = exp.train.status(state_repo=repo)
    assert status.final_state_ref is None
    assert status.checkpoint_state_ref is not None


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_terminal_publication_failure_keeps_accepted_fit_without_claiming_receipt(tmp_path, monkeypatch):
    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(MeanModel(), MeanTraining(), train_data=_data(), repo=repo)

    def fail_publication(*args, **kwargs):
        raise OSError("checkpoint publication failed")

    monkeypatch.setattr(exp, "_stage_checkpoint", fail_publication)
    with pytest.raises(OSError, match="checkpoint publication failed"):
        exp.train(managed=ManagedConfig(state_repo=repo))

    assert (exp.state.step, exp.state.epoch, exp.state.examples_seen) == (1, 1, 3)
    np.testing.assert_array_equal(exp.model(np.zeros((1, 1))), [3.0])
    assert exp.train.status(state_repo=repo).final_state_ref is None


def test_statsmodels_pure_specs_and_closed_configuration():
    """Both algorithms have known pure output shapes before native construction."""
    from dryml.core import TensorSpec
    from dryml.models.statsmodels import Model, ClassifierModel, RegressionModel, BasicTraining

    spec = TensorSpec("float32", shape=(3,), batch=5, backend="numpy")
    for cls, shape in ((RegressionModel, ()), (ClassifierModel, (2,))):
        model = cls()
        assert model.results is None
        assert model.definition.parameters["intercept"] is False
        assert model.infer_output_spec(spec).equal_exact(
            TensorSpec("float64", shape=shape, batch=5, backend="numpy"))
        assert model.find_implementation(spec).conversion_edge is None
    for options in ({"intercept": 1}, {"threshold": True}, {"threshold": 0},
                    {"threshold": 1}, {"threshold": float("nan")},
                    {"fit_options": {"maxiter": True}}, {"fit_options": {"maxiter": 0}},
                    {"fit_options": {"tol": float("inf")}}, {"fit_options": {"disp": 1}},
                    {"fit_options": {"method": "bfgs"}}, {"fit_options": {"start_params": [1]}}):
        with pytest.raises((TypeError, ValueError)):
            ClassifierModel(**options)
    with pytest.raises(ValueError):
        RegressionModel(fit_options={"maxiter": 1})
    with pytest.raises(ValueError):
        Model("GLM")
    with pytest.raises(TypeError):
        BasicTraining(maxiter=2)
    for output in (TensorSpec("float32", shape=(2,), backend="numpy"),
                   TensorSpec("float64", shape=(2,), backend="torch"),
                   TensorSpec("float64", shape=(1,), backend="numpy")):
        with pytest.raises((TypeError, ValueError)):
            ClassifierModel(output_spec=output)


def test_statsmodels_recipe_selection_and_map_are_native_free():
    """Foreign producer declarations fail before conversion planning or source open."""
    script = """
import sys
class RejectNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'statsmodels', 'sklearn', 'xgboost', 'torch', 'tensorflow', 'jax'}:
            raise AssertionError('unexpected native import: ' + fullname)
sys.meta_path.insert(0, RejectNative())
import dryml
import dryml.models
from dryml.models.statsmodels import ClassifierModel, RegressionModel, BasicTraining, base
from dryml.core import TensorSpec
from dryml.data import Map
from dryml.methods import ImplementationSelectionError, conversion
from tests.data.test_materialize_supervised_numpy import TrackingDataset, pair
base.version = lambda name: (_ for _ in ()).throw(AssertionError('package inspection'))
def reject_conversion(*args, **kwargs):
    raise AssertionError('unexpected conversion planning/activation')
conversion.make_edge = conversion.convert = reject_conversion
spec = TensorSpec('float32', shape=(2,), backend='numpy')
for model in (ClassifierModel(), RegressionModel()):
    assert model.results is None
    assert model.infer_output_spec(spec).dtype.name == 'float64'
    model.find_implementation(spec).prepared_invoker()
    model.learn(spec)
    for backend, shape in ((b, s) for b in ('torch', 'tf', 'jax', None) for s in ((2,), (3, 2))):
        foreign = TensorSpec('float32', shape=shape, backend=backend)
        source = TrackingDataset((pair(),), spec=foreign)
        for action in (lambda: model.find_implementation(foreign), lambda: model.learn(foreign),
                       lambda: model._prepare_implementation(foreign, backend=foreign.backend, batch_mode=None),
                       lambda: Map(source, model).prepare()):
            try:
                action()
            except ImplementationSelectionError:
                pass
            else:
                raise AssertionError('foreign selection admitted')
        assert source.opens == 0
    for batch in (None, 3):
        invalid = TensorSpec('float32', shape=(3, 2), batch=batch, backend='numpy')
        for action in (lambda: model.find_implementation(invalid), lambda: model.learn(invalid)):
            try:
                action()
            except ImplementationSelectionError:
                pass
            else:
                raise AssertionError('declared rank-two sample admitted')
assert BasicTraining().supports_safe_points is False
assert BasicTraining().supports_observers is False
assert not {'statsmodels', 'sklearn', 'xgboost', 'torch', 'tensorflow', 'jax'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_statsmodels_raw_foreign_refusal_and_missing_extra_error(monkeypatch):
    """Foreign __array__ never runs; admitted fit gets actionable dependency errors."""
    from importlib.metadata import PackageNotFoundError
    from dryml.models.statsmodels import RegressionModel, base

    class Convertible:
        def __array__(self, *args, **kwargs):
            pytest.fail("foreign coercion attempted")

    model = RegressionModel()
    x, y = np.ones((3, 1)), np.arange(3, dtype=np.float64)
    for action in (model, model.predict):
        with pytest.raises(TypeError, match="dense NumPy"):
            action(Convertible())
        with pytest.raises(ValueError, match="unfitted"):
            action(x)
    with pytest.raises(TypeError, match="dense NumPy"):
        model.fit(Convertible(), y)
    with pytest.raises(TypeError, match="dense NumPy"):
        model.fit(x, Convertible())

    def missing(name):
        raise PackageNotFoundError(name)

    with monkeypatch.context() as guard:
        guard.setattr(base, "version", missing)
        with pytest.raises(ImportError, match=r"dryml\[statsmodels\]"):
            model.fit(x, y)
    assert model.results is None


def test_statsmodels_unsupported_controls_reject_before_source_acquisition(monkeypatch):
    from dryml.models.statsmodels import RegressionModel, BasicTraining

    exp = Experiment(RegressionModel(), BasicTraining(), train_data=_data())
    monkeypatch.setattr(exp.train_data, "iterator", lambda *a, **k: pytest.fail("opened source"))
    for controls in ({"callbacks": (lambda: None,)}, {"observer_session": object()},
                     {"callbacks": None}, {"sample_weight": np.ones(3)}, {"maxiter": 1}):
        with pytest.raises((TypeError, NotImplementedError)):
            exp.train_fn(exp, **controls)
    assert exp.model.results is None and exp.state == TrainState()


def test_routine_estimator_selection_keeps_fast_contracts_without_native_imports(tmp_path):
    """Audit actual file/tier/profile selection over only the named estimator owners."""
    script = """
import json
from pathlib import Path
import sys

class RejectNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tensorflow', 'torch', 'jax', 'jaxlib', 'xgboost', 'statsmodels', 'sklearn'}:
            raise AssertionError('unexpected native collection import: ' + fullname)
sys.meta_path.insert(0, RejectNative())
import pytest
from tests.tools import test_buckets as buckets

paths = (
    'tests/models/test_estimator_contracts.py',
    'tests/data/test_materialize_supervised_numpy.py',
    'tests/models/test_xgb_models.py',
    'tests/models/test_statsmodels_models.py',
    'tests/models/test_estimator_experiments.py',
    'tests/models/test_experiment_recovery.py',
    'tests/core/test_state_ref_restore_into.py',
)
# Limit the runner's filesystem input to this contract's explicit owners.
buckets.iter_test_files = lambda: [Path(path) for path in paths]
baseline = json.loads(Path('tests/test_tiers.json').read_text())
profile = json.loads(Path('tests/test_profiles.json').read_text())['profiles']['good-enough']
selected = buckets.select_profile_files(baseline, {'smoke', 'medium'}, profile)
assert set(selected) == {'./' + paths[index] for index in (0, 1, 6)}, selected

class Inventory:
    def pytest_collection_finish(self, session):
        nodes = [item.nodeid for item in session.items]
        Path(sys.argv[1]).write_text(json.dumps(nodes))

raise SystemExit(pytest.main([
    *selected, '--collect-only', '-q', '--no-cov', '--import-mode=importlib',
    '--dryml-runner-tiers=smoke,medium', '--dryml-test-profile=good-enough',
    '-o', 'cache_dir=' + sys.argv[2],
], plugins=[Inventory()]))
"""
    inventory = tmp_path / "routine-nodes.json"
    result = subprocess.run(
        [sys.executable, "-c", script, str(inventory), str(tmp_path / "cache")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = json.loads(inventory.read_text())
    assert len([node for node in nodes if "/test_materialize_supervised_numpy.py::" in node]) == 47
    functions = {node.split("::", 1)[1].split("[", 1)[0] for node in nodes}
    assert {
        "test_one_shot_accepts_exact_collected_exposure_once_without_telemetry",
        "test_preacceptance_failure_preserves_prior_predictions_facts_and_success_counts",
        "test_xgb_recipe_selection_and_inference_are_native_free",
        "test_xgb_pure_specs_and_closed_configuration",
        "test_xgb_missing_extra_and_reader_version_errors_are_actionable",
        "test_statsmodels_recipe_selection_and_map_are_native_free",
        "test_statsmodels_pure_specs_and_closed_configuration",
        "test_statsmodels_raw_foreign_refusal_and_missing_extra_error",
        "test_statsmodels_unsupported_controls_reject_before_source_acquisition",
    } <= functions
    assert not any("test_native_" in node for node in nodes)
