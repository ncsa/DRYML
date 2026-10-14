"""Private one-shot acceptance exercised through Models, Data, and Experiment."""

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
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
