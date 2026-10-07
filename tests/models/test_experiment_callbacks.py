"""Experiment-managed observer composition and interruption ordering tests."""

from __future__ import annotations

from asyncio import CancelledError as AsyncCancelledError
from concurrent.futures import CancelledError as FuturesCancelledError

import pytest

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedInterrupted
from dryml.models import Experiment, ExperimentData, TrainFunction
from dryml.models.experiment import _notify_host_observers

from .test_experiment_artifacts import CounterModel, TerminalOnly


pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")


class TelemetryTrainer(TrainFunction):
    """Tiny trainer that emits one host log after a truthful safe point."""

    supports_observers = True

    def __call__(self, exp, *, callbacks=(), observer_session=None):
        """Retain one update, checkpoint it, then report its accepted position."""

        exp.model.value += 1
        exp.state.record_update(examples=1, loss=2.0)
        for callback in callbacks:
            callback()
        _notify_host_observers(observer_session, {
            "event": "train_batch_end",
            "step": exp.state.step,
            "loss": 2.0,
        })


def test_caller_observer_runs_after_completed_experiment_history_without_mutation(tmp_path):
    """Experiment prepends only its invocation-local observer before caller order."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    exp = Experiment(CounterModel(), TerminalOnly(), repo=repo)
    seen = []

    def first(obj, context):
        history = ExperimentData.find(context.checkpoint_state_ref.object_projection(), repo=context.state_repo)
        seen.append(("first", history.data.iloc[0].evaluation_status))

    def second(obj, context):
        seen.append(("second", context.checkpoint_state_ref))

    callbacks = [first, second]
    config = ManagedConfig(state_repo=repo, callbacks=callbacks)
    final = exp.train(managed=config)

    assert callbacks == [first, second]
    assert seen == [("first", "completed"), ("second", final)]


def test_caller_interrupt_is_decided_only_after_experiment_observer(tmp_path):
    """An external request sees the terminal facts-only row before interruption."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    exp = Experiment(CounterModel(), TerminalOnly(), repo=repo)
    observed = []

    def request(obj, context):
        history = ExperimentData.find(context.checkpoint_state_ref.object_projection(), repo=context.state_repo)
        observed.append(history.data.iloc[0].evaluation_status)
        assert obj.train.request_interrupt(state_repo=context.state_repo).outcome == "requested"

    with pytest.raises(ManagedInterrupted):
        exp.train(managed=ManagedConfig(state_repo=repo, callbacks=[request]))

    assert observed == ["completed"]


def test_caller_failure_after_terminal_observer_preserves_completed_history(tmp_path):
    """Caller callback failure follows, but cannot undo, Experiment terminal work."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    exp = Experiment(CounterModel(), TerminalOnly(), repo=repo)
    callbacks = []

    def fail(obj, context):
        history = ExperimentData.find(context.checkpoint_state_ref.object_projection(), repo=context.state_repo)
        callbacks.append(("fail", history.data.iloc[0].evaluation_status))
        raise RuntimeError("caller failed")

    config = ManagedConfig(state_repo=repo, callbacks=[fail])
    with pytest.raises(RuntimeError, match="caller failed"):
        exp.train(managed=config)

    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    history = ExperimentData.find(checkpoint.object_projection(), repo=repo)
    assert callbacks == [("fail", "completed")]
    assert config.callbacks == [fail]
    assert history.data.iloc[0].evaluation_status == "completed"


def test_best_effort_telemetry_warns_once_closes_and_allows_completion(tmp_path):
    """Routine observer failures remain visible but nonauthoritative by default."""

    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(CounterModel(), TelemetryTrainer(), repo=repo)
    events = []

    class Observer:
        def __call__(self, logs):
            events.append(("event", dict(logs)))
            raise RuntimeError("private telemetry detail")

        def close(self):
            events.append(("close", None))
            raise RuntimeError("private close detail")

    with pytest.warns(RuntimeWarning, match="observer 0 failed.*train_batch_end") as warnings:
        final = exp.train(
            callbacks=[Observer()],
            managed=ManagedConfig(state_repo=repo),
        )

    assert len(warnings) == 1
    assert events == [
        ("event", {"event": "train_batch_end", "step": 1, "loss": 2.0}),
        ("close", None),
    ]
    assert exp.state.is_trained
    assert exp.train.status(state_repo=repo).final_state_ref == final


def test_strict_telemetry_fails_after_retained_update_without_false_completion(tmp_path):
    """Strict telemetry propagates after the safe point and closes exactly once."""

    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(
        CounterModel(), TelemetryTrainer(), checkpoint_every_steps=1, repo=repo,
    )
    closed = []

    class Observer:
        def __call__(self, logs):
            assert logs["step"] == 1
            raise RuntimeError("strict telemetry")

        def close(self):
            closed.append("closed")

    with pytest.raises(RuntimeError, match="strict telemetry"):
        exp.train(
            callbacks=[Observer()],
            observer_strict=True,
            managed=ManagedConfig(state_repo=repo),
        )

    status = exp.train.status(state_repo=repo)
    assert (exp.model.value, exp.state.step, closed) == (1, 1, ["closed"])
    assert status.state == "failed"
    assert status.checkpoint_state_ref is not None
    assert status.final_state_ref is None


def test_telemetry_cleanup_does_not_mask_an_escaping_failure(tmp_path):
    """Cleanup interruption cannot replace the strict delivery failure."""

    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(CounterModel(), TelemetryTrainer(), repo=repo)

    class Observer:
        def __call__(self, logs):
            del logs
            raise RuntimeError("delivery failure")

        def close(self):
            raise KeyboardInterrupt()

    with pytest.raises(RuntimeError, match="delivery failure"):
        exp.train(
            callbacks=[Observer()],
            observer_strict=True,
            managed=ManagedConfig(state_repo=repo),
        )


def test_telemetry_keyboard_interrupt_is_not_downgraded_to_best_effort(tmp_path):
    """Caller interruption escapes observer policy and still closes resources."""

    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(CounterModel(), TelemetryTrainer(), repo=repo)
    closed = []

    class Observer:
        def __call__(self, logs):
            del logs
            raise KeyboardInterrupt()

        def close(self):
            closed.append("closed")

    with pytest.raises(ManagedInterrupted):
        exp.train(
            callbacks=[Observer()],
            managed=ManagedConfig(state_repo=repo),
        )

    assert (exp.model.value, exp.state.step, closed) == (1, 1, ["closed"])


@pytest.mark.parametrize("error_type", [AsyncCancelledError, FuturesCancelledError])
def test_telemetry_cancellation_is_not_downgraded_to_best_effort(
        tmp_path, error_type):
    """Async and concurrent cancellation escape while cleanup still runs."""

    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(CounterModel(), TelemetryTrainer(), repo=repo)
    closed = []

    class Observer:
        def __call__(self, logs):
            del logs
            raise error_type()

        def close(self):
            closed.append("closed")

    with pytest.raises(error_type):
        exp.train(
            callbacks=[Observer()],
            managed=ManagedConfig(state_repo=repo),
        )

    assert closed == ["closed"]


def test_telemetry_cleanup_interruption_still_closes_remaining_observers(tmp_path):
    """Reverse cleanup remains exhaustive before its interruption propagates."""

    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(CounterModel(), TelemetryTrainer(), repo=repo)
    closed = []

    class RecordingObserver:
        def __call__(self, logs):
            del logs

        def close(self):
            closed.append("recording")

    class InterruptingObserver:
        def __call__(self, logs):
            del logs

        def close(self):
            closed.append("interrupting")
            raise KeyboardInterrupt()

    with pytest.raises(ManagedInterrupted):
        exp.train(
            callbacks=[RecordingObserver(), InterruptingObserver()],
            managed=ManagedConfig(state_repo=repo),
        )

    assert closed == ["interrupting", "recording"]


def test_telemetry_rejects_a_trainer_without_observer_support_before_mutation(tmp_path):
    """Telemetry is not silently passed to trainers without a qualified hook."""

    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(CounterModel(), TerminalOnly(), repo=repo)

    with pytest.raises(TypeError, match="does not support"):
        exp.train(
            callbacks=[lambda logs: None],
            managed=ManagedConfig(state_repo=repo),
        )

    assert (exp.model.value, exp.state.step, exp.state.phase) == (0, 0, None)
