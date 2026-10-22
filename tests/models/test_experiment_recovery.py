"""Recovery boundaries for managed Experiment Artifact evaluation."""

from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from dryml.artifacts import Value
from dryml.core import Definition, Par, Ref, Repo, StateRef
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedRerunRequiredError, managed_operation
from dryml.models import Experiment, ExperimentData, TrainFunction
from dryml.models import experiment as experiment_module
from dryml.models.experiment import _notify_host_observers


pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")


class RecoveryModel(Pickleable):
    """Minimal stateful model used to retain a recoverable Experiment checkpoint."""

    def __init__(self, value=0):
        self.value = value


class CheckpointTrainer(TrainFunction):
    """Produce one update and expose its safe point to Experiment."""

    def __call__(self, exp, *, callbacks=()):
        exp.model.value += 1
        exp.state.record_update(examples=1, loss=1.0)
        for callback in callbacks:
            callback()


class OrderedValue(Value):
    """Artifact recording declaration order and optionally failing its first call."""

    events = []
    failures = set()

    def __init__(self, name, model: Ref[StateRef]):
        self.name = name
        self.model = model

    @property
    def ready(self):
        """Return whether the public scalar result has been installed."""

        return self._value_is_present()

    @managed_operation(resumable=True)
    def compute(self, *, managed) -> None:
        """Record this receiver and fail only the configured initial invocation."""

        type(self).events.append(self.name)
        if self.name in type(self).failures:
            type(self).failures.remove(self.name)
            raise RuntimeError(f"{self.name} failed")
        self._install_value_payload({
            "format": "dryml.artifacts.value", "version": 1,
            "present": True, "result": self.name,
        })


class TerminalTrainer(TrainFunction):
    """Mutate once without an intermediate safe point and count invocations."""

    supports_safe_points = False
    calls = 0

    def __call__(self, exp, *, callbacks=()):
        """Apply the one terminal update that retry must not repeat."""

        type(self).calls += 1
        exp.model.value += 1
        exp.state.record_update(examples=1, loss=1.0)


class CadencedRecoveryTrainer(TrainFunction):
    """Advance to three retained steps without replaying restored updates."""

    def __call__(self, exp, *, callbacks=()):
        while exp.state.step < 3:
            exp.model.value += 1
            exp.state.record_update(examples=1, loss=1.0)
            for callback in callbacks:
                callback()


class TelemetryRecoveryTrainer(TrainFunction):
    """Two-update trainer whose telemetry is outside retained progress identity."""

    supports_observers = True

    def __call__(self, exp, *, callbacks=(), observer_session=None):
        """Resume from accepted steps while never replaying observer delivery."""

        while exp.state.step < 2:
            exp.model.value += 1
            exp.state.record_update(examples=1, loss=1.0)
            for callback in callbacks:
                callback()
            _notify_host_observers(observer_session, {
                "event": "train_batch_end",
                "step": exp.state.step,
            })


class FailOnceValue(Value):
    """Terminal Artifact whose first managed invocation fails before a checkpoint."""

    calls = 0

    @property
    def ready(self):
        """Return whether a complete result has been persisted."""

        return self._value_is_present()

    @managed_operation(resumable=True)
    def compute(self, *, managed) -> None:
        """Fail once, then install a complete terminal result on retry."""

        type(self).calls += 1
        if type(self).calls == 1:
            raise RuntimeError("terminal artifact failed")
        self._install_value_payload({
            "format": "dryml.artifacts.value", "version": 1,
            "present": True, "result": 1,
        })


class _RetainedCheckpointAttempt:
    """Narrow managed-checkpoint seam for revisiting one retained state."""

    def __init__(self, checkpoint, attempt_id):
        self.checkpoint_state_ref = checkpoint
        self.attempt_id = attempt_id

    def checkpoint(self):
        """Return the already retained checkpoint observed by this trajectory arm."""

        return self.checkpoint_state_ref


def _observation_context(repo, checkpoint):
    """Build the callback fields consumed after a managed checkpoint association."""

    return SimpleNamespace(
        checkpoint_state_ref=checkpoint, state_repo=repo, control_store=repo.default_store,
    )


def test_revisited_checkpoint_occurrences_keep_trajectory_predecessors(tmp_path):
    """Managed attempt occurrences do not collapse when the same state is revisited."""

    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(RecoveryModel(), TerminalTrainer(), repo=repo)
    checkpoint = repo.save_object(exp, deep_capture=True)
    ExperimentData.get_or_create(checkpoint.object_projection(), repo=repo)
    context = _observation_context(repo, checkpoint)

    exp._stage_checkpoint(_RetainedCheckpointAttempt(checkpoint, "trajectory-a"))
    exp._evaluate_checkpoint(context)
    first = ExperimentData.find(checkpoint.object_projection(), repo=repo).data.iloc[0]

    # This globally appended row is not the second branch's trajectory predecessor.
    foreign = ExperimentData.find(checkpoint.object_projection(), repo=repo)
    foreign.add_row(
        expected_artifacts=(), state_ref=checkpoint, prev_state_ref=None,
        time=0, examples_seen=0, row_key="other-trajectory",
    )
    foreign.publish(repo=repo)

    exp._stage_checkpoint(_RetainedCheckpointAttempt(checkpoint, "trajectory-b"))
    exp._evaluate_checkpoint(context)
    history = ExperimentData.find(checkpoint.object_projection(), repo=repo)
    second = history.data.loc[history.data.row_key == "v1:trajectory-b:2"].iloc[0]
    exp._evaluate_checkpoint(context)
    history = ExperimentData.find(checkpoint.object_projection(), repo=repo)
    retried = history.data.loc[history.data.row_key == second.row_key].iloc[0]

    assert first.row_key == "v1:trajectory-a:1"
    assert first.state_ref == second.state_ref == checkpoint
    assert second.row_key != first.row_key
    assert (second.prev_row_key, second.prev_state_ref) == (first.row_key, checkpoint)
    assert (retried.row_key, retried.time, retried.state_ref, retried.examples_seen) == (
        second.row_key, second.time, second.state_ref, second.examples_seen,
    )


def test_changed_telemetry_resumes_same_attempt_without_replaying_update(tmp_path):
    """Observer configuration is transient while accepted progress stays authoritative."""

    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(
        RecoveryModel(), TelemetryRecoveryTrainer(),
        checkpoint_every_steps=1, repo=repo,
    )
    first_steps = []
    resumed_steps = []

    def fail(logs):
        first_steps.append(logs["step"])
        raise RuntimeError("telemetry failed")

    with pytest.raises(RuntimeError, match="telemetry failed"):
        exp.train(
            callbacks=[fail], observer_strict=True,
            managed=ManagedConfig(state_repo=repo),
        )
    interrupted = exp.train.status(state_repo=repo)

    final = exp.train(
        callbacks=[lambda logs: resumed_steps.append(logs["step"])],
        managed=ManagedConfig(state_repo=repo),
    )
    completed = exp.train.status(state_repo=repo)

    assert interrupted.attempt_id == completed.attempt_id
    assert first_steps == [1]
    assert resumed_steps == [2]
    assert (exp.model.value, exp.state.step) == (2, 2)
    assert completed.final_state_ref == final


def test_failed_second_artifact_reuses_the_same_pending_history_occurrence(tmp_path):
    """Completed predecessors survive a failed Artifact and retry repairs one row."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    OrderedValue.events.clear()
    OrderedValue.failures = {"second"}
    exp = Experiment(
        RecoveryModel(), CheckpointTrainer(),
        artifacts={
            name: Definition(OrderedValue, name, Par("this.model"))
            for name in ("first", "second", "third")
        },
        repo=repo,
    )

    with pytest.raises(RuntimeError, match="second failed"):
        exp.train(managed=ManagedConfig(state_repo=repo))
    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    history = ExperimentData.find(checkpoint.object_projection(), repo=repo)
    failed = history.data.iloc[0]

    assert OrderedValue.events == ["first", "third", "second"]
    assert failed.evaluation_status == "failed"
    assert tuple(failed.eval_artifacts) == ("first", "third")

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    repaired = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]

    assert repaired.row_key == failed.row_key
    assert repaired.state_ref == checkpoint
    assert repaired.evaluation_status == "completed"
    assert tuple(repaired.eval_artifacts) == ("first", "third", "second")
    # The default cadence is terminal-only; retry repairs its retained row without
    # scheduling a second intermediate/final pair for the one completed update.
    assert OrderedValue.events == ["first", "third", "second", "second"]


def test_completed_admission_race_adopts_receipt_without_rerunning_artifact(
        tmp_path, monkeypatch):
    """A concurrent completion result is adopted without rerunning authored work."""

    import dryml.managed.runtime as managed_runtime

    repo = Repo(DirStore(tmp_path / "store"))
    OrderedValue.events.clear()
    OrderedValue.failures = set()
    exp = Experiment(
        RecoveryModel(), TerminalTrainer(),
        artifacts={"race": Definition(OrderedValue, "race", Par("this.model"))},
        repo=repo,
    )
    original_invoke = managed_runtime.invoke
    completed = []

    def race(descriptor, instance, args, managed, kwargs, **invoke_kwargs):
        result = original_invoke(
            descriptor, instance, args, managed, kwargs, **invoke_kwargs,
        )
        if isinstance(instance, OrderedValue) and not completed:
            completed.append(result)
            raise ManagedRerunRequiredError(
                "already_completed", "simulated concurrent completion",
            )
        return result

    monkeypatch.setattr(managed_runtime, "invoke", race)

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    row = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]

    assert OrderedValue.events == ["race"]
    assert row.evaluation_status == "completed"
    assert row.eval_artifacts["race"] == completed[0]


def test_terminal_artifact_retry_replays_one_occurrence_without_retraining(tmp_path):
    """A failed terminal observer retries its receiver but never invokes the trainer."""

    repo = Repo(DirStore(tmp_path / "store"))
    TerminalTrainer.calls = 0
    FailOnceValue.calls = 0
    exp = Experiment(
        RecoveryModel(), TerminalTrainer(), artifacts={"fail-once": Definition(FailOnceValue)}, repo=repo,
    )

    with pytest.raises(RuntimeError, match="terminal artifact failed"):
        exp.train(managed=ManagedConfig(state_repo=repo))
    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    failed = ExperimentData.find(checkpoint.object_projection(), repo=repo).data.iloc[0]
    final = exp.train(managed=ManagedConfig(state_repo=repo))
    repaired = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]

    assert TerminalTrainer.calls == 1
    assert FailOnceValue.calls == 2
    assert (repaired.row_key, repaired.time, repaired.state_ref) == (
        failed.row_key, failed.time, checkpoint,
    )
    assert repaired.evaluation_status == "completed"
    assert final == checkpoint


def test_cadenced_checkpoint_retry_restores_step_and_finishes_without_duplicate_update(tmp_path):
    """A failed step-two checkpoint retries before the remaining step and final row."""

    repo = Repo(DirStore(tmp_path / "store"))
    FailOnceValue.calls = 0
    exp = Experiment(
        RecoveryModel(), CadencedRecoveryTrainer(), artifacts={"fail-once": Definition(FailOnceValue)},
        checkpoint_every_steps=2, repo=repo,
    )

    with pytest.raises(RuntimeError, match="terminal artifact failed"):
        exp.train(managed=ManagedConfig(state_repo=repo))
    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    assert checkpoint is not None

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    history = ExperimentData.find(final.object_projection(), repo=repo).data

    assert exp.model.value == exp.state.step == 3
    assert list(history.examples_seen) == [2, 3]
    assert len(history) == 2


@pytest.mark.parametrize("stage", (
    "pending_row_published", "artifact_input_published", "artifact_completed",
    "result_row_published", "completed_status_published",
))
def test_terminal_recovery_reuses_one_occurrence_after_each_experiment_boundary(
        tmp_path, monkeypatch, stage):
    """Every post-association U9 boundary retries one durable terminal occurrence."""

    repo = Repo(DirStore(tmp_path / "store"))
    TerminalTrainer.calls = 0
    OrderedValue.events.clear()
    OrderedValue.failures = set()
    exp = Experiment(
        RecoveryModel(), TerminalTrainer(),
        artifacts={"only": Definition(OrderedValue, "only", Par("this.model"))},
        repo=repo,
    )
    original = experiment_module._experiment_boundary
    triggered = []

    def interrupt(boundary):
        if boundary == stage and not triggered:
            triggered.append(boundary)
            raise RuntimeError(f"injected {boundary}")
        original(boundary)

    monkeypatch.setattr(experiment_module, "_experiment_boundary", interrupt)
    with pytest.raises(RuntimeError, match=stage):
        exp.train(managed=ManagedConfig(state_repo=repo))
    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    failed = ExperimentData.find(checkpoint.object_projection(), repo=repo).data.iloc[0]
    final = exp.train(managed=ManagedConfig(state_repo=repo))
    repaired = ExperimentData.find(final.object_projection(), repo=repo).data

    assert triggered == [stage]
    assert TerminalTrainer.calls == 1
    assert len(repaired) == 1
    assert repaired.iloc[0].row_key == failed.row_key
    assert repaired.iloc[0].time == failed.time
    assert repaired.iloc[0].state_ref == checkpoint == final
    assert repaired.iloc[0].evaluation_status == "completed"


def test_association_failure_replays_the_staged_terminal_occurrence(tmp_path, monkeypatch):
    """A crash immediately after association keeps the same staged facts on retry."""

    from dryml.managed import context as managed_context

    repo = Repo(DirStore(tmp_path / "store"))
    TerminalTrainer.calls = 0
    exp = Experiment(RecoveryModel(), TerminalTrainer(), repo=repo)
    original = managed_context._checkpoint_boundary
    triggered = []

    def interrupt(stage):
        if stage == "checkpoint_associated" and not triggered:
            triggered.append(stage)
            raise RuntimeError("injected association")
        original(stage)

    monkeypatch.setattr(managed_context, "_checkpoint_boundary", interrupt)
    with pytest.raises(RuntimeError, match="injected association"):
        exp.train(managed=ManagedConfig(state_repo=repo))
    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    final = exp.train(managed=ManagedConfig(state_repo=repo))
    row = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]

    assert triggered == ["checkpoint_associated"]
    assert TerminalTrainer.calls == 1
    assert (row.state_ref, final) == (checkpoint, checkpoint)
    assert row.evaluation_status == "completed"


def test_failed_status_publication_boundary_recovers_without_retraining(tmp_path, monkeypatch):
    """A post-failure status interruption keeps the receiver and occurrence durable."""

    repo = Repo(DirStore(tmp_path / "store"))
    TerminalTrainer.calls = 0
    OrderedValue.events.clear()
    OrderedValue.failures = {"only"}
    exp = Experiment(
        RecoveryModel(), TerminalTrainer(),
        artifacts={"only": Definition(OrderedValue, "only", Par("this.model"))},
        repo=repo,
    )
    original = experiment_module._experiment_boundary
    triggered = []

    def interrupt(stage):
        if stage == "failed_status_published" and not triggered:
            triggered.append(stage)
            raise RuntimeError("injected failed status")
        original(stage)

    monkeypatch.setattr(experiment_module, "_experiment_boundary", interrupt)
    with pytest.raises(RuntimeError, match="injected failed status"):
        exp.train(managed=ManagedConfig(state_repo=repo))
    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    failed = ExperimentData.find(checkpoint.object_projection(), repo=repo).data.iloc[0]
    final = exp.train(managed=ManagedConfig(state_repo=repo))
    repaired = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]

    assert triggered == ["failed_status_published"]
    assert TerminalTrainer.calls == 1
    assert (repaired.row_key, repaired.time, repaired.state_ref) == (
        failed.row_key, failed.time, checkpoint,
    )
    assert repaired.evaluation_status == "completed"


def test_failed_status_publish_error_preserves_completed_results_for_retry(tmp_path, monkeypatch):
    """A failed-status write error leaves durable result evidence for one retry."""

    repo = Repo(DirStore(tmp_path / "store"))
    TerminalTrainer.calls = 0
    OrderedValue.events.clear()
    OrderedValue.failures = {"second"}
    exp = Experiment(
        RecoveryModel(), TerminalTrainer(),
        artifacts={
            name: Definition(OrderedValue, name, Par("this.model"))
            for name in ("first", "second")
        },
        repo=repo,
    )
    original = ExperimentData.publish
    triggered = []

    def fail_failed_status_publish(history, *args, **kwargs):
        if not triggered and any(
                operation[0] == "update" and operation[2]["evaluation_status"] == "failed"
                for operation in history._pending_operations):
            triggered.append("failed")
            raise OSError("injected failed history publish")
        return original(history, *args, **kwargs)

    monkeypatch.setattr(ExperimentData, "publish", fail_failed_status_publish)
    with pytest.raises(OSError, match="injected failed history publish"):
        exp.train(managed=ManagedConfig(state_repo=repo))
    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    history = ExperimentData.find(checkpoint.object_projection(), repo=repo)
    pending = history.data.iloc[0]
    first_result = pending.eval_artifacts["first"]

    assert triggered == ["failed"]
    assert TerminalTrainer.calls == 1
    assert pending.evaluation_status == "pending"
    assert OrderedValue.events == ["first", "second"]

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    repaired = ExperimentData.find(final.object_projection(), repo=repo).data
    row = repaired.loc[repaired.row_key == pending.row_key].iloc[0]

    assert TerminalTrainer.calls == 1
    assert row.row_key == pending.row_key
    assert row.eval_artifacts["first"] == first_result
    assert row.evaluation_status == "completed"
    assert OrderedValue.events[:3] == ["first", "second", "second"]


def test_completed_status_publish_error_reuses_completed_artifact_without_retraining(tmp_path, monkeypatch):
    """A completed-status write error retries one row without rerunning its Artifact."""

    repo = Repo(DirStore(tmp_path / "store"))
    TerminalTrainer.calls = 0
    OrderedValue.events.clear()
    OrderedValue.failures = set()
    exp = Experiment(
        RecoveryModel(), TerminalTrainer(),
        artifacts={"only": Definition(OrderedValue, "only", Par("this.model"))},
        repo=repo,
    )
    original = ExperimentData.publish
    triggered = []

    def fail_completed_status_publish(history, *args, **kwargs):
        if not triggered and any(
                operation[0] == "update" and operation[2]["evaluation_status"] == "completed"
                for operation in history._pending_operations):
            triggered.append("completed")
            raise OSError("injected completed history publish")
        return original(history, *args, **kwargs)

    monkeypatch.setattr(ExperimentData, "publish", fail_completed_status_publish)
    with pytest.raises(OSError, match="injected completed history publish"):
        exp.train(managed=ManagedConfig(state_repo=repo))
    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    pending = ExperimentData.find(checkpoint.object_projection(), repo=repo).data.iloc[0]
    result = pending.eval_artifacts["only"]

    assert triggered == ["completed"]
    assert TerminalTrainer.calls == 1
    assert pending.evaluation_status == "pending"
    assert OrderedValue.events == ["only"]

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    repaired = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]

    assert TerminalTrainer.calls == 1
    assert (repaired.row_key, repaired.time, repaired.eval_artifacts["only"]) == (
        pending.row_key, pending.time, result,
    )
    assert repaired.evaluation_status == "completed"
    assert OrderedValue.events == ["only"]


@pytest.mark.speed_heavy
@pytest.mark.parametrize("kind", ("xgb-regressor", "stats-logit"))
@pytest.mark.parametrize("stage", (
    "checkpoint_associated", "artifact_completed", "pending_row_published",
    "artifact_compute", "completed_status_publish",
))
def test_native_terminal_replay_reemits_exact_associated_checkpoint(
        tmp_path, monkeypatch, kind, stage, record_property):
    """Real codec replay preserves one fit, occurrence, timestamp and exact pin."""
    from dryml.managed import context as managed_context
    from .test_estimator_experiments import (
        _arrays, _assert_row, _experiment, _forbid_native_encoding, _native_fit_counter,
    )
    from .test_experiment_artifacts import SavedModelValue

    repo = Repo(DirStore(tmp_path / "store"))
    SavedModelValue.calls.clear()
    exp = _experiment(kind, repo)
    calls = _native_fit_counter(kind, monkeypatch)
    triggered = []

    def interrupt(boundary):
        if boundary == stage and not triggered:
            triggered.append(boundary)
            raise RuntimeError(f"injected native {stage}")

    owner = managed_context if stage == "checkpoint_associated" else experiment_module
    name = "_checkpoint_boundary" if owner is managed_context else "_experiment_boundary"
    monkeypatch.setattr(owner, name, interrupt)
    if stage == "artifact_compute":
        install = SavedModelValue._install_value_payload

        def fail_compute(artifact, payload):
            if payload["present"]:
                interrupt("artifact_compute")
            return install(artifact, payload)

        monkeypatch.setattr(SavedModelValue, "_install_value_payload", fail_compute)
    elif stage == "completed_status_publish":
        publish = ExperimentData.publish

        def fail_history(history, *args, **kwargs):
            if any(operation[0] == "update" and operation[2]["evaluation_status"] == "completed"
                   for operation in history._pending_operations):
                interrupt("completed_status_publish")
            return publish(history, *args, **kwargs)

        monkeypatch.setattr(ExperimentData, "publish", fail_history)
    try:
        with pytest.raises(RuntimeError, match=f"injected native {stage}"):
            exp.train(managed=ManagedConfig(state_repo=repo))
        failed_status = exp.train.status(state_repo=repo)
        checkpoint = failed_status.checkpoint_state_ref
        assert checkpoint is not None and failed_status.final_state_ref is None
        retained = repo.load_state_ref(checkpoint, reuse_live="never", cache="none")
        observation = retained.state.pending_observation
        occurrence = retained._occurrence_key(retained.state.pending_observation_attempt_id, observation.sequence)
        timestamp = retained.state.pending_observation_time
        payload, metadata = retained.model._accepted_fit.payload, retained.model._accepted_fit.metadata
        expected = retained.model(_arrays(kind)[0])
        if kind == "stats-logit":
            assert np.isfinite(retained.model.results.bse).all()
            assert np.isfinite(retained.model.results.conf_int()).all()
        _forbid_native_encoding(retained.model, monkeypatch)
        replayed = []
        save = type(retained.model).save_state_to_dir_imp

        def reemit(model, directory, *, codec):
            assert model._accepted_fit.payload == payload
            assert model._accepted_fit.metadata == metadata
            replayed.append(codec)
            return save(model, directory, codec=codec)

        monkeypatch.setattr(type(retained.model), "save_state_to_dir_imp", reemit)
        # Independent handles and a fresh exact graph rule out recovery by live reuse.
        resumed_repo = Repo(DirStore(tmp_path / "store"))
        try:
            resumed = resumed_repo.load_state_ref(checkpoint, reuse_live="never", cache="none")
            publication_refs = []
            publisher_factory = managed_context._publish_managed_state

            def checked_publisher():
                publisher = publisher_factory()

                def publish(*args, **kwargs):
                    result = publisher(*args, **kwargs)
                    assert result.state_ref == checkpoint
                    publication_refs.append(result.state_ref)
                    return result

                return publish

            monkeypatch.setattr(managed_context, "_publish_managed_state", checked_publisher)
            final = resumed.train(managed=ManagedConfig(state_repo=resumed_repo))
            completed = resumed.train.status(state_repo=resumed_repo)
            rows = ExperimentData.find(final.object_projection(), repo=resumed_repo).data
            assert calls == [48] and replayed and publication_refs == [checkpoint]
            assert triggered == [stage]
            assert completed.attempt_id == failed_status.attempt_id
            assert completed.final_state_ref == final
            assert len(rows) == 1
            row = rows.iloc[0]
            assert len(SavedModelValue.calls) == (2 if stage == "artifact_compute" else 1)
            assert (row.row_key, row.time, row.state_ref, row.examples_seen) == (occurrence, timestamp, checkpoint, 48)
            _assert_row(resumed_repo, row)
            assert (resumed.state.step, resumed.state.epoch, resumed.state.examples_seen) == (1, 1, 48)
            np.testing.assert_array_equal(resumed.model(_arrays(kind)[0]), expected)
            # The Experiment replay guard checks this same full StateRef before
            # continuing the row; independently republishing proves codec identity.
            assert resumed_repo.save_object(retained, deep_capture=True) == checkpoint
            record_property("native_replay", json.dumps({
                "kind": kind, "stage": stage, "checkpoint": checkpoint.digest(),
                "final": final.digest(), "row_key": occurrence, "time": timestamp,
                "native_fit_rows": calls, "examples_seen": int(row.examples_seen),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "replayed_checkpoint": publication_refs[0].digest(),
            }, sort_keys=True))
        finally:
            resumed_repo.close(flush=False)
    finally:
        repo.close(flush=False)


@pytest.mark.speed_heavy
@pytest.mark.parametrize("kind", ("xgb-regressor", "stats-logit"))
def test_native_preassociation_failure_retains_live_fit_but_prior_authority_refits(
        tmp_path, monkeypatch, kind, record_property):
    """Published-but-unassociated native work is not a completed-fit resume pin."""
    from dryml.managed import context as managed_context
    from .test_estimator_experiments import _arrays, _experiment, _native_fit_counter

    repo = Repo(DirStore(tmp_path / "store"))
    exp = _experiment(kind, repo)
    prior = repo.save_object(exp, deep_capture=True)
    calls = _native_fit_counter(kind, monkeypatch)
    triggered = []

    def fail(stage):
        if stage == "state_published" and not triggered:
            triggered.append(stage)
            raise OSError("native checkpoint not associated")

    monkeypatch.setattr(managed_context, "_checkpoint_boundary", fail)
    try:
        with pytest.raises(OSError, match="not associated"):
            exp.train(managed=ManagedConfig(state_repo=repo))
        status = exp.train.status(state_repo=repo)
        assert status.state == "failed"
        assert status.checkpoint_state_ref is status.final_state_ref is None
        assert exp.model._accepted_fit is not None
        assert (exp.state.step, exp.state.epoch, exp.state.examples_seen) == (1, 1, 48)
        assert exp.state.pending_observation_terminal
        assert exp.model(_arrays(kind)[0]).shape[0] == 48
        assert ExperimentData.find(prior.object_projection(), repo=repo) is None
        fresh_repo = Repo(DirStore(tmp_path / "store"))
        try:
            restarted = fresh_repo.load_state_ref(prior, reuse_live="never", cache="none")
            assert restarted.model._accepted_fit is None and restarted.state.step == 0
            final = restarted.train(managed=ManagedConfig(state_repo=fresh_repo, rerun=True))
            assert calls == [48, 48]
            assert (restarted.state.step, restarted.state.epoch, restarted.state.examples_seen) == (1, 1, 48)
            assert restarted.train.status(state_repo=fresh_repo).final_state_ref == final
            record_property("preassociation_restart", json.dumps({
                "kind": kind, "prior": prior.digest(), "final": final.digest(),
                "native_fit_rows": calls, "restarted_exposure": restarted.state.examples_seen,
                "failed_checkpoint_associated": status.checkpoint_state_ref is not None,
            }, sort_keys=True))
        finally:
            fresh_repo.close(flush=False)
    finally:
        repo.close(flush=False)
