"""Recovery boundaries for managed Experiment Artifact evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dryml.artifacts import Value
from dryml.core import Definition, Par, Ref, Repo, StateRef
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedRerunRequiredError, managed_operation
from dryml.models import Experiment, ExperimentData, TrainFunction
from dryml.models import experiment as experiment_module


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

    @managed_operation(resumable=True, return_state_ref=True)
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


class FailOnceValue(Value):
    """Terminal Artifact whose first managed invocation fails before a checkpoint."""

    calls = 0

    @property
    def ready(self):
        """Return whether a complete result has been persisted."""

        return self._value_is_present()

    @managed_operation(resumable=True, return_state_ref=True)
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
