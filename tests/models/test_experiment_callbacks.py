"""Experiment-managed observer composition and interruption ordering tests."""

from __future__ import annotations

import pytest

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedInterrupted
from dryml.models import Experiment, ExperimentData

from .test_experiment_artifacts import CounterModel, TerminalOnly


pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")


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
