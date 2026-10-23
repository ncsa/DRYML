import inspect

import pytest

from dryml.core import Node, Repo
from dryml.managed import InvocationResult
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig
from dryml.models import Experiment, TrainFunction, TrainState
from dryml.models.utils import advance_train_state
from dryml.core.object import Pickleable


class CountedModel(Pickleable):
    captures = 0
    restores = 0

    def __init__(self):
        self.value = 0

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).captures += 1
        return super().save_state_to_dir_imp(dest_dir, codec=codec)

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        type(self).restores += 1
        return super().restore_state_from_dir_imp(src_dir, codec=codec)


class SuccessfulTrain(TrainFunction):
    def __call__(self, exp, *, callbacks=()):
        advance_train_state(exp, epochs=1, steps=2)
        return "ok"


class NoStateTrain(TrainFunction):
    def __call__(self, exp, *, callbacks=()):
        return "ok"


class FailingTrain(TrainFunction):
    def __call__(self, exp, *, callbacks=()):
        raise RuntimeError("boom")


def test_train_state_phase_constants_and_predicates():
    state = TrainState()

    assert state == TrainState.initial
    assert state != TrainState.trained
    assert state.is_initial

    state.phase = TrainState.trained

    assert state == TrainState.trained
    assert state.is_trained


def test_train_function_requires_a_logical_call_without_breaking_current_training():
    """TrainFunction declares one mandatory call and direct subclasses satisfy it."""

    class MissingCall(TrainFunction):
        pass

    assert inspect.isabstract(TrainFunction)
    assert inspect.isabstract(MissingCall)
    with pytest.raises(TypeError, match="__call__"):
        MissingCall()
    assert not inspect.isabstract(SuccessfulTrain)


def test_experiment_train_sets_trained_on_success(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(None, SuccessfulTrain(), repo=repo)

    result = exp.train(managed=ManagedConfig(state_repo=repo))
    assert isinstance(result, InvocationResult)
    assert isinstance(result.final_state_ref, Node)
    assert result.value is None
    assert exp.state == TrainState.trained
    assert exp.state.is_trained
    assert exp.state.epoch == 1
    assert exp.state.step == 2


def test_experiment_train_marks_trained_if_train_fn_does_not_set_phase(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(None, NoStateTrain(), repo=repo)

    exp.train(managed=ManagedConfig(state_repo=repo))

    assert exp.state == TrainState.trained


def test_experiment_train_sets_failed_on_exception(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(None, FailingTrain(), repo=repo)

    with pytest.raises(RuntimeError, match="boom"):
        exp.train(managed=ManagedConfig(state_repo=repo))

    assert exp.state == TrainState.failed
    assert exp.state.is_failed


def test_direct_train_function_calls_retain_backend_return_values():
    """Backend-facing direct trainer calls do not inherit Experiment's StateRef API."""

    exp = Experiment(None, SuccessfulTrain())

    assert exp.train_fn(exp, callbacks=()) == "ok"


def test_terminal_selection_and_completed_replay_do_no_extra_capture_or_callbacks(tmp_path):
    """One terminal checkpoint is also final; replay only changes caller identity."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    exp = Experiment(CountedModel(), NoStateTrain(), repo=repo)
    CountedModel.captures = CountedModel.restores = 0
    callbacks = []
    config = ManagedConfig(state_repo=repo, callbacks=[lambda _obj, ctx: callbacks.append(ctx.invocation_id)])
    result = exp.train(managed=config)
    status = exp.train.status(state_repo=repo)
    assert result.final_state_ref.graph_equal(status.checkpoint)
    assert (CountedModel.captures, CountedModel.restores, len(callbacks)) == (1, 0, 1)
    replay = exp.train(managed=config)
    assert replay.execution.run_id == result.execution.run_id
    assert replay.execution.invocation_id != result.execution.invocation_id
    assert (CountedModel.captures, CountedModel.restores, len(callbacks)) == (1, 0, 1)


@pytest.mark.parametrize("damage", ("old", "missing", "unknown", "counter"))
def test_run_aware_train_state_rejects_incompatible_records_atomically(damage):
    """The current schema has no legacy/default-filling conversion lane."""
    state = TrainState(step=7)
    before = state.__getstate__()
    payload = TrainState().__getstate__()
    if damage == "old":
        payload = {"epoch": 0, "step": 0, "phase": None}
    elif damage == "missing":
        del payload["fields"]["pending_observation_run_id"]
    elif damage == "unknown":
        payload["fields"]["pending_observation_attempt_id"] = "retired"
    else:
        payload["fields"]["step"] = True
    with pytest.raises((TypeError, ValueError)):
        state.__setstate__(payload)
    assert state.__getstate__() == before
