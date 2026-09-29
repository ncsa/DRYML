import inspect

import pytest

from dryml.core import Repo, StateRef
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig
from dryml.models import Experiment, TrainFunction, TrainState
from dryml.models.utils import advance_train_state


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

    assert isinstance(exp.train(managed=ManagedConfig(state_repo=repo)), StateRef)
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
