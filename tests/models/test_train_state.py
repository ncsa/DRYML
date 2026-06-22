import pytest

from dryml.models import Experiment, TrainFunction, TrainState
from dryml.models.utils import advance_train_state


class SuccessfulTrain(TrainFunction):
    def __call__(self, exp):
        advance_train_state(exp, epochs=1, steps=2)
        return "ok"


class NoStateTrain(TrainFunction):
    def __call__(self, exp):
        return "ok"


class FailingTrain(TrainFunction):
    def __call__(self, exp):
        raise RuntimeError("boom")


def test_train_state_phase_constants_and_predicates():
    state = TrainState()

    assert state == TrainState.initial
    assert state != TrainState.trained
    assert state.is_initial

    state.phase = TrainState.trained

    assert state == TrainState.trained
    assert state.is_trained


def test_experiment_train_sets_trained_on_success():
    exp = Experiment(None, SuccessfulTrain())

    assert exp.train() == "ok"
    assert exp.state == TrainState.trained
    assert exp.state.is_trained
    assert exp.state.epoch == 1
    assert exp.state.step == 2


def test_experiment_train_marks_trained_if_train_fn_does_not_set_phase():
    exp = Experiment(None, NoStateTrain())

    exp.train()

    assert exp.state == TrainState.trained


def test_experiment_train_sets_failed_on_exception():
    exp = Experiment(None, FailingTrain())

    with pytest.raises(RuntimeError, match="boom"):
        exp.train()

    assert exp.state == TrainState.failed
    assert exp.state.is_failed
