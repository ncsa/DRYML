import pytest

from dryml.core2 import Repo
from dryml.core2.store.dir import DirStore
from dryml.models import Experiment, Model, TrainFunction, TrainState
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


class DummyModel(Model):
    pass


def test_train_state_phase_constants_and_predicates():
    state = TrainState()

    assert state == TrainState.initial
    assert state != TrainState.trained
    assert state.is_initial

    state.phase = TrainState.trained

    assert state == TrainState.trained
    assert state.is_trained


def test_experiment_train_sets_trained_on_success():
    exp = Experiment(DummyModel(), SuccessfulTrain())

    assert exp.train() == "ok"
    assert exp.state == TrainState.trained
    assert exp.state.is_trained
    assert exp.state.epoch == 1
    assert exp.state.step == 2


def test_experiment_train_marks_trained_if_train_fn_does_not_set_phase():
    exp = Experiment(DummyModel(), NoStateTrain())

    exp.train()

    assert exp.state == TrainState.trained


def test_experiment_train_sets_failed_on_exception():
    exp = Experiment(DummyModel(), FailingTrain())

    with pytest.raises(RuntimeError, match="boom"):
        exp.train()

    assert exp.state == TrainState.failed
    assert exp.state.is_failed


def test_experiment_direct_train_state_is_not_persisted_as_lifecycle_authority(tmp_path):
    store = DirStore(tmp_path / "store")
    exp = Experiment(DummyModel(), SuccessfulTrain())
    exp.train()
    Repo(store).save_object(exp, record_policy="none")

    loaded = Repo(store).load(exp.definition)

    assert loaded.state == TrainState.initial
