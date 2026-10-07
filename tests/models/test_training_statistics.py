import pickle

import pytest

from dryml.models import TrainState


def test_train_state_retains_successful_exposure_and_weighted_loss_window():
    state = TrainState()

    state.record_update(examples=64, loss=2.0)
    state.record_update(examples=17, loss=5.0)
    observation = state.stage_observation()

    assert state.examples_seen == 81
    assert state.step == 2
    assert observation.training_loss == pytest.approx((64 * 2.0 + 17 * 5.0) / 81)
    assert state.loss_numerator == 0.0
    assert state.loss_denominator == 0


def test_train_state_round_trip_retains_progress_and_pending_loss_window():
    state = TrainState()
    state.record_update(examples=64, loss=1.0)
    state.record_update(examples=17, loss=3.0)
    state.next_batch = 32
    state.finish_epoch()
    state.record_update(examples=5, loss=4.0)

    restored = pickle.loads(pickle.dumps(state))

    assert restored == state
    assert restored.examples_seen == 86
    assert restored.next_batch == 1
    assert restored.loss_numerator == pytest.approx(135.0)
    assert restored.loss_denominator == 86


def test_train_state_retains_one_invocation_target_across_restore_then_allows_fresh_work():
    state = TrainState()

    assert state.begin_invocation(2) == 2
    state.record_update(examples=1, loss=1.0)
    state.finish_epoch()
    restored = pickle.loads(pickle.dumps(state))

    assert restored.begin_invocation(2) == 2
    restored.finish_epoch()
    restored.finish_invocation(2)
    assert restored.target_epoch is None
    assert restored.begin_invocation(1) == 3


def test_train_state_rejects_failed_or_invalid_updates_without_advancing_accounting():
    state = TrainState()

    with pytest.raises(ValueError):
        state.record_update(examples=0, loss=1.0)
    with pytest.raises(ValueError):
        state.record_update(examples=1, loss=float("nan"))

    assert state.examples_seen == 0
    assert state.step == 0


def test_train_state_records_one_shot_fit_without_optimizer_loss_telemetry():
    state = TrainState()

    state.record_fit(examples=3)

    assert (state.step, state.examples_seen, state.next_batch) == (1, 3, 1)
    assert (state.loss_numerator, state.loss_denominator) == (0.0, 0)


def test_train_state_retains_normalized_epoch_postlude_until_completion():
    state = TrainState()
    state.record_update(examples=1, loss=1.0)
    state.finish_epoch(postlude_pending=True)

    assert (state.epoch, state.next_batch, state.pending_epoch_postlude) == (1, 0, 0)
    with pytest.raises(ValueError, match="incomplete"):
        state.finish_invocation(1)

    state.target_epoch = 1
    state.finish_epoch_postlude(0)
    state.finish_invocation(1)
    assert state.target_epoch is None


def test_train_state_round_trip_retains_pending_epoch_postlude():
    state = TrainState(target_epoch=1)
    state.record_update(examples=1, loss=1.0)
    state.finish_epoch(postlude_pending=True)

    restored = pickle.loads(pickle.dumps(state))

    assert restored.pending_epoch_postlude == 0
    assert (restored.epoch, restored.next_batch, restored.target_epoch) == (1, 0, 1)


def test_train_state_restores_legacy_slotted_schema_with_new_defaults():
    state = TrainState.__new__(TrainState)

    state.__setstate__({"epoch": 3, "step": 5, "phase": TrainState.training})

    assert (state.epoch, state.step, state.phase) == (3, 5, TrainState.training)
    assert state.examples_seen == 0
    assert state.pending_epoch_postlude is None
    assert state.pending_epoch_metrics is None
    assert state.pending_observation is None


def test_train_state_restores_genuine_three_slot_pickle_state_with_defaults():
    state = TrainState.__new__(TrainState)

    state.__setstate__((3, 5, TrainState.training))

    assert (state.epoch, state.step, state.phase) == (3, 5, TrainState.training)
    assert state.examples_seen == state.loss_denominator == state.next_batch == 0
    assert state.target_epoch is state.pending_epoch_postlude is None


@pytest.mark.parametrize(
    ("state", "error"),
    [
        ({"unexpected": 1}, ValueError),
        ({"epoch": True}, TypeError),
        ({"step": -1}, ValueError),
        ({"loss_numerator": float("nan")}, ValueError),
        ({"phase": "unknown"}, ValueError),
        ({"pending_epoch_postlude_phase": "unknown"}, ValueError),
        ({"target_epoch": 0, "epoch": 1}, ValueError),
        ({"epoch": 1, "next_batch": 1, "pending_epoch_postlude": 0,
          "pending_epoch_postlude_phase": "start"}, ValueError),
        ({"pending_observation": object()}, TypeError),
        (({"unexpected": 1}, {"epoch": 1}), TypeError),
    ],
)
def test_train_state_rejects_malformed_persisted_state_before_installation(state, error):
    restored = TrainState.__new__(TrainState)

    with pytest.raises(error):
        restored.__setstate__(state)
