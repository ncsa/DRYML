from __future__ import annotations

import pytest

from dryml.managed import (
    CallbackCoordinator,
    CallbackFailure,
    ControlRequest,
    ManagedCallback,
    ManagedCapabilityError,
    OperationEvent,
    preflight_callbacks,
)


def test_controls_coalesce_by_precedence_and_checkpoint_is_one_shot():
    callbacks = (
        ManagedCallback(lambda event: ControlRequest.CHECKPOINT, controls={ControlRequest.CHECKPOINT}),
        ManagedCallback(lambda event: ControlRequest.INTERRUPT, controls={ControlRequest.INTERRUPT}),
        ManagedCallback(lambda event: ControlRequest.GRACEFUL_STOP, controls={ControlRequest.GRACEFUL_STOP}),
    )
    coordinator = CallbackCoordinator(callbacks)

    coordinator.publish(OperationEvent(1, "safe_point"))
    assert coordinator.poll() is ControlRequest.INTERRUPT
    assert coordinator.poll() is ControlRequest.INTERRUPT


def test_fail_soft_callback_records_bounded_diagnostics_and_continues():
    def fail(_event):
        raise RuntimeError("observer failed")

    coordinator = CallbackCoordinator((ManagedCallback(fail, fail_soft=True),), max_diagnostics=4)
    for sequence in range(20):
        coordinator.publish(OperationEvent(sequence + 1, "progress"))

    assert coordinator.poll() is ControlRequest.NONE
    assert len(coordinator.diagnostics) == 4
    assert all(
        item == "callback RuntimeError: execution_failed"
        for item in coordinator.diagnostics
    )
    assert "observer failed" not in str(coordinator.diagnostics)


def test_strict_callback_failure_requests_checkpoint_then_failure():
    def fail(_event):
        raise RuntimeError("strict failed")

    coordinator = CallbackCoordinator((ManagedCallback(fail),))
    coordinator.publish(OperationEvent(1, "progress"))

    assert coordinator.poll() is ControlRequest.FAIL
    with pytest.raises(CallbackFailure, match="RuntimeError") as raised:
        coordinator.raise_failure()
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "strict failed"


def test_callback_guarantees_are_rejected_during_preflight():
    strict = ManagedCallback(lambda event: None)
    early = ManagedCallback(
        lambda event: ControlRequest.GRACEFUL_STOP,
        controls={ControlRequest.GRACEFUL_STOP},
    )

    with pytest.raises(ManagedCapabilityError, match="strict callback"):
        preflight_callbacks((strict,), resumable=False, checkpoint_schema=None, early_completion=False)
    with pytest.raises(ManagedCapabilityError, match="early completion"):
        preflight_callbacks((ManagedCallback(early.callback, controls=early.controls, fail_soft=True),), resumable=True, checkpoint_schema="fake-v1", early_completion=False)


def test_callback_may_not_return_an_undeclared_control():
    callback = ManagedCallback(lambda event: ControlRequest.INTERRUPT, fail_soft=True)
    coordinator = CallbackCoordinator((callback,))

    coordinator.publish(OperationEvent(1, "progress"))

    assert coordinator.poll() is ControlRequest.NONE
    assert coordinator.diagnostics[-1] == "callback ValueError: execution_failed"
