"""Caller configuration tests for managed operations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dryml.core.store.dir import DirStore
from dryml.managed import InterruptRequestResult, ManagedConfig, ManagedConfigError, ManagedStatus, ManagedStoreError


def test_config_is_immutable_and_snapshots_callbacks_without_mutating_caller(tmp_path):
    """Config retains caller policy while invocation snapshots its callback list."""

    callback = lambda object_, context: None
    callbacks = [callback]
    config = ManagedConfig(
        state_store=DirStore(tmp_path / "state"),
        control_store=DirStore(tmp_path / "control"),
        rerun=True,
        callbacks=callbacks,
    )

    snapshot = config.snapshot()
    callbacks.append(callback)

    assert snapshot.callbacks == (callback,)
    assert config.callbacks == callbacks
    with pytest.raises(FrozenInstanceError):
        config.rerun = False


def test_config_accepts_only_exact_boolean_dirstore_and_callback_list(tmp_path):
    """Unsupported caller policy fails before a workload can be entered."""

    class BoolLike(int):
        pass

    for rerun in (1, BoolLike(1), "true"):
        with pytest.raises(ManagedConfigError, match="rerun"):
            ManagedConfig(rerun=rerun)
    with pytest.raises(ManagedStoreError):
        ManagedConfig(state_store=object())
    with pytest.raises(ManagedStoreError):
        ManagedConfig(control_store=object())
    with pytest.raises(ManagedConfigError, match="list"):
        ManagedConfig(callbacks=())
    with pytest.raises(ManagedConfigError, match="callable"):
        ManagedConfig(callbacks=[object()])
    with pytest.raises(ManagedConfigError, match="64"):
        ManagedConfig(callbacks=[lambda object_, context: None] * 65)


def test_config_does_not_coerce_or_retain_tuple_callback_input():
    """The public callback grammar deliberately remains caller-list-only."""

    with pytest.raises(ManagedConfigError):
        ManagedConfig(callbacks=(lambda object_, context: None,))


def test_public_error_reasons_and_result_values_are_stable_and_immutable():
    """Caller-visible failures use static reasons and projections are value types."""

    error = ManagedConfigError(message="detail may be human-readable")
    status = ManagedStatus("not_started", "a" * 64, None, 0, None, None, False, None)
    result = InterruptRequestResult("not_running", "a" * 64, None, 0)

    assert error.reason == "invalid_config"
    assert "invalid_config" in str(error)
    with pytest.raises(FrozenInstanceError):
        status.state = "running"
    with pytest.raises(FrozenInstanceError):
        result.outcome = "requested"
