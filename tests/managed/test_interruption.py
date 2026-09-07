"""U7 cooperative safe-point interruption coverage."""

from __future__ import annotations

from threading import Event, Thread

import pytest

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedInterrupted, ManagedRerunRequiredError, managed_operation


class InterruptValue(Pickleable):
    """Receiver providing explicit and externally requested safe points."""

    entered = Event()
    release = Event()

    def __init__(self, value=0):
        self.value = value

    @managed_operation(resumable=True)
    def explicit(self, *, managed):
        """Publish a checkpoint then explicitly terminate with the supplied cause."""

        self.value = 7
        managed.interrupt(cause=self.cause)

    @managed_operation(resumable=True)
    def requested(self, *, managed):
        """Wait for another caller before reaching its checkpoint boundary."""

        type(self).entered.set()
        assert type(self).release.wait(10)
        self.value = 8
        managed.checkpoint()

    @managed_operation()
    def keyboard(self, *, managed):
        """Mutate then receive an interrupt outside a declared safe point."""

        self.value = 9
        raise KeyboardInterrupt()

    @managed_operation()
    def invalid_cause(self, *, managed):
        """Pass an invalid cause before any safe-point mutation may happen."""

        managed.interrupt(cause="not an exception")

    @managed_operation(resumable=True)
    def keyboard_after_checkpoint(self, *, managed):
        """Raise outside a safe point after retaining a restorable checkpoint."""

        if managed.is_resuming:
            return self.value
        self.value = 1
        managed.checkpoint()
        self.value = 2
        raise KeyboardInterrupt()


class NestedInterruptValue(Pickleable):
    """Disjoint receiver whose safe point raises the public interruption signal."""

    @managed_operation()
    def stop(self, *, managed):
        """Commit its own interruption for an enclosing-method regression test."""

        managed.interrupt()


class NestedCallerValue(Pickleable):
    """Receiver that invokes a disjoint managed operation from ordinary workload."""

    target = None

    @managed_operation()
    def call_nested(self, *, managed):
        """Allow a disjoint public interruption to escape this ordinary method."""

        type(self).target.stop(managed=ManagedConfig(state_store=managed.state_store))


def test_explicit_interrupt_associates_checkpoint_then_preserves_exact_cause(tmp_path):
    """Method initiated interruption is terminal only after save and callbacks succeed."""

    store = DirStore(tmp_path / "store")
    value = InterruptValue(repo=Repo((store,)))
    cause = ValueError("stop now")
    value.cause = cause

    with pytest.raises(ManagedInterrupted) as caught:
        value.explicit(managed=ManagedConfig(state_store=store))
    assert caught.value.__cause__ is cause
    status = value.explicit.status(state_store=store)
    assert status.state == "interrupted"
    assert status.checkpoint_state_ref is not None
    assert status.final_state_ref is None


def test_external_request_is_honored_at_the_next_checkpoint(tmp_path):
    """A concurrent request remains cooperative and stops at its associated save."""

    InterruptValue.entered, InterruptValue.release = Event(), Event()
    store = DirStore(tmp_path / "store")
    value = InterruptValue(repo=Repo((store,)))
    outcome = []
    thread = Thread(target=lambda: outcome.append(pytest.raises(ManagedInterrupted, value.requested, managed=ManagedConfig(state_store=store)).value))
    thread.start()
    try:
        assert InterruptValue.entered.wait(10)
        assert value.requested.request_interrupt(state_store=store).outcome == "requested"
        InterruptValue.release.set()
        thread.join(10)
        assert not thread.is_alive()
    finally:
        InterruptValue.release.set()
        thread.join(10)
    assert type(outcome[0]) is ManagedInterrupted
    assert value.requested.status(state_store=store).state == "interrupted"


def test_keyboard_interrupt_without_checkpoint_requires_explicit_rerun(tmp_path):
    """Caller interruption is terminal without publishing arbitrary live mutation."""

    store = DirStore(tmp_path / "store")
    value = InterruptValue(repo=Repo((store,)))

    with pytest.raises(ManagedInterrupted) as caught:
        value.keyboard(managed=ManagedConfig(state_store=store))
    assert type(caught.value.__cause__) is KeyboardInterrupt
    status = value.keyboard.status(state_store=store)
    assert (status.state, status.checkpoint_state_ref, status.final_state_ref) == ("interrupted", None, None)
    with pytest.raises(ManagedRerunRequiredError, match="rerun_required"):
        value.keyboard(managed=ManagedConfig(state_store=store))


def test_keyboard_interrupt_retains_checkpoint_for_resume_restoration(tmp_path):
    """A caller interruption preserves the last checkpoint without saving later mutation."""

    store = DirStore(tmp_path / "store")
    value = InterruptValue(repo=Repo((store,)))

    with pytest.raises(ManagedInterrupted) as caught:
        value.keyboard_after_checkpoint(managed=ManagedConfig(state_store=store))
    interrupted = value.keyboard_after_checkpoint.status(state_store=store)
    assert type(caught.value.__cause__) is KeyboardInterrupt
    assert (interrupted.state, interrupted.checkpoint_state_ref is not None, value.value) == ("interrupted", True, 2)
    assert value.keyboard_after_checkpoint(managed=ManagedConfig(state_store=store)) == 1
    assert (value.value, value.keyboard_after_checkpoint.status(state_store=store).state) == (1, "completed")


def test_disjoint_nested_interruption_fails_the_enclosing_method(tmp_path):
    """Only the context that committed interruption skips terminal failure cleanup."""

    store = DirStore(tmp_path / "store")
    nested = NestedInterruptValue(repo=Repo((store,)))
    caller = NestedCallerValue(repo=Repo((store,)))
    NestedCallerValue.target = nested
    try:
        with pytest.raises(ManagedInterrupted):
            caller.call_nested(managed=ManagedConfig(state_store=store))
    finally:
        NestedCallerValue.target = None
    assert caller.call_nested.status(state_store=store).failure_code == "method_error"
    assert nested.stop.status(state_store=store).state == "interrupted"


def test_invalid_explicit_interrupt_cause_fails_before_checkpoint_mutation(tmp_path):
    """Cause validation cannot create a checkpoint or an interrupted authority."""

    from dryml.managed import ManagedContextError

    store = DirStore(tmp_path / "store")
    value = InterruptValue(repo=Repo((store,)))
    callbacks = []
    with pytest.raises(ManagedContextError, match="invalid_cause"):
        value.invalid_cause(managed=ManagedConfig(state_store=store, callbacks=[lambda *args: callbacks.append(args)]))
    status = value.invalid_cause.status(state_store=store)
    assert (value.value, callbacks, status.state, status.checkpoint_state_ref) == (0, [], "failed", None)
