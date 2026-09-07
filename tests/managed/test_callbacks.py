"""U7 checkpoint callback ordering and retained-receipt coverage."""

from __future__ import annotations

import pytest

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import (
    ManagedConfig,
    ManagedConflictError,
    ManagedContextError,
    ManagedInterrupted,
    managed_operation,
)
from dryml.managed import context as context_module


class CallbackValue(Pickleable):
    """Receiver that publishes one checkpoint before final mutation."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation()
    def run(self, *, managed):
        """Save a callback-visible intermediate state then complete normally."""

        self.value = 4
        checkpoint = managed.checkpoint()
        self.value = 5
        return checkpoint

    @managed_operation()
    def no_checkpoint(self, *, managed):
        """Complete without creating a callback safe point."""

        self.value += 1

    @managed_operation()
    def twice(self, *, managed):
        """Publish two safe points so a late request has a later service boundary."""

        self.value = 1
        managed.checkpoint()
        self.value = 2
        managed.checkpoint()

    @managed_operation()
    def catch_terminal_interrupt(self, *, managed):
        """Catch a local interruption to prove runtime terminality remains enforced."""

        try:
            managed.interrupt()
        except ManagedInterrupted:
            for safe_point in (managed.checkpoint, managed.interrupt):
                try:
                    safe_point()
                except ManagedContextError as error:
                    self.rejected_safe_points.append(error.reason)


class DerivedValue(Pickleable):
    """Independent object used to prove callback work can hold disjoint ownership."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation()
    def bump(self, *, managed):
        """Persist a disjoint object through a nested managed invocation."""

        self.value += 1


def test_callbacks_receive_exact_committed_receipt_in_list_order(tmp_path):
    """Observers run after association and see the same live object and StateRef."""

    store = DirStore(tmp_path / "store")
    value = CallbackValue(repo=Repo((store,)))
    observed = []

    def first(obj, context):
        observed.append(("first", obj is value, context.checkpoint_state_ref, context.state_store))

    def second(obj, context):
        observed.append(("second", obj.value, context.checkpoint_state_ref))

    checkpoint = value.run(managed=ManagedConfig(state_store=store, callbacks=[first, second]))

    assert observed == [
        ("first", True, checkpoint, store),
        ("second", 4, checkpoint),
    ]


def test_callback_error_stops_later_callbacks_retains_checkpoint_and_fails(tmp_path):
    """A failed observer is not replayed or hidden behind final completion."""

    store = DirStore(tmp_path / "store")
    value = CallbackValue(repo=Repo((store,)))
    calls = []

    def fail(obj, context):
        calls.append(context.checkpoint_state_ref)
        raise RuntimeError("observer failed")

    def never(obj, context):
        calls.append("late")

    with pytest.raises(RuntimeError, match="observer failed"):
        value.run(managed=ManagedConfig(state_store=store, callbacks=[fail, never]))
    status = value.run.status(state_store=store)
    assert (status.state, status.failure_code) == ("failed", "callback_error")
    assert status.checkpoint_state_ref == calls[0]
    assert calls != ["late"]


def test_final_save_never_notifies_checkpoint_callbacks(tmp_path):
    """Completion is not an implicit checkpoint or interruption service point."""

    store = DirStore(tmp_path / "store")
    value = CallbackValue(repo=Repo((store,)))
    calls = []

    assert value.no_checkpoint(managed=ManagedConfig(state_store=store, callbacks=[lambda *args: calls.append(args)])) is None
    assert calls == []


def test_callback_can_save_and_manage_a_disjoint_derived_object(tmp_path):
    """Callbacks run outside short control locks while lifetime ownership remains safe."""

    store = DirStore(tmp_path / "store")
    repo = Repo((store,))
    value = CallbackValue(repo=repo)
    derived = DerivedValue(repo=repo)

    def save_derived(obj, context):
        assert derived.bump(managed=ManagedConfig(state_store=context.state_store)) is None

    value.run(managed=ManagedConfig(state_store=store, callbacks=[save_derived]))
    assert derived.value == 1
    assert derived.bump.status(state_store=store).state == "completed"


def test_callback_cannot_begin_an_overlapping_managed_operation(tmp_path):
    """Nested work sharing the active object's graph fails immediately, not by waiting."""

    store = DirStore(tmp_path / "store")
    value = CallbackValue(repo=Repo((store,)))
    errors = []

    def overlap(obj, context):
        with pytest.raises(ManagedConflictError) as caught:
            value.no_checkpoint(managed=ManagedConfig(state_store=context.state_store))
        errors.append(caught.value.reason)

    value.run(managed=ManagedConfig(state_store=store, callbacks=[overlap]))
    assert errors == ["state_graph_conflict"]


def test_request_arriving_during_callbacks_is_honored_at_that_checkpoint(tmp_path):
    """A callback-time request precedes the post-callback safe-point decision."""

    store = DirStore(tmp_path / "store")
    value = CallbackValue(repo=Repo((store,)))
    outcomes = []

    def request(obj, context):
        outcomes.append(obj.run.request_interrupt(state_store=context.state_store).outcome)

    with pytest.raises(ManagedInterrupted):
        value.run(managed=ManagedConfig(state_store=store, callbacks=[request]))
    assert outcomes == ["requested"]
    assert value.run.status(state_store=store).state == "interrupted"


def test_request_after_callback_decision_waits_for_the_next_checkpoint(tmp_path, monkeypatch):
    """A request after a no-interrupt decision cannot retroactively stop that save."""

    store = DirStore(tmp_path / "store")
    value = CallbackValue(repo=Repo((store,)))
    outcomes = []

    def request_after_decision(stage):
        if stage == "interruption_decided" and not outcomes:
            outcomes.append(value.twice.request_interrupt(state_store=store).outcome)

    monkeypatch.setattr(context_module, "_checkpoint_boundary", request_after_decision)
    with pytest.raises(ManagedInterrupted):
        value.twice(managed=ManagedConfig(state_store=store))
    status = value.twice.status(state_store=store)
    assert outcomes == ["requested"]
    assert (status.state, status.checkpoint_state_ref is not None) == ("interrupted", True)


def test_caught_local_interruption_remains_terminal_and_rejects_safe_points(tmp_path):
    """User code cannot catch a committed interruption and resume context mutation."""

    store = DirStore(tmp_path / "store")
    value = CallbackValue(repo=Repo((store,)))
    value.rejected_safe_points = []
    callbacks = []

    with pytest.raises(ManagedInterrupted):
        value.catch_terminal_interrupt(
            managed=ManagedConfig(state_store=store, callbacks=[lambda *args: callbacks.append(args)]),
        )
    status = value.catch_terminal_interrupt.status(state_store=store)
    assert value.rejected_safe_points == ["inactive_context", "inactive_context"]
    assert len(callbacks) == 1
    assert (status.state, status.final_state_ref) == ("interrupted", None)
