"""U7 checkpoint and terminal-association failure-window coverage."""

from __future__ import annotations

import multiprocessing
import os

import pytest

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedControlError, ManagedPublicationError, managed_operation
from dryml.managed.control import ManagedControlStore
from dryml.managed import context as context_module
from dryml.managed import runtime as runtime_module


class FailureWindowValue(Pickleable):
    """Receiver with one safe point for persistence fault injection."""

    @managed_operation()
    def checkpoint(self, *, managed):
        """Save one state at a controlled checkpoint boundary."""

        self.value = 1
        managed.checkpoint()

    def __init__(self):
        self.value = 0


class CrashWindowValue(Pickleable):
    """Spawn-safe receiver used to terminate real lifecycle publication windows."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation(resumable=True)
    def checkpoint_then_return(self, *, managed):
        """Checkpoint once on fresh entry and avoid a callback on resumed entry."""

        if managed.is_resuming:
            return "resumed"
        self.value = 1
        managed.checkpoint()
        return "fresh"

    @managed_operation(resumable=True)
    def interrupt(self, *, managed):
        """Create a checkpoint and terminal interruption in one explicit safe point."""

        self.value = 1
        managed.interrupt()

    @managed_operation()
    def complete(self, *, managed):
        """Finish without a checkpoint to isolate final-publication windows."""

        self.value = 2
        return "complete"


def _crash_window_worker(store_root, state_ref, member, boundary, reached, release):
    """Run one managed method until a deterministic publication barrier is reached."""

    store = DirStore(store_root)
    repo = Repo._for_state_io((store,))
    try:
        value = repo.load_state_ref(state_ref, reuse_live="never")

        def checkpoint_boundary(stage):
            if stage == boundary:
                reached.set()
                release.wait()

        def completion_boundary(stage):
            if stage == boundary:
                reached.set()
                release.wait()

        context_module._checkpoint_boundary = checkpoint_boundary
        runtime_module._completion_boundary = completion_boundary
        getattr(value, member)(managed=ManagedConfig(state_repo=store))
    finally:
        repo.close()


def _terminate_at_boundary(tmp_path, member, boundary):
    """Start a spawned lifecycle and terminate it only after its named barrier."""

    store = DirStore(tmp_path / "store")
    repo = Repo((store,))
    value = CrashWindowValue(repo=repo)
    seed = repo.save_object(value, store=store, deep_capture=True)
    ctx = multiprocessing.get_context("spawn")
    reached, release = ctx.Event(), ctx.Event()
    process = ctx.Process(
        target=_crash_window_worker,
        args=(os.fspath(store.base_dir), seed, member, boundary, reached, release),
    )
    process.start()
    assert reached.wait(20), f"worker did not reach {boundary}"
    process.terminate()
    process.join(20)
    assert process.exitcode is not None and process.exitcode != 0
    return store, value


def test_checkpoint_association_failure_never_advertises_or_notifies_orphan_state(tmp_path, monkeypatch):
    """A saved-but-unassociated StateRef cannot become managed recovery evidence."""

    store = DirStore(tmp_path / "store")
    value = FailureWindowValue(repo=Repo((store,)))
    original = ManagedControlStore.transition_running_owner
    calls = 0

    def fail_checkpoint(self, operation_id, **kwargs):
        nonlocal calls
        if kwargs["build"].__name__ == "<lambda>" and calls == 0:
            calls += 1
            raise OSError("association failed")
        return original(self, operation_id, **kwargs)

    monkeypatch.setattr(ManagedControlStore, "transition_running_owner", fail_checkpoint)
    with pytest.raises(OSError, match="association failed"):
        value.checkpoint(managed=ManagedConfig(state_repo=store))
    status = value.checkpoint.status(state_repo=store)
    assert (status.state, status.failure_code, status.checkpoint_state_ref) == ("failed", "publication_error", None)
    assert status.final_state_ref is None


def test_second_checkpoint_association_failure_retains_prior_checkpoint_without_callbacks(tmp_path, monkeypatch):
    """A failed replacement checkpoint cannot notify observers or discard recovery state."""

    class TwoCheckpoints(FailureWindowValue):
        """Receiver that has an already-associated checkpoint before the injected fault."""

        @managed_operation()
        def checkpoint(self, *, managed):
            """Associate one checkpoint, then attempt a second one that may fail."""

            self.value = 1
            managed.checkpoint()
            self.value = 2
            managed.checkpoint()

    store = DirStore(tmp_path / "store")
    value = TwoCheckpoints(repo=Repo((store,)))
    original = ManagedControlStore.transition_running_owner
    transitions = 0
    callbacks = []

    def fail_second_association(self, operation_id, **kwargs):
        nonlocal transitions
        transitions += 1
        if transitions == 3:
            raise OSError("second association failed")
        return original(self, operation_id, **kwargs)

    monkeypatch.setattr(ManagedControlStore, "transition_running_owner", fail_second_association)
    with pytest.raises(OSError, match="second association failed"):
        value.checkpoint(managed=ManagedConfig(state_repo=store, callbacks=[lambda *args: callbacks.append(args)]))
    status = value.checkpoint.status(state_repo=store)
    assert len(callbacks) == 1
    assert (status.state, status.failure_code, status.checkpoint_state_ref is not None) == (
        "failed", "publication_error", True,
    )


def test_checkpoint_save_failure_runs_no_callbacks_and_records_failure(tmp_path, monkeypatch):
    """State publication failure leaves no association and never starts observer delivery."""

    store = DirStore(tmp_path / "store")
    value = FailureWindowValue(repo=Repo((store,)))
    callbacks = []
    original = Repo.save_object

    def fail_checkpoint_save(self, *args, **kwargs):
        if kwargs.get("report_stores") is True:
            raise OSError("checkpoint state write failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Repo, "save_object", fail_checkpoint_save)
    with pytest.raises(OSError, match="checkpoint state write failed"):
        value.checkpoint(managed=ManagedConfig(state_repo=store, callbacks=[lambda *args: callbacks.append(args)]))
    status = value.checkpoint.status(state_repo=store)
    assert callbacks == []
    assert (status.state, status.checkpoint_state_ref) == ("failed", None)


def test_interrupted_transition_failure_is_control_failure_not_success(tmp_path, monkeypatch):
    """An explicit safe point cannot report interruption until its terminal write commits."""

    class Interrupting(FailureWindowValue):
        @managed_operation()
        def checkpoint(self, *, managed):
            self.value = 1
            managed.interrupt()

    store = DirStore(tmp_path / "store")
    value = Interrupting(repo=Repo((store,)))
    original = ManagedControlStore.transition_running_owner
    calls = 0

    def fail_terminal(self, operation_id, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interruption control write failed")
        return original(self, operation_id, **kwargs)

    monkeypatch.setattr(ManagedControlStore, "transition_running_owner", fail_terminal)
    with pytest.raises(ManagedControlError, match="interruption_recording_failed") as caught:
        value.checkpoint(managed=ManagedConfig(state_repo=store))
    assert type(caught.value.__cause__) is OSError
    status = value.checkpoint.status(state_repo=store)
    assert status.state == "failed"
    assert status.checkpoint_state_ref is not None


def test_indeterminate_checkpoint_association_is_not_replaced_by_generic_failure(tmp_path, monkeypatch):
    """Pending checkpoint authority remains recoverable rather than being overwritten as failed."""

    store = DirStore(tmp_path / "store")
    value = FailureWindowValue(repo=Repo((store,)))
    original = ManagedControlStore._clear_intent_and_acknowledge

    def leave_checkpoint_pending(self, operation, snapshot, digest):
        if snapshot.checkpoint_digest is not None and snapshot.state == "running":
            raise ManagedPublicationError("indeterminate", "checkpoint acknowledgement is uncertain")
        return original(self, operation, snapshot, digest)

    monkeypatch.setattr(ManagedControlStore, "_clear_intent_and_acknowledge", leave_checkpoint_pending)
    with pytest.raises(ManagedPublicationError) as caught:
        value.checkpoint(managed=ManagedConfig(state_repo=store))
    assert caught.value.outcome == "indeterminate"
    with pytest.raises(ManagedControlError, match="pending_reconciliation"):
        value.checkpoint.status(state_repo=store)
    monkeypatch.setattr(ManagedControlStore, "_clear_intent_and_acknowledge", original)
    from dryml.managed.identity import operation_digest

    ManagedControlStore(store, Repo((store,))).reconcile(operation_digest(value.object_ref, "checkpoint"))
    status = value.checkpoint.status(state_repo=store)
    assert (status.state, status.checkpoint_state_ref is not None) == ("failed", True)


@pytest.mark.parametrize(
    ("member", "boundary", "expected_checkpoint", "expected_final"),
    [
        ("checkpoint_then_return", "state_published", False, False),
        ("checkpoint_then_return", "checkpoint_associated", True, False),
        ("checkpoint_then_return", "callbacks_started", True, False),
        ("checkpoint_then_return", "callbacks_finished", True, False),
        ("interrupt", "interrupted_transition", True, False),
        ("complete", "final_state_published", False, False),
        ("complete", "final_associated", False, True),
    ],
)
def test_spawned_crash_preserves_only_associated_checkpoint_or_final_state(
        tmp_path, member, boundary, expected_checkpoint, expected_final,
):
    """Process death at every publication boundary never promotes orphan state authority."""

    store, value = _terminate_at_boundary(tmp_path, member, boundary)
    status = getattr(value, member).status(state_repo=store)
    assert (status.checkpoint_state_ref is not None) is expected_checkpoint
    assert (status.final_state_ref is not None) is expected_final
    if boundary == "interrupted_transition":
        assert status.state == "interrupted"
    elif expected_final:
        assert status.state == "completed"
    else:
        assert (status.state, status.failure_code) == ("failed", "owner_lost")


def test_spawned_callback_window_recovery_does_not_replay_observers(tmp_path):
    """A callback delivered before process death is not redelivered merely by resume."""

    store, value = _terminate_at_boundary(tmp_path, "checkpoint_then_return", "callbacks_finished")
    calls = []
    recovered = Repo._for_state_io((store,))
    try:
        live = recovered.load_state_ref(value.checkpoint_then_return.status(state_repo=store).checkpoint_state_ref, reuse_live="never")
        assert live.checkpoint_then_return(
            managed=ManagedConfig(state_repo=store, callbacks=[lambda *args: calls.append(args)]),
        ) == "resumed"
    finally:
        recovered.close()
    assert calls == []
