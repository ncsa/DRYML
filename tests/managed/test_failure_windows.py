"""U7 checkpoint and terminal-association failure-window coverage."""

from __future__ import annotations

import multiprocessing
import os
import shutil
import warnings

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedControlError, ManagedInterrupted, ManagedPublicationError, managed_operation
from dryml.managed.control import ManagedControlStore
from dryml.managed import context as context_module
from dryml.managed import runtime as runtime_module
from dryml.managed import storage as storage_module
from dryml.managed.errors import ManagedRecoveryError


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


class HookWindowValue(Pickleable):
    """Receiver proving failed lifecycle windows never deliver a completion handoff."""

    def __init__(self):
        """Initialize observable hook delivery count."""

        self.handoffs = 0

    @managed_operation(return_state_ref=True)
    def complete(self, *, managed) -> None:
        """Return normally so injected publication windows are reachable."""

    @managed_operation(return_state_ref=True)
    def fail(self, *, managed) -> None:
        """Fail in the workload body before final publication."""

        raise ValueError("body failed")

    @managed_operation(return_state_ref=True)
    def warn(self, *, managed) -> None:
        """Return an unwanted value to exercise warning-filter failure."""

        return object()

    @managed_operation(return_state_ref=True)
    def interrupt(self, *, managed) -> None:
        """Stop at a managed safe point before completion."""

        managed.interrupt()

    def _managed_post_publication(self, final_state, state_repo, report) -> None:
        """Count only successful post-completion delivery attempts."""

        del final_state, state_repo, report
        self.handoffs += 1


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
    original = ManagedControlStore._transition_running_fields
    calls = 0

    def fail_checkpoint(self, operation_id, **kwargs):
        nonlocal calls
        if kwargs["kind"] == "checkpoint" and calls == 0:
            calls += 1
            raise OSError("association failed")
        return original(self, operation_id, **kwargs)

    monkeypatch.setattr(ManagedControlStore, "_transition_running_fields", fail_checkpoint)
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
    original = ManagedControlStore._transition_running_fields
    transitions = 0
    callbacks = []

    def fail_second_association(self, operation_id, **kwargs):
        nonlocal transitions
        transitions += 1
        if transitions == 3:
            raise OSError("second association failed")
        return original(self, operation_id, **kwargs)

    monkeypatch.setattr(ManagedControlStore, "_transition_running_fields", fail_second_association)
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
    original = ManagedControlStore._transition_running_fields
    calls = 0

    def fail_terminal(self, operation_id, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interruption control write failed")
        return original(self, operation_id, **kwargs)

    monkeypatch.setattr(ManagedControlStore, "_transition_running_fields", fail_terminal)
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


def test_spawned_callback_window_recovery_does_not_replay_observers(
        tmp_path, monkeypatch, record_testsuite_property):
    """Characterize restore ownership across materialization and managed reentry.

    The current two owning boundaries each restore the selected receiver. R19's
    later cutover must strengthen this to one selected restore across the handoff;
    metadata/preparation/hash observations below are measurements, not budgets.
    """

    from tests.core.test_exact_load_metadata_evidence import exact_work_counts

    store, value = _terminate_at_boundary(tmp_path, "checkpoint_then_return", "callbacks_finished")
    calls = []
    hooks = []
    publications = []
    phase = "materialization"
    original_restore = CrashWindowValue.restore_state_from_dir_imp
    original_capture = CrashWindowValue.save_state_to_dir_imp
    original_publish = runtime_module._publish_managed_state_result

    def restore(obj, directory, *, codec):
        original_restore(obj, directory, codec=codec)
        hooks.append((phase, "restore", id(obj), obj.value))

    def capture(obj, directory, *, codec):
        hooks.append((phase, "capture", id(obj), obj.value))
        return original_capture(obj, directory, codec=codec)

    def publish(*args, **kwargs):
        result = original_publish(*args, **kwargs)
        publications.append(result)
        return result

    monkeypatch.setattr(CrashWindowValue, "restore_state_from_dir_imp", restore)
    monkeypatch.setattr(CrashWindowValue, "save_state_to_dir_imp", capture)
    monkeypatch.setattr(runtime_module, "_publish_managed_state_result", publish)
    recovered = Repo._for_state_io((store,))
    try:
        selected = value.checkpoint_then_return.status(state_repo=store).checkpoint_state_ref
        with exact_work_counts() as materialization:
            live = recovered.load_state_ref(selected, reuse_live="never")
        assert live.object_ref == value.object_ref and live.value == 1
        assert hooks == [("materialization", "restore", id(live), 1)]
        assert publications == []

        assert live.last_state_ref == selected
        phase = "managed"
        with exact_work_counts() as managed_entry:
            assert live.checkpoint_then_return(
                managed=ManagedConfig(state_repo=store, callbacks=[lambda *args: calls.append(args)]),
            ) == "resumed"
        assert hooks == [
            ("materialization", "restore", id(live), 1),
            ("managed", "restore", id(live), 1),
            ("managed", "capture", id(live), 1),
        ]
        assert live.value == 1
        status = live.checkpoint_then_return.status(state_repo=store)
        assert status.state == "completed"
        assert status.checkpoint_state_ref == selected
        assert len(publications) == 1
        assert publications[0].state_ref == status.final_state_ref
        assert publications[0].report is not None

        for stage, counts in (("materialization", materialization), ("managed", managed_entry)):
            record_testsuite_property(f"u01_{stage}_work", repr(dict(sorted(counts.items()))))
        record_testsuite_property("u01_hook_work", repr([
            (stage, hook, value) for stage, hook, _, value in hooks
        ]))
        record_testsuite_property("u01_publications", len(publications))
    finally:
        recovered.close(flush=False)
    assert calls == []


@pytest.mark.parametrize("window", ("body", "interruption", "warning", "publication", "validation", "association"))
def test_private_post_publication_hook_is_not_called_before_completed_association(
        tmp_path, monkeypatch, window,
):
    """Every pre-completion failure window suppresses the optional receiver hook."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = HookWindowValue(repo=repo)
    config = ManagedConfig(state_repo=repo)

    if window == "body":
        with pytest.raises(ValueError, match="body failed"):
            value.fail(managed=config)
    elif window == "interruption":
        with pytest.raises(ManagedInterrupted):
            value.interrupt(managed=config)
    elif window == "warning":
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(RuntimeWarning, match="discarded"):
                value.warn(managed=config)
    elif window == "publication":
        monkeypatch.setattr(
            runtime_module, "_publish_managed_state_result",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("publication failed")),
        )
        with pytest.raises(OSError, match="publication failed"):
            value.complete(managed=config)
    elif window == "validation":
        monkeypatch.setattr(
            storage_module, "_validate_publication_closure",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("validation failed")),
        )
        with pytest.raises(RepoSaveError) as caught:
            value.complete(managed=config)
        assert str(caught.value.__cause__.__cause__) == "validation failed"
        assert caught.value.report is not None
    else:
        original = ManagedControlStore._transition_running_fields
        calls = 0

        def fail_final_association(self, operation_id, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("association failed")
            return original(self, operation_id, **kwargs)

        monkeypatch.setattr(
            ManagedControlStore, "_transition_running_fields", fail_final_association,
        )
        with pytest.raises(OSError, match="association failed"):
            value.complete(managed=config)

    assert value.handoffs == 0


@pytest.mark.parametrize("member, stage", [("checkpoint_then_return", "state_published"), ("complete", "final_state_published")])
def test_deletion_at_publication_hook_fails_fresh_control_association(tmp_path, monkeypatch, member, stage):
    """An already-consumed event never substitutes for current control validation."""

    store = DirStore(tmp_path / "state")
    repo = Repo(store)
    value = CrashWindowValue(repo=repo)
    callbacks = []
    deleted = []

    def boundary(current):
        if current == stage:
            state_ref = value.last_state_ref
            shutil.rmtree(store._snapshot_path(state_ref.digest()))
            deleted.append(state_ref)

    monkeypatch.setattr(context_module, "_checkpoint_boundary", boundary)
    monkeypatch.setattr(runtime_module, "_completion_boundary", boundary)
    with pytest.raises(ManagedRecoveryError) as caught:
        getattr(value, member)(managed=ManagedConfig(
            state_repo=repo, callbacks=[lambda *args: callbacks.append(args)],
        ))
    assert len(deleted) == 1 and callbacks == []
    assert caught.value.report is not None
    status = getattr(value, member).status(state_repo=repo)
    assert status.state == "failed"
    assert status.checkpoint_state_ref is None and status.final_state_ref is None
    assert repo._topology_leases == 0


def test_later_status_and_invocation_revalidate_deleted_completed_state(tmp_path):
    """A successful event leaves no continuing authority for independent recovery."""

    store = DirStore(tmp_path / "state")
    repo = Repo(store)
    value = CrashWindowValue(repo=repo)
    assert value.complete(managed=ManagedConfig(state_repo=repo)) == "complete"
    shutil.rmtree(store._snapshot_path(value.last_state_ref.digest()))
    with pytest.raises(ManagedRecoveryError):
        value.complete.status(state_repo=repo)
    with pytest.raises(ManagedRecoveryError):
        value.complete(managed=ManagedConfig(state_repo=repo, rerun=True))
