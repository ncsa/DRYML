"""U7 checkpoint and terminal-association failure-window coverage."""

from __future__ import annotations

import multiprocessing
import os
import shutil
import warnings
from uuid import UUID

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


@pytest.mark.parametrize("boundary", ["validation", "restore", "association", "ready", "before_cas", "after_cas"])
def test_u15_preparation_failure_preserves_exact_authority_and_restart_never_executes(tmp_path, monkeypatch, boundary):
    """Each start cut retains exact intent evidence and terminates recovery-only."""
    from tests.managed.u15_fixtures import Value, enter, seed, setup
    from dryml.core.store.retention import RetentionAuthority
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, state="completed")
    live.value = 91
    original_admit = control.admit_prepared
    original_write = control.write_preparation
    original_adopt = RetentionAuthority.adopt
    original_prepare = control.state_repo.prepare_restore
    original_read = control.control_store.read_states
    def fail(*args, **kwargs):
        raise OSError("U15 injected " + boundary)
    if boundary == "validation":
        monkeypatch.setattr(control.control_store, "read_states", fail)
    elif boundary == "restore":
        monkeypatch.setattr(control.state_repo, "prepare_restore", fail)
    elif boundary == "association":
        monkeypatch.setattr(RetentionAuthority, "adopt", fail)
    elif boundary == "ready":
        def write(proposed, **kwargs):
            if proposed.phase == "ready":
                fail()
            return original_write(proposed, **kwargs)
        monkeypatch.setattr(control, "write_preparation", write)
    else:
        def admit(prepared, **kwargs):
            if boundary == "after_cas":
                original_admit(prepared, **kwargs)
            fail()
        monkeypatch.setattr(control, "admit_prepared", admit)
    with pytest.raises(Exception):
        enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))
    pending = control.read_preparation(old.operation_id)
    current = control.inspect(old.operation_id)
    assert current == (pending.proposed if boundary == "after_cas" else old)
    assert not pending.settled
    monkeypatch.setattr(control, "admit_prepared", original_admit)
    monkeypatch.setattr(control, "write_preparation", original_write)
    monkeypatch.setattr(RetentionAuthority, "adopt", original_adopt)
    monkeypatch.setattr(control.state_repo, "prepare_restore", original_prepare)
    monkeypatch.setattr(control.control_store, "read_states", original_read)
    counts = Value.restores, Value.captures, Value.bodies
    callbacks = []
    result = enter(control, live, config=ManagedConfig(callbacks=[lambda *args: callbacks.append(args)]),
                   realize=lambda _: pytest.fail("recovery realized"),
                   execute=lambda *args: pytest.fail("recovery executed"))
    assert result.disposition == ("run_admitted" if boundary == "after_cas" else "preparation_abandoned")
    assert (Value.restores, Value.captures, Value.bodies) == counts
    assert callbacks == []
    assert control.read_preparation(old.operation_id).settled
    ledger = control.control_store.retention.read(str(UUID(pending.token)))
    if ledger is not None:
        assert ledger.disposition == ("adopted" if boundary == "after_cas" else "released")


def test_u15_post_hook_failure_invalidates_live_but_preserves_old_current(tmp_path):
    from tests.managed.u15_fixtures import Value, enter, seed, setup
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, state="completed")
    Value.fail_restore = True
    try:
        with pytest.raises(Exception, match="fresh materialization"):
            enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))
    finally:
        Value.fail_restore = False
    assert live._restore_failed and control.inspect(old.operation_id) == old
    from dryml.managed.identity import AdmissionIdentity
    result = runtime_module.admit_execution(AdmissionIdentity(old.receiver, old.member, old.argument_digest),
        Value.run, ManagedConfig().snapshot(), control,
        realize=lambda _: pytest.fail("recovery must not require a usable live receiver"),
        execute=lambda *args: pytest.fail("recovery executed"))
    assert result.disposition == "preparation_abandoned"
    assert (Value.restores, Value.captures, Value.bodies) == (1, 0, 0)


def test_u15_interrupt_during_preparation_fences_next_admission_cas(tmp_path, monkeypatch):
    from tests.managed.u15_fixtures import Value, enter, seed, setup
    from dataclasses import replace
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, state="running")
    original = control.state_repo.prepare_restore
    def prepare(*args, **kwargs):
        outcome, requested = control.request_interrupt_if_running(old.operation_id)
        assert outcome == "requested" and requested.generation == old.generation + 1
        return original(*args, **kwargs)
    monkeypatch.setattr(control.state_repo, "prepare_restore", prepare)
    with pytest.raises(Exception, match="generation_conflict"):
        enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))
    interrupted = control.inspect(old.operation_id)
    assert interrupted.run_id == old.run_id and interrupted.interrupt_request is not None
    assert enter(control, live).disposition == "preparation_abandoned"
    assert Value.bodies == 0


def test_u15_interrupted_preparation_recovery_never_transfers_restore_receipt(tmp_path, monkeypatch):
    from tests.managed.u15_fixtures import Value, enter, seed, setup
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved)
    original = control.admit_prepared
    monkeypatch.setattr(control, "admit_prepared", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))
    token = control.read_preparation(old.operation_id).token
    assert Value.restores == 1
    monkeypatch.setattr(control, "admit_prepared", original)
    result = enter(control, live)
    assert result.disposition == "preparation_abandoned" and Value.restores == 1
    current, value, resuming = enter(control, live)
    assert current.run_id == old.run_id and resuming and value == 1
    assert Value.restores == 2 and control.read_preparation(old.operation_id).token != token


@pytest.mark.parametrize("after", [False, True])
def test_u15_uncertain_cas_keeps_both_fingerprints_and_reconciles_without_execution(tmp_path, monkeypatch, after):
    from tests.managed.u15_fixtures import Value, enter, seed, setup
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved)
    original = control._replace_file
    def publish(path, payload):
        if path.endswith("current.json"):
            if after:
                original(path, payload)
            raise OSError("CAS uncertain")
        return original(path, payload)
    monkeypatch.setattr(control, "_replace_file", publish)
    with pytest.raises(Exception):
        enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))
    preparation = control.read_preparation(old.operation_id)
    assert preparation.previous == old and preparation.phase == "ready"
    monkeypatch.setattr(control, "_replace_file", original)
    result = enter(control, live, realize=lambda _: pytest.fail("pending receiver work"),
        execute=lambda *args: pytest.fail("pending execution"))
    assert result.disposition == ("run_admitted" if after else "preparation_abandoned")
    assert result.current == (preparation.proposed if after else old)
    assert (Value.restores, Value.captures, Value.bodies) == (1, 0, 0)
    resumed, value, resuming = enter(control, live)
    assert resuming and value == 1 and resumed.run_id == result.current.run_id
    assert resumed.invocation_id not in {result.invocation_id, preparation.proposed.invocation_id}
    assert (Value.restores, Value.captures, Value.bodies) == (2, 0, 0)


def test_u15_incomplete_finalization_records_failure_and_returns_latest_without_hooks(tmp_path):
    from tests.managed.u15_fixtures import Value, enter, seed, setup
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, state="finalizing")
    result = enter(control, live, realize=lambda _: pytest.fail("finalizing realized"),
        execute=lambda *args: pytest.fail("finalizing executed"))
    assert result.disposition == "finalization_incomplete"
    assert result.current.run_id == old.run_id and result.current.state == "failed"
    assert result.current.checkpoint.graph_equal(saved)
    assert (Value.restores, Value.captures, Value.bodies) == (0, 0, 0)


def test_u15_interrupt_after_admission_preserves_new_run_and_blocks_execution(tmp_path, monkeypatch):
    from tests.managed.u15_fixtures import Value, enter, seed, setup
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, state="completed")
    original = control.admit_prepared
    def admit(*args, **kwargs):
        current = original(*args, **kwargs)
        assert control.request_interrupt_if_running(current.operation_id)[0] == "requested"
        return current
    monkeypatch.setattr(control, "admit_prepared", admit)
    with pytest.raises(Exception, match="interrupted"):
        enter(control, live, config=ManagedConfig(rerun=True, start_from=saved),
              execute=lambda *args: pytest.fail("interrupted admission executed"))
    current = control.inspect(old.operation_id)
    assert current.run_id != old.run_id and current.interrupt_request is not None
    assert current.checkpoint.graph_equal(saved)
    assert control.read_preparation(old.operation_id).settled
    assert (Value.restores, Value.captures, Value.bodies) == (1, 0, 0)


@pytest.mark.parametrize("failure", ["grant_error", "disconnected", "physical_garbage"])
def test_u15_only_unreferenced_physical_garbage_failure_is_a_cleanup_report(tmp_path, monkeypatch, failure):
    from tests.managed.u15_fixtures import Value, enter, setup
    from dryml.core.store.retention import RetentionAuthority, StateRemovalOutcome
    from dataclasses import replace
    control, live, saved = setup(tmp_path)
    first, _, _ = enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))
    from dryml.core.store.retention import grants
    for grant in grants(control.control_store, saved.state.key):
        if grant.token != str(UUID(first.intents[-1].token)):
            control.control_store.retention.release(grant.token, (control.control_store,))
    live.value = 2
    next_start = control.state_repo.save(live)
    original_release = RetentionAuthority.release
    def release(self, token, stores):
        if failure == "grant_error":
            raise OSError("grant removal failed")
        if failure == "disconnected":
            return replace(self.read(token), disposition="release-pending")
        return original_release(self, token, stores)
    monkeypatch.setattr(RetentionAuthority, "release", release)
    if failure == "physical_garbage":
        monkeypatch.setattr(control.control_store, "delete_state", lambda key:
            StateRemovalOutcome("cleanup-pending", cleanup_error=OSError("physical garbage removal failed")))
        reports = []
        enter(control, live, config=ManagedConfig(rerun=True, start_from=next_start),
              execute=lambda owner, *args: reports.append(owner.cleanup_report))
        assert len(reports[0]) == 1
        assert control.read_preparation(first.operation_id).settled
    else:
        with pytest.raises((OSError, ManagedRecoveryError)):
            enter(control, live, config=ManagedConfig(rerun=True, start_from=next_start),
                  execute=lambda *args: pytest.fail("unresolved release fell into execution"))
        assert not control.read_preparation(first.operation_id).settled
        assert control.inspect(first.operation_id).run_id != first.run_id


def _u15_prepared_worker(root, saved, reached, release):
    """Spawn target stopped after the one-shot restore but before current CAS."""
    from tests.managed.u15_fixtures import enter
    store = DirStore(root, query_index="none")
    repo = Repo(store)
    live = repo.load(saved)
    control = ManagedControlStore(store, repo)
    def prepared(*args, **kwargs):
        reached.set()
        release.recv()
        raise AssertionError("parent must terminate the prepared owner")
    control.admit_prepared = prepared
    enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))


def test_u15_spawned_prepared_owner_loss_reconciles_without_hooks_or_receipt(tmp_path):
    from tests.managed.u15_fixtures import Value, enter, seed, setup
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, state="completed")
    ctx = multiprocessing.get_context("spawn")
    reached = ctx.Event()
    release_reader, release_writer = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_u15_prepared_worker,
        args=(control.control_store.base_dir, saved, reached, release_reader))
    try:
        process.start()
        assert reached.wait(20)
        process.terminate()
        process.join(10)
        assert process.exitcode != 0
        prepared = control.read_preparation(old.operation_id)
        assert prepared.phase == "ready" and control.inspect(old.operation_id) == old
        recovered = enter(control, live, realize=lambda _: pytest.fail("restart receiver work"),
                          execute=lambda *args: pytest.fail("restart body work"))
        assert recovered.disposition == "preparation_abandoned"
        assert (Value.restores, Value.captures, Value.bodies) == (0, 0, 0)
        assert control.control_store.retention.read(str(UUID(prepared.token))).disposition == "released"
        fresh, value, resuming = enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))
        assert fresh.run_id != old.run_id and not resuming and Value.restores == 1
        assert control.read_preparation(old.operation_id).token != prepared.token
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        release_reader.close()
        release_writer.close()


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
