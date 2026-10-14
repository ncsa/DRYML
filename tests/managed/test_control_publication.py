"""Fault, pending-intent, and concurrent bootstrap proof for U4 publication."""

from __future__ import annotations

import errno
import hashlib
import multiprocessing
import os
import shutil
import sys
from dataclasses import replace
from contextlib import contextmanager
from threading import Barrier, Event, Thread, current_thread
from uuid import uuid4

import pytest

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.reference_values import StateRef
from dryml.core.store.dir import DirStore
from dryml.managed import control as control_module
from dryml.managed.control import ControlSnapshot, ManagedControlStore
from dryml.managed.errors import ManagedConflictError, ManagedControlError, ManagedPublicationError, ManagedRecoveryError


def _operation_id(object_digest="b" * 64, member="run"):
    """Encode the U3 operation identity for closed-control test fixtures."""

    def atom(tag, payload):
        return len(tag).to_bytes(2, "big") + tag + len(payload).to_bytes(8, "big") + payload

    return hashlib.sha256(
        b"dryml-managed-operation-v1\x00"
        + atom(b"object-ref", object_digest.encode("ascii"))
        + atom(b"member", member.encode("utf-8"))
    ).hexdigest()


def _snapshot(*, operation=None, generation=1, owner=None):
    """Create a minimal running current record with no StateRef association."""

    return ControlSnapshot(operation or _operation_id(), "b" * 64, "c" * 64, "run", uuid4().hex, owner or uuid4().hex, generation, "running", None, None, None, None, {"version": 1, "store_keys": ["d" * 64], "object_keys": ["e" * 64]})


def _next(snapshot, *, generation=None, owner_id=None):
    """Build a same-attempt v2 replacement while retaining ownership evidence."""

    return replace(
        snapshot, generation=snapshot.generation + 1 if generation is None else generation,
        owner_id=snapshot.owner_id if owner_id is None else owner_id,
    )


def _bootstrap_worker(root, ready, start, queue):
    """Spawn target that races only managed namespace bootstrap publication."""

    store = DirStore(root)
    control = ManagedControlStore(store, Repo((store,)))
    ready.set()
    start.wait(10)
    try:
        control.initialize()
        queue.put("ok")
    except Exception as error:  # pragma: no cover - asserted through parent result.
        queue.put(type(error).__name__)


def _initial_worker(root, ready, start, queue):
    """Spawn target that races generation-one publication for one operation."""

    store = DirStore(root)
    control = ManagedControlStore(store, Repo((store,)))
    ready.set()
    start.wait(10)
    try:
        control.create_initial(_snapshot())
        queue.put("ok")
    except ManagedConflictError:
        queue.put("conflict")
    except Exception as error:  # pragma: no cover - asserted through parent result.
        queue.put(type(error).__name__)


def test_initial_operation_and_expected_transition_are_atomic(tmp_path):
    """Initial staging settles pending and stale transitions cannot overwrite current."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = _snapshot()
    assert control.create_initial(initial) == initial
    assert control.inspect(initial.operation_id) == initial
    proposed = _next(initial)
    assert control.transition(initial.operation_id, proposed, expected_generation=1, expected_attempt_id=initial.attempt_id, expected_owner_id=initial.owner_id) == proposed
    with pytest.raises(ManagedConflictError):
        control.transition(initial.operation_id, proposed, expected_generation=1)


def test_v2_ownership_evidence_has_a_closed_bounded_json_shape():
    """Current authority retains only bounded hashed Store/Object lock evidence."""

    snapshot = _snapshot()
    assert snapshot.to_data()["ownership"] == {
        "version": 1, "store_keys": ["d" * 64], "object_keys": ["e" * 64],
    }
    for ownership in (
        {"version": 1, "store_keys": ["d" * 64] * 257, "object_keys": ["e" * 64]},
        {"version": 1, "store_keys": ["d" * 64], "object_keys": ["e" * 64] * 4097},
        {"version": 1, "store_keys": [f"{index:064x}" for index in range(256)], "object_keys": [f"{index:064x}" for index in range(257)]},
    ):
        with pytest.raises(ManagedControlError, match="ownership topology"):
            ControlSnapshot(
                snapshot.operation_id, snapshot.object_ref_digest, snapshot.argument_digest,
                snapshot.member, snapshot.attempt_id, snapshot.owner_id, snapshot.generation,
                snapshot.state, snapshot.interrupt_request, snapshot.checkpoint_digest,
                snapshot.final_digest, snapshot.failure_code, ownership,
            )


@pytest.mark.parametrize("invalid_ancestor", ("operations", "v1", "shard"))
def test_operation_ancestor_files_are_never_absent_authority(tmp_path, invalid_ancestor):
    """Existing non-directory operation ancestors fail closed in every control path."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = _snapshot()
    operations = os.path.join(control.root, "operations")
    control.initialize()
    if invalid_ancestor == "operations":
        ancestor = operations
    elif invalid_ancestor == "v1":
        os.mkdir(operations)
        ancestor = os.path.join(operations, "v1")
    else:
        os.makedirs(os.path.join(operations, "v1"))
        ancestor = os.path.join(operations, "v1", initial.operation_id[:2])
    with open(ancestor, "xb"):
        pass

    proposed = _next(initial)
    for operation in (
        lambda: control.inspect(initial.operation_id),
        lambda: control.reconcile(initial.operation_id),
        lambda: control.create_initial(initial),
        lambda: control.transition(initial.operation_id, proposed, expected_generation=1),
    ):
        with pytest.raises(ManagedControlError, match="directory"):
            operation()


def test_absent_operation_hierarchy_is_not_started_without_bootstrap(tmp_path):
    """A valid gate with no operation directories remains read-only absent authority."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    control.initialize()
    operation_id = _operation_id()

    assert control.inspect(operation_id) is None
    assert control.reconcile(operation_id) is None
    assert not os.path.exists(os.path.join(control.root, "operations"))


def test_unreadable_operation_ancestor_is_a_typed_control_error(tmp_path, monkeypatch):
    """An unreadable operation ancestor is not inferred to be absent authority."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    control.initialize()
    operations = os.path.join(control.root, "operations")
    original_lstat = control_module.os.lstat

    def deny_operations(path):
        if path == operations:
            raise PermissionError("operation hierarchy is unreadable")
        return original_lstat(path)

    monkeypatch.setattr(control_module.os, "lstat", deny_operations)
    with pytest.raises(ManagedControlError, match="could not inspect"):
        control.inspect(_operation_id())


def test_replace_failure_preserves_old_authority_then_reconciles(tmp_path, monkeypatch):
    """A failure after intent publication blocks inspection and accepts only old current."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = control.create_initial(_snapshot())
    proposed = _next(initial)
    monkeypatch.setattr(control_module, "_replace_file", lambda path, payload: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(ManagedPublicationError) as caught:
        control.transition(initial.operation_id, proposed, expected_generation=1)
    assert caught.value.outcome == "indeterminate"
    with pytest.raises(ManagedControlError, match="pending"):
        control.inspect(initial.operation_id)
    assert control.reconcile(initial.operation_id) == initial


def test_short_write_and_initial_staging_failure_are_not_committed(tmp_path, monkeypatch):
    """Incomplete writes preserve prior authority and classify pre-publish failure."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    control.initialize()
    original_write = control_module._write_new
    monkeypatch.setattr(control_module, "_write_new", lambda path, payload: (_ for _ in ()).throw(OSError("short write")))
    with pytest.raises(ManagedPublicationError) as caught:
        control.create_initial(_snapshot())
    assert caught.value.outcome == "not_committed"
    monkeypatch.setattr(control_module, "_write_new", original_write)
    initial = control.create_initial(_snapshot())
    proposed = _next(initial)
    monkeypatch.setattr(control_module, "_replace_file", lambda path, payload: (_ for _ in ()).throw(OSError("short write")))
    with pytest.raises(ManagedPublicationError) as caught:
        control.transition(initial.operation_id, proposed, expected_generation=1)
    assert caught.value.outcome == "indeterminate"
    assert control.reconcile(initial.operation_id) == initial


def test_partial_pending_write_preserves_old_current_without_malformed_intent(tmp_path, monkeypatch):
    """An interrupted pending write never exposes a malformed authoritative intent."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = control.create_initial(_snapshot())
    proposed = _next(initial)
    monkeypatch.setattr(control_module.os, "write", lambda fd, payload: len(payload) - 1)
    with pytest.raises(ManagedPublicationError) as caught:
        control.transition(initial.operation_id, proposed, expected_generation=1)
    assert caught.value.outcome == "not_committed"
    assert not os.path.exists(control._pending_path(control._operation_path(initial.operation_id)))
    assert control.inspect(initial.operation_id) == initial


def test_pending_fsync_failure_preserves_old_current_with_a_typed_outcome(tmp_path, monkeypatch):
    """A portable temporary-file sync failure cannot expose a partial intent or current."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = control.create_initial(_snapshot())
    proposed = _next(initial)
    monkeypatch.setattr(control_module.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(ManagedPublicationError) as caught:
        control.transition(initial.operation_id, proposed, expected_generation=1)
    assert caught.value.outcome == "not_committed"
    assert control.inspect(initial.operation_id) == initial


def test_initial_reconciliation_syncs_current_with_write_handle_without_changing_bytes(tmp_path, monkeypatch):
    """Simulate Windows fsync requirements while retaining the exact initial bytes."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = _snapshot()
    current = control._current_path(control._operation_path(initial.operation_id))
    current_modes = {}
    synced_current_modes = []
    original_open = open
    original_fsync = control_module.os.fsync

    def windows_open(path, mode="r", *args, **kwargs):
        source = original_open(path, mode, *args, **kwargs)
        if os.fspath(path) == current:
            current_modes[source.fileno()] = mode
        return source

    def windows_fsync(fd):
        if fd in current_modes and "+" not in current_modes[fd]:
            raise OSError(errno.EBADF, "Windows fsync requires a write-capable handle")
        if fd in current_modes:
            synced_current_modes.append(current_modes.pop(fd))
        return original_fsync(fd)

    monkeypatch.setattr(control_module, "open", windows_open, raising=False)
    monkeypatch.setattr(control_module.os, "fsync", windows_fsync)
    assert control.create_initial(initial) == initial
    with open(current, "rb") as source:
        assert source.read() == initial.to_bytes()
    assert synced_current_modes == ["r+b"]
    assert control.reconcile(initial.operation_id) == initial


def test_replace_failure_after_current_swap_reconciles_exact_new_authority(tmp_path, monkeypatch):
    """A post-replace fault retains the complete intent and recovers only its new current."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = control.create_initial(_snapshot())
    proposed = _next(initial)
    original_replace = control_module._replace_file

    def replace_then_fail(path, payload):
        original_replace(path, payload)
        raise OSError("replace acknowledgement failed")

    monkeypatch.setattr(control_module, "_replace_file", replace_then_fail)
    with pytest.raises(ManagedPublicationError) as caught:
        control.transition(initial.operation_id, proposed, expected_generation=1)
    assert caught.value.outcome == "indeterminate"
    assert control.reconcile(initial.operation_id) == proposed


def test_initial_directory_publication_retains_pending_until_reconciliation(tmp_path, monkeypatch):
    """A fault after generation-one directory publication leaves recoverable intent evidence."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = _snapshot()
    operation = control._operation_path(initial.operation_id)
    operation_parent = os.path.dirname(operation)
    original_sync = control_module._sync_directory

    def fail_operation_parent(path):
        if path == operation_parent:
            raise OSError("operation directory acknowledgement failed")
        original_sync(path)

    monkeypatch.setattr(control_module, "_sync_directory", fail_operation_parent)
    with pytest.raises(ManagedPublicationError) as caught:
        control.create_initial(initial)
    assert caught.value.outcome == "indeterminate"
    with pytest.raises(ManagedControlError, match="pending"):
        control.inspect(initial.operation_id)
    monkeypatch.setattr(control_module, "_sync_directory", original_sync)
    assert control.reconcile(initial.operation_id) == initial


def test_ack_failure_and_new_pending_recovery_do_not_replay_or_guess(tmp_path, monkeypatch):
    """Acknowledgement uncertainty preserves exact new authority for reconciliation."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = control.create_initial(_snapshot())
    proposed = _next(initial)
    calls = [0]
    original_sync = control_module._sync_directory

    def fail_after_replace(path):
        calls[0] += 1
        if calls[0] == 2:
            raise OSError("directory acknowledgement failed")
        original_sync(path)

    monkeypatch.setattr(control_module, "_sync_directory", fail_after_replace)
    with pytest.raises(ManagedPublicationError) as caught:
        control.transition(initial.operation_id, proposed, expected_generation=1)
    assert caught.value.outcome == "indeterminate"
    monkeypatch.setattr(control_module, "_sync_directory", original_sync)
    assert control.reconcile(initial.operation_id) == proposed
    assert control.inspect(initial.operation_id) == proposed


def test_absent_pending_after_ack_failure_is_a_committed_exact_generation(tmp_path, monkeypatch):
    """A failed receipt cannot turn an already acknowledged current into uncertainty."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = control.create_initial(_snapshot())
    proposed = _next(initial)
    monkeypatch.setattr(control_module, "_publication_acknowledged", lambda: (_ for _ in ()).throw(OSError("receipt failed")))
    assert control.transition(initial.operation_id, proposed, expected_generation=1) == proposed
    assert control.inspect(initial.operation_id) == proposed


def test_recovered_pending_issues_a_completed_acknowledgement(tmp_path, monkeypatch):
    """Reconciliation recognizes an exact new current as committed after intent removal."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = control.create_initial(_snapshot())
    proposed = _next(initial)
    operation = control._operation_path(initial.operation_id)
    control._replace_file(control._current_path(operation), proposed.to_bytes())
    intent = control_module._PendingIntent(initial.operation_id, 1, control_module._payload_digest(initial.to_bytes()), 2, control_module._payload_digest(proposed.to_bytes()))
    control._write_new(control._pending_path(operation), intent.to_bytes())
    acknowledgements = []
    monkeypatch.setattr(control_module, "_publication_acknowledged", lambda: acknowledgements.append("committed"))
    assert control.reconcile(initial.operation_id) == proposed
    assert acknowledgements == ["committed"]


def test_inspection_holds_the_shared_control_lock_through_current_validation(tmp_path, monkeypatch):
    """Inspection cannot accept a generation while a writer changes pending/current authority."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = control.create_initial(_snapshot())
    proposed = _next(initial)
    entered, release, writer_done = Event(), Event(), Event()
    observed = []
    original_read_current = ManagedControlStore._read_current

    def delay_first_inspection(self, operation, operation_id):
        if self is control and not entered.is_set():
            entered.set()
            assert release.wait(10)
        return original_read_current(self, operation, operation_id)

    monkeypatch.setattr(ManagedControlStore, "_read_current", delay_first_inspection)
    inspector = Thread(target=lambda: observed.append(control.inspect(initial.operation_id)))

    def transition():
        try:
            ManagedControlStore(store, Repo((store,))).transition(initial.operation_id, proposed, expected_generation=1)
        finally:
            writer_done.set()

    inspector.start()
    assert entered.wait(10)
    writer = Thread(target=transition)
    writer.start()
    assert not writer_done.wait(0.2)
    release.set()
    inspector.join(10)
    writer.join(10)
    assert observed == [initial]
    assert writer_done.is_set()
    assert control.inspect(initial.operation_id) == proposed


def test_pending_neither_current_fails_closed_and_generation_overflow(tmp_path):
    """Reconciliation never chooses an unrelated current or wraps generation."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    initial = control.create_initial(_snapshot())
    operation = control._operation_path(initial.operation_id)
    intent = control_module._PendingIntent(initial.operation_id, 1, "d" * 64, 2, "e" * 64)
    control._write_new(os.path.join(operation, "pending.json"), intent.to_bytes())
    with pytest.raises(ManagedControlError, match="does not identify"):
        control.reconcile(initial.operation_id)
    other = DirStore(tmp_path / "other")
    control = ManagedControlStore(other, Repo((other,)))
    base = control.create_initial(_snapshot())
    overflow = _next(base, generation=2**63 - 1)
    control._replace_file(control._current_path(control._operation_path(base.operation_id)), overflow.to_bytes())
    with pytest.raises(ManagedControlError, match="overflow"):
        control.transition(overflow.operation_id, overflow, expected_generation=2**63 - 1)


def test_spawned_bootstrap_uses_writer_lock_and_one_final_gate(tmp_path):
    """Spawned peers synchronize bootstrap at a barrier without partial authority."""

    ctx = multiprocessing.get_context("spawn")
    ready_one, ready_two, start, queue = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Queue()
    root = os.fspath(tmp_path / "store")
    first = ctx.Process(target=_bootstrap_worker, args=(root, ready_one, start, queue))
    second = ctx.Process(target=_bootstrap_worker, args=(root, ready_two, start, queue))
    first.start()
    second.start()
    assert ready_one.wait(10) and ready_two.wait(10)
    start.set()
    first.join(20)
    second.join(20)
    assert first.exitcode == second.exitcode == 0
    assert sorted((queue.get(timeout=2), queue.get(timeout=2))) == ["ok", "ok"]
    store = DirStore(root)
    assert ManagedControlStore(store, Repo((store,))).inspect("a" * 64) is None


def test_spawned_initializers_publish_one_generation_one_lineage(tmp_path):
    """Spawned peers create one operation directory and one gets an explicit conflict."""

    ctx = multiprocessing.get_context("spawn")
    ready_one, ready_two, start, queue = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Queue()
    root = os.fspath(tmp_path / "store")
    first = ctx.Process(target=_initial_worker, args=(root, ready_one, start, queue))
    second = ctx.Process(target=_initial_worker, args=(root, ready_two, start, queue))
    first.start()
    second.start()
    assert ready_one.wait(10) and ready_two.wait(10)
    start.set()
    first.join(20)
    second.join(20)
    assert first.exitcode == second.exitcode == 0
    assert sorted((queue.get(timeout=2), queue.get(timeout=2))) == ["conflict", "ok"]
    store = DirStore(root)
    assert ManagedControlStore(store, Repo((store,))).inspect(_operation_id()).generation == 1


def test_concurrent_initial_operation_has_one_lineage_gate(tmp_path):
    """Threaded callers race one operation start without creating two generation ones."""

    store = DirStore(tmp_path / "store")
    barrier = Barrier(2)
    outcomes = []

    def create():
        control = ManagedControlStore(store, Repo((store,)))
        barrier.wait()
        try:
            outcomes.append(control.create_initial(_snapshot()))
        except ManagedConflictError:
            outcomes.append("conflict")

    first = Thread(target=create)
    second = Thread(target=create)
    first.start()
    second.start()
    first.join(10)
    second.join(10)
    assert sum(item == "conflict" for item in outcomes) == 1
    assert sum(isinstance(item, ControlSnapshot) for item in outcomes) == 1


def test_owner_transition_rereads_current_under_control_before_store_fences(tmp_path, monkeypatch):
    """Fresh generation, owner and request reads precede builder invocation under control."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    control = ManagedControlStore(store, repo)
    initial = control.create_initial(_snapshot())
    requested = replace(initial, generation=2, interrupt_request=(initial.attempt_id, initial.owner_id))
    control.transition(initial.operation_id, requested, expected_generation=1)
    held = []
    events = []
    original_lock = control._control_lock
    original_fences = repo._authority_read_fences
    original_read = control._read_current

    @contextmanager
    def lock():
        with original_lock():
            held.append("control")
            try:
                yield
            finally:
                held.pop()

    @contextmanager
    def fences(stores):
        assert held == ["control"]
        events.append("fences")
        with original_fences(stores):
            yield

    def read(*args):
        assert held == ["control"]
        events.append("current")
        return original_read(*args)

    def build(current):
        assert held == ["control"]
        assert current == requested
        events.append("build")
        return _next(current)

    monkeypatch.setattr(control, "_control_lock", lock)
    monkeypatch.setattr(repo, "_authority_read_fences", fences)
    monkeypatch.setattr(control, "_read_current", read)
    result = control.transition_running_owner(
        initial.operation_id, attempt_id=initial.attempt_id, owner_id=initial.owner_id, build=build,
    )
    assert result.generation == 3 and result.interrupt_request == requested.interrupt_request
    assert events == ["current", "build"]
    with pytest.raises(ManagedConflictError, match="owner_conflict"):
        control.transition_running_owner(
            initial.operation_id, attempt_id=initial.attempt_id, owner_id=uuid4().hex,
            build=lambda current: pytest.fail("stale owner builder ran"),
        )
    operation = control._operation_path(initial.operation_id)
    intent = control_module._PendingIntent(initial.operation_id, 3, control_module._payload_digest(result.to_bytes()), 4, "a" * 64)
    control._write_new(control._pending_path(operation), intent.to_bytes())
    with pytest.raises(ManagedControlError, match="pending_reconciliation"):
        control.transition_running_owner(
            initial.operation_id, attempt_id=initial.attempt_id, owner_id=initial.owner_id,
            build=lambda current: pytest.fail("pending builder ran"),
        )


class ControlValue(Pickleable):
    """Small persisted state for real control-reference authority tests."""

    def __init__(self, value=0):
        self.value = value


@pytest.fixture
def associated_control(tmp_path, fixed_managed_snapshot_environment):
    """Create a running owner with one real, already-associated checkpoint."""

    store = DirStore(tmp_path / "state")
    repo = Repo(store)
    value = ControlValue(repo=repo)
    checkpoint = repo.save_object(value, deep_capture=True)
    control = ManagedControlStore(store, repo)
    initial = replace(
        _snapshot(), operation_id=_operation_id(value.object_ref.digest()),
        object_ref_digest=value.object_ref.digest(), checkpoint_digest=checkpoint.digest(),
    )
    control.create_initial(initial)
    yield control, initial, value, checkpoint
    repo.close(flush=False)


def _owner_fields(control, current, **kwargs):
    return control._transition_running_fields(
        current.operation_id, attempt_id=current.attempt_id, owner_id=current.owner_id, **kwargs,
    )


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("kind", ["checkpoint", "completion"])
def test_closed_completion_shares_only_equal_references_in_one_control_first_cut(associated_control, monkeypatch, changed, kind):
    """Current/proposed checkpoint and identical final share; changed final remains distinct."""

    from dryml.core.materialization import _build_exact_state_load_plan

    control, initial, value, checkpoint = associated_control
    repo = control.state_repo
    if changed:
        value.value = 1
        final = repo.save_object(value, deep_capture=True)
    else:
        final = checkpoint
    calls, cuts, events = [], [], []
    held = []
    original_lock, original_cut = control._control_lock, repo._read_cut
    original_fences, original_read = repo._authority_read_fences, control._read_current

    @contextmanager
    def lock():
        with original_lock():
            held.append("control")
            try:
                yield
            finally:
                held.pop()

    def read_cut(**kwargs):
        assert held == ["control"]
        events.append("cut")
        cut = original_cut(**kwargs)
        cuts.append(cut)
        return cut

    @contextmanager
    def fences(stores):
        assert held == ["control"] and events == ["cut"]
        events.append("fences")
        with original_fences(stores):
            yield

    def read(*args):
        assert held == ["control"] and cuts[-1].active
        events.append("current")
        return original_read(*args)

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is _build_exact_state_load_plan.__code__:
            assert cuts[-1].active and held == ["control"]
            calls.append(frame.f_locals["state_ref"])

    monkeypatch.setattr(control, "_control_lock", lock)
    monkeypatch.setattr(repo, "_read_cut", read_cut)
    monkeypatch.setattr(repo, "_authority_read_fences", fences)
    monkeypatch.setattr(control, "_read_current", read)
    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        completed = _owner_fields(control, initial, kind=kind, digest=final.digest())
    finally:
        sys.setprofile(previous)
    assert calls == ([checkpoint, final] if changed else [checkpoint])
    assert events == ["cut", "fences", "current"]
    assert completed.generation == 2
    assert (completed.final_digest if kind == "completion" else completed.checkpoint_digest) == final.digest()
    assert len(cuts) == 1 and not cuts[0].active
    assert cuts[0].resolved_state_refs == {} and cuts[0].validated_state_refs == []
    assert repo._topology_leases == 0


def test_every_object_association_is_checked_even_after_a_closure_hit(associated_control):
    """A prior successful digest/closure match cannot authorize another receiver."""

    control, initial, _, _ = associated_control
    wrong_digest = "a" * 64
    wrong = replace(initial, operation_id=_operation_id(wrong_digest), object_ref_digest=wrong_digest)
    with control._control_lock(), control.state_repo._read_cut() as cut:
        control._validate_references(initial, _read_cut=cut)
        assert len(cut.validated_state_refs) == 1
        with pytest.raises(ManagedRecoveryError, match="state_object_mismatch"):
            control._validate_references(wrong, _read_cut=cut)


def test_control_closure_equality_does_not_accept_a_digest_collision(associated_control, monkeypatch):
    """A lookup-hash collision never substitutes a different complete StateRef."""

    from dryml.managed.storage import _validate_control_state_ref

    control, _, _, checkpoint = associated_control
    unequal = StateRef(checkpoint.object, {
        path: state.rsplit("-", 1)[0] + "-" + "0" * 64 for path, state in checkpoint.states.items()
    })
    with control.state_repo._read_cut() as cut:
        _validate_control_state_ref(control.state_repo, checkpoint, cut)
        monkeypatch.setattr(StateRef, "digest", lambda self: checkpoint._digest_cache)
        assert checkpoint != unequal and checkpoint.digest() == unequal.digest()
        with pytest.raises(ManagedRecoveryError):
            _validate_control_state_ref(control.state_repo, unequal, cut)
        assert cut.validated_state_refs == []


def test_control_ties_hash_based_public_equality_with_complete_content(associated_control, tmp_path, monkeypatch):
    """Colliding graph and reference hashes cannot merge unequal replica records."""

    from dryml.core.definition import ConcreteDefinition
    from dryml.core.reference_values import ObjectRef
    from dryml.core.store.records import StateRefRecord
    from dryml.managed.storage import _same_control_state_ref, _state_ref_for_digest

    control, _, _, checkpoint = associated_control
    other = ControlValue(2, repo=control.state_repo)
    unequal = StateRef(ObjectRef(other.definition, checkpoint.object.objects), checkpoint.states)
    equal = StateRef.from_data(checkpoint.to_data())
    assert equal is not checkpoint and equal == checkpoint and unequal != checkpoint
    digest = checkpoint.digest()
    monkeypatch.setattr(ConcreteDefinition, "stable_hash", lambda self: "a" * 64)
    monkeypatch.setattr(StateRef, "digest", lambda self: digest)
    assert unequal == checkpoint  # Public equality includes hash-based graph facts.
    assert _same_control_state_ref(equal, checkpoint)
    assert not _same_control_state_ref(unequal, checkpoint)

    second = DirStore(tmp_path / "replica")
    repo = Repo._for_state_io((control.control_store, second))
    try:
        with repo._read_cut() as cut:
            # Isolate resolution's collision tie-breaker from bundle validators.
            monkeypatch.setattr(cut, "snapshot", lambda store, key: (
                StateRefRecord(unequal if store is second else checkpoint), None, {},
            ))
            with pytest.raises(ManagedRecoveryError, match="one consistent retained StateRef"):
                _state_ref_for_digest(repo, digest, cut=cut)
            assert cut.resolved_state_refs == {}
    finally:
        repo.close(flush=False)


@pytest.mark.parametrize("outcome", ["mutate", "raise", "no_change"])
def test_arbitrary_builder_keeps_separate_validation_and_original_error_order(associated_control, outcome):
    """Caller builders run after current validation, without an inherited cut."""

    control, initial, _, checkpoint = associated_control
    error = KeyboardInterrupt("builder stopped")
    called = []

    def build(current):
        assert current == initial and control.state_repo._topology_leases == 0
        called.append(current)
        if outcome == "raise":
            raise error
        if outcome == "no_change":
            return None
        shutil.rmtree(control.control_store._snapshot_path(checkpoint.digest()))
        return _next(current)

    if outcome == "no_change":
        assert control.transition_running_owner(initial.operation_id, attempt_id=initial.attempt_id,
                                                owner_id=initial.owner_id, build=build) is None
    elif outcome == "raise":
        with pytest.raises(KeyboardInterrupt) as caught:
            control.transition_running_owner(initial.operation_id, attempt_id=initial.attempt_id,
                                             owner_id=initial.owner_id, build=build)
        assert caught.value is error
    else:
        with pytest.raises(ManagedRecoveryError):
            control.transition_running_owner(initial.operation_id, attempt_id=initial.attempt_id,
                                             owner_id=initial.owner_id, build=build)
    assert called == [initial]
    assert not os.path.exists(control._pending_path(control._operation_path(initial.operation_id)))


def test_overridden_reader_keeps_uncached_control_boundaries(associated_control, monkeypatch):
    """An overridden built-in reader is called at every original reference occurrence."""

    from dryml.core.materialization import _build_exact_state_load_plan

    control, initial, _, checkpoint = associated_control
    store = control.control_store
    original = store.read_state_ref_record
    reads = []

    def read(digest):
        reads.append(digest)
        return original(digest)

    monkeypatch.setattr(store, "read_state_ref_record", read)
    assert not store._can_reuse_exact_validation()
    plans = []
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is _build_exact_state_load_plan.__code__:
            plans.append(True)

    sys.setprofile(observe)
    try:
        _owner_fields(control, initial, kind="completion", digest=checkpoint.digest())
    finally:
        sys.setprofile(previous)
    assert len(plans) == 3
    assert reads.count(checkpoint.digest()) >= 3
    assert control.state_repo._topology_leases == 0


@pytest.mark.parametrize("reader", ["read_state_ref_record", "read_snapshot_metadata", "open_local_state"])
def test_control_fallback_preserves_overridden_reader_errors(associated_control, monkeypatch, reader):
    """Custom record, metadata and payload errors cannot be bypassed by field reuse."""

    control, initial, _, checkpoint = associated_control
    error = OSError("custom authority unavailable")

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(control.control_store, reader, unavailable)
    assert not control.control_store._can_reuse_exact_validation()
    with pytest.raises(ManagedRecoveryError) as caught:
        _owner_fields(control, initial, kind="completion", digest=checkpoint.digest())
    with pytest.raises(ManagedRecoveryError) as original:
        control.transition_running_owner(
            initial.operation_id, attempt_id=initial.attempt_id, owner_id=initial.owner_id,
            build=lambda current: pytest.fail("unreadable current reached builder"),
        )
    assert str(caught.value) == str(original.value)
    assert type(caught.value.__cause__) is type(original.value.__cause__)
    assert str(caught.value.__cause__) == str(original.value.__cause__)
    assert control.state_repo._topology_leases == 0
    assert not os.path.exists(control._pending_path(control._operation_path(initial.operation_id)))


@pytest.mark.parametrize("changed", ["request", "owner", "attempt", "pending", "overflow"])
def test_closed_updates_use_fresh_current_pending_owner_generation_and_request(associated_control, changed):
    """The invocation snapshot is never a substitute for locked current authority."""

    control, initial, _, _ = associated_control
    current = replace(initial, generation=2, interrupt_request=(initial.attempt_id, initial.owner_id))
    if changed == "owner":
        current = replace(current, owner_id=uuid4().hex, interrupt_request=None)
    elif changed == "attempt":
        current = replace(current, attempt_id=uuid4().hex, interrupt_request=None)
    elif changed == "overflow":
        current = replace(current, generation=2**63 - 1)
    operation = control._operation_path(initial.operation_id)
    control._replace_file(control._current_path(operation), current.to_bytes())
    if changed == "pending":
        intent = control_module._PendingIntent(initial.operation_id, 2, control_module._payload_digest(current.to_bytes()), 3, "a" * 64)
        control._write_new(control._pending_path(operation), intent.to_bytes())
    if changed == "request":
        interrupted = _owner_fields(control, initial, kind="interruption")
        assert interrupted.generation == 3 and interrupted.state == "interrupted"
        assert interrupted.checkpoint_digest == initial.checkpoint_digest
    else:
        reason = "owner_conflict" if changed in {"owner", "attempt"} else (
            "pending_reconciliation" if changed == "pending" else "generation_overflow")
        with pytest.raises((ManagedConflictError, ManagedControlError), match=reason):
            _owner_fields(control, initial, kind="interruption", force_interrupt=True)
        assert control._read_current(operation, initial.operation_id)[0] == current
    assert control.state_repo._topology_leases == 0


@pytest.mark.parametrize("delete", [False, True])
def test_acknowledgement_fault_retires_control_cut_before_fresh_confirmation(associated_control, monkeypatch, delete):
    """Even an acknowledged identical final must be revalidated after a receipt fault."""

    control, initial, _, checkpoint = associated_control
    cuts = []
    original_cut = control.state_repo._read_cut

    def read_cut(**kwargs):
        cut = original_cut(**kwargs)
        cuts.append(cut)
        return cut

    def fail_ack():
        assert cuts and all(not cut.active for cut in cuts)
        assert control.state_repo._topology_leases == 0
        if delete:
            shutil.rmtree(control.control_store._snapshot_path(checkpoint.digest()))
        raise KeyboardInterrupt("acknowledgement stopped")

    monkeypatch.setattr(control.state_repo, "_read_cut", read_cut)
    monkeypatch.setattr(control_module, "_publication_acknowledged", fail_ack)
    if delete:
        with pytest.raises(ManagedRecoveryError):
            _owner_fields(control, initial, kind="completion", digest=checkpoint.digest())
    else:
        assert _owner_fields(control, initial, kind="completion", digest=checkpoint.digest()).state == "completed"
        assert len(cuts) == 3  # transition, then independent checkpoint and final confirmation
    assert all(not cut.active for cut in cuts)


def test_cancelled_replacement_retires_cut_and_reconciliation_reads_fresh(associated_control, monkeypatch):
    """A pending replacement survives cancellation; old proof cannot heal deleted state."""

    control, initial, _, checkpoint = associated_control

    def cancel(*args):
        raise KeyboardInterrupt("replacement stopped")

    monkeypatch.setattr(control, "_replace_file", cancel)
    with pytest.raises(ManagedPublicationError, match="reconciliation") as caught:
        _owner_fields(control, initial, kind="completion", digest=checkpoint.digest())
    assert caught.value.outcome == "indeterminate"
    assert isinstance(caught.value.__cause__, KeyboardInterrupt)
    assert control.state_repo._topology_leases == 0
    shutil.rmtree(control.control_store._snapshot_path(checkpoint.digest()))
    with pytest.raises(ManagedRecoveryError):
        control.reconcile(initial.operation_id)
    assert os.path.exists(control._pending_path(control._operation_path(initial.operation_id)))


def test_cut_release_fault_after_current_replace_preserves_pending_uncertainty(associated_control, monkeypatch):
    """A Store-fence cleanup failure cannot misreport an associated current as uncommitted."""

    control, initial, _, checkpoint = associated_control
    repo = control.state_repo
    original_fences = repo._authority_read_fences
    error = OSError("Store-fence release failed")

    @contextmanager
    def fences(stores):
        with original_fences(stores):
            yield
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(repo, "_authority_read_fences", fences)
        with pytest.raises(ManagedPublicationError) as caught:
            _owner_fields(control, initial, kind="completion", digest=checkpoint.digest())
    assert caught.value.outcome == "indeterminate" and caught.value.__cause__ is error
    assert repo._topology_leases == 0
    with pytest.raises(ManagedControlError, match="pending_reconciliation"):
        control.inspect(initial.operation_id)
    settled = control.reconcile(initial.operation_id)
    assert settled.generation == 2 and settled.state == "completed"


def test_publisher_releases_store_fence_before_waiting_for_control(associated_control, monkeypatch, fixed_managed_snapshot_environment):
    """A barrier-controlled publisher/control overlap cannot invert control-before-Store."""

    from dryml.managed import storage

    control, initial, value, checkpoint = associated_control
    repo = control.state_repo
    publishing, control_held = Event(), Event()
    errors, results = [], []
    original_closure = storage._validate_publication_closure
    original_lock = control._control_lock

    @contextmanager
    def lock():
        with original_lock():
            if current_thread().name == "control-transition":
                assert publishing.is_set()
                control_held.set()
            yield

    def closure(*args):
        if current_thread().name == "publisher":
            assert args[2].active
            publishing.set()
            assert control_held.wait(10)
        return original_closure(*args)

    def publish():
        try:
            with storage._acquire_state_ownership(repo, value) as ownership:
                storage.publish_managed_state(
                    repo, value, reservation=ownership.reservation,
                    snapshot_observer=lambda: fixed_managed_snapshot_environment,
                )
                assert repo._topology_leases <= 2  # ownership plus the other thread's cut
                results.append(control.inspect(initial.operation_id).state)
        except BaseException as error:
            errors.append(error)

    def transition():
        try:
            results.append(_owner_fields(control, initial, kind="completion", digest=checkpoint.digest()).state)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(control, "_control_lock", lock)
    monkeypatch.setattr(storage, "_validate_publication_closure", closure)
    publisher = Thread(target=publish, name="publisher")
    controller = Thread(target=transition, name="control-transition")
    publisher.start()
    assert publishing.wait(10)
    controller.start()
    publisher.join(15)
    controller.join(15)
    assert not publisher.is_alive() and not controller.is_alive()
    assert errors == [] and results == ["completed", "completed"]
    assert repo._topology_leases == 0
