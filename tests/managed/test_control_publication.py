"""Fault, pending-intent, and concurrent bootstrap proof for U4 publication."""

from __future__ import annotations

import multiprocessing
import os
import hashlib
from threading import Barrier, Event, Thread
from uuid import uuid4

import pytest

from dryml.core.store.dir import DirStore
from dryml.managed import control as control_module
from dryml.managed.control import ControlSnapshot, ManagedControlStore
from dryml.managed.errors import ManagedConflictError, ManagedControlError, ManagedPublicationError


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

    return ControlSnapshot(operation or _operation_id(), "b" * 64, "c" * 64, "run", uuid4().hex, owner or uuid4().hex, generation, "running", None, None, None, None)


def _bootstrap_worker(root, ready, start, queue):
    """Spawn target that races only managed namespace bootstrap publication."""

    store = DirStore(root)
    control = ManagedControlStore(store, store)
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
    control = ManagedControlStore(store, store)
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
    control = ManagedControlStore(store, store)
    initial = _snapshot()
    assert control.create_initial(initial) == initial
    assert control.inspect(initial.operation_id) == initial
    proposed = _snapshot(generation=2, owner=initial.owner_id)
    proposed = ControlSnapshot(initial.operation_id, initial.object_ref_digest, initial.argument_digest, initial.member, initial.attempt_id, initial.owner_id, 2, "running", None, None, None, None)
    assert control.transition(initial.operation_id, proposed, expected_generation=1, expected_attempt_id=initial.attempt_id, expected_owner_id=initial.owner_id) == proposed
    with pytest.raises(ManagedConflictError):
        control.transition(initial.operation_id, proposed, expected_generation=1)


def test_replace_failure_preserves_old_authority_then_reconciles(tmp_path, monkeypatch):
    """A failure after intent publication blocks inspection and accepts only old current."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, store)
    initial = control.create_initial(_snapshot())
    proposed = ControlSnapshot(initial.operation_id, initial.object_ref_digest, initial.argument_digest, initial.member, initial.attempt_id, initial.owner_id, 2, "running", None, None, None, None)
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
    control = ManagedControlStore(store, store)
    control.initialize()
    original_write = control_module._write_new
    monkeypatch.setattr(control_module, "_write_new", lambda path, payload: (_ for _ in ()).throw(OSError("short write")))
    with pytest.raises(ManagedPublicationError) as caught:
        control.create_initial(_snapshot())
    assert caught.value.outcome == "not_committed"
    monkeypatch.setattr(control_module, "_write_new", original_write)
    initial = control.create_initial(_snapshot())
    proposed = ControlSnapshot(initial.operation_id, initial.object_ref_digest, initial.argument_digest, initial.member, initial.attempt_id, initial.owner_id, 2, "running", None, None, None, None)
    monkeypatch.setattr(control_module, "_replace_file", lambda path, payload: (_ for _ in ()).throw(OSError("short write")))
    with pytest.raises(ManagedPublicationError) as caught:
        control.transition(initial.operation_id, proposed, expected_generation=1)
    assert caught.value.outcome == "indeterminate"
    assert control.reconcile(initial.operation_id) == initial


def test_partial_pending_write_preserves_old_current_without_malformed_intent(tmp_path, monkeypatch):
    """An interrupted pending write never exposes a malformed authoritative intent."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, store)
    initial = control.create_initial(_snapshot())
    proposed = ControlSnapshot(initial.operation_id, initial.object_ref_digest, initial.argument_digest, initial.member, initial.attempt_id, initial.owner_id, 2, "running", None, None, None, None)
    monkeypatch.setattr(control_module.os, "write", lambda fd, payload: len(payload) - 1)
    with pytest.raises(ManagedPublicationError) as caught:
        control.transition(initial.operation_id, proposed, expected_generation=1)
    assert caught.value.outcome == "not_committed"
    assert not os.path.exists(control._pending_path(control._operation_path(initial.operation_id)))
    assert control.inspect(initial.operation_id) == initial


def test_pending_fsync_failure_preserves_old_current_with_a_typed_outcome(tmp_path, monkeypatch):
    """A portable temporary-file sync failure cannot expose a partial intent or current."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, store)
    initial = control.create_initial(_snapshot())
    proposed = ControlSnapshot(initial.operation_id, initial.object_ref_digest, initial.argument_digest, initial.member, initial.attempt_id, initial.owner_id, 2, "running", None, None, None, None)
    monkeypatch.setattr(control_module.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(ManagedPublicationError) as caught:
        control.transition(initial.operation_id, proposed, expected_generation=1)
    assert caught.value.outcome == "not_committed"
    assert control.inspect(initial.operation_id) == initial


def test_replace_failure_after_current_swap_reconciles_exact_new_authority(tmp_path, monkeypatch):
    """A post-replace fault retains the complete intent and recovers only its new current."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, store)
    initial = control.create_initial(_snapshot())
    proposed = ControlSnapshot(initial.operation_id, initial.object_ref_digest, initial.argument_digest, initial.member, initial.attempt_id, initial.owner_id, 2, "running", None, None, None, None)
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
    control = ManagedControlStore(store, store)
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
    control = ManagedControlStore(store, store)
    initial = control.create_initial(_snapshot())
    proposed = ControlSnapshot(initial.operation_id, initial.object_ref_digest, initial.argument_digest, initial.member, initial.attempt_id, initial.owner_id, 2, "running", None, None, None, None)
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
    control = ManagedControlStore(store, store)
    initial = control.create_initial(_snapshot())
    proposed = ControlSnapshot(initial.operation_id, initial.object_ref_digest, initial.argument_digest, initial.member, initial.attempt_id, initial.owner_id, 2, "running", None, None, None, None)
    monkeypatch.setattr(control_module, "_publication_acknowledged", lambda: (_ for _ in ()).throw(OSError("receipt failed")))
    assert control.transition(initial.operation_id, proposed, expected_generation=1) == proposed
    assert control.inspect(initial.operation_id) == proposed


def test_recovered_pending_issues_a_completed_acknowledgement(tmp_path, monkeypatch):
    """Reconciliation recognizes an exact new current as committed after intent removal."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, store)
    initial = control.create_initial(_snapshot())
    proposed = ControlSnapshot(initial.operation_id, initial.object_ref_digest, initial.argument_digest, initial.member, initial.attempt_id, initial.owner_id, 2, "running", None, None, None, None)
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
    control = ManagedControlStore(store, store)
    initial = control.create_initial(_snapshot())
    proposed = ControlSnapshot(initial.operation_id, initial.object_ref_digest, initial.argument_digest, initial.member, initial.attempt_id, initial.owner_id, 2, "running", None, None, None, None)
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
            ManagedControlStore(store, store).transition(initial.operation_id, proposed, expected_generation=1)
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
    control = ManagedControlStore(store, store)
    initial = control.create_initial(_snapshot())
    operation = control._operation_path(initial.operation_id)
    intent = control_module._PendingIntent(initial.operation_id, 1, "d" * 64, 2, "e" * 64)
    control._write_new(os.path.join(operation, "pending.json"), intent.to_bytes())
    with pytest.raises(ManagedControlError, match="does not identify"):
        control.reconcile(initial.operation_id)
    other = DirStore(tmp_path / "other")
    control = ManagedControlStore(other, other)
    base = control.create_initial(_snapshot())
    overflow = ControlSnapshot(base.operation_id, base.object_ref_digest, base.argument_digest, base.member, base.attempt_id, base.owner_id, 2**63 - 1, "running", None, None, None, None)
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
    assert ManagedControlStore(store, store).inspect("a" * 64) is None


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
    assert ManagedControlStore(store, store).inspect(_operation_id()).generation == 1


def test_concurrent_initial_operation_has_one_lineage_gate(tmp_path):
    """Threaded callers race one operation start without creating two generation ones."""

    store = DirStore(tmp_path / "store")
    barrier = Barrier(2)
    outcomes = []

    def create():
        control = ManagedControlStore(store, store)
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
