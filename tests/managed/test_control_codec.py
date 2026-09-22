"""Closed managed-current and namespace inspection tests for U4."""

from __future__ import annotations

import os
import hashlib
from uuid import uuid4

import pytest

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.managed import control as control_module
from dryml.managed.control import ControlSnapshot, ManagedControlStore
from dryml.managed.errors import ManagedControlError, ManagedRecoveryError


_OWNERSHIP = {"version": 1, "store_keys": ["d" * 64], "object_keys": ["e" * 64]}


def _operation_id(object_digest="b" * 64, member="run"):
    """Encode the U3 operation identity for closed-control test fixtures."""

    def atom(tag, payload):
        return len(tag).to_bytes(2, "big") + tag + len(payload).to_bytes(8, "big") + payload

    return hashlib.sha256(
        b"dryml-managed-operation-v1\x00"
        + atom(b"object-ref", object_digest.encode("ascii"))
        + atom(b"member", member.encode("utf-8"))
    ).hexdigest()


def _snapshot(*, generation=1, state="running", owner=True, checkpoint=None, final=None, failure=None):
    """Build one valid closed snapshot with independently selected terminal shape."""

    attempt = uuid4().hex
    owner_id = uuid4().hex if owner else None
    return ControlSnapshot(
        _operation_id(), "b" * 64, "c" * 64, "run", attempt, owner_id, generation,
        state, None, checkpoint, final, failure, _OWNERSHIP,
    )


class CodecValue(Serializable):
    """Small stateful value that supplies a real complete StateRef closure."""

    def __init__(self, value=1):
        self.value = value


def test_snapshot_round_trip_is_closed_and_duplicate_aware():
    """Codec rejects duplicate, unknown, oversized, and cross-field authority."""

    snapshot = _snapshot()
    assert ControlSnapshot.from_bytes(snapshot.to_bytes(), operation_id=snapshot.operation_id) == snapshot
    duplicate = snapshot.to_bytes().replace(b'"schema"', b'"schema":"x","schema"', 1)
    with pytest.raises(ManagedControlError):
        ControlSnapshot.from_bytes(duplicate)
    with pytest.raises(ManagedControlError):
        ControlSnapshot.from_bytes(snapshot.to_bytes()[:-1] + b',"extra":1}')
    with pytest.raises(ManagedControlError):
        ControlSnapshot.from_bytes(b"{" + b"x" * (64 * 1024) + b"}")
    with pytest.raises(ManagedControlError):
        _snapshot(state="completed", owner=False, final=None)
    with pytest.raises(ManagedControlError):
        _snapshot(generation=2**63)


def test_absent_inspection_is_read_only_and_bad_namespace_fails_closed(tmp_path):
    """Inspection never bootstraps absent control authority or masks bad gates."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    assert control.inspect("a" * 64) is None
    assert not os.path.exists(os.path.join(store.base_dir, "managed"))
    os.mkdir(os.path.join(store.base_dir, "managed"))
    with pytest.raises(ManagedRecoveryError, match="format"):
        control.inspect("a" * 64)


def test_existing_operation_without_current_never_restarts(tmp_path):
    """An operation directory is a lineage gate even if its current file vanished."""

    store = DirStore(tmp_path / "store")
    control = ManagedControlStore(store, Repo((store,)))
    control.initialize()
    path = os.path.join(control.root, "operations", "v1", "aa", "a" * 64)
    os.makedirs(path)
    with pytest.raises(ManagedRecoveryError, match="lacks current"):
        control.inspect("a" * 64)


def test_snapshot_path_identity_request_and_generation_are_checked():
    """Path, UUID, request tuple, and overflow fields cannot be forged loosely."""

    snapshot = _snapshot()
    with pytest.raises(ManagedControlError):
        ControlSnapshot.from_bytes(snapshot.to_bytes(), operation_id="d" * 64)
    with pytest.raises(ManagedControlError):
        ControlSnapshot(_operation_id(), "b" * 64, "c" * 64, "run", "not-a-uuid", None, 1, "interrupted", None, None, None, None, _OWNERSHIP)
    with pytest.raises(ManagedControlError):
        ControlSnapshot(_operation_id(), "b" * 64, "c" * 64, "run", uuid4().hex, uuid4().hex, 1, "running", (uuid4().hex, uuid4().hex), None, None, None, _OWNERSHIP)


def test_snapshot_rejects_malformed_json_state_and_wrong_operation_identity():
    """Supported JSON values produce typed errors and cannot forge a path identity."""

    snapshot = _snapshot()
    malformed_state = snapshot.to_bytes().replace(b'"state":"running"', b'"state":[]')
    with pytest.raises(ManagedControlError):
        ControlSnapshot.from_bytes(malformed_state)
    with pytest.raises(ManagedControlError):
        ControlSnapshot(_operation_id(), "b" * 64, "c" * 64, "run", uuid4().hex, uuid4().hex, 1, [], None, None, None, None, _OWNERSHIP)
    with pytest.raises(ManagedControlError, match="operation_id"):
        ControlSnapshot("a" * 64, "b" * 64, "c" * 64, "run", uuid4().hex, uuid4().hex, 1, "running", None, None, None, None, _OWNERSHIP)


def test_completed_current_requires_the_selected_exact_state_closure(tmp_path):
    """Completion control cannot be read after its exact StateRef authority is lost."""

    store = DirStore(tmp_path / "store")
    repo = Repo((store,))
    state = repo.save_object(CodecValue(repo=repo), deep_capture=True)
    snapshot = ControlSnapshot(
        _operation_id(state.object.digest()), state.object.digest(), "c" * 64, "run", uuid4().hex, None,
        1, "completed", None, None, state.digest(), None, _OWNERSHIP,
    )
    control = ManagedControlStore(store, repo)
    assert control.create_initial(snapshot) == snapshot
    (store.get_snapshot_directory(state) / "state-ref.record").unlink()
    with pytest.raises(ManagedRecoveryError, match="StateRef"):
        control.inspect(snapshot.operation_id)


def test_initial_and_reconciled_old_current_require_matching_state_object(tmp_path):
    """Control never accepts a checkpoint for a different receiver object graph."""

    store = DirStore(tmp_path / "store")
    repo = Repo((store,))
    first = repo.save_object(CodecValue(1), deep_capture=True)
    second = repo.save_object(CodecValue(2), deep_capture=True)
    control = ManagedControlStore(store, repo)
    mismatched = ControlSnapshot(
        _operation_id(first.object.digest()), first.object.digest(), "c" * 64, "run", uuid4().hex, uuid4().hex,
        1, "running", None, second.digest(), None, None, _OWNERSHIP,
    )
    with pytest.raises(ManagedRecoveryError, match="object"):
        control.create_initial(mismatched)
    assert not os.path.exists(control.root)

    initial = ControlSnapshot(
        _operation_id(first.object.digest()), first.object.digest(), "c" * 64, "run", uuid4().hex, uuid4().hex,
        1, "running", None, first.digest(), None, None, _OWNERSHIP,
    )
    assert control.create_initial(initial) == initial
    operation = control._operation_path(initial.operation_id)
    proposed = ControlSnapshot(
        initial.operation_id, initial.object_ref_digest, initial.argument_digest, initial.member, initial.attempt_id,
        initial.owner_id, 2, "running", None, initial.checkpoint_digest, None, None, _OWNERSHIP,
    )
    intent = control_module._PendingIntent(initial.operation_id, 1, control_module._payload_digest(initial.to_bytes()), 2, control_module._payload_digest(proposed.to_bytes()))
    control._write_new(control._pending_path(operation), intent.to_bytes())
    (store.get_snapshot_directory(first) / "state-ref.record").unlink()
    with pytest.raises(ManagedRecoveryError, match="StateRef"):
        control.reconcile(initial.operation_id)
