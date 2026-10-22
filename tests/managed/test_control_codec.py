"""Managed area2/current3/pending2 and nested Node/history/intent admission."""

from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4

import pytest

from dryml.core import Definition
from dryml.core.utils.graph.path import Parameter
from dryml.core.store.retention import RetentionResource
from dryml.managed.control import ControlHistory, ControlIntent, ControlSnapshot, _PendingIntent
from dryml.managed.errors import ManagedControlError
from tests.managed.test_u14_authority import Dataset, adapter, facts, receiver, running


def saved_receiver(control):
    dataset = Definition(Dataset).concretize()
    staging = control.control_store.create_local_state_staging()
    Path(staging, "data", "value").write_bytes(b"checkpoint")
    state = control.control_store.install_state(dataset, staging, codec="Test", codec_version=1, interpretation={}).state
    from dryml.core import ObjectId
    return receiver(dataset.with_identity(ObjectId()).with_state(state))


def snapshot_with_history(control):
    node = saved_receiver(control)
    identity = facts(node)
    with control.admission(identity) as owner:
        initial = running(identity, owner)
        history = ControlHistory(owner.run_id, owner.invocation_id, 1, "running", node)
        resource = RetentionResource(control.control_store.store_uuid, (node.at((Parameter("dataset"),)).state.key,))
        intent = ControlIntent(uuid4().hex, owner.run_id, owner.invocation_id, 1, "prepared", (resource,))
        return replace(initial, checkpoint=node, history=(history,), intents=(intent,))


def test_closed_current_roundtrip_and_lifecycle_cross_fields(tmp_path):
    control = adapter(tmp_path)
    snapshot = snapshot_with_history(control)
    assert json.loads(snapshot.to_bytes())["version"] == 3
    assert ControlSnapshot.from_bytes(snapshot.to_bytes()) == snapshot
    duplicate = snapshot.to_bytes().replace(b'"schema"', b'"schema":"bad","schema"', 1)
    with pytest.raises(ManagedControlError): ControlSnapshot.from_bytes(duplicate)
    for changes in ({"generation": True}, {"generation": 2**63}, {"run_id": "bad"},
                    {"state": "completed"}, {"failure_code": "secret arbitrary text"},
                    {"interrupt_request": (uuid4().hex, uuid4().hex)}):
        with pytest.raises(ManagedControlError): replace(snapshot, **changes)
    with pytest.raises(ManagedControlError):
        ControlSnapshot.from_bytes(snapshot.to_bytes(), operation_id="a" * 64)


def test_operation_digest_is_computed_once_per_snapshot_constructor(monkeypatch):
    from dryml.managed import control as module

    identity = facts(receiver())
    calls = []
    original = module.operation_digest

    def digest(node, member):
        calls.append((node, member))
        return original(node, member)

    monkeypatch.setattr(module, "operation_digest", digest)
    snapshot = ControlSnapshot(identity.receiver, identity.member, identity.argument_digest,
                               uuid4().hex, uuid4().hex, 1, "running")
    assert len(calls) == 1
    assert snapshot.operation_id == identity.operation_id
    payload = snapshot.to_bytes()
    assert snapshot.to_bytes() == payload and len(calls) == 1
    restored = ControlSnapshot.from_bytes(payload, operation_id=identity.operation_id)
    assert len(calls) == 2
    assert restored.to_bytes() == payload and restored == snapshot and len(calls) == 2
    replacement = replace(snapshot, member="other")
    assert len(calls) == 3
    assert replacement.operation_id == original(identity.receiver, "other") != snapshot.operation_id
    replacement.to_bytes()
    assert len(calls) == 3
    for record_mismatch in (False, True):
        data = json.loads(payload)
        if record_mismatch:
            data["operation_id"] = replacement.operation_id
        with pytest.raises(ManagedControlError, match="record/path identity"):
            ControlSnapshot.from_bytes(module._bytes(data), operation_id=replacement.operation_id)
    assert len(calls) == 5
    print("control: 5 constructors (2 rejected mismatches), 5 digest computations, byte-identical roundtrip")


@pytest.mark.parametrize("location", ["current", "receiver", "graph", "checkpoint", "history", "history_node", "intent", "resource"])
def test_nested_versions_reject_intact_bytes(tmp_path, location):
    control = adapter(tmp_path)
    snapshot = snapshot_with_history(control)
    data = snapshot.to_data()
    targets = {"current": data, "receiver": data["receiver"], "graph": data["receiver"]["graph"],
               "checkpoint": data["checkpoint"], "history": data["history"][0],
               "history_node": data["history"][0]["checkpoint"], "intent": data["intents"][0],
               "resource": data["intents"][0]["resources"][0]["keys"][0]}
    target = targets[location]
    target["identity_version" if location == "graph" else "version"] = 2 if location == "current" else 999
    payload = json.dumps(data).encode()
    path = Path(control.root, "incompatible-witness.json")
    path.write_bytes(payload)
    with pytest.raises(ManagedControlError):
        ControlSnapshot.from_bytes(path.read_bytes())
    assert path.read_bytes() == payload


def test_pending_v2_roundtrip_rejects_old_version_and_wrong_identity():
    intent = _PendingIntent("a" * 64, 1, "b" * 64, 2, "c" * 64, uuid4().hex)
    assert _PendingIntent.from_bytes(intent.to_bytes(), operation_id=intent.operation_id) == intent
    for version in (1, True, 3):
        data = json.loads(intent.to_bytes()); data["version"] = version
        with pytest.raises(ManagedControlError):
            _PendingIntent.from_bytes(json.dumps(data).encode(), operation_id=intent.operation_id)


def test_incompatible_area_rejected_by_mutating_bootstrap_without_changes(tmp_path):
    control = adapter(tmp_path)
    Path(control.root).mkdir()
    old = b'{"schema":"dryml-managed","version":1}'
    path = Path(control.root, "format.json"); path.write_bytes(old)
    with pytest.raises(ManagedControlError): control.initialize()
    assert path.read_bytes() == old
    assert tuple(Path(control.root).iterdir()) == (path,)
