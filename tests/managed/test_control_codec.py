"""Managed area2/current3/pending2 and nested Node/history/intent admission."""

from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4

import pytest

from dryml.core import Definition
from dryml.core.utils.graph.path import Key, Parameter
from dryml.core.store.retention import RetentionResource
from dryml.managed.control import ControlHistory, ControlIntent, ControlSnapshot, StartPreparation, _PendingIntent
from dryml.managed.errors import ManagedControlError
from dryml.managed.results import ResultReference
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
    assert json.loads(snapshot.to_bytes())["version"] == 4
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


def test_unordered_receiver_roundtrip_preserves_current_cas_bytes(tmp_path):
    """Fresh unordered handles cannot change a persisted current-record witness."""
    from dryml.core import ObjectId

    control = adapter(tmp_path)
    children = {Definition(Dataset, value=index).concretize().with_identity(ObjectId())
                for index in range(9)}
    identity = facts(receiver(children))
    with control.admission(identity) as admitted:
        snapshot = running(identity, admitted)
        payload = snapshot.to_bytes()
        for _ in range(3):
            decoded = ControlSnapshot.from_bytes(payload, operation_id=identity.operation_id)
            assert decoded.to_bytes() == payload
            assert decoded == snapshot
            payload = decoded.to_bytes()
        control.create_initial(snapshot, owner=admitted)
        observed = control.inspect(identity.operation_id)
        proposed = replace(observed, generation=observed.generation + 1)
        control.transition(identity.operation_id, proposed, owner=admitted,
                            expected_generation=observed.generation,
                            expected_run_id=observed.run_id, expected_invocation_id=observed.invocation_id)


def unordered_saved_receiver():
    """Metadata-only saved graph with ties, cross-aliases and retained state pins."""
    from dryml.core import ObjectId, ObjectRef, StateRef, Serializable
    from dryml.core.cdef_graph import EdgeKind
    from dryml.core.links import DefLink
    from dryml.core.state_values import LocalState, StateKey
    from dryml.core.utils.graph.path import GraphPath
    from tests.managed.test_u14_authority import Experiment

    leaves = []
    for index in range(9):
        creation = Definition(Dataset, value=index).concretize()
        state = LocalState(StateKey(creation.creation_digest(), "Test-" + "a" * 64), creation)
        latest = LocalState(StateKey(creation.creation_digest(), "Test-" + "b" * 64), creation)
        leaves.append(creation.with_identity(ObjectId()).with_state(state).with_realization(states={"$": latest}))
    first, second = (Definition(Experiment, leaves[0]).concretize() for _ in range(2))
    reference = ObjectRef(Definition(Serializable).concretize(), {GraphPath(): ObjectId()})
    return receiver({"members": set(leaves + [first, second]), "external": leaves[0],
        "chosen": first, "ordered": first, "reference": DefLink.finalized(EdgeKind.REF, leaves[1]),
        "opaque": {"bytes": b"\x00\xff\x80", "tuple": (1, -0.0), "set": frozenset({3, 5})},
        "quoted": Definition(Dataset, value=[reference, StateRef(reference, {GraphPath(): "Test-" + "c" * 64})]).quote()})


def force_fresh_handle_order(monkeypatch):
    """Give each new handle a stable hash in a caller-selected allocation order."""
    from dryml.core.node import Node
    from dryml.core.definition import ConcreteNode

    hashes, direction = {}, [1]
    def allocated_hash(node):
        if node._view not in hashes:
            hashes[node._view] = direction[0] * (len(hashes) + 1)
        return hashes[node._view]
    monkeypatch.setattr(Node, "__hash__", allocated_hash)
    monkeypatch.setattr(ConcreteNode, "__hash__", allocated_hash)
    return lambda: direction.__setitem__(0, -direction[0])


def test_nested_unordered_wire_witness_is_lossless_and_replacement_discards_it(tmp_path, monkeypatch):
    flip = force_fresh_handle_order(monkeypatch)
    node = unordered_saved_receiver()
    control, identity = adapter(tmp_path), facts(node)
    with control.admission(identity) as owner:
        history = ControlHistory(owner.run_id, owner.invocation_id, 1, "completed", node, node, ResultReference("f" * 64))
        intent = ControlIntent(uuid4().hex, owner.run_id, owner.invocation_id, 1,
                               "adopted", kind="initial", selection=node)
        snapshot = replace(running(identity, owner), checkpoint=node, history=(history,), intents=(intent,))
        payload = snapshot.to_bytes()
        independent = []
        placement_changed = False
        for _ in range(3):
            flip()
            decoded = ControlSnapshot.from_bytes(payload)
            independent.append(decoded)
            assert decoded.to_bytes() == payload
            assert decoded.to_data() == json.loads(payload)
            assert decoded == snapshot and decoded.history == snapshot.history and decoded.intents == snapshot.intents
            detached = decoded.to_data()
            detached["generation"] = 99
            assert decoded.to_bytes() == payload
            encoded_copy = replace(decoded).to_bytes()
            placement_changed |= encoded_copy != payload
            assert ControlSnapshot.from_bytes(encoded_copy) == snapshot
            changed = replace(decoded, generation=2, checkpoint=None)
            assert changed._source_bytes is None
            assert json.loads(changed.to_bytes())["generation"] == 2
            assert json.loads(changed.to_bytes())["checkpoint"] is None
            external = decoded.receiver.parameters["dataset"]["external"]
            assert external.state.key.codec_hash == "Test-" + "b" * 64
            assert external.construction_view.state.key.codec_hash == "Test-" + "a" * 64
            details = decoded.receiver.parameters["dataset"]
            assert details["chosen"] in details["members"]
            assert details["chosen"].parameters["dataset"] is external
            assert details["opaque"]["bytes"] == b"\x00\xff\x80"
        assert placement_changed, "proof must exercise different authenticated packet placement"
        assert len({id(value.receiver) for value in independent}) == 3
        assert all(value.to_bytes() == payload for value in independent)
        completed = replace(snapshot, generation=2, state="completed", final=node,
                            producer_invocation_id=owner.invocation_id, result_reference=ResultReference("f" * 64))
        final_bytes = completed.to_bytes()
        flip()
        restored_final = ControlSnapshot.from_bytes(final_bytes)
        assert restored_final == completed and restored_final.to_bytes() == final_bytes
        assert restored_final.final.graph_equal(node)


def test_start_preparation_preserves_nested_source_and_resets_only_own_witness(tmp_path, monkeypatch):
    flip = force_fresh_handle_order(monkeypatch)
    control, identity = adapter(tmp_path), facts(unordered_saved_receiver())
    with control.admission(identity) as owner:
        initial = running(identity, owner)
        previous_bytes = initial.to_bytes()
        previous = ControlSnapshot.from_bytes(previous_bytes)
        proposed = replace(previous, generation=2, invocation_id=uuid4().hex)
        preparation = StartPreparation(uuid4().hex, previous, proposed)
        payload = preparation.to_bytes()
        for _ in range(3):
            flip()
            decoded = StartPreparation.from_bytes(payload, operation_id=identity.operation_id)
            assert decoded.to_bytes() == payload and decoded == preparation
            assert decoded.previous.to_bytes() == previous_bytes
            assert decoded.proposed.to_bytes() == proposed.to_bytes()
            ready = replace(decoded, phase="ready")
            assert ready._source_bytes is None
            assert ready.previous.to_bytes() == previous_bytes
            assert json.loads(ready.to_bytes())["phase"] == "ready"
            assert ready.to_data()["previous"] == json.loads(previous_bytes)


@pytest.mark.parametrize("difference", ["state", "identity", "alias", "quotation"])
def test_logical_snapshot_history_and_root_equality_reject_digest_collisions(monkeypatch, difference):
    from dryml.core import ObjectId
    from dryml.core.state_values import LocalState, StateKey
    from dryml.core import node_prepare
    from dryml.managed import control as module

    base = unordered_saved_receiver()
    path = (Parameter("dataset"),)
    details = base.parameters["dataset"]
    external = details["external"]
    if difference == "state":
        different = base.replace(path + (Key("external"),),
            external.replace_bindings(states={"$": LocalState(StateKey(external.state.key.cdef_digest,
                "Test-" + "d" * 64), external.state.creation_cdef)}))
    elif difference == "identity":
        different = base.replace(path + (Key("external"),),
                                 external.replace_bindings(identities={"$": ObjectId()}))
    else:
        different = base.lens(path).set(dict(details, **(
            {"chosen": next(value for value in details["members"] if value is not details["chosen"]
                and value.graph_equal(details["chosen"]))} if difference == "alias" else
            {"quoted": Definition(Dataset, value="different").quote()})))
    snapshot = ControlSnapshot(base, "run", "a" * 64, uuid4().hex, uuid4().hex, 1, "running")
    copied_snapshot, wrong_snapshot = replace(snapshot, receiver=base.copy_graph()), replace(snapshot, receiver=different)
    history = ControlHistory(snapshot.run_id, snapshot.invocation_id, 1, "running", base)
    copied_history, wrong_history = replace(history, checkpoint=base.copy_graph()), replace(history, checkpoint=different)
    intent = ControlIntent(uuid4().hex, snapshot.run_id, snapshot.invocation_id, 1,
                           "adopted", kind="initial", selection=base)
    copied_intent, wrong_intent = replace(intent, selection=base.copy_graph()), replace(intent, selection=different)
    monkeypatch.setattr(node_prepare, "_digest", lambda *args: "0" * 64)
    monkeypatch.setattr(module, "operation_digest", lambda *args: "0" * 64)
    snapshot, copied_snapshot, wrong_snapshot = (replace(value) for value in
        (snapshot, copied_snapshot, wrong_snapshot))
    assert snapshot.operation_id == copied_snapshot.operation_id == wrong_snapshot.operation_id == "0" * 64
    assert base.exact_key() == different.exact_key()
    assert snapshot == copied_snapshot and snapshot != wrong_snapshot
    assert history == copied_history and history != wrong_history
    assert intent == copied_intent and intent != wrong_intent


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


@pytest.mark.parametrize("field", ["old_current", "old_root", "result", "old_result", "result_ack", "provisional_ack"])
def test_u17_nested_result_and_publication_versions_refuse_without_mutation(tmp_path, field):
    from dryml.managed import ManagedConfig
    from tests.managed.u16_fixtures import Value, setup
    control, live = setup(tmp_path)
    Value.action = lambda obj, ctx: 1
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    snapshot = control.inspect(result.execution.operation_id)
    data = snapshot.to_data()
    if field == "old_current": data["version"] = 3
    elif field == "old_root": data["intents"][-1]["version"] = 2
    elif field == "result": data["result_reference"]["version"] = 99
    elif field == "old_result": data["result_reference"]["version"] = 1
    elif field == "result_ack": data["result_ack"]["version"] = 99
    else: data["intents"][-1]["publications"][0]["version"] = 99
    payload = json.dumps(data).encode()
    with pytest.raises(ManagedControlError): ControlSnapshot.from_bytes(payload)
    assert control.inspect(result.execution.operation_id).to_bytes() == snapshot.to_bytes()
