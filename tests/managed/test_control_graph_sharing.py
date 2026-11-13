"""Control records share repeated graph blocks without changing exact CAS evidence."""

from dataclasses import replace
import json
from uuid import uuid4

import numpy as np
import pytest

from dryml.core import Definition, ObjectId
from dryml.core.state_values import LocalState, StateKey
from dryml.formats import canonical_json_bytes
from dryml.managed.control import ControlIntent, ControlSnapshot, StartPreparation, _bytes
from dryml.managed.control_codec import SCHEMA, ControlSizeError, decode_control
from dryml.managed.errors import ManagedControlError
from tests.managed.test_u14_authority import Dataset, Experiment, adapter, facts, receiver


def large_receiver():
    """A 43-row metadata-only recipe, with 120 KiB of coordinate payload."""
    layers = tuple(Definition(Experiment, dataset=index).concretize() for index in range(41))
    creation = Definition(Dataset, value={"coordinates": np.arange(30000, dtype="float32").reshape(10000, 3),
                                          "layers": layers}).concretize()
    return saved_recipe(creation)


def saved_recipe(creation):
    saved = creation.with_identity(ObjectId()).with_state(
        LocalState(StateKey(creation.creation_digest(), "Test-" + "a" * 64), creation))
    return receiver(saved)


def association(node, *, owner=None):
    run, invocation = (uuid4().hex, uuid4().hex) if owner is None else (owner.run_id, owner.invocation_id)
    root = ControlIntent(uuid4().hex, run, invocation, 1, "adopted", kind="latest", selection=node)
    previous = ControlSnapshot(node, "run", facts(node).argument_digest, run, invocation, 1,
                               "finalizing", checkpoint=node, intents=(root,))
    final = ControlIntent(uuid4().hex, run, invocation, 2, "adopted", kind="final", selection=node)
    proposed = replace(previous, generation=2, intents=(root, final))
    return StartPreparation(uuid4().hex, previous, proposed)


def test_final_association_shares_graph_blocks_below_one_mib(large_preparation):
    preparation = large_preparation
    expanded = canonical_json_bytes(preparation.to_data(), max_depth=96, max_nodes=65536,
                                    max_entries=16384, max_string=1024 * 1024, max_int_bits=64)
    assert len(expanded) > 1024 * 1024
    payload = preparation.to_bytes()
    assert len(payload) < 1024 * 1024
    assert len(payload) * 2 < len(expanded)
    restored = StartPreparation.from_bytes(payload, operation_id=preparation.proposed.operation_id)
    assert restored == preparation
    assert restored.to_bytes() == payload
    assert restored.previous.to_bytes() == preparation.previous.to_bytes()
    assert restored.proposed.to_bytes() == preparation.proposed.to_bytes()
    print(f"control sharing: expanded={len(expanded)}, wire={len(payload)}")


@pytest.fixture(scope="module")
def large_preparation():
    return association(large_receiver())


@pytest.fixture(scope="module")
def large_wire(large_preparation):
    return large_preparation.to_bytes()


def test_plain_current_and_legacy_preparation_remain_exact(large_preparation):
    previous = large_preparation.previous
    legacy = canonical_json_bytes(previous.to_data(), max_depth=96, max_nodes=65536,
                                  max_entries=16384, max_string=1024 * 1024, max_int_bits=64)
    assert len(legacy) < 1024 * 1024
    old = ControlSnapshot.from_bytes(legacy)
    preparation = replace(large_preparation, previous=old)
    wire = preparation.to_bytes()
    restored = StartPreparation.from_bytes(wire, operation_id=previous.operation_id)
    assert restored.previous.to_bytes() == legacy
    assert restored.proposed.to_bytes() == preparation.proposed.to_bytes()
    assert replace(restored, phase="ready").previous.to_bytes() == legacy

    # Old start v1 has ordinary inline snapshots, not packet descriptors.
    small = association(saved_recipe(Definition(Dataset).concretize()))
    legacy_start = canonical_json_bytes(small.to_data(), max_depth=96, max_nodes=65536,
        max_entries=16384, max_string=1024 * 1024, max_int_bits=64)
    decoded = StartPreparation.from_bytes(legacy_start, operation_id=small.proposed.operation_id)
    assert decoded == small and decoded.to_bytes() == legacy_start


def test_noncanonical_snapshot_bytes_are_preserved_and_cannot_be_substituted():
    source = association(saved_recipe(Definition(Dataset).concretize())).previous
    spelling = json.dumps(source.to_data(), indent=1).encode() + b"\n"
    previous = ControlSnapshot.from_bytes(spelling)
    preparation = StartPreparation(uuid4().hex, previous, replace(previous, generation=2))
    payload = preparation.to_bytes()
    restored = StartPreparation.from_bytes(payload, operation_id=source.operation_id)
    assert restored.previous.to_bytes() == spelling
    assert replace(restored, phase="ready").previous.to_bytes() == spelling
    changed = json.loads(payload)
    changed["previous"]["packet"]["generation"] = 9
    with pytest.raises(ManagedControlError, match="source differs"):
        StartPreparation.from_bytes(_bytes(changed), operation_id=source.operation_id)
    changed = json.loads(payload)
    changed["previous"]["source"] = "\ud800"
    with pytest.raises(ManagedControlError, match="UTF-8"):
        StartPreparation.from_bytes(json.dumps(changed).encode(), operation_id=source.operation_id)


def test_distinct_saved_states_are_not_coalesced_by_shared_recipe_blocks(large_preparation):
    original = large_preparation.proposed.receiver
    dataset = original.parameters["dataset"]
    changed = receiver(dataset.replace_bindings(states={"$": LocalState(
        StateKey(dataset.state.key.cdef_digest, "Test-" + "b" * 64), dataset.state.creation_cdef)}))
    final = replace(large_preparation.proposed.intents[-1], selection=changed)
    preparation = replace(large_preparation, proposed=replace(large_preparation.proposed,
                                                            intents=(*large_preparation.proposed.intents[:-1], final)))
    restored = StartPreparation.from_bytes(preparation.to_bytes(), operation_id=preparation.proposed.operation_id)
    assert restored.proposed.receiver.graph_equal(original)
    assert restored.proposed.intents[-1].selection.graph_equal(changed)
    assert not restored.proposed.intents[-1].selection.graph_equal(original)


@pytest.mark.parametrize("corruption", ("version", "extra", "missing", "reference", "unused", "content", "cycle", "duplicate"))
def test_malformed_pools_fail_closed_without_altering_bytes(large_preparation, large_wire, corruption, tmp_path):
    data = json.loads(large_wire)
    assert data["schema"] == SCHEMA
    key = next(iter(data["blocks"]))
    if corruption == "version":
        data["version"] = True
    elif corruption == "extra":
        data["extra"] = None
    elif corruption == "missing":
        del data["blocks"][key]
    elif corruption == "reference":
        values = data["record"]["previous"]["packet"]["blocks"]
        slot = next(key for key, value in values.items() if isinstance(value, str))
        values[slot] = "0" * 64
    elif corruption == "unused":
        data["blocks"]["f" * 64] = data["blocks"][key]
    elif corruption == "content":
        data["blocks"][key] = {"tag": "leaf", "value": "wrong block content"}
    elif corruption == "cycle":
        data["blocks"][key] = key
    payload = json.dumps(data).encode()
    if corruption == "duplicate":
        payload = payload.replace(b'"schema"', b'"schema":"duplicate","schema"', 1)
    path = tmp_path / "retained-control.json"
    path.write_bytes(payload)
    with pytest.raises(ManagedControlError):
        StartPreparation.from_bytes(path.read_bytes(), operation_id=large_preparation.proposed.operation_id)
    assert path.read_bytes() == payload


def test_limits_apply_to_unique_wire_bytes_and_expanded_reference_budget():
    with pytest.raises(ManagedControlError, match="snapshot_too_large"):
        _bytes({"data": ["x" * 600000, "y" * 600000]})
    key = "a" * 64
    bomb = {"schema": SCHEMA, "version": 1,
            "record": [{"format": "dryml-node-graph/v1", "values": {key: key}} for _ in range(20)],
            "blocks": {key: {"tag": "leaf", "value": "x" * 500000}}}
    payload = json.dumps(bomb).encode()
    assert len(payload) < 1024 * 1024
    with pytest.raises(ControlSizeError):
        decode_control(payload, max_bytes=1024 * 1024)


def test_same_key_different_block_content_is_rejected_before_pooling():
    key = "a" * 64
    data = [{"format": "dryml-node-graph/v1", "values": {key: {"tag": "leaf", "value": char * 140000}}}
            for char in "xy"]
    with pytest.raises(ManagedControlError, match="invalid_control"):
        _bytes(data)


@pytest.mark.parametrize("bound", ("depth", "nodes"))
def test_reference_expansion_retains_json_grammar_bounds(bound):
    key = "a" * 64
    graph = {"format": "dryml-node-graph/v1", "values": {key: key}}
    block = {"items": [0] * 400} if bound == "nodes" else {"value": 0}
    record = [graph] * 200 if bound == "nodes" else graph
    if bound == "depth":
        for _ in range(50):
            block, record = {"child": block}, {"child": record}
    wire = {"schema": SCHEMA, "version": 1, "record": record, "blocks": {key: block}}
    with pytest.raises(ValueError, match="grammar budget"):
        decode_control(json.dumps(wire).encode(), max_bytes=1024 * 1024)


def test_ready_large_association_recovers_exact_bytes_after_owner_exit(tmp_path, large_preparation, monkeypatch):
    from dryml.core.store.dir import DirStore
    from dryml.managed.results import ResultReference
    control = adapter(tmp_path)
    identity = facts(large_preparation.proposed.receiver)
    def forbidden(*args, **kwargs):
        pytest.fail("metadata association recovery opened private state")
    monkeypatch.setattr(DirStore, "read_state", forbidden)
    with control.admission(identity) as owner:
        preparation = association(identity.receiver, owner=owner)
        previous = replace(preparation.previous, result_reference=ResultReference("f" * 64))
        proposed = replace(preparation.proposed, result_reference=previous.result_reference)
        preparation = replace(preparation, previous=previous, proposed=proposed)
        control.create_initial(previous, owner=owner)
        recorded = control.write_preparation(preparation, owner=owner, expected=None)
        ready = control.write_preparation(replace(recorded, phase="ready"), owner=owner, expected=recorded)
        before = previous.to_bytes()
    with control.admission(identity) as recovery:
        retained = control.read_preparation(identity.operation_id)
        assert retained.to_bytes() == ready.to_bytes()
        assert retained.previous.to_bytes() == before == recovery.current.to_bytes()
        current = control.recover_final_association(retained, owner=recovery)
        assert current.to_bytes() == retained.proposed.to_bytes()
        assert current.invocation_id != recovery.invocation_id
        assert current.state == "finalizing"
        assert not control.control_store.pending_mutations()
    control.state_repo.close(flush=False)


def test_real_large_recipe_checkpoints_completion_sqlite_projection_and_replay(
        tmp_path, monkeypatch, fixed_managed_snapshot_environment):
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.core.query.authority import capture_store_projection
    from dryml.managed import ManagedConfig
    from dryml.managed.control import ManagedControlStore
    from tests.managed.u16_fixtures import Value

    store = DirStore(tmp_path / "state", query_index="sqlite")
    repo = Repo(store)
    live = Value(b"coordinate-metadata" * 7000, repo=repo)
    live.value = 0
    for counter in ("captures", "restores", "bodies"):
        monkeypatch.setattr(Value, counter, 0)
    def body(obj, context):
        context.checkpoint(initial=True)
        obj.value += 1
        return 7
    monkeypatch.setattr(Value, "action", body)
    original = ManagedControlStore.write_preparation
    sizes = []
    def observe(self, preparation, **kwargs):
        logical = canonical_json_bytes(preparation.to_data(), max_depth=96, max_nodes=65536,
            max_entries=16384, max_string=1024 * 1024, max_int_bits=64)
        sizes.append((len(logical), len(preparation.to_bytes())))
        return original(self, preparation, **kwargs)
    monkeypatch.setattr(ManagedControlStore, "write_preparation", observe)
    config = ManagedConfig(state_repo=repo)
    result = live.run(managed=config)
    assert result.value == 7
    assert (Value.bodies, Value.captures, Value.restores) == (1, 2, 0)
    assert max(expanded for expanded, _ in sizes) > 1024 * 1024
    assert max(wire for _, wire in sizes) < 1024 * 1024
    control = ManagedControlStore(store, repo)
    snapshot = control.inspect(result.execution.operation_id)
    assert snapshot.state == "completed"
    projection = capture_store_projection(store, rebuild=True)
    assert store._relative(control._preparation_path(snapshot.operation_id)) in projection.sources
    def forbidden(*args, **kwargs):
        pytest.fail("completion replay opened private state")
    monkeypatch.setattr(DirStore, "read_state", forbidden)
    assert live.run(managed=config).value == 7
    assert (Value.bodies, Value.captures, Value.restores) == (1, 2, 0)
    print(f"large managed workflow: expanded/wire max={max(sizes)}, metadata-only replay confirmed")
    repo.close(flush=False)
