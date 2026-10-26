"""Closed durable-result grammar, alias, representation and malformed-input proofs."""

import json
import struct
from dataclasses import dataclass

import numpy as np
import pytest

from dryml.managed.result_codec import encode_result, decode_result
from dryml.managed.errors import ManagedResultError


def test_closed_grammar_preserves_tags_aliases_order_and_float_bits():
    nan = struct.unpack("!d", bytes.fromhex("7ff8000000000042"))[0]
    shared = [None, True, 2**180, -0.0, nan, float("inf"), complex(-0.0, nan), "s", b"x"]
    value = {"second": shared, "first": shared, (1, frozenset({2})): ({3}, (4,), frozenset({5}))}
    restored = decode_result(encode_result(value))
    assert list(restored) == list(value) and restored["first"] is restored["second"]
    for index in (3, 4, 5):
        assert struct.pack("!d", restored["first"][index]) == struct.pack("!d", shared[index])
    assert struct.pack("!dd", restored["first"][6].real, restored["first"][6].imag) == struct.pack("!dd", shared[6].real, shared[6].imag)
    assert tuple(type(v) for v in restored[(1, frozenset({2}))]) == (set, tuple, frozenset)


@pytest.mark.parametrize("dtype", [">f8", "<c16", "datetime64[ns]", "U3", np.dtype([("x", ">i4"), ("y", "<f8")])])
def test_numpy_logical_representation_and_aliases(dtype):
    from dryml.core.node_codec import _logical_bytes
    array = np.zeros((3, 4), dtype=dtype)[:, ::2]
    restored = decode_result(encode_result([array, array, array[0, 0]]))
    assert restored[0] is restored[1]
    assert restored[0].dtype == array.dtype and restored[0].shape == array.shape
    assert _logical_bytes(restored[0]) == _logical_bytes(array)
    assert type(restored[2]) is type(array[0, 0]) and restored[2].tobytes() == array[0, 0].tobytes()


def test_explicit_metadata_is_inert_and_keeps_retained_construction(tmp_path, monkeypatch):
    from tests.managed.u16_fixtures import setup
    from dryml.core.state_values import IdentitySpec, StateSpec
    control, live = setup(tmp_path)
    saved = control.state_repo.save(live)
    def forbidden(*args, **kwargs): pytest.fail("codec materialized metadata")
    monkeypatch.setattr(control.state_repo, "load", forbidden)
    values = [saved, saved.state, saved.state.key, saved.identity_spec(), saved.state_spec()]
    restored = decode_result(encode_result(values))
    from dryml.managed.control import _node_equal
    assert _node_equal(restored[0], saved)
    assert restored[1] == saved.state and restored[2] == saved.state.key
    assert type(restored[3]) is IdentitySpec and type(restored[4]) is StateSpec
    assert _node_equal(restored[3].root, saved) and _node_equal(restored[4].root, saved)


@dataclass(frozen=True)
class Summary:
    value: int


class Sublist(list):
    pass


@pytest.mark.parametrize("value", [Summary(1), Sublist([1]), object(), lambda: 1, (v for v in ()), np.array([object()]), np.ma.array([1])])
def test_unsupported_values_have_no_fallback(value):
    with pytest.raises(ManagedResultError): encode_result(value)


def test_cycles_and_budget_fail_deterministically():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ManagedResultError): encode_result(cyclic)
    with pytest.raises(ManagedResultError): encode_result(b"x" * 100, max_bytes=32)
    with pytest.raises(ManagedResultError): decode_result(b"x" * 100, max_bytes=32)


@pytest.mark.parametrize("change", ["version", "tag", "ref", "cycle", "extra", "unreachable"])
def test_malformed_packets_fail_closed(change):
    packet = json.loads(encode_result([1]))
    if change == "version": packet["version"] = 99
    elif change == "tag": packet["nodes"][0] = ["opaque", "ignored"]
    elif change == "ref": packet["root"] = 999
    elif change == "cycle": packet["nodes"] = [["list", [0]]]; packet["root"] = 0
    elif change == "extra": packet["extra"] = True
    else: packet["nodes"].append(["none"])
    with pytest.raises(ManagedResultError): decode_result(json.dumps(packet).encode())


def test_unordered_metadata_keeps_independent_equal_nodes_aliases_quotes_and_construction(monkeypatch):
    from tests.managed.test_control_codec import unordered_saved_receiver, force_fresh_handle_order
    from dryml.managed.control import _node_equal
    flip = force_fresh_handle_order(monkeypatch)
    node = unordered_saved_receiver()
    packet = encode_result([node, node])
    flip()
    copied = decode_result(packet)
    assert copied[0] is copied[1] and _node_equal(copied[0], node)
    details = copied[0].parameters["dataset"]
    assert details["external"] is details["chosen"].parameters["dataset"]
    equal = [v for v in details["members"] if v.graph_equal(details["chosen"])]
    assert len(equal) == 2 and equal[0] is not equal[1]
    assert details["external"].state != details["external"].construction_view.state


def related_metadata_result():
    """Return related roots, an independent equal root, specs and container aliases."""
    from dryml.core import Definition
    from tests.managed.test_control_codec import unordered_saved_receiver
    from tests.managed.test_u14_authority import Experiment
    parent = unordered_saved_receiver()
    details = parent.parameters["dataset"]
    child = details["external"]
    independent = next(v for v in details["members"] if v is not details["chosen"]
                       and v.graph_equal(details["chosen"]))
    roots = [parent, child, details["chosen"], Definition(Experiment, child).concretize(),
             independent, parent.identity_spec(), parent.state_spec(), child.construction_view,
             child.state, child.state.key, child.object_id, details["quoted"].value,
             details["reference"].target]
    return {"roots": roots, "again": roots}


def assert_related_metadata_result(value):
    """Check cross-root occurrences, exact facets, roles and retained construction."""
    (parent, child, chosen, second_parent, independent, identities, states,
     construction, state, key, identity, quoted, referenced) = value["roots"]
    assert value["again"] is value["roots"]
    details = parent.parameters["dataset"]
    assert details["external"] is child
    assert details["chosen"] is chosen
    assert chosen.parameters["dataset"] is second_parent.parameters["dataset"] is child
    assert independent.graph_equal(chosen) and independent is not chosen
    assert independent.parameters["dataset"] is child
    assert identities.root is states.root is parent
    assert construction is child.construction_view
    assert state == child.state and key == child.state.key and identity == child.object_id
    assert child.state != child.construction_view.state
    assert details["reference"].target is referenced
    assert any(v is referenced for v in details["members"])
    assert details["quoted"].value is quoted


def test_related_metadata_roots_share_occurrences_and_recompose_exactly(monkeypatch):
    from dryml.core import Definition
    from tests.managed.test_control_codec import force_fresh_handle_order
    from tests.managed.test_u14_authority import Experiment
    flip = force_fresh_handle_order(monkeypatch)
    value = related_metadata_result()
    payload = encode_result(value)
    flip()
    restored = decode_result(payload)
    assert_related_metadata_result(restored)
    assert Definition(Experiment, value["roots"][:5]).graph_equal(Definition(Experiment, restored["roots"][:5]))


def test_metadata_is_encoded_and_decoded_once_for_all_result_views(monkeypatch):
    """Result transport owns one table; quotation hashes are a separate codec use."""
    import sys
    from dryml.core import node_codec
    value = related_metadata_result()
    calls = []
    original_encode, original_decode = node_codec.encode_node_graph, node_codec._decode_node_graph_views
    def encode(prepared, *args, **kwargs):
        if sys._getframe(1).f_globals["__name__"] == "dryml.managed.result_codec":
            calls.append(("encode", len(prepared.roots)))
        return original_encode(prepared, *args, **kwargs)
    def decode(graph, *args, **kwargs):
        calls.append(("decode", len(graph["views"])))
        return original_decode(graph, *args, **kwargs)
    monkeypatch.setattr(node_codec, "encode_node_graph", encode)
    monkeypatch.setattr(node_codec, "_decode_node_graph_views", decode)
    payload = encode_result(value)
    packet = json.loads(payload)
    rows = [row for row in packet["nodes"] if row[0] == "metadata"]
    assert rows == [["metadata", index] for index in range(len(rows))]
    assert len(packet["metadata"]["tables"]) == 1
    restored = decode_result(payload)
    assert calls == [("encode", len(rows)), ("decode", len(rows))]
    assert_related_metadata_result(restored)


@pytest.mark.parametrize("change", ["old_result", "old_graph", "missing_graph", "shared_blocks",
    "block_hash", "unused_block", "missing_block", "view_kind", "view_key", "view_table",
    "view_slot", "bool_slot", "duplicate_view", "unadvertised_root", "result_view", "bool_view",
    "unused_view", "broader_metadata", "wrapper_facets", "extra_table"])
def test_malformed_shared_metadata_refuses_before_resources_or_materialization(monkeypatch, change):
    from dryml.core import Definition, ObjectId, Repo
    from dryml.core import node_codec, node_prepare, symbol
    from dryml.core.store.dir import DirStore
    from tests.managed.test_u14_authority import Dataset
    node = Definition(Dataset).concretize()
    packet = json.loads(encode_result([node, Definition(Dataset, value=2).concretize()]))
    graph = packet["metadata"]
    row = next(row for row in packet["nodes"] if row[0] == "metadata")
    if change == "old_result": packet["version"] = 1
    elif change == "old_graph": graph["identity_version"] = 2
    elif change == "missing_graph": packet["metadata"] = None
    elif change == "shared_blocks": graph["tables"] = []
    elif change == "block_hash": next(iter(graph["values"].values()))["extra"] = True
    elif change == "unused_block":
        record = {"tag": "str", "value": "unused"}
        graph["values"][node_prepare._digest("dryml-node-values/v1", record)] = record
    elif change == "missing_block": graph["values"].pop(next(iter(graph["values"])))
    elif change == "view_kind": graph["views"][0]["view"] = "creation"
    elif change == "view_key": graph["views"][0]["expected_key"] = "forged"
    elif change == "view_table": graph["views"][0]["table"] = "0" * 64
    elif change == "view_slot": graph["views"][0]["slot"] = 99999
    elif change == "bool_slot": graph["views"][0]["slot"] = False
    elif change == "duplicate_view": graph["views"].append(dict(graph["views"][0]))
    elif change == "unadvertised_root": graph["views"].pop()
    elif change == "result_view": row[1] = 99999
    elif change == "bool_view": row[1] = False
    elif change == "unused_view": row[:] = ["none"]
    elif change in {"broader_metadata", "wrapper_facets"}:
        wrapper = Definition._from_prepared_parameters(None, {"value": (1,) if change == "broader_metadata" else node})
        if change == "wrapper_facets": wrapper = wrapper.with_identity(ObjectId())
        packet = json.loads(encode_result(node))
        packet["metadata"] = node_codec.encode_node_graph(node_prepare.NodePreparation((wrapper,)))
    else: graph["tables"]["0" * 64] = next(iter(graph["tables"].values()))
    payload = json.dumps(packet).encode()
    def forbidden(*args, **kwargs): pytest.fail("result decoder opened a resource or materialized metadata")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(DirStore, "__init__", forbidden)
    monkeypatch.setattr(Repo, "load", forbidden)
    monkeypatch.setattr(symbol, "resolve_symbol", forbidden)
    with pytest.raises(ManagedResultError): decode_result(payload)


def test_result_and_shared_metadata_bounds_remain_closed(monkeypatch):
    from dryml.managed import result_codec as codec
    from dryml.core import Definition
    from tests.managed.test_u14_authority import Dataset
    shallow = [Definition(Dataset).concretize()]
    payload = encode_result(shallow)
    with monkeypatch.context() as patch:
        patch.setattr(codec, "_MAX_NODES", 3)
        with pytest.raises(ManagedResultError): encode_result(shallow)
        with pytest.raises(ManagedResultError): decode_result(payload)
    graph = Definition(Dataset)
    for _ in range(4): graph = Definition(Dataset, value=graph)
    payload = encode_result(graph)
    with monkeypatch.context() as patch:
        patch.setattr(codec, "_MAX_DEPTH", 3)
        with pytest.raises(ManagedResultError): encode_result(graph)
        with pytest.raises(ManagedResultError): decode_result(payload)
    nested = 1
    for _ in range(81): nested = [nested]
    with pytest.raises(ManagedResultError): encode_result(nested)
    packet = {"schema": "dryml-managed-result", "version": 2, "metadata": None,
              "nodes": [["none"]] + [["list", [i]] for i in range(81)], "root": 81}
    with pytest.raises(ManagedResultError): decode_result(json.dumps(packet).encode())
    packet["nodes"] = [["none"]] * (codec._MAX_NODES + 1)
    with pytest.raises(ManagedResultError): decode_result(json.dumps(packet).encode())


def test_v1_result_bytes_references_and_namespace_refuse_intact(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from uuid import uuid4
    from dryml.managed import results
    from dryml.managed.control import _projection_reader, _validate_projection_namespace
    from dryml.managed.errors import ManagedControlError
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.managed.control import ManagedControlStore
    store = DirStore(tmp_path / "state", query_index="none")
    ManagedControlStore(store, Repo(store)).initialize()
    payload = b'{"nodes":[["str","old"]],"root":0,"schema":"dryml-managed-result","version":1}'
    key = hashlib.sha256(payload).hexdigest()
    path = Path(store.base_dir, "managed", "results", "v1", key[:2], key, "current.json")
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    with pytest.raises(ManagedResultError): decode_result(path.read_bytes())
    with pytest.raises(ManagedResultError): results.ResultReference.from_data({"kind": "dryml-managed-result", "version": 1, "key": key})
    with pytest.raises(ManagedControlError): _projection_reader(store._relative(path))
    with pytest.raises(ManagedControlError, match="legacy result"): _validate_projection_namespace(store)
    def forbidden(*args, **kwargs): pytest.fail("old result reached source open/publication")
    monkeypatch.setattr(store, "_prepare_and_apply", forbidden)
    with pytest.raises(ManagedResultError): results.publish_result(store, payload)
    assert "/results/v2/" in results.result_path(store, key)
    reference = results.ResultReference(key)
    object.__setattr__(reference, "version", 1)
    source = results.ResultSource(str(tmp_path / "missing"), str(uuid4()))
    monkeypatch.setattr(results, "DirStore", forbidden)
    monkeypatch.setattr(results.os.path, "isfile", forbidden)
    with pytest.raises(ManagedResultError, match="invalid_result_reference"): source.read(reference)
    assert path.read_bytes() == payload


def test_content_address_collision_never_substitutes_hash_for_complete_bytes(tmp_path, monkeypatch):
    from tests.managed.u16_fixtures import setup
    from dryml.managed import results
    control, live = setup(tmp_path)
    control.initialize()
    class Collision:
        def hexdigest(self): return "f" * 64
    from types import SimpleNamespace
    monkeypatch.setattr(results, "hashlib", SimpleNamespace(sha256=lambda *a: Collision()))
    first = results.publish_result(control.control_store, encode_result(1))
    with pytest.raises(ManagedResultError, match="collision"):
        results.publish_result(control.control_store, encode_result(2))
    assert decode_result(results.read_result_bytes(control.control_store, first)) == 1
