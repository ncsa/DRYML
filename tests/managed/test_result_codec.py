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
