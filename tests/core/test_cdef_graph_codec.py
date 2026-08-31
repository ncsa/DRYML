from copy import deepcopy
import pickle

import pytest

from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.cdef_codec import (
    CDefGraphCodecError,
    decode_cdef_graph,
    encode_cdef_graph,
)
from dryml.core.cdef_identity import cdef_node_key
from dryml.core.symbol import ImportRef, SourceSpec


class CodecLeaf(Object):
    def __init__(self, value):
        self.value = value


class CodecStatefulLeaf(Serializable):
    def __init__(self, value):
        self.value = value


class CodecParent(Object):
    def __init__(self, children):
        self.children = children


def test_graph_codec_round_trip_preserves_topology_and_regenerates_tokens():
    """Graph authority labels retain aliases while decoding creates fresh private tokens."""

    shared = Definition(CodecStatefulLeaf, "same")
    root = Definition(
        CodecParent, [shared, shared, Definition(CodecStatefulLeaf, "same")]
    ).concretize()
    encoded = encode_cdef_graph(root)
    restored = decode_cdef_graph(encoded)

    assert restored == root
    assert restored.graph_equal(root)
    assert restored.graph_hash() == root.graph_hash()
    assert cdef_node_key(restored) is not cdef_node_key(root)
    assert cdef_node_key(restored.parameters["children"][0]) is cdef_node_key(
        restored.parameters["children"][1]
    )
    assert cdef_node_key(
        restored.parameters["children"][0]
    ) is not cdef_node_key(restored.parameters["children"][2])
    assert [node["stateful_role"] for node in encoded["nodes"]].count(
        True
    ) == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["nodes"].append(deepcopy(data["nodes"][0])),
        lambda data: data.__setitem__("root", "n404"),
        lambda data: data["nodes"][0].__setitem__("stateful_role", "yes"),
        lambda data: data["nodes"][0].__setitem__(
            "parameters", {"kind": "cdef", "label": "n404"}
        ),
    ],
)
def test_graph_codec_rejects_malformed_authority(mutate):
    """Malformed declarations and dangling topology references fail closed."""

    root = Definition(CodecParent, [Definition(CodecLeaf, "x")]).concretize()
    encoded = encode_cdef_graph(root)
    mutate(encoded)

    with pytest.raises(CDefGraphCodecError):
        decode_cdef_graph(encoded)


def test_repr_uses_deterministic_monikers_without_private_tokens():
    """Independent equal children are distinguishable without exposing tokens."""

    root = Definition(
        CodecParent,
        [Definition(CodecLeaf, "same"), Definition(CodecLeaf, "same")],
    ).concretize()

    rendered = repr(root)

    assert "@n1=ConcreteDefinition" in rendered
    assert "@n2=ConcreteDefinition" in rendered
    assert "same" in rendered
    assert "_node_id" not in rendered
    assert "object at 0x" not in rendered


def test_repr_declares_one_shared_node_then_reuses_its_moniker():
    """Shared occurrences reuse one declared moniker without losing payload."""

    child = Definition(CodecLeaf, "same")
    root = Definition(CodecParent, [child, child]).concretize()

    rendered = repr(root)

    assert rendered.count("@n1=ConcreteDefinition") == 1
    assert rendered.count("@n1") == 2
    assert "same" in rendered


def test_function_source_target_is_rejected_before_cdef_creation():
    """Callable source values are allowed as parameters, never CDef class targets."""

    target = SourceSpec.from_source("lambda value: value", kind="function")

    with pytest.raises(TypeError, match="must resolve to a class"):
        Definition(target, "value").concretize()


@pytest.mark.parametrize(
    ("base", "expected_role"),
    [("Object", False), ("Serializable", True)],
)
def test_source_class_target_records_its_stateful_role(base, expected_role):
    """Source-defined class targets preserve role bits without changing CDef identity."""

    target = SourceSpec.from_source(
        f"class SourceTarget({base}):\n    def __init__(self, value):\n        self.value = value",
        kind="class",
        name="SourceTarget",
        imports={base: ImportRef("dryml.core.object", base)},
    )
    cdef = Definition(target, "value").concretize()

    assert (
        encode_cdef_graph(cdef)["nodes"][0]["stateful_role"] is expected_role
    )


def test_recorded_role_mismatch_fails_before_constructor_execution():
    """Decoded role authority is checked at the admitted materialization seam."""

    root = Definition(CodecStatefulLeaf, "x").concretize()
    encoded = encode_cdef_graph(root)
    encoded["nodes"][0]["stateful_role"] = False
    mismatched = decode_cdef_graph(encoded)

    with pytest.raises(Exception, match="stateful role mismatch"):
        mismatched.build(repo=Repo())


def test_graph_authority_inspection_never_resolves_symbols(monkeypatch):
    """Graph hashing and codec hydration remain definition-only inspection paths."""

    root = Definition(CodecParent, [Definition(CodecLeaf, "x")]).concretize()

    def fail_resolution(*args, **kwargs):
        raise AssertionError(
            "graph authority inspection must not resolve symbols"
        )

    monkeypatch.setattr(ImportRef, "resolve", fail_resolution)

    encoded = encode_cdef_graph(root)
    restored = decode_cdef_graph(encoded)

    assert root.graph_equal(restored)
    assert root.graph_hash() == restored.graph_hash()


def test_v2_pickle_round_trip_preserves_stateful_role_authority():
    """Ordinary CDef serialization must retain the role checked at build time."""

    root = Definition(CodecStatefulLeaf, "x").concretize()
    restored = pickle.loads(pickle.dumps(root))

    assert encode_cdef_graph(restored)["nodes"][0]["stateful_role"] is True
    restored.build(repo=Repo())
