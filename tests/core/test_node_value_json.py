"""Semantic JSON leaf validation remains strict within current Node transport."""

import json
from enum import Enum

import numpy as np
import pytest

from dryml.core import Definition, ObjectId, Serializable
from dryml.core._value_json import ReferenceJSONCodecError, _value_from_json, _value_to_json
from dryml.core.backend import Backend
from dryml.core.cardinality import Cardinality
from dryml.core.config import ConfigRef
from dryml.core.dtype import DType
from dryml.core.factory import FactorySpec
from dryml.core.node_codec import decode_node_graph, encode_node_graph
from dryml.core.node_prepare import NodePreparation
from dryml.core.symbol import SourceSpec
from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec


class SemanticLeaf(Serializable):
    """Stateful fixture retaining representative canonical atom families."""

    def __init__(self, values):
        self.values = values


def test_node_transport_round_trips_canonical_atoms_and_identity():
    source = SourceSpec.from_source("def build(value):\n    return value\n", kind="function", name="build")
    values = {
        "bytes": b"portable", "cardinality": Cardinality.INFINITE,
        "config": ConfigRef("batch_size", np.int16(8)), "core_type": DType,
        "dtype": DType("float", 32), "enum": Backend.jax,
        "factory": FactorySpec(source, 3), "integer_key": {7: "seven"},
        "scalar": np.float32(1.25),
        "spec": TensorSpec("float32", shape=(Dynamic, 3), backend="numpy", layout=Layout.DENSE),
        "array": np.arange(4, dtype=np.int16).reshape(2, 2),
    }
    node = Definition(SemanticLeaf, values).concretize().with_identity(ObjectId(("codec",)))
    wire = json.loads(json.dumps(encode_node_graph(NodePreparation((node,))), allow_nan=False))
    restored = decode_node_graph(wire)
    assert restored.graph_equal(node)
    assert restored.object_id == node.object_id


def test_semantic_json_rejects_duplicate_members_and_extra_fields():
    with pytest.raises(ReferenceJSONCodecError, match="repeats"):
        _value_from_json({"type": "frozen_set", "items": [
            {"type": "int", "value": "1"}, {"type": "int", "value": "1"},
        ]}, grammar=3)
    with pytest.raises(ReferenceJSONCodecError, match="invalid fields"):
        _value_from_json({"type": "core_type", "module": "dryml.core.dtype",
                          "qualname": "DType", "member": "ignored"}, grammar=3)


def test_semantic_json_rejects_process_local_enum_authority():
    notebook_enum = Enum("NotebookEnum", {"MEMBER": "value"}, module="__main__")
    with pytest.raises(ReferenceJSONCodecError, match="local enum"):
        _value_to_json(notebook_enum.MEMBER, grammar=3)


@pytest.mark.parametrize("encoded, match", (
    ({"type": "dtype", "kind": "float", "bits": None}, "DType"),
    ({"type": "dtype", "kind": "float", "bits": True}, "DType"),
    ({"type": "cardinality", "kind": "finite", "value": "01"}, "canonical"),
    ({"type": "cardinality", "kind": "infinite", "value": "0"}, "Cardinality"),
))
def test_semantic_json_rejects_malformed_atoms(encoded, match):
    with pytest.raises(ReferenceJSONCodecError, match=match):
        _value_from_json(encoded, grammar=3)
