"""Portable core-owned ObjectRef and StateRef JSON codec contracts."""

import json
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from dryml.core import Definition, ObjectId, ObjectRef, Serializable, StateRef
from dryml.core.backend import Backend
from dryml.core.cardinality import Cardinality
from dryml.core.config import ConfigRef
from dryml.core.dtype import DType
from dryml.core.factory import FactorySpec
from dryml.core.reference_json import (
    REFERENCE_JSON_SCHEMA,
    ReferenceJSONCodecError,
    _decode_legacy_reference_json,
    _encode_legacy_reference_json,
    decode_reference_json,
    encode_reference_json,
    _value_from_json,
    _value_to_json,
)
from dryml.core.symbol import SourceSpec
from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec
from dryml.core.utils.graph.path import GraphPath
from dryml.data import GeneratorDataset


class ReferenceJSONSubject(Serializable):
    """Stateful definition fixture retaining representative canonical atoms."""

    def __init__(self, values):
        self.values = values


def _factory_source():
    return SourceSpec.from_source(
        "def build(value):\n    return value\n",
        kind="function",
        name="build",
    )


def _reference_values():
    return {
        "bytes": b"portable",
        "cardinality": Cardinality.INFINITE,
        "config": ConfigRef("batch_size", np.int16(8)),
        "core_type": DType,
        "dtype": DType("float", 32),
        "enum": Backend.jax,
        "factory": FactorySpec(_factory_source(), 3),
        "integer_key": {7: "seven"},
        "scalar": np.float32(1.25),
        "spec": TensorSpec(
            "float32", shape=(Dynamic, 3), backend="numpy", layout=Layout.DENSE,
        ),
        "array": np.arange(4, dtype=np.int16).reshape(2, 2),
    }


def _references():
    definition = Definition(ReferenceJSONSubject, _reference_values()).concretize()
    path = GraphPath()
    object_ref = ObjectRef(definition, {path: ObjectId(("codec",))})
    state_ref = StateRef(object_ref, {path: "pkl-" + "a" * 64})
    return object_ref, state_ref


@pytest.mark.parametrize("index", (0, 1))
def test_reference_json_round_trips_exact_authority_and_canonical_atoms(index):
    """The public codec preserves ObjectRef and StateRef identity using JSON only."""

    reference = _references()[index]
    encoded = encode_reference_json(reference)

    assert encoded["schema"] == REFERENCE_JSON_SCHEMA
    json.dumps(encoded, ensure_ascii=False, allow_nan=False)
    assert decode_reference_json(encoded) == reference


def test_reference_json_rejects_malformed_envelopes_and_values():
    """Closed envelope and value tags fail before constructing reference authority."""

    _, state_ref = _references()
    encoded = encode_reference_json(state_ref)
    encoded["value"] = {"type": "unknown"}

    with pytest.raises(ReferenceJSONCodecError, match="unknown"):
        decode_reference_json(encoded)
    with pytest.raises(ReferenceJSONCodecError, match="exactly"):
        decode_reference_json({"schema": REFERENCE_JSON_SCHEMA})
    for field, malformed in (("version", True), ("kind", [])):
        invalid = encode_reference_json(state_ref)
        invalid[field] = malformed
        with pytest.raises(ReferenceJSONCodecError, match="schema, version, or kind"):
            decode_reference_json(invalid)
    with pytest.raises(ReferenceJSONCodecError, match="repeats"):
        _value_from_json({
            "type": "frozen_set",
            "items": [
                {"type": "int", "value": "1"},
                {"type": "int", "value": "1"},
            ],
        }, grammar=3)
    with pytest.raises(ReferenceJSONCodecError, match="invalid fields"):
        _value_from_json({
            "type": "core_type",
            "module": "dryml.core.dtype",
            "qualname": "DType",
            "member": "ignored",
        }, grammar=3)


def test_reference_json_rejects_process_local_enum_authority():
    """Portable references cannot retain notebook or process-main Enum classes."""

    notebook_enum = Enum("NotebookEnum", {"MEMBER": "value"}, module="__main__")

    with pytest.raises(ReferenceJSONCodecError, match="local enum"):
        _value_to_json(notebook_enum.MEMBER, grammar=3)


def test_legacy_reference_json_requires_exact_controls():
    """Compatibility selection rejects bool versions and malformed kind values."""

    _, state_ref = _references()
    with pytest.raises(ReferenceJSONCodecError, match="version"):
        _encode_legacy_reference_json(state_ref, version=True)
    with pytest.raises(ReferenceJSONCodecError, match="version"):
        _decode_legacy_reference_json({}, kind="state_ref", version=True)
    with pytest.raises(ReferenceJSONCodecError, match="kind"):
        _decode_legacy_reference_json({}, kind=[], version=2)


def test_legacy_reference_json_v2_round_trips_and_v1_stays_closed():
    """Core owns pre-envelope compatibility without broadening the v1 grammar."""

    definition = Definition(
        ReferenceJSONSubject,
        {
            "cardinality": Cardinality.UNKNOWN,
            "factory": FactorySpec(
                _factory_source(),
                [1, {"members": {2, 3}}],
                ("left", "right"),
                spec=TensorSpec("float32", shape=(1,), backend="jax"),
            ),
        },
    ).concretize()
    path = GraphPath()
    reference = StateRef(
        ObjectRef(definition, {path: ObjectId(("legacy",))}),
        {path: "pkl-" + "b" * 64},
    )
    encoded = _encode_legacy_reference_json(reference, version=2)

    assert _decode_legacy_reference_json(
        encoded, kind="state_ref", version=2,
    ) == reference
    with pytest.raises(ReferenceJSONCodecError):
        _decode_legacy_reference_json(encoded, kind="state_ref", version=1)
    with pytest.raises(ReferenceJSONCodecError, match="not valid in v1"):
        _value_from_json({"type": "frozen_list", "items": []}, grammar=1)
    with pytest.raises(ReferenceJSONCodecError, match="not valid in v2"):
        _value_from_json(
            {"type": "dtype", "kind": "float", "bits": 32}, grammar=2,
        )


def test_fixed_historical_v2_reference_json_decodes_exactly():
    """The v2 reader remains pinned to an encoder-independent persisted vector."""

    fixture = Path(__file__).parents[1] / "fixtures" / "reference_json_v2.json"
    decoded = _decode_legacy_reference_json(
        json.loads(fixture.read_text(encoding="ascii")),
        kind="state_ref",
        version=2,
    )
    expected = StateRef(
        ObjectRef(
            Definition(
                GeneratorDataset, list, cardinality=Cardinality.INFINITE,
            ).concretize(),
            {},
        ),
        {},
    )

    assert decoded == expected


@pytest.mark.parametrize(
    "encoded, match, grammar",
    (
        ({"type": "dtype", "kind": "float", "bits": None}, "DType", 3),
        ({"type": "dtype", "kind": "float", "bits": True}, "DType", 3),
        ({"type": "cardinality", "kind": "finite", "value": "01"}, "canonical", 2),
        ({"type": "cardinality", "kind": "infinite", "value": "0"}, "Cardinality", 2),
    ),
)
def test_reference_json_rejects_malformed_semantic_atoms(encoded, match, grammar):
    """Semantic tags retain strict canonical field validation in core."""

    with pytest.raises(ReferenceJSONCodecError, match=match):
        _value_from_json(encoded, grammar=grammar)
