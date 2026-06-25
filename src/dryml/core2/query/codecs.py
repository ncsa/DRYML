from __future__ import annotations

import hashlib
from typing import Any

from ..definition import ConcreteDefinition
from ..utils.general import pickler, unpickler
from .model import FeatureToken
from .path import GRAPH_PATH_SCHEMA_VERSION, GraphPath


CDEF_CODEC_VERSION = 1
FEATURE_CODEC_VERSION = 1
PATH_CODEC_VERSION = GRAPH_PATH_SCHEMA_VERSION


class QueryCodecError(ValueError):
    pass


def encode_cdef(cdef: ConcreteDefinition) -> bytes:
    if not isinstance(cdef, ConcreteDefinition):
        raise TypeError(f"encode_cdef expected ConcreteDefinition, got {type(cdef).__name__}.")
    return _pack("cdef", CDEF_CODEC_VERSION, cdef)


def decode_cdef(blob: bytes) -> ConcreteDefinition:
    value = _unpack(blob, expected_kind="cdef", expected_version=CDEF_CODEC_VERSION)
    if not isinstance(value, ConcreteDefinition):
        raise QueryCodecError(f"Decoded CDef payload has type {type(value).__name__}, expected ConcreteDefinition.")
    return value


def encode_feature_token(token: FeatureToken) -> bytes:
    if not isinstance(token, FeatureToken):
        raise TypeError(f"encode_feature_token expected FeatureToken, got {type(token).__name__}.")
    payload = {
        "kind": token.kind,
        "path": None if token.path is None else token.path.to_data(),
        "payload": token.payload,
    }
    return _pack("feature-token", FEATURE_CODEC_VERSION, payload)


def decode_feature_token(blob: bytes) -> FeatureToken:
    value = _unpack(blob, expected_kind="feature-token", expected_version=FEATURE_CODEC_VERSION)
    if not isinstance(value, dict):
        raise QueryCodecError(f"Decoded feature token payload has type {type(value).__name__}, expected dict.")
    path_data = value.get("path")
    path = None if path_data is None else GraphPath.from_data(path_data)
    token = FeatureToken(value.get("kind"), path, value.get("payload"))
    if not isinstance(token.kind, str):
        raise QueryCodecError("Decoded feature token kind must be a string.")
    return token


def encode_graph_path(path: GraphPath) -> bytes:
    if not isinstance(path, GraphPath):
        raise TypeError(f"encode_graph_path expected GraphPath, got {type(path).__name__}.")
    return _pack("graph-path", PATH_CODEC_VERSION, path.to_data())


def decode_graph_path(blob: bytes) -> GraphPath:
    value = _unpack(blob, expected_kind="graph-path", expected_version=PATH_CODEC_VERSION)
    try:
        return GraphPath.from_data(value)
    except Exception as exc:
        raise QueryCodecError("Decoded graph path payload is invalid.") from exc


def digest_blob(blob: bytes) -> bytes:
    if not isinstance(blob, bytes):
        raise TypeError(f"digest_blob expected bytes, got {type(blob).__name__}.")
    return hashlib.sha256(blob).digest()


def _pack(kind: str, version: int, payload: Any) -> bytes:
    return pickler({"kind": kind, "version": version, "payload": payload})


def _unpack(blob: bytes, *, expected_kind: str, expected_version: int) -> Any:
    if not isinstance(blob, bytes):
        raise TypeError(f"codec blob must be bytes, got {type(blob).__name__}.")
    try:
        envelope = unpickler(blob)
    except Exception as exc:
        raise QueryCodecError("Could not decode query-index blob.") from exc
    if not isinstance(envelope, dict):
        raise QueryCodecError(f"Decoded codec envelope has type {type(envelope).__name__}, expected dict.")
    kind = envelope.get("kind")
    if kind != expected_kind:
        raise QueryCodecError(f"Expected {expected_kind!r} blob, got {kind!r}.")
    version = envelope.get("version")
    if version != expected_version:
        raise QueryCodecError(f"Unsupported {expected_kind} codec version {version!r}.")
    if "payload" not in envelope:
        raise QueryCodecError("Decoded codec envelope is missing 'payload'.")
    return envelope["payload"]
