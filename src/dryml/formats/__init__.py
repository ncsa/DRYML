"""Dependency-light canonical v1.1 metadata codec layer."""

from .canonical import FrozenJson, JsonPrimitive, canonical_json_bytes, canonical_json_dumps, canonical_json_load_bytes, canonical_json_loads, deep_freeze_json, freeze_mapping, json_ready
from .envelope import make_envelope, validate_envelope
from .errors import CanonicalJSONError, ContentIDError, DrymlFormatError, EnvelopeError
from .ids import CONTRACT_VERSION, semantic_id, verify_semantic_id

__all__ = ["CONTRACT_VERSION", "CanonicalJSONError", "ContentIDError", "DrymlFormatError", "EnvelopeError", "FrozenJson", "JsonPrimitive", "canonical_json_bytes", "canonical_json_dumps", "canonical_json_load_bytes", "canonical_json_loads", "deep_freeze_json", "freeze_mapping", "json_ready", "make_envelope", "semantic_id", "validate_envelope", "verify_semantic_id"]
