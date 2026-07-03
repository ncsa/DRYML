"""Shared canonical format helpers for DRYML metadata.

The :mod:`dryml.formats` package provides dependency-light canonical JSON,
stable content IDs, generic envelopes, and reserved-reference parsing utilities.
It intentionally does not import higher-level DRYML subsystems.
"""

from .canonical import (
    FrozenJson,
    JsonPrimitive,
    canonical_json_bytes,
    canonical_json_dumps,
    canonical_json_load_bytes,
    canonical_json_loads,
    deep_freeze_json,
    freeze_mapping,
    json_ready,
)
from .envelope import EnvelopeSpec, envelope_payload_for_id, make_envelope, validate_envelope
from .errors import (
    CanonicalJSONError,
    ContentIDError,
    DrymlFormatError,
    EnvelopeError,
    ReferenceParseError,
)
from .ids import (
    ContentIDParts,
    content_id,
    is_content_id,
    parse_content_id,
    stable_hash,
    validate_id_prefix,
    validate_schema_version,
)
from .refs import (
    CDefID,
    ReservedRef,
    format_cdef_id,
    format_ref_cdef,
    is_cdef_id,
    is_literal_escape,
    is_ref_cdef,
    is_reserved_ref,
    literal_escape,
    parse_cdef_id,
    parse_ref_cdef,
    parse_reserved_ref,
    unwrap_literal_escape,
)

__all__ = [
    "CDefID",
    "CanonicalJSONError",
    "ContentIDError",
    "ContentIDParts",
    "DrymlFormatError",
    "EnvelopeError",
    "EnvelopeSpec",
    "FrozenJson",
    "JsonPrimitive",
    "ReferenceParseError",
    "ReservedRef",
    "canonical_json_bytes",
    "canonical_json_dumps",
    "canonical_json_load_bytes",
    "canonical_json_loads",
    "content_id",
    "deep_freeze_json",
    "envelope_payload_for_id",
    "format_cdef_id",
    "format_ref_cdef",
    "freeze_mapping",
    "is_cdef_id",
    "is_content_id",
    "is_literal_escape",
    "is_ref_cdef",
    "is_reserved_ref",
    "json_ready",
    "literal_escape",
    "make_envelope",
    "parse_cdef_id",
    "parse_content_id",
    "parse_ref_cdef",
    "parse_reserved_ref",
    "stable_hash",
    "unwrap_literal_escape",
    "validate_envelope",
    "validate_id_prefix",
    "validate_schema_version",
]
