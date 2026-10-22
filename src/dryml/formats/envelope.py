"""Closed v1.1 envelopes and separately framed versioned JSON metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .canonical import canonical_json_bytes, json_ready
from .errors import EnvelopeError
from .ids import CONTRACT_VERSION, verify_semantic_id
import json

_FIELDS = frozenset({"contract_version", "schema", "kind", "payload", "id", "metadata"})


def make_envelope(*, schema: str, kind: str, prefix: str, payload: Mapping[str, Any], semantic_id: str, identifying_payload: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None, max_bytes: int = 4_194_304, **bounds: Any) -> dict[str, Any]:
    """Build one closed v1.1 envelope with a self-validating semantic ID.

    Args:
        schema: Exact family schema ending in ``.v1.1``.
        kind: Exact family kind.
        prefix: Exact family ID prefix.
        payload: Complete closed family payload.
        semantic_id: Attached family ID to verify before emission.
        identifying_payload: Identity projection, or the full payload by default.
        metadata: Optional non-identifying envelope metadata.
        max_bytes: Maximum canonical UTF-8 envelope size.
        **bounds: Canonical JSON validation bounds.

    Returns:
        A detached JSON-compatible envelope mapping.

    Raises:
        ContentIDError: If ``semantic_id`` does not match the projection.
        EnvelopeError: If the complete envelope exceeds ``max_bytes``.
    """

    ready_payload = json_ready(payload, **bounds)
    verify_semantic_id(
        semantic_id,
        prefix=prefix,
        schema=schema,
        kind=kind,
        identifying_payload=ready_payload if identifying_payload is None else identifying_payload,
        **bounds,
    )
    result: dict[str, Any] = {"contract_version": CONTRACT_VERSION, "schema": schema, "kind": kind, "payload": ready_payload, "id": semantic_id}
    if metadata is not None:
        result["metadata"] = json_ready(metadata, **bounds)
    if len(canonical_json_bytes(result, **_envelope_bounds(bounds))) > max_bytes:
        raise EnvelopeError("v1.1 envelope exceeds byte bound", context={"limit": max_bytes})
    return result


def validate_envelope(data: Mapping[str, Any], *, schema: str, kind: str, prefix: str, identifying_payload: Mapping[str, Any], max_bytes: int = 4_194_304, **bounds: Any) -> dict[str, Any]:
    """Validate a closed family envelope and its optional attached ID.

    Raises an error containing observed and supported versions whenever the
    contract version is absent, old, or future.
    """

    if not isinstance(data, Mapping):
        raise EnvelopeError("v1.1 envelope must be a mapping")
    observed = data.get("contract_version")
    if observed != CONTRACT_VERSION:
        raise EnvelopeError("unsupported metadata contract version", context={"observed_version": observed, "supported_version": CONTRACT_VERSION})
    unknown = set(data) - _FIELDS
    missing = {"schema", "kind", "payload"} - set(data)
    if unknown or missing:
        raise EnvelopeError("v1.1 envelope fields are closed", context={"unknown": sorted(unknown), "missing": sorted(missing)})
    if data["schema"] != schema or not schema.endswith(".v1.1"):
        raise EnvelopeError("v1.1 envelope schema mismatch", context={"expected": schema, "observed": data["schema"]})
    if data["kind"] != kind:
        raise EnvelopeError("v1.1 envelope kind mismatch", context={"expected": kind, "observed": data["kind"]})
    if not isinstance(data["payload"], Mapping):
        raise EnvelopeError("v1.1 envelope payload must be a mapping")
    result = {key: json_ready(value, **bounds) for key, value in data.items()}
    if len(canonical_json_bytes(result, **_envelope_bounds(bounds))) > max_bytes:
        raise EnvelopeError("v1.1 envelope exceeds byte bound", context={"limit": max_bytes})
    if "metadata" in result and not isinstance(result["metadata"], Mapping):
        raise EnvelopeError("v1.1 envelope metadata must be a mapping")
    if "id" in result:
        verify_semantic_id(result["id"], prefix=prefix, schema=schema, kind=kind, identifying_payload=identifying_payload, **bounds)
    return result


def _envelope_bounds(bounds: Mapping[str, Any]) -> dict[str, Any]:
    """Add only the fixed envelope wrapper overhead to payload bounds."""

    result = dict(bounds)
    if "max_depth" in result:
        result["max_depth"] += 1
    if "max_nodes" in result:
        result["max_nodes"] += 6
    return result


def encode_metadata_frame(schema: str, version: int, data: Mapping[str, Any]) -> bytes:
    """Frame a closed metadata mapping without executing a pickle codec.

    Args:
        schema: ASCII, path-free family name.
        version: Positive exact integer family version.
        data: JSON-compatible family authority; the owning decoder validates fields.
    Returns:
        bytes: Versioned frame followed by canonical JSON.
    Raises:
        EnvelopeError: If the family/version or JSON representation is invalid.
    Side Effects:
        None; this is distinct from the v1.1 envelope/semantic-ID contract.
    """
    if (type(schema) is not str or not schema.isascii() or not schema
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in schema)
            or type(version) is not int or version < 1):
        raise EnvelopeError("invalid metadata frame family/version")
    try:
        payload = canonical_json_bytes(data, max_depth=128, max_nodes=1_000_000,
                                       max_entries=100_000, max_string=16 * 1024 * 1024)
    except ValueError as error:
        raise EnvelopeError("invalid metadata JSON") from error
    return f"DRYML-METADATA/{schema}/{version}\n".encode("ascii") + payload


def decode_metadata_frame(data: bytes, *, schema: str, version: int,
                          max_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
    """Decode one exact bounded metadata frame, rejecting duplicate/trailing JSON.

    Args:
        data: Complete framed bytes from trusted metadata authority.
        schema: Required family name.
        version: Required exact integer version.
        max_bytes: Maximum accepted complete frame size.
    Returns:
        dict: Fresh primitive metadata mapping; owner validates its closed schema.
    Raises:
        EnvelopeError: For incompatible frames, size, malformed or noncanonical JSON.
    Side Effects:
        Allocates primitive values only, without deserialization hooks or IO.
    """
    prefix = f"DRYML-METADATA/{schema}/{version}\n".encode("ascii")
    if type(data) is not bytes or len(data) > max_bytes or not data.startswith(prefix):
        raise EnvelopeError("unsupported or malformed metadata frame")
    try:
        payload = data[len(prefix):]
        result = json.loads(payload)
        if type(result) is not dict or canonical_json_bytes(
                result, max_depth=128, max_nodes=1_000_000,
                max_entries=100_000, max_string=16 * 1024 * 1024) != payload:
            raise ValueError("noncanonical metadata JSON")
        return result
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise EnvelopeError("malformed or noncanonical metadata JSON") from error


__all__ = ["make_envelope", "validate_envelope", "encode_metadata_frame", "decode_metadata_frame"]
