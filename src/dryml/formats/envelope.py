"""Closed v1.1 envelope construction and validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .canonical import canonical_json_bytes, json_ready
from .errors import EnvelopeError
from .ids import CONTRACT_VERSION, verify_semantic_id

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


__all__ = ["make_envelope", "validate_envelope"]
