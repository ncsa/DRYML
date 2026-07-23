"""Shared lightweight validators for dispatch and execution metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.formats import CanonicalJSONError, deep_freeze_json, json_ready
from dryml.formats.errors import ContentIDError, ReferenceParseError
from dryml.formats.ids import parse_content_id
from dryml.formats.refs import parse_cdef_id

from .errors import DispatchSpecError


EXECUTION_STATUSES = frozenset({"ok", "failed", "cancelled", "timeout", "unsupported", "skipped", "degraded"})
EXECUTION_KINDS = frozenset({"python", "probe", "adapter", "compiler", "lowering", "internal", "unknown"})


def normalize_execution_status(status: Any) -> str:
    """Return a normalized immutable execution status."""

    if not isinstance(status, str):
        raise DispatchSpecError("execution status must be a string", context={"type": type(status).__name__})
    value = status.strip().lower()
    if value not in EXECUTION_STATUSES:
        raise DispatchSpecError("invalid execution status", context={"status": status, "allowed": sorted(EXECUTION_STATUSES)})
    return value


def normalize_execution_kind(kind: Any) -> str:
    """Return a normalized execution-kind token."""

    if not isinstance(kind, str):
        raise DispatchSpecError("execution_kind must be a string", context={"type": type(kind).__name__})
    value = kind.strip().lower()
    if value not in EXECUTION_KINDS:
        raise DispatchSpecError("invalid execution_kind", context={"execution_kind": kind, "allowed": sorted(EXECUTION_KINDS)})
    return value


def normalize_backend_identity(backend: Any) -> Mapping[str, Any]:
    """Validate and freeze backend identity metadata."""

    if not isinstance(backend, Mapping):
        raise DispatchSpecError("backend must be a mapping", context={"type": type(backend).__name__})
    unknown = set(backend) - {"name", "kind", "version", "provider", "metadata"}
    if unknown:
        raise DispatchSpecError("backend contains unknown fields", context={"fields": sorted(unknown)})
    name = backend.get("name")
    if not isinstance(name, str) or not name:
        raise DispatchSpecError("backend.name must be a non-empty string")
    for key in ("kind", "version", "provider"):
        if key in backend and backend[key] is not None and not isinstance(backend[key], str):
            raise DispatchSpecError(f"backend.{key} must be a string", context={"type": type(backend[key]).__name__})
    if "metadata" in backend and backend["metadata"] is not None and not isinstance(backend["metadata"], Mapping):
        raise DispatchSpecError("backend.metadata must be a mapping", context={"type": type(backend['metadata']).__name__})
    return _freeze_mapping(backend, "backend")


def validate_id(value: Any, prefixes: tuple[str, ...], field_name: str) -> str:
    """Validate a schema-versioned content ID by prefix."""

    if not isinstance(value, str):
        raise DispatchSpecError(f"{field_name} must be a string", context={"type": type(value).__name__})
    try:
        parts = parse_content_id(value)
    except ContentIDError as exc:
        raise DispatchSpecError(f"invalid {field_name}", context=exc.context) from exc
    if parts.prefix not in prefixes or parts.schema_version != 1:
        raise DispatchSpecError(f"{field_name} prefix mismatch", context={"value": value, "expected": prefixes})
    return value


def validate_cdef_id(value: Any, field_name: str) -> str:
    """Validate and return a raw CDef ID."""

    try:
        return parse_cdef_id(value).raw
    except ReferenceParseError as exc:
        raise DispatchSpecError(f"invalid {field_name}", context=exc.context) from exc


def json_ready_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Return a frozen JSON-ready mapping."""

    if not isinstance(value, Mapping):
        raise DispatchSpecError(f"{field_name} must be a mapping", context={"type": type(value).__name__})
    return _freeze_mapping(value, field_name)


def json_ready_value(value: Any, field_name: str) -> Any:
    """Return a JSON-ready value or raise a dispatch metadata error."""

    try:
        return json_ready(value)
    except CanonicalJSONError as exc:
        raise DispatchSpecError(f"{field_name} is not canonical JSON", context=exc.context) from exc


def _freeze_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    try:
        frozen = deep_freeze_json(value)
    except CanonicalJSONError as exc:
        raise DispatchSpecError(f"{field_name} is not canonical JSON", context=exc.context) from exc
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = [
    "EXECUTION_KINDS",
    "EXECUTION_STATUSES",
    "json_ready_mapping",
    "json_ready_value",
    "normalize_backend_identity",
    "normalize_execution_kind",
    "normalize_execution_status",
    "validate_cdef_id",
    "validate_id",
]
