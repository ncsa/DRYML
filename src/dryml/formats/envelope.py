"""Generic schema/kind/payload envelope helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .canonical import json_ready
from .errors import CanonicalJSONError, ContentIDError, EnvelopeError
from .ids import validate_schema_version


@dataclass(frozen=True, slots=True)
class EnvelopeSpec:
    """Expected schema and kind for a generic DRYML envelope."""

    schema: str
    kind: str
    schema_version: int | None = None


def make_envelope(
    *,
    schema: str,
    kind: str,
    payload: Mapping[str, Any] | None = None,
    schema_version: int | None = None,
    id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical JSON-compatible envelope data.

    Parameters
    ----------
    schema:
        Schema name for the envelope.
    kind:
        Kind tag within the schema.
    payload:
        Optional JSON-compatible payload mapping. Defaults to an empty mapping.
    schema_version:
        Optional positive schema version.
    id:
        Optional externally computed content ID.
    metadata:
        Optional JSON-compatible metadata mapping.

    Returns
    -------
    dict[str, Any]
        Envelope data using ``schema``, ``kind``, ``payload``, and optional keys.
    """

    _validate_required_string("schema", schema)
    _validate_required_string("kind", kind)
    if payload is not None and not isinstance(payload, Mapping):
        raise EnvelopeError("envelope payload must be a mapping", context={"type": type(payload).__name__})
    if metadata is not None and not isinstance(metadata, Mapping):
        raise EnvelopeError("envelope metadata must be a mapping", context={"type": type(metadata).__name__})
    envelope: dict[str, Any] = {
        "schema": schema,
        "kind": kind,
        "payload": _json_ready_for_envelope({} if payload is None else payload, "payload"),
    }
    if schema_version is not None:
        envelope["schema_version"] = _validate_schema_version_for_envelope(schema_version)
    if id is not None:
        _validate_required_string("id", id)
        envelope["id"] = id
    if metadata is not None:
        envelope["metadata"] = _json_ready_for_envelope(metadata, "metadata")
    return envelope


def validate_envelope(
    data: Mapping[str, Any],
    *,
    schema: str | None = None,
    kind: str | None = None,
    require_payload: bool = True,
    require_json_ready: bool = True,
) -> Mapping[str, Any]:
    """Validate a generic envelope and return it unchanged.

    Parameters
    ----------
    data:
        Candidate envelope mapping.
    schema:
        Optional expected schema.
    kind:
        Optional expected kind.
    require_payload:
        Whether the ``payload`` key must be present.
    require_json_ready:
        Whether ``payload`` and ``metadata`` values must be canonical
        JSON-compatible during validation.

    Returns
    -------
    Mapping[str, Any]
        The original envelope mapping when validation succeeds.
    """

    if not isinstance(data, Mapping):
        raise EnvelopeError("envelope must be a mapping", context={"type": type(data).__name__})
    if "schema" not in data:
        raise EnvelopeError("envelope missing schema")
    if "kind" not in data:
        raise EnvelopeError("envelope missing kind")
    _validate_required_string("schema", data["schema"])
    _validate_required_string("kind", data["kind"])
    if schema is not None and data["schema"] != schema:
        raise EnvelopeError(
            "envelope schema mismatch",
            context={"expected": schema, "observed": data["schema"]},
        )
    if kind is not None and data["kind"] != kind:
        raise EnvelopeError(
            "envelope kind mismatch",
            context={"expected": kind, "observed": data["kind"]},
        )
    if require_payload and "payload" not in data:
        raise EnvelopeError("envelope missing payload")
    if "payload" in data and not isinstance(data["payload"], Mapping):
        raise EnvelopeError("envelope payload must be a mapping", context={"type": type(data["payload"]).__name__})
    if "metadata" in data and not isinstance(data["metadata"], Mapping):
        raise EnvelopeError("envelope metadata must be a mapping", context={"type": type(data["metadata"]).__name__})
    if "schema_version" in data:
        _validate_schema_version_for_envelope(data["schema_version"])
    if require_json_ready:
        if "payload" in data:
            _json_ready_for_envelope(data["payload"], "payload")
        if "metadata" in data:
            _json_ready_for_envelope(data["metadata"], "metadata")
    return data


def envelope_payload_for_id(data: Mapping[str, Any], *, include_id: bool = False) -> dict[str, Any]:
    """Return stable envelope fields intended for content-ID hashing.

    Parameters
    ----------
    data:
        Envelope mapping.
    include_id:
        Include the existing ``id`` key when explicitly requested.

    Returns
    -------
    dict[str, Any]
        Canonical JSON-compatible ID payload. Metadata is excluded because it is
        often volatile; include such fields inside ``payload`` when they are part
        of semantic identity.
    """

    validate_envelope(data)
    keys = ("schema", "kind", "schema_version", "payload")
    payload = {key: data[key] for key in keys if key in data}
    if include_id and "id" in data:
        payload["id"] = data["id"]
    return _json_ready_for_envelope(payload, "id_payload")


def _validate_required_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise EnvelopeError(f"envelope {name} must be a non-empty string", context={name: value})


def _json_ready_for_envelope(value: Any, field: str) -> Any:
    try:
        return json_ready(value)
    except CanonicalJSONError as exc:
        raise EnvelopeError(f"envelope {field} is not JSON serializable", context=exc.context) from exc


def _validate_schema_version_for_envelope(schema_version: Any) -> int:
    try:
        return validate_schema_version(schema_version)
    except ContentIDError as exc:
        raise EnvelopeError("envelope schema_version is invalid", context=exc.context) from exc


__all__ = [
    "EnvelopeSpec",
    "envelope_payload_for_id",
    "make_envelope",
    "validate_envelope",
]
