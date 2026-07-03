"""Canonical record envelope helpers for store-owned DRYML sidecars."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.formats.envelope import envelope_payload_for_id, make_envelope, validate_envelope
from dryml.formats.errors import ContentIDError, EnvelopeError
from dryml.formats.ids import content_id, parse_content_id

from .errors import RecordValidationError


RECORD_SCHEMA = "dryml.record.v1"
RECORD_SCHEMA_VERSION = 1
RECORD_ID_PREFIX = "record"


def make_record(
    *,
    kind: str,
    payload: Mapping[str, Any] | None = None,
    id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical JSON-ready Sprint 1 record envelope."""

    try:
        record = make_envelope(
            schema=RECORD_SCHEMA,
            schema_version=RECORD_SCHEMA_VERSION,
            kind=kind,
            payload=payload,
            id=id,
            metadata=metadata,
        )
    except EnvelopeError as exc:
        raise RecordValidationError("invalid record envelope", context=exc.context) from exc
    return validate_record(record)


def validate_record(record: Mapping[str, Any], *, kind: str | None = None) -> Mapping[str, Any]:
    """Validate a record envelope and any attached record ID."""

    _validate_record_shape(record, kind=kind)
    if "id" in record:
        _validate_record_id(record["id"])
        computed = compute_record_id(record)
        if record["id"] != computed:
            raise RecordValidationError(
                "record ID does not match record payload",
                context={"expected": computed, "observed": record["id"]},
            )
    return record


def record_payload_for_id(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return record envelope fields that participate in record identity."""

    _validate_record_shape(record)
    return envelope_payload_for_id(record)


def compute_record_id(record: Mapping[str, Any]) -> str:
    """Compute the canonical ``record-v1-*`` ID for a record envelope."""

    return content_id(RECORD_ID_PREFIX, RECORD_SCHEMA_VERSION, record_payload_for_id(record))


def attach_record_id(record: Mapping[str, Any], *, verify_existing: bool = True) -> dict[str, Any]:
    """Return a copy of *record* with its canonical record ID attached."""

    _validate_record_shape(record)
    computed = compute_record_id(record)
    if verify_existing and "id" in record and record["id"] != computed:
        raise RecordValidationError(
            "record ID does not match record payload",
            context={"expected": computed, "observed": record["id"]},
        )
    result = dict(record)
    result["id"] = computed
    return result


def _validate_record_shape(record: Mapping[str, Any], *, kind: str | None = None) -> None:
    try:
        validate_envelope(record, schema=RECORD_SCHEMA, kind=kind)
    except EnvelopeError as exc:
        raise RecordValidationError(str(exc), context=exc.context) from exc
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise RecordValidationError(
            "record schema_version mismatch",
            context={"expected": RECORD_SCHEMA_VERSION, "observed": record.get("schema_version")},
        )


def _validate_record_id(record_id: str) -> None:
    try:
        parts = parse_content_id(record_id)
    except ContentIDError as exc:
        raise RecordValidationError("invalid record ID", context=exc.context) from exc
    if parts.prefix != RECORD_ID_PREFIX or parts.schema_version != RECORD_SCHEMA_VERSION:
        raise RecordValidationError(
            "record ID must use record-v1 prefix",
            context={"prefix": parts.prefix, "schema_version": parts.schema_version},
        )


__all__ = [
    "RECORD_ID_PREFIX",
    "RECORD_SCHEMA",
    "RECORD_SCHEMA_VERSION",
    "attach_record_id",
    "compute_record_id",
    "make_record",
    "record_payload_for_id",
    "validate_record",
]
