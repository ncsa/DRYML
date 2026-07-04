"""Canonical dispatch-intent spec helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.operations import validate_operation_spec
from dryml.operations.errors import OperationSpecError
from dryml.records import attach_spec_id, compute_spec_id, make_spec, spec_payload_for_id, validate_spec
from dryml.records.errors import RecordPolicyError, SpecValidationError
from dryml.records.policy import normalize_record_policy

from .errors import DispatchSpecError
from .links import json_ready_mapping, validate_id


DISPATCH_SCHEMA = "dryml.dispatch.v1"
DISPATCH_SCHEMA_VERSION = 1
DISPATCH_SPEC_FAMILY = "dispatch"
DISPATCH_KIND = "dispatch"

_PAYLOAD_FIELDS = frozenset({"operation_id", "operation", "environment", "world", "runtime", "records", "providers", "outputs", "execution", "metadata"})
_MAPPING_FIELDS = frozenset({"environment", "world", "runtime", "records", "providers", "outputs", "execution", "metadata"})


def make_dispatch_spec(
    *,
    operation_id: str,
    operation: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    world: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    records: Mapping[str, Any] | None = None,
    providers: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    execution: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    id: str | None = None,
    envelope_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical JSON-ready dispatch request-intent spec."""

    _report_step("dryml.dispatch.spec.build", "Building dispatch spec", operation_id=operation_id)
    payload: dict[str, Any] = {"operation_id": operation_id}
    if operation is not None:
        payload["operation"] = operation
    for name, value in (
        ("environment", environment),
        ("world", world),
        ("runtime", runtime),
        ("records", records),
        ("providers", providers),
        ("outputs", outputs),
        ("execution", execution),
        ("metadata", metadata),
    ):
        if value is not None:
            payload[name] = value
    spec = make_spec(family=DISPATCH_SPEC_FAMILY, kind=DISPATCH_KIND, payload=payload, id=id, metadata=envelope_metadata)
    result = validate_dispatch_spec(spec)
    _report_detail("dryml.dispatch.spec.build", "Dispatch spec built", operation_id=operation_id)
    return result


def validate_dispatch_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized dispatch spec."""

    shape_spec = dict(spec)
    existing_id = shape_spec.pop("id", None)
    try:
        validate_spec(shape_spec, family=DISPATCH_SPEC_FAMILY, kind=DISPATCH_KIND)
    except SpecValidationError as exc:
        raise DispatchSpecError(str(exc), context=exc.context) from exc
    payload = spec.get("payload")
    if not isinstance(payload, Mapping):
        raise DispatchSpecError("dispatch payload must be a mapping", context={"type": type(payload).__name__})
    unknown = set(payload) - _PAYLOAD_FIELDS
    if unknown:
        raise DispatchSpecError("dispatch payload contains unknown fields", context={"fields": sorted(unknown)})
    if "operation_id" not in payload:
        raise DispatchSpecError("dispatch payload requires operation_id")
    operation_id = validate_id(payload["operation_id"], ("op",), "operation_id")
    normalized_payload: dict[str, Any] = {"operation_id": operation_id}
    if "operation" in payload:
        try:
            operation = validate_operation_spec(payload["operation"])
        except OperationSpecError as exc:
            raise DispatchSpecError("embedded operation spec is invalid", context=exc.context) from exc
        if operation.get("id") != operation_id:
            raise DispatchSpecError("embedded operation ID must match operation_id", context={"operation_id": operation_id, "embedded_id": operation.get("id")})
        normalized_payload["operation"] = operation
    for field in sorted(_MAPPING_FIELDS):
        if field in payload:
            normalized_payload[field] = dict(json_ready_mapping(payload[field], field))
    records = normalized_payload.get("records")
    if isinstance(records, Mapping) and "record_policy" in records:
        try:
            normalize_record_policy(records["record_policy"])
        except RecordPolicyError as exc:
            raise DispatchSpecError("invalid dispatch records.record_policy", context=exc.context) from exc
    normalized = dict(spec)
    normalized["payload"] = normalized_payload
    if existing_id is not None:
        normalized["id"] = existing_id
    try:
        validate_spec(normalized, family=DISPATCH_SPEC_FAMILY, kind=DISPATCH_KIND)
    except SpecValidationError as exc:
        raise DispatchSpecError(str(exc), context=exc.context) from exc
    return normalized


def compute_dispatch_id(spec: Mapping[str, Any]) -> str:
    """Compute the canonical ``dispatch-v1-*`` ID for a dispatch spec."""

    return compute_spec_id(validate_dispatch_spec(spec), family=DISPATCH_SPEC_FAMILY)


def attach_dispatch_id(spec: Mapping[str, Any], *, verify_existing: bool = True) -> dict[str, Any]:
    """Return a copy of *spec* with its canonical dispatch ID attached."""

    normalized = validate_dispatch_spec(spec)
    try:
        return attach_spec_id(normalized, family=DISPATCH_SPEC_FAMILY, verify_existing=verify_existing)
    except SpecValidationError as exc:
        raise DispatchSpecError(str(exc), context=exc.context) from exc


def dispatch_payload_for_id(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return dispatch fields that participate in dispatch identity."""

    return spec_payload_for_id(validate_dispatch_spec(spec), family=DISPATCH_SPEC_FAMILY)


def _report_step(name: str, message: str, *, operation_id: str | None = None) -> None:
    try:
        from dryml import reporting

        reporting.step(name, message, operation_id=operation_id)
    except Exception:
        pass


def _report_detail(name: str, message: str, *, operation_id: str | None = None) -> None:
    try:
        from dryml import reporting

        reporting.detail(name, message, operation_id=operation_id)
    except Exception:
        pass


__all__ = [
    "DISPATCH_KIND",
    "DISPATCH_SCHEMA",
    "DISPATCH_SCHEMA_VERSION",
    "DISPATCH_SPEC_FAMILY",
    "attach_dispatch_id",
    "compute_dispatch_id",
    "dispatch_payload_for_id",
    "make_dispatch_spec",
    "validate_dispatch_spec",
]
