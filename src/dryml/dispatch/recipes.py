"""Canonical resolved execution-recipe spec helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.records import attach_spec_id, compute_spec_id, make_spec, spec_payload_for_id, validate_spec
from dryml.records.errors import SpecValidationError

from .errors import DispatchSpecError
from .links import json_ready_mapping, normalize_backend_identity, validate_id


EXECUTION_RECIPE_SCHEMA = "dryml.execution_recipe.v1"
EXECUTION_RECIPE_SCHEMA_VERSION = 1
EXECUTION_RECIPE_SPEC_FAMILY = "execution_recipe"
EXECUTION_RECIPE_KIND = "execution_recipe"

_PAYLOAD_FIELDS = frozenset(
    {
        "dispatch_id",
        "operation_id",
        "backend",
        "environment_id",
        "environment_record_id",
        "environment_spec_id",
        "environment_requirement_id",
        "world_requirement_id",
        "world_id",
        "world_allocation_id",
        "runtime_id",
        "annotation_report",
        "probe_report_ids",
        "input_plan",
        "record_resolution",
        "output_plan",
        "log_plan",
        "store_plan",
        "constraints",
    }
)
_MAPPING_FIELDS = frozenset({"annotation_report", "input_plan", "record_resolution", "output_plan", "log_plan", "store_plan", "constraints"})
_ID_FIELDS = {
    "dispatch_id": ("dispatch",),
    "operation_id": ("op",),
    "environment_id": ("envrec", "env"),
    "environment_record_id": ("envrec",),
    "environment_spec_id": ("envspec",),
    "environment_requirement_id": ("envreq",),
    "world_requirement_id": ("worldreq",),
    "world_id": ("world",),
    "world_allocation_id": ("worldalloc",),
    "runtime_id": ("runtime",),
}


def make_execution_recipe(
    *,
    dispatch_id: str,
    operation_id: str,
    backend: Mapping[str, Any],
    id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build a canonical JSON-ready resolved execution-recipe spec."""

    _report_step("dryml.dispatch.recipe.build", "Building execution recipe", operation_id=operation_id, data={"dispatch_id": dispatch_id})
    payload = {"dispatch_id": dispatch_id, "operation_id": operation_id, "backend": backend}
    payload.update({key: value for key, value in fields.items() if value is not None})
    spec = make_spec(family=EXECUTION_RECIPE_SPEC_FAMILY, kind=EXECUTION_RECIPE_KIND, payload=payload, id=id, metadata=metadata)
    result = validate_execution_recipe(spec)
    _report_detail("dryml.dispatch.recipe.build", "Execution recipe built", operation_id=operation_id, data={"dispatch_id": dispatch_id})
    return result


def validate_execution_recipe(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized execution-recipe spec."""

    shape_spec = dict(spec)
    existing_id = shape_spec.pop("id", None)
    try:
        validate_spec(shape_spec, family=EXECUTION_RECIPE_SPEC_FAMILY, kind=EXECUTION_RECIPE_KIND)
    except SpecValidationError as exc:
        raise DispatchSpecError(str(exc), context=exc.context) from exc
    payload = spec.get("payload")
    if not isinstance(payload, Mapping):
        raise DispatchSpecError("execution recipe payload must be a mapping", context={"type": type(payload).__name__})
    unknown = set(payload) - _PAYLOAD_FIELDS
    if unknown:
        raise DispatchSpecError("execution recipe payload contains unknown fields", context={"fields": sorted(unknown)})
    for required in ("dispatch_id", "operation_id", "backend"):
        if required not in payload:
            raise DispatchSpecError("execution recipe payload missing required field", context={"field": required})
    normalized_payload: dict[str, Any] = {
        "dispatch_id": validate_id(payload["dispatch_id"], ("dispatch",), "dispatch_id"),
        "operation_id": validate_id(payload["operation_id"], ("op",), "operation_id"),
        "backend": dict(normalize_backend_identity(payload["backend"])),
    }
    for field_name, prefixes in _ID_FIELDS.items():
        if field_name in {"dispatch_id", "operation_id"} or field_name not in payload:
            continue
        normalized_payload[field_name] = validate_id(payload[field_name], prefixes, field_name)
    if "probe_report_ids" in payload:
        normalized_payload["probe_report_ids"] = _id_list(payload["probe_report_ids"], ("record",), "probe_report_ids")
    for field in sorted(_MAPPING_FIELDS):
        if field in payload:
            normalized_payload[field] = dict(json_ready_mapping(payload[field], field))
    normalized = dict(spec)
    normalized["payload"] = normalized_payload
    if existing_id is not None:
        normalized["id"] = existing_id
    try:
        validate_spec(normalized, family=EXECUTION_RECIPE_SPEC_FAMILY, kind=EXECUTION_RECIPE_KIND)
    except SpecValidationError as exc:
        raise DispatchSpecError(str(exc), context=exc.context) from exc
    return normalized


def compute_recipe_id(spec: Mapping[str, Any]) -> str:
    """Compute the canonical ``recipe-v1-*`` ID for an execution recipe."""

    return compute_spec_id(validate_execution_recipe(spec), family=EXECUTION_RECIPE_SPEC_FAMILY)


def attach_recipe_id(spec: Mapping[str, Any], *, verify_existing: bool = True) -> dict[str, Any]:
    """Return a copy of *spec* with its canonical recipe ID attached."""

    normalized = validate_execution_recipe(spec)
    try:
        return attach_spec_id(normalized, family=EXECUTION_RECIPE_SPEC_FAMILY, verify_existing=verify_existing)
    except SpecValidationError as exc:
        raise DispatchSpecError(str(exc), context=exc.context) from exc


def recipe_payload_for_id(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return recipe fields that participate in recipe identity."""

    return spec_payload_for_id(validate_execution_recipe(spec), family=EXECUTION_RECIPE_SPEC_FAMILY)


def _id_list(value: Any, prefixes: tuple[str, ...], field_name: str) -> list[str]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise DispatchSpecError(f"{field_name} must be a list", context={"type": type(value).__name__})
    return [validate_id(item, prefixes, field_name) for item in value]


def _report_step(name: str, message: str, *, operation_id: str | None = None, data: Mapping[str, Any] | None = None) -> None:
    try:
        from dryml import reporting

        reporting.step(name, message, operation_id=operation_id, data=data or {})
    except Exception:
        pass


def _report_detail(name: str, message: str, *, operation_id: str | None = None, data: Mapping[str, Any] | None = None) -> None:
    try:
        from dryml import reporting

        reporting.detail(name, message, operation_id=operation_id, data=data or {})
    except Exception:
        pass


__all__ = [
    "EXECUTION_RECIPE_KIND",
    "EXECUTION_RECIPE_SCHEMA",
    "EXECUTION_RECIPE_SCHEMA_VERSION",
    "EXECUTION_RECIPE_SPEC_FAMILY",
    "attach_recipe_id",
    "compute_recipe_id",
    "make_execution_recipe",
    "recipe_payload_for_id",
    "validate_execution_recipe",
]
