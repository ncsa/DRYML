"""Canonical operation-call spec helpers.

Operation specs are normal ``dryml.records`` specs in the existing
``operation`` family. This module validates the operation-specific payload
shape and delegates envelope normalization and ID computation to the spec
family machinery.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from dryml.formats.canonical import json_ready
from dryml.formats.errors import CanonicalJSONError, ReferenceParseError
from dryml.formats.refs import parse_cdef_id, parse_reserved_ref
from dryml.records import attach_spec_id, compute_spec_id, make_spec, spec_payload_for_id, validate_spec
from dryml.records.errors import SpecValidationError
from dryml.records.kinds import SPEC_FAMILIES

from .errors import OperationSpecError


OPERATION_SCHEMA = "dryml.operation.v1"
OPERATION_SCHEMA_VERSION = 1
OPERATION_SPEC_FAMILY = "operation"
OPERATION_KINDS = frozenset({"function_call", "method_call"})

_IMPORT_PATH_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_ATTR_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def make_function_call_spec(
    function: str,
    *,
    args: list[Any] | tuple[Any, ...] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical JSON-ready ``function_call`` operation spec.

    Parameters are semantic operation fields and are stored under ``payload``.
    Missing ``args`` and ``kwargs`` are normalized to ``[]`` and ``{}``.
    """

    payload = {
        "function": _validate_function_path(function),
        "args": _normalize_args(args),
        "kwargs": _normalize_kwargs(kwargs),
    }
    return make_spec(
        family=OPERATION_SPEC_FAMILY,
        kind="function_call",
        payload=payload,
        id=id,
        metadata=metadata,
    )


def make_method_call_spec(
    subject: str,
    method: str,
    *,
    args: list[Any] | tuple[Any, ...] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical JSON-ready ``method_call`` operation spec."""

    payload = {
        "subject": _validate_method_subject(subject),
        "method": _validate_method_path(method),
        "args": _normalize_args(args),
        "kwargs": _normalize_kwargs(kwargs),
    }
    return make_spec(
        family=OPERATION_SPEC_FAMILY,
        kind="method_call",
        payload=payload,
        id=id,
        metadata=metadata,
    )


def validate_operation_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized operation spec mapping.

    The returned mapping has defaulted ``args``/``kwargs`` fields. Any attached
    operation ID is checked against that normalized semantic payload.
    """

    normalized = _normalize_operation_spec(spec)
    try:
        validate_spec(normalized, family=OPERATION_SPEC_FAMILY)
    except SpecValidationError as exc:
        raise OperationSpecError(str(exc), context=exc.context) from exc
    return normalized


def compute_operation_id(spec: Mapping[str, Any]) -> str:
    """Compute the canonical ``op-v1-*`` ID for an operation spec."""

    return compute_spec_id(validate_operation_spec(spec), family=OPERATION_SPEC_FAMILY)


def attach_operation_id(spec: Mapping[str, Any], *, verify_existing: bool = True) -> dict[str, Any]:
    """Return a copy of *spec* with its canonical operation ID attached."""

    normalized = validate_operation_spec(spec)
    try:
        return attach_spec_id(normalized, family=OPERATION_SPEC_FAMILY, verify_existing=verify_existing)
    except SpecValidationError as exc:
        raise OperationSpecError(str(exc), context=exc.context) from exc


def operation_payload_for_id(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return operation spec fields that participate in operation identity."""

    return spec_payload_for_id(validate_operation_spec(spec), family=OPERATION_SPEC_FAMILY)


def _normalize_operation_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    _check_operation_family_constants()
    if not isinstance(spec, Mapping):
        raise OperationSpecError("operation spec must be a mapping", context={"type": type(spec).__name__})
    shape_spec = dict(spec)
    existing_id = shape_spec.pop("id", None)
    try:
        validate_spec(shape_spec, family=OPERATION_SPEC_FAMILY)
    except SpecValidationError as exc:
        raise OperationSpecError(str(exc), context=exc.context) from exc
    kind = spec.get("kind")
    if kind not in OPERATION_KINDS:
        raise OperationSpecError("unknown operation kind", context={"kind": kind})
    payload = spec.get("payload")
    if not isinstance(payload, Mapping):
        raise OperationSpecError("operation payload must be a mapping", context={"type": type(payload).__name__})
    normalized_payload = _normalize_payload(kind, payload)
    result = dict(spec)
    result["payload"] = normalized_payload
    if existing_id is not None:
        result["id"] = existing_id
    return result


def _normalize_payload(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"args", "kwargs", "function"} if kind == "function_call" else {"args", "kwargs", "method", "subject"}
    unknown = set(payload) - allowed
    if unknown:
        raise OperationSpecError("operation payload contains unknown fields", context={"fields": sorted(unknown)})
    if kind == "function_call":
        if "function" not in payload:
            raise OperationSpecError("function_call requires function")
        return {
            "function": _validate_function_path(payload["function"]),
            "args": _normalize_args(payload.get("args")),
            "kwargs": _normalize_kwargs(payload.get("kwargs")),
        }
    if "subject" not in payload:
        raise OperationSpecError("method_call requires subject")
    if "method" not in payload:
        raise OperationSpecError("method_call requires method")
    return {
        "subject": _validate_method_subject(payload["subject"]),
        "method": _validate_method_path(payload["method"]),
        "args": _normalize_args(payload.get("args")),
        "kwargs": _normalize_kwargs(payload.get("kwargs")),
    }


def _normalize_args(args: Any) -> list[Any]:
    if args is None:
        args = []
    if not isinstance(args, list | tuple):
        raise OperationSpecError("operation args must be a list", context={"type": type(args).__name__})
    try:
        normalized = json_ready(list(args))
    except CanonicalJSONError as exc:
        raise OperationSpecError("operation args are not canonical JSON", context=exc.context) from exc
    _validate_reserved_refs(normalized)
    return normalized


def _normalize_kwargs(kwargs: Any) -> dict[str, Any]:
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise OperationSpecError("operation kwargs must be a mapping", context={"type": type(kwargs).__name__})
    try:
        normalized = json_ready(dict(kwargs))
    except CanonicalJSONError as exc:
        raise OperationSpecError("operation kwargs are not canonical JSON", context=exc.context) from exc
    _validate_reserved_refs(normalized)
    return normalized


def _validate_reserved_refs(value: Any) -> None:
    if isinstance(value, Mapping):
        if "$literal" in value:
            try:
                parse_reserved_ref(value)
            except ReferenceParseError as exc:
                raise OperationSpecError("invalid literal escape", context=exc.context) from exc
            return
        for item in value.values():
            _validate_reserved_refs(item)
        return
    if isinstance(value, list):
        for item in value:
            _validate_reserved_refs(item)
        return
    if isinstance(value, str):
        try:
            parse_reserved_ref(value)
        except ReferenceParseError as exc:
            raise OperationSpecError("invalid reserved reference", context=exc.context) from exc


def _validate_function_path(function: Any) -> str:
    if not isinstance(function, str) or _IMPORT_PATH_RE.fullmatch(function) is None:
        raise OperationSpecError("function must be a non-empty module:qualname import path", context={"function": function})
    return function


def _validate_method_subject(subject: Any) -> str:
    if not isinstance(subject, str):
        raise OperationSpecError("method subject must be a CDef ID string", context={"type": type(subject).__name__})
    if subject.startswith("ref("):
        raise OperationSpecError("method subject must be a raw CDef ID, not ref(cdef)", context={"subject": subject})
    try:
        return parse_cdef_id(subject).raw
    except ReferenceParseError as exc:
        raise OperationSpecError("method subject must be a valid CDef ID", context=exc.context) from exc


def _validate_method_path(method: Any) -> str:
    if not isinstance(method, str) or _ATTR_PATH_RE.fullmatch(method) is None:
        raise OperationSpecError("method must be a dotted Python attribute path", context={"method": method})
    return method


def _check_operation_family_constants() -> None:
    info = SPEC_FAMILIES[OPERATION_SPEC_FAMILY]
    if info.schema != OPERATION_SCHEMA or info.schema_version != OPERATION_SCHEMA_VERSION:
        raise OperationSpecError("operation constants do not match spec-family metadata")


__all__ = [
    "OPERATION_KINDS",
    "OPERATION_SCHEMA",
    "OPERATION_SCHEMA_VERSION",
    "OPERATION_SPEC_FAMILY",
    "attach_operation_id",
    "compute_operation_id",
    "make_function_call_spec",
    "make_method_call_spec",
    "operation_payload_for_id",
    "validate_operation_spec",
]
