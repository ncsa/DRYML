"""Resolver skeleton for operation-call argument semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from dryml.formats.errors import ReferenceParseError
from dryml.formats.refs import parse_reserved_ref, unwrap_literal_escape

from .errors import OperationResolutionError, OperationSpecError
from .specs import validate_operation_spec


@dataclass(frozen=True, slots=True)
class MaterializeCDefArg:
    """CDef ID placeholder returned when resolution has no materialization callback.

    ``resolve_call_arguments(...)`` returns this placeholder for a CDef reference
    when ``materialize_cdef`` is omitted. Current dispatch supplies that callback
    to materialize the referenced CDef instead.
    """

    cdef_id: str


@dataclass(frozen=True, slots=True)
class CDefRefArg:
    """Placeholder for passing a CDef identity rather than materializing it."""

    cdef_id: str


@dataclass(frozen=True, slots=True)
class ResolvedOperationCall:
    """Structured result of resolving an operation-call spec's arguments."""

    kind: Literal["function_call", "method_call"]
    function: str | None
    subject: MaterializeCDefArg | Any | None
    method: str | None
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    operation_id: str | None = None


def plan_call_resolution(spec: Mapping[str, Any]) -> ResolvedOperationCall:
    """Resolve operation arguments to placeholders without loading objects."""

    return resolve_call_arguments(spec)


def resolve_call_arguments(
    spec: Mapping[str, Any],
    *,
    materialize_cdef: Callable[[str], Any] | None = None,
    make_cdef_ref: Callable[[str], Any] | None = None,
) -> ResolvedOperationCall:
    """Resolve CDef/ref/literal semantics in an operation spec recursively.

    Supplied callbacks are applied to CDef materialization and non-materializing
    CDef refs. Without callbacks, placeholder dataclasses are returned.
    """

    try:
        normalized = validate_operation_spec(spec)
    except OperationSpecError as exc:
        raise OperationResolutionError(str(exc), context=exc.context) from exc
    payload = normalized["payload"]
    args = tuple(_resolve_value(item, materialize_cdef=materialize_cdef, make_cdef_ref=make_cdef_ref) for item in payload["args"])
    kwargs = {
        key: _resolve_value(value, materialize_cdef=materialize_cdef, make_cdef_ref=make_cdef_ref)
        for key, value in payload["kwargs"].items()
    }
    if normalized["kind"] == "function_call":
        return ResolvedOperationCall(
            kind="function_call",
            function=payload["function"],
            subject=None,
            method=None,
            args=args,
            kwargs=kwargs,
            operation_id=normalized.get("id"),
        )
    subject = _resolve_value(payload["subject"], materialize_cdef=materialize_cdef, make_cdef_ref=make_cdef_ref)
    return ResolvedOperationCall(
        kind="method_call",
        function=None,
        subject=subject,
        method=payload["method"],
        args=args,
        kwargs=kwargs,
        operation_id=normalized.get("id"),
    )


def _resolve_value(
    value: Any,
    *,
    materialize_cdef: Callable[[str], Any] | None,
    make_cdef_ref: Callable[[str], Any] | None,
) -> Any:
    if isinstance(value, Mapping):
        if "$literal" in value:
            try:
                return unwrap_literal_escape(value)
            except ReferenceParseError as exc:
                raise OperationResolutionError("invalid literal escape", context=exc.context) from exc
        return {key: _resolve_value(item, materialize_cdef=materialize_cdef, make_cdef_ref=make_cdef_ref) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, materialize_cdef=materialize_cdef, make_cdef_ref=make_cdef_ref) for item in value]
    if isinstance(value, str):
        try:
            ref = parse_reserved_ref(value)
        except ReferenceParseError as exc:
            raise OperationResolutionError("invalid reserved reference", context=exc.context) from exc
        if ref is None:
            return value
        if ref.kind == "cdef":
            return materialize_cdef(ref.target) if materialize_cdef is not None else MaterializeCDefArg(ref.target)  # type: ignore[arg-type]
        if ref.kind == "ref_cdef":
            return make_cdef_ref(ref.target) if make_cdef_ref is not None else CDefRefArg(ref.target)  # type: ignore[arg-type]
        return value
    return value


__all__ = [
    "CDefRefArg",
    "MaterializeCDefArg",
    "ResolvedOperationCall",
    "plan_call_resolution",
    "resolve_call_arguments",
]
