"""Operation execution helpers used inside local dispatch workers."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable

from dryml.core2.canonical import to_canonical
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.object import Object
from dryml.core2.repo import Repo
from dryml.core2.utils.general import pickle_load, pickle_save
from dryml.formats.refs import format_cdef_id, parse_cdef_id
from dryml.operations import resolve_call_arguments
from dryml.operations.errors import OperationResolutionError

from .errors import DispatchPlanningError, WorkerProtocolError


@dataclass(frozen=True, slots=True)
class PickledCallable:
    """Explicit same-environment convenience transport for a Python callable."""

    callable: Callable[..., Any]
    portable: bool = False
    transport: str = "pickle_small"

    def __post_init__(self) -> None:
        if not callable(self.callable):
            raise DispatchPlanningError("PickledCallable requires a callable")


def write_pickled_callable(func: Callable[..., Any], path: str) -> None:
    """Write a callable pickle as launch-time data, not operation identity."""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pickle_save(func, path)


def execute_operation(operation_spec: dict[str, Any], *, repo: Repo, envelope_launch: dict[str, Any] | None = None) -> tuple[Any, tuple[str, ...]]:
    """Resolve and execute a function_call or method_call operation in a worker."""

    launch = envelope_launch or {}
    if launch.get("call_transport") == "pickle_small":
        path = launch.get("pickle_path")
        if not isinstance(path, str):
            raise WorkerProtocolError("pickle_small transport requires pickle_path")
        func = pickle_load(path)
        call = resolve_call_arguments(operation_spec, materialize_cdef=lambda cdef_id: _materialize_cdef(repo, cdef_id), make_cdef_ref=lambda cdef_id: cdef_id)
        identity_arg_count = launch.get("identity_arg_count")
        args = call.args[:identity_arg_count] if isinstance(identity_arg_count, int) else call.args
        return func(*args, **call.kwargs), _consumed_cdefs(operation_spec)

    call = resolve_call_arguments(operation_spec, materialize_cdef=lambda cdef_id: _materialize_cdef(repo, cdef_id), make_cdef_ref=lambda cdef_id: cdef_id)
    if call.kind == "function_call":
        func = import_function(call.function or "")
        return func(*call.args, **call.kwargs), _consumed_cdefs(operation_spec)
    subject = call.subject
    method = resolve_attr(subject, call.method or "")
    return method(*call.args, **call.kwargs), _consumed_cdefs(operation_spec)


def import_function(path: str) -> Callable[..., Any]:
    """Import ``module:qualname`` after worker runtime activation."""

    module_name, sep, qualname = path.partition(":")
    if not sep or not module_name or not qualname:
        raise OperationResolutionError("function path must be module:qualname", context={"function": path})
    module = importlib.import_module(module_name)
    target = resolve_attr(module, qualname)
    if not callable(target):
        raise OperationResolutionError("imported function target is not callable", context={"function": path})
    return target


def resolve_attr(obj: Any, attr_path: str) -> Any:
    """Resolve a dotted attribute path without accepting private empty parts."""

    if not attr_path or any(part == "" for part in attr_path.split(".")):
        raise OperationResolutionError("attribute path must be dotted Python names", context={"attribute": attr_path})
    current = obj
    for part in attr_path.split("."):
        current = getattr(current, part)
    return current


def canonicalize_result(result: Any, *, repo: Repo, store: Any, record_policy: str = "descriptive") -> tuple[Any, tuple[str, ...]]:
    """Save object results when needed and return compact canonical refs."""

    produced: list[str] = []

    def convert(value: Any) -> Any:
        if isinstance(value, Object):
            repo.save(value, store=store, record_policy=record_policy)
            cdef_id = format_cdef_id(value.definition.stable_hash())
            produced.append(cdef_id)
            return cdef_id
        if isinstance(value, ConcreteDefinition):
            cdef_id = format_cdef_id(value.stable_hash())
            produced.append(cdef_id)
            return cdef_id
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [convert(item) for item in value]
        return value

    compact = convert(result)
    return to_canonical(compact, repo=repo), tuple(dict.fromkeys(produced))


def _materialize_cdef(repo: Repo, cdef_id: str) -> Any:
    parsed = parse_cdef_id(cdef_id)
    for store in repo.stores:
        try:
            def_path = os.path.join(store.object_dir_for_cdef_id(parsed.raw), "def.pkl")
        except Exception:
            continue
        if os.path.exists(def_path):
            cdef = pickle_load(def_path)
            if not isinstance(cdef, ConcreteDefinition):
                raise OperationResolutionError("stored def.pkl is not a ConcreteDefinition", context={"cdef_id": cdef_id})
            return repo.load(cdef)
    raise OperationResolutionError("CDef is not available in worker stores", context={"cdef_id": cdef_id})


def _consumed_cdefs(operation_spec: dict[str, Any]) -> tuple[str, ...]:
    consumed: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "$literal" in value:
                return
            for item in value.values():
                visit(item)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if isinstance(value, str):
            try:
                parsed = parse_cdef_id(value)
            except Exception:
                return
            consumed.append(parsed.raw)

    payload = operation_spec.get("payload", {})
    if operation_spec.get("kind") == "method_call" and isinstance(payload.get("subject"), str):
        visit(payload["subject"])
    visit(payload.get("args", []))
    visit(payload.get("kwargs", {}))
    return tuple(dict.fromkeys(consumed))


__all__ = [
    "PickledCallable",
    "canonicalize_result",
    "execute_operation",
    "import_function",
    "resolve_attr",
    "write_pickled_callable",
]
