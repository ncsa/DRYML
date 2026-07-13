from __future__ import annotations

import importlib
import inspect
import types
from dataclasses import dataclass
from typing import Any

from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, FunctionAnalyzer
from dryml.code.facts import CallableFact, DiagnosticFact
from dryml.code.targets import CodeTarget


@dataclass(frozen=True)
class CallableInfo:
    """Normalized information about a Python callable.

    Args:
        original: Original object passed by the caller.
        func: Underlying function or ``type(obj).__call__`` for callable instances.
        bound_self: Bound ``self``/``cls`` or callable instance when present.
        signature: Inspected signature for the underlying callable.
        qualname: Qualified name for the underlying callable.
        module: Module for the underlying callable.
        is_bound_method: Whether the original object is a bound method.
        is_function: Whether the original object is a function.
        is_callable_instance: Whether the original object is a callable instance.
    """

    original: object
    func: object
    bound_self: object | None
    signature: inspect.Signature
    qualname: str | None
    module: str | None
    is_bound_method: bool
    is_function: bool
    is_callable_instance: bool


def analyze_callable(obj) -> CallableInfo:
    """Inspect *obj* without invoking its callable or metadata hooks."""

    if type(obj) is types.MethodType:
        func = object.__getattribute__(obj, "__func__")
        bound_self = object.__getattribute__(obj, "__self__")
        return CallableInfo(
            original=obj,
            func=func,
            bound_self=bound_self,
            signature=inspect.signature(func),
            qualname=_safe_callable_metadata(func, "__qualname__"),
            module=_safe_callable_metadata(func, "__module__"),
            is_bound_method=True,
            is_function=False,
            is_callable_instance=False,
        )

    if type(obj) is types.FunctionType:
        return CallableInfo(
            original=obj,
            func=obj,
            bound_self=None,
            signature=inspect.signature(obj),
            qualname=_safe_callable_metadata(obj, "__qualname__"),
            module=_safe_callable_metadata(obj, "__module__"),
            is_bound_method=False,
            is_function=True,
            is_callable_instance=False,
        )

    if callable(obj):
        call = inspect.getattr_static(type(obj), "__call__", None)
        if type(call) in {staticmethod, classmethod}:
            call = object.__getattribute__(call, "__func__")
        if call is None:
            raise TypeError("Callable object has no statically discoverable type.__call__")
        return CallableInfo(
            original=obj,
            func=call,
            bound_self=obj,
            signature=inspect.signature(call),
            qualname=_safe_callable_metadata(call, "__qualname__"),
            module=_safe_callable_metadata(call, "__module__"),
            is_bound_method=False,
            is_function=False,
            is_callable_instance=True,
        )

    raise TypeError("Object is not callable")


def analyze_target(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Produce a callable fact for a normalized target."""

    obj = target.obj
    if obj is None or not callable(obj):
        return CodeAnalysisResult(target=target.spec)

    diagnostics: list[DiagnosticFact] = []
    info: CallableInfo | None = None
    signature = None
    try:
        info = analyze_callable(obj)
        signature = str(info.signature)
    except Exception as exc:
        diagnostics.append(DiagnosticFact(
            severity="warning",
            code="dryml.code.signature_unavailable",
            message="Callable signature could not be inspected.",
            source={"analyzer": "callables", "target_kind": target.spec.kind},
            data={"error": repr(exc)},
        ))

    analyzed = info.func if info is not None else obj
    import_path = _import_path_for(analyzed)
    if import_path is None and target.spec.import_path is not None:
        import_path = target.spec.import_path
    importable = import_path is not None and _verify_import_path(import_path, analyzed)
    if not importable:
        diagnostics.append(DiagnosticFact(
            severity="warning",
            code="dryml.code.not_importable",
            message="Callable does not have a verified stable import path.",
            source={"analyzer": "callables", "target_kind": target.spec.kind},
            data={
                "module": _safe_callable_metadata(analyzed, "__module__"),
                "qualname": _safe_callable_metadata(analyzed, "__qualname__"),
            },
        ))

    owner_module, owner_qualname = _safe_owner_metadata(target.owner)
    data = {
        "module": _safe_callable_metadata(analyzed, "__module__"),
        "qualname": _safe_callable_metadata(analyzed, "__qualname__"),
        "name": _safe_callable_metadata(analyzed, "__name__") or _safe_type_name(obj),
        "is_callable": callable(obj),
        "is_function": type(obj) is types.FunctionType,
        "is_bound_method": type(obj) is types.MethodType,
        "is_lambda": _safe_callable_metadata(analyzed, "__name__") == "<lambda>",
        "is_callable_instance": bool(info.is_callable_instance) if info is not None else False,
        "owner_module": owner_module,
        "owner_qualname": owner_qualname,
        "signature": signature,
        "importable": importable,
        "import_path": import_path if importable else None,
    }
    return CodeAnalysisResult(
        target=target.spec,
        facts=(CallableFact(source=_source(target), data=data),),
        diagnostics=tuple(diagnostics),
    )


def _safe_owner_metadata(owner: type | None) -> tuple[str | None, str | None]:
    """Return ordinary-class owner metadata without metaclass dispatch."""

    if type(owner) is not type:
        return None, None
    module = object.__getattribute__(owner, "__module__")
    qualname = object.__getattribute__(owner, "__qualname__")
    return (
        module if isinstance(module, str) else None,
        qualname if isinstance(qualname, str) else None,
    )


def _safe_callable_metadata(obj: Any, name: str) -> str | None:
    """Read standard callable metadata without arbitrary attribute dispatch."""

    if type(obj) not in {
        types.FunctionType,
        types.MethodType,
        types.BuiltinFunctionType,
        types.BuiltinMethodType,
    }:
        return None
    value = object.__getattribute__(obj, name)
    return value if isinstance(value, str) else None


def _safe_type_name(obj: Any) -> str:
    """Return a type name without consulting a target object's hooks."""

    name = object.__getattribute__(type(obj), "__name__")
    return name if isinstance(name, str) else "callable"


def can_analyze(target: CodeTarget, context: CodeAnalysisContext) -> bool:
    """Return true for live callable targets."""

    return target.obj is not None and callable(target.obj)


def _source(target: CodeTarget) -> dict[str, Any]:
    return {
        "analyzer": "callables",
        "target_kind": target.spec.kind,
        "module": target.spec.metadata.get("module"),
        "qualname": target.spec.metadata.get("qualname"),
    }


def _import_path_for(obj: Any) -> str | None:
    module = _safe_callable_metadata(obj, "__module__")
    qualname = _safe_callable_metadata(obj, "__qualname__")
    if not module or not qualname or module == "__main__" or "<locals>" in qualname:
        return None
    return f"{module}:{qualname}"


def _verify_import_path(path: str, obj: Any) -> bool:
    try:
        module_name, qualname = path.split(":", 1)
        module = importlib.import_module(module_name)
        resolved: Any = module
        for part in qualname.split("."):
            resolved = inspect.getattr_static(resolved, part)
        if type(resolved) in {staticmethod, classmethod}:
            resolved = object.__getattribute__(resolved, "__func__")
    except Exception:
        return False
    return resolved is obj


ANALYZER = FunctionAnalyzer("callables", analyze_target, can_analyze)


__all__ = ["ANALYZER", "CallableInfo", "analyze_callable", "analyze_target"]
