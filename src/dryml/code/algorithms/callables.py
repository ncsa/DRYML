from __future__ import annotations

import importlib
import inspect
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
    """Inspect *obj* and return compatibility callable information."""

    if inspect.ismethod(obj):
        func = obj.__func__
        bound_self = obj.__self__
        return CallableInfo(
            original=obj,
            func=func,
            bound_self=bound_self,
            signature=inspect.signature(func),
            qualname=getattr(func, "__qualname__", None),
            module=getattr(func, "__module__", None),
            is_bound_method=True,
            is_function=False,
            is_callable_instance=False,
        )

    if inspect.isfunction(obj):
        return CallableInfo(
            original=obj,
            func=obj,
            bound_self=None,
            signature=inspect.signature(obj),
            qualname=getattr(obj, "__qualname__", None),
            module=getattr(obj, "__module__", None),
            is_bound_method=False,
            is_function=True,
            is_callable_instance=False,
        )

    if callable(obj):
        call = getattr(type(obj), "__call__", None)
        if call is None:
            raise TypeError(f"Callable object {obj!r} has no type.__call__")
        return CallableInfo(
            original=obj,
            func=call,
            bound_self=obj,
            signature=inspect.signature(call),
            qualname=getattr(call, "__qualname__", None),
            module=getattr(call, "__module__", None),
            is_bound_method=False,
            is_function=False,
            is_callable_instance=True,
        )

    raise TypeError(f"Object {obj!r} is not callable")


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

    analyzed = info.func if info is not None else getattr(obj, "__func__", obj)
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
            data={"module": getattr(analyzed, "__module__", None), "qualname": getattr(analyzed, "__qualname__", None)},
        ))

    data = {
        "module": getattr(analyzed, "__module__", None),
        "qualname": getattr(analyzed, "__qualname__", None),
        "name": getattr(analyzed, "__name__", type(obj).__name__),
        "is_callable": callable(obj),
        "is_function": inspect.isfunction(obj),
        "is_bound_method": inspect.ismethod(obj),
        "is_lambda": getattr(analyzed, "__name__", None) == "<lambda>",
        "is_callable_instance": bool(info.is_callable_instance) if info is not None else False,
        "owner_module": getattr(target.owner, "__module__", None) if target.owner else None,
        "owner_qualname": getattr(target.owner, "__qualname__", None) if target.owner else None,
        "signature": signature,
        "importable": importable,
        "import_path": import_path if importable else None,
    }
    return CodeAnalysisResult(
        target=target.spec,
        facts=(CallableFact(source=_source(target), data=data),),
        diagnostics=tuple(diagnostics),
    )


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
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not module or not qualname or module == "__main__" or "<locals>" in qualname:
        return None
    return f"{module}:{qualname}"


def _verify_import_path(path: str, obj: Any) -> bool:
    try:
        module_name, qualname = path.split(":", 1)
        module = importlib.import_module(module_name)
        resolved = module
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except Exception:
        return False
    return resolved is obj


ANALYZER = FunctionAnalyzer("callables", analyze_target, can_analyze)


__all__ = ["ANALYZER", "CallableInfo", "analyze_callable", "analyze_target"]
