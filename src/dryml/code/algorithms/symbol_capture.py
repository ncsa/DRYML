from __future__ import annotations

import inspect
from typing import Any

from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, FunctionAnalyzer
from dryml.code.facts import DiagnosticFact, SymbolFact
from dryml.code.targets import CodeTarget
from dryml.core2.symbol import ImportRef, SourceSpec, symbol_ref


def analyze_target(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Expose core symbol references as reusable code facts."""

    obj = target.unwrapped if target.unwrapped is not None else target.obj
    if target.spec.import_path and obj is None:
        return CodeAnalysisResult(target=target.spec, facts=(SymbolFact(
            source={"analyzer": "symbol_capture", "target_kind": target.spec.kind},
            data={"symbol_kind": "import_ref", "import_path": target.spec.import_path, "portable": True},
        ),))
    if obj is None:
        return CodeAnalysisResult(target=target.spec)

    try:
        ref = symbol_ref(obj)
    except Exception as exc:
        code = "dryml.code.closure_unsupported" if _has_closure(obj) else "dryml.code.symbol_capture_failed"
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="warning",
            code=code,
            message="Could not capture a stable symbol reference for the target.",
            source={"analyzer": "symbol_capture", "target_kind": target.spec.kind},
            data={"error": repr(exc)},
        ),))

    return CodeAnalysisResult(target=target.spec, facts=(SymbolFact(
        source={"analyzer": "symbol_capture", "target_kind": target.spec.kind},
        data=_symbol_to_data(ref),
    ),))


def can_analyze(target: CodeTarget, context: CodeAnalysisContext) -> bool:
    """Return true for import-path, function, and class targets."""

    obj = target.unwrapped if target.unwrapped is not None else target.obj
    return target.spec.import_path is not None or inspect.isfunction(obj) or inspect.isclass(obj)


def _has_closure(obj: Any) -> bool:
    return bool(getattr(getattr(obj, "__code__", None), "co_freevars", ()))


def _symbol_to_data(ref: ImportRef | SourceSpec) -> dict[str, Any]:
    if isinstance(ref, ImportRef):
        return {"symbol_kind": "import_ref", "import_path": ref.import_path(), "portable": True}
    return {
        "symbol_kind": "source_spec",
        "portable": False,
        "source_spec": {
            "kind": ref.kind,
            "source": ref.source,
            "name": ref.name,
            "imports": {name: dep.import_path() for name, dep in (ref.imports or {}).items()},
        },
        "imports": [dep.import_path() for dep in (ref.imports or {}).values()],
    }


ANALYZER = FunctionAnalyzer("symbol_capture", analyze_target, can_analyze)


__all__ = ["ANALYZER", "analyze_target"]
