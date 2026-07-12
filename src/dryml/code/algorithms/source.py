from __future__ import annotations

import inspect
import textwrap
import types
from dataclasses import dataclass

from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, FunctionAnalyzer
from dryml.code.facts import DiagnosticFact, SourceFact
from dryml.code.targets import CodeTarget


@dataclass(frozen=True)
class SourceInfo:
    """Source text and location for a Python object."""

    source: str
    filename: str | None
    start_line: int | None


def get_source_info(obj) -> SourceInfo | None:
    """Return source text/location for *obj*, or ``None`` if unavailable."""

    # inspect.getsourcelines(function) follows __wrapped__ by default. Reading
    # the code object directly avoids user-controlled wrapper metadata.
    if type(obj) is types.FunctionType:
        obj = object.__getattribute__(obj, "__code__")
    elif issubclass(type(obj), type) and type(obj) is not type:
        return None
    elif type(obj) is not types.CodeType and not issubclass(type(obj), type):
        return None
    try:
        lines, start_line = inspect.getsourcelines(obj)
        filename = inspect.getsourcefile(obj)
    except (OSError, TypeError):
        return None

    return SourceInfo(
        source=textwrap.dedent("".join(lines)),
        filename=filename,
        start_line=start_line,
    )


def func_source_extract(func):
    """Return dedented source text for *func* using the legacy helper behavior."""

    lines, _ = inspect.getsourcelines(func)
    return textwrap.dedent("".join(lines))


def analyze_target(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Produce a source fact for a normalized target."""

    if not context.allow_source:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="info",
            code="dryml.code.source_disabled",
            message="Source extraction is disabled by context.",
            source={"analyzer": "source", "target_kind": target.spec.kind},
        ),))
    obj = target.unwrapped or target.obj
    if obj is None:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="warning",
            code="dryml.code.source_unavailable",
            message="No live object is available for source extraction.",
            source={"analyzer": "source", "target_kind": target.spec.kind},
        ),))
    info = get_source_info(obj)
    if info is None:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="warning",
            code="dryml.code.source_unavailable",
            message="Source text is unavailable for the target.",
            source={"analyzer": "source", "target_kind": target.spec.kind},
            data={"object_type": type(obj).__name__},
        ),))
    line_count = len(info.source.splitlines())
    end_line = info.start_line + line_count - 1 if info.start_line is not None else None
    return CodeAnalysisResult(target=target.spec, facts=(SourceFact(
        source={"analyzer": "source", "target_kind": target.spec.kind, "filename": info.filename, "line": info.start_line},
        data={
            "filename": info.filename,
            "start_line": info.start_line,
            "end_line": end_line,
            "source": info.source,
            "dedented": True,
        },
    ),))


def can_analyze(target: CodeTarget, context: CodeAnalysisContext) -> bool:
    """Return true for targets that can potentially expose source."""

    return (
        target.obj is not None
        or target.unwrapped is not None
        or target.spec.source_spec is not None
        # An explicitly selected source analyzer must explain why an import-path
        # target cannot yield source while imports are disabled. Keep the default
        # lightweight tuple's established unavailable-target behavior unchanged.
        or (target.spec.import_path is not None and "source" in context.algorithms)
    )


ANALYZER = FunctionAnalyzer("source", analyze_target, can_analyze)


__all__ = ["ANALYZER", "SourceInfo", "func_source_extract", "get_source_info", "analyze_target"]
