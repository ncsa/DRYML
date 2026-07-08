from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .facts import CodeFact, DiagnosticFact, json_compatible
from .targets import CodeTarget, CodeTargetSpec, normalize_target


DEFAULT_ALGORITHMS = (
    "callables",
    "source",
    "ast_access",
    "symbol_capture",
    "direct_annotations",
    "method_contracts",
)


class CodeAnalysisError(RuntimeError):
    """Raised when code analysis is configured to raise instead of collect."""


@dataclass(frozen=True, slots=True)
class CodeAnalysisContext:
    """Options controlling code-analysis execution.

    Args:
        algorithms: Ordered analyzer names. Empty means the default lightweight set.
        allow_import: Whether import-path targets may be imported.
        allow_source: Whether source extraction is allowed.
        allow_dynamic_execution: Reserved for future dynamic analyzers; defaults false.
        include_annotations: Whether annotation analyzers should collect fragments.
        include_method_contracts: Whether method-contract analyzers should run.
        diagnostics_policy: ``"collect"`` converts failures to diagnostics; ``"raise"`` raises.
        metadata: JSON-compatible caller metadata.
    """

    algorithms: tuple[str, ...] = ()
    allow_import: bool = True
    allow_source: bool = True
    allow_dynamic_execution: bool = False
    include_annotations: bool = True
    include_method_contracts: bool = True
    diagnostics_policy: str = "collect"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "algorithms", tuple(self.algorithms or ()))
        object.__setattr__(self, "metadata", json_compatible(self.metadata))


@dataclass(frozen=True, slots=True)
class CodeAnalysisResult:
    """Serializable aggregate output from one or more code analyzers."""

    target: CodeTargetSpec
    facts: tuple[CodeFact, ...] = ()
    diagnostics: tuple[DiagnosticFact, ...] = ()

    @property
    def ok(self) -> bool:
        """Return true when no error-severity diagnostics were emitted."""

        return not any(d.severity == "error" for d in self.diagnostics)

    def facts_of_kind(self, kind: str) -> tuple[CodeFact, ...]:
        """Return facts matching *kind* in original order."""

        return tuple(fact for fact in self.facts if fact.kind == kind)

    def diagnostics_of_code(self, code: str) -> tuple[DiagnosticFact, ...]:
        """Return diagnostics matching *code* in original order."""

        return tuple(diagnostic for diagnostic in self.diagnostics if diagnostic.code == code)

    def extend(self, other: "CodeAnalysisResult") -> "CodeAnalysisResult":
        """Return a result containing this result followed by *other*."""

        return CodeAnalysisResult(
            target=self.target,
            facts=self.facts + other.facts,
            diagnostics=self.diagnostics + other.diagnostics,
        )

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-compatible result representation."""

        return {
            "target": self.target.to_data(),
            "facts": [fact.to_data() for fact in self.facts],
            "diagnostics": [diagnostic.to_data() for diagnostic in self.diagnostics],
            "ok": self.ok,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CodeAnalysisResult":
        """Build an analysis result from :meth:`to_data` output."""

        return cls(
            target=CodeTargetSpec.from_data(data.get("target") or {}),
            facts=tuple(CodeFact.from_data(item) for item in data.get("facts") or ()),
            diagnostics=tuple(DiagnosticFact.from_data(item) for item in data.get("diagnostics") or ()),
        )


class CodeAnalyzer(Protocol):
    """Protocol implemented by pluggable code analyzers."""

    name: str

    def can_analyze(self, target: CodeTarget, context: CodeAnalysisContext) -> bool:
        """Return true when this analyzer can inspect *target*."""

    def analyze(self, target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
        """Analyze *target* and return facts/diagnostics."""


@dataclass(frozen=True, slots=True)
class FunctionAnalyzer:
    """Small adapter for function-based analyzers."""

    name: str
    fn: Callable[[CodeTarget, CodeAnalysisContext], CodeAnalysisResult]
    can_fn: Callable[[CodeTarget, CodeAnalysisContext], bool] | None = None

    def can_analyze(self, target: CodeTarget, context: CodeAnalysisContext) -> bool:
        """Return true when this analyzer should run for *target*."""

        return True if self.can_fn is None else self.can_fn(target, context)

    def analyze(self, target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
        """Run the wrapped analyzer function."""

        return self.fn(target, context)


_ANALYZERS: dict[str, CodeAnalyzer] = {}
_BUILTINS_REGISTERED = False


def register_analyzer(analyzer: CodeAnalyzer, *, replace: bool = False) -> None:
    """Register *analyzer* by name.

    Args:
        analyzer: Analyzer object implementing :class:`CodeAnalyzer`.
        replace: Whether an existing analyzer with the same name may be replaced.
    """

    name = analyzer.name
    if name in _ANALYZERS and not replace:
        raise ValueError(f"Analyzer {name!r} is already registered.")
    _ANALYZERS[name] = analyzer


def get_analyzer(name: str) -> CodeAnalyzer:
    """Return a registered analyzer by name."""

    _ensure_builtin_analyzers()
    try:
        return _ANALYZERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown code analyzer {name!r}.") from exc


def available_analyzers() -> tuple[str, ...]:
    """Return registered analyzer names in registration order."""

    _ensure_builtin_analyzers()
    return tuple(_ANALYZERS)


def analyze(
    target: Any,
    *,
    algorithms: Iterable[str] | None = None,
    context: CodeAnalysisContext | None = None,
) -> CodeAnalysisResult:
    """Analyze a Python/DRYML code target and return reusable facts.

    Args:
        target: Live object, import path string, target spec, or target wrapper.
        algorithms: Optional ordered analyzer names overriding the context.
        context: Optional analysis options.

    Returns:
        A serializable result containing facts and diagnostics.
    """

    context = context or CodeAnalysisContext()
    if algorithms is not None:
        context = replace(context, algorithms=tuple(algorithms))
    selected = context.algorithms or DEFAULT_ALGORITHMS
    code_target = normalize_target(target, allow_import=context.allow_import)
    result = CodeAnalysisResult(target=code_target.spec, diagnostics=code_target.diagnostics)

    for name in selected:
        try:
            analyzer = get_analyzer(name)
        except Exception as exc:
            if context.diagnostics_policy == "raise":
                raise
            result = result.extend(CodeAnalysisResult(
                target=code_target.spec,
                diagnostics=(DiagnosticFact(
                    severity="error",
                    code="dryml.code.unknown_analyzer",
                    message=f"Unknown code analyzer {name!r}.",
                    data={"error": repr(exc), "analyzer": name},
                ),),
            ))
            continue

        try:
            if not analyzer.can_analyze(code_target, context):
                continue
            partial = analyzer.analyze(code_target, context)
        except Exception as exc:
            if context.diagnostics_policy == "raise":
                raise CodeAnalysisError(f"Analyzer {name!r} failed.") from exc
            partial = CodeAnalysisResult(
                target=code_target.spec,
                diagnostics=(DiagnosticFact(
                    severity="error",
                    code="dryml.code.algorithm_failed",
                    message=f"Analyzer {name!r} failed.",
                    source={"analyzer": name, "target_kind": code_target.spec.kind},
                    data={"error": repr(exc)},
                ),),
            )
        result = result.extend(partial)

    return result


def _ensure_builtin_analyzers() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True
    from .algorithms import ast_access, callables, direct_annotations, method_contracts, source, symbol_capture

    for module in (callables, source, ast_access, symbol_capture, direct_annotations, method_contracts):
        register_analyzer(module.ANALYZER, replace=True)


__all__ = [
    "CodeAnalysisContext",
    "CodeAnalysisError",
    "CodeAnalysisResult",
    "CodeAnalyzer",
    "DEFAULT_ALGORITHMS",
    "FunctionAnalyzer",
    "analyze",
    "available_analyzers",
    "get_analyzer",
    "register_analyzer",
]
