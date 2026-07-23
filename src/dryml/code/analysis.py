from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
import inspect
import types
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
        allow_dynamic_execution: Explicit permission for invocation through
            :func:`dryml.code.trace`; ordinary analysis remains non-invoking.
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
        if isinstance(self.algorithms, str):
            algorithms = (self.algorithms,)
        else:
            algorithms = tuple(self.algorithms or ())
        if self.diagnostics_policy not in {"collect", "raise"}:
            raise ValueError("diagnostics_policy must be 'collect' or 'raise'.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "algorithms", algorithms)
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
_REGISTERING_BUILTINS = False


def register_analyzer(analyzer: CodeAnalyzer, *, replace: bool = False) -> None:
    """Register *analyzer* by name.

    Args:
        analyzer: Analyzer object implementing :class:`CodeAnalyzer`.
        replace: Whether an existing analyzer with the same name may be replaced.
    """

    if not _BUILTINS_REGISTERED and not _REGISTERING_BUILTINS:
        _ensure_builtin_analyzers()
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

    Analysis inspects metadata, source, and ASTs without intentionally invoking
    the submitted target body. Resolving an import-path target can still execute
    module-level import code.

    Args:
        target: Live object, import path string, target spec, or target wrapper.
        algorithms: Optional ordered analyzer names overriding the context.
        context: Optional analysis options.

    Returns:
        A serializable result containing facts and diagnostics.
    """

    context = context or CodeAnalysisContext()
    if algorithms is not None:
        context = replace(context, algorithms=(algorithms,) if isinstance(algorithms, str) else tuple(algorithms))
    selected = context.algorithms or DEFAULT_ALGORITHMS
    code_target = normalize_target(target, allow_import=context.allow_import, metadata=context.metadata)
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


def trace(
    target: Any,
    *,
    args=(),
    kwargs=None,
    context=None,
    policy=None,
) -> CodeAnalysisResult:
    """Invoke a supported trusted function once and record proxy method calls.

    Tracing runs synchronously in the current process without sandboxing or a
    hard timeout. It is disabled unless ``context`` is a
    :class:`CodeAnalysisContext` with ``allow_dynamic_execution=True``. Only
    exact tuple/dict invocation containers and the bounded Definition/CDef trace
    grammar are accepted. Ordinary :func:`analyze` and probe APIs never acquire
    these invocation semantics.

    Args:
        target: Live synchronous Python function, import path, or ``CodeTarget``
            wrapping one. Bound methods, classes, callable instances, builtins,
            generators, and async functions are unsupported.
        args: Exact tuple of invocation arguments.
        kwargs: ``None`` or an exact dict with string keys.
        context: Analysis context granting dynamic execution. Empty algorithms
            or exactly ``("dynamic_trace",)`` are accepted.
        policy: Optional validated :class:`DynamicTracePolicy`.

    Returns:
        A :class:`CodeAnalysisResult` containing bounded ``DynamicCallFact``
        records and, once execution starts, one ``dynamic_trace_summary``.
        Disabled execution, incompatible trace-algorithm selection, unsupported
        targets or trace grammar, bounded-result failures, and ordinary target
        exceptions are returned as structured diagnostics rather than raised.

    Raises:
        TypeError: If ``context`` is not a :class:`CodeAnalysisContext` or
            ``None``; ``policy`` is not a :class:`DynamicTracePolicy` or
            ``None``; ``args`` is not an exact tuple; ``kwargs`` is neither an
            exact dict nor ``None``; or a keyword key is not an exact string.
        CodeAnalysisError: If an unexpected trace implementation or analyzer
            failure occurs while ``context.diagnostics_policy`` is ``"raise"``,
            after trace state is restored. Ordinary target failures and
            supported trace failures are returned as structured diagnostics.
        BaseException: An interruption raised by the invoked target propagates
            after trace state is restored.
    """

    from .algorithms.dynamic_trace import DynamicTracePolicy, _InvocationRequest, run_trace

    if context is not None and not isinstance(context, CodeAnalysisContext):
        raise TypeError("context must be a CodeAnalysisContext or None")
    selected_context = context if context is not None else CodeAnalysisContext()
    if policy is not None and not isinstance(policy, DynamicTracePolicy):
        raise TypeError("policy must be a DynamicTracePolicy or None")
    selected_policy = policy if policy is not None else DynamicTracePolicy()
    if type(args) is not tuple:
        raise TypeError("args must be an exact tuple")
    if kwargs is not None and type(kwargs) is not dict:
        raise TypeError("kwargs must be an exact dict or None")
    selected_kwargs = {} if kwargs is None else kwargs
    if any(type(key) is not str for key in selected_kwargs):
        raise TypeError("kwargs keys must be exact strings")

    target_spec = _trace_target_spec_without_resolution(target)
    if selected_context.allow_dynamic_execution is not True:
        return CodeAnalysisResult(
            target=target_spec,
            diagnostics=(DiagnosticFact(
                severity="error",
                code="dryml.code.dynamic_trace_disabled",
                message="Dynamic tracing requires explicit execution permission.",
                source={"analyzer": "dynamic_trace", "target_kind": _trace_diagnostic_target_kind(target_spec)},
            ),),
        )
    if selected_context.algorithms not in ((), ("dynamic_trace",)):
        return CodeAnalysisResult(
            target=target_spec,
            diagnostics=(DiagnosticFact(
                severity="error",
                code="dryml.code.dynamic_trace_invalid_context",
                message="Dynamic tracing requires an empty or dynamic_trace-only algorithm selection.",
                source={"analyzer": "dynamic_trace", "target_kind": _trace_diagnostic_target_kind(target_spec)},
            ),),
        )

    try:
        code_target = normalize_target(
            target,
            allow_import=selected_context.allow_import,
            metadata=selected_context.metadata,
        )
    except Exception:
        return _unsupported_trace_target(target_spec)
    if code_target.diagnostics:
        # Target normalization can contain general-purpose import diagnostics
        # with exception repr data. The trace contract instead exposes one
        # fixed, bounded target diagnostic and never serializes import exception
        # messages or repr output.
        return _unsupported_trace_target(code_target.spec)
    if _trace_diagnostic_target_kind(code_target.spec) != code_target.spec.kind:
        return _unsupported_trace_target(code_target.spec)
    if type(code_target.obj) is not types.FunctionType:
        return _unsupported_trace_target(code_target.spec)
    if (
        inspect.iscoroutinefunction(code_target.obj)
        or inspect.isasyncgenfunction(code_target.obj)
        or inspect.isgeneratorfunction(code_target.obj)
    ):
        return _unsupported_trace_target(code_target.spec)
    return run_trace(_InvocationRequest(
        target=code_target,
        args=args,
        kwargs=dict(selected_kwargs),
        context=selected_context,
        policy=selected_policy,
    ))


def _trace_target_spec_without_resolution(target: Any) -> CodeTargetSpec:
    """Describe a trace target without importing or invoking target hooks."""

    if type(target) is CodeTarget:
        return target.spec
    if type(target) is CodeTargetSpec:
        return target
    if type(target) is str:
        return CodeTargetSpec("import_path", import_path=target)
    if type(target) is types.FunctionType:
        name = object.__getattribute__(target, "__name__")
        qualname = object.__getattribute__(target, "__qualname__")
        kind = "lambda" if name == "<lambda>" else "local_function" if "<locals>" in qualname else "unbound_method" if "." in qualname else "function"
        return CodeTargetSpec(kind)
    if type(target) is types.MethodType:
        return CodeTargetSpec("bound_method")
    if isinstance(target, type):
        return CodeTargetSpec("class")
    return CodeTargetSpec("unknown")


def _unsupported_trace_target(target_spec: CodeTargetSpec) -> CodeAnalysisResult:
    return CodeAnalysisResult(
        target=target_spec,
        diagnostics=(DiagnosticFact(
            severity="error",
            code="dryml.code.dynamic_trace_unsupported_target",
            message="Dynamic trace target is unsupported.",
            source={"analyzer": "dynamic_trace", "target_kind": _trace_diagnostic_target_kind(target_spec)},
        ),),
    )


def _trace_diagnostic_target_kind(target_spec: CodeTargetSpec) -> str:
    value = target_spec.kind
    return value if isinstance(value, str) and value and len(value) <= 4_096 else "unknown"


def _ensure_builtin_analyzers() -> None:
    global _BUILTINS_REGISTERED, _REGISTERING_BUILTINS
    if _BUILTINS_REGISTERED:
        return
    _REGISTERING_BUILTINS = True
    try:
        from .algorithms import ast_access, callables, direct_annotations, dynamic_trace, method_contracts, source, static_calls, symbol_capture

        for module in (callables, source, ast_access, symbol_capture, direct_annotations, method_contracts, static_calls, dynamic_trace):
            register_analyzer(module.ANALYZER, replace=True)
        _BUILTINS_REGISTERED = True
    finally:
        _REGISTERING_BUILTINS = False


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
    "trace",
]
