"""Opt-in conservative static call resolution.

Only direct calls through a function's real globals mapping and direct methods on
concretely annotated parameters can resolve. Every other Python call form is a
source-level possibility, never evidence that the call executes at runtime.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping
from typing import Any

from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, FunctionAnalyzer
from dryml.code.facts import CodeFact, DiagnosticFact, StaticCallFact
from dryml.code.targets import CodeTarget, CodeTargetSpec, normalize_target

from .source import get_source_info
from .static_analysis import (
    MAX_AST_NODES,
    MAX_CALL_SITES,
    MAX_CHAIN_COMPONENTS,
    MAX_RESOLUTION_DIAGNOSTICS,
    MAX_SOURCE_BYTES,
    MAX_STATIC_SCALAR_CHARS,
    STATIC_ANALYSIS_LIMITS,
    absolute_line,
    bounded_string,
    bounded_target_mapping,
    limit_diagnostic,
    parse_static_source,
)


class _CallCollector(ast.NodeVisitor):
    """Collect call expressions in deterministic source traversal order."""

    def __init__(self) -> None:
        self.calls: list[ast.Call] = []
        self.call_sites_seen = 0
        self.exhausted = False

    def visit(self, node: ast.AST):
        if not self.exhausted:
            return super().visit(node)
        return None

    def visit_Call(self, node: ast.Call) -> None:
        self.call_sites_seen += 1
        if self.call_sites_seen > MAX_CALL_SITES:
            self.exhausted = True
            return
        self.calls.append(node)
        self.generic_visit(node)


def analyze_target(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Emit bounded static call possibilities without executing user code.

    Direct global calls and direct methods on concretely annotated parameters may
    resolve. Attribute chains, aliases, call-result receivers, string annotations,
    and dynamic lookup remain non-resolved by design.
    """

    if not context.allow_source:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="info",
            code="dryml.code.source_disabled",
            message="Static call analysis requires source extraction.",
            source={"analyzer": "static_calls", "target_kind": target.spec.kind},
        ),))
    obj = target.unwrapped or target.obj
    info = get_source_info(obj) if obj is not None else None
    if info is None:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="warning",
            code="dryml.code.source_unavailable",
            message="No source is available for static call analysis.",
            source={"analyzer": "static_calls", "target_kind": target.spec.kind},
        ),))
    parsed, parse_diagnostic = parse_static_source(
        target,
        analyzer="static_calls",
        source=info.source,
        filename=info.filename,
        start_line=info.start_line,
    )
    if parse_diagnostic is not None:
        return CodeAnalysisResult(target=target.spec, diagnostics=(parse_diagnostic,))
    assert parsed is not None

    collector = _CallCollector()
    collector.visit(parsed.tree)
    function = _function_for_target(target)
    globals_mapping = getattr(function, "__globals__", {}) if function is not None else {}
    annotations = _parameter_annotations(function)
    parameter_names = _parameter_names(function)
    reassigned_names = _reassigned_names(parsed.tree)
    facts = tuple(
        _fact_for_call(
            target,
            call,
            globals_mapping=globals_mapping,
            annotations=annotations,
            parameter_names=parameter_names,
            reassigned_names=reassigned_names,
            filename=parsed.filename,
            start_line=parsed.start_line,
        )
        for call in collector.calls
    )
    complete = not collector.exhausted
    summary = CodeFact(
        kind="static_call_summary",
        source={"analyzer": "static_calls", "target_kind": target.spec.kind, "filename": parsed.filename},
        data={
            "complete": complete,
            "call_sites_seen": collector.call_sites_seen,
            "facts_emitted": len(facts),
            "limits": STATIC_ANALYSIS_LIMITS,
        },
    )
    diagnostics: tuple[DiagnosticFact, ...] = ()
    if collector.exhausted:
        diagnostics = (limit_diagnostic(
            target,
            "static_calls",
            limit_name="call_sites",
            limit=MAX_CALL_SITES,
            observed_lower_bound=collector.call_sites_seen,
        ),)
    return CodeAnalysisResult(target=target.spec, facts=(*facts, summary), diagnostics=diagnostics)


def can_analyze(target: CodeTarget, context: CodeAnalysisContext) -> bool:
    """Return true for live targets whose source can be inspected."""

    return target.obj is not None or target.unwrapped is not None or target.spec.source_spec is not None


def _function_for_target(target: CodeTarget) -> Any | None:
    candidate = target.unwrapped or target.obj
    if inspect.ismethod(candidate):
        return candidate.__func__
    return candidate if inspect.isfunction(candidate) else None


def _parameter_annotations(function: Any | None) -> Mapping[str, Any]:
    if function is None:
        return {}
    try:
        return {
            name: parameter.annotation
            for name, parameter in inspect.signature(function).parameters.items()
            if parameter.annotation is not inspect.Parameter.empty
        }
    except (TypeError, ValueError):
        return {}


def _parameter_names(function: Any | None) -> set[str]:
    if function is None:
        return set()
    try:
        return set(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return set()


def _fact_for_call(
    target: CodeTarget,
    call: ast.Call,
    *,
    globals_mapping: Mapping[str, Any],
    annotations: Mapping[str, Any],
    parameter_names: set[str],
    reassigned_names: set[str],
    filename: str | None,
    start_line: int | None,
) -> StaticCallFact:
    func = call.func
    status = "unresolved"
    syntax = "other"
    display: str | None = None
    receiver: str | None = None
    method_name: str | None = None
    target_data: dict[str, str | None] | None = None
    reason: str | None = None

    if isinstance(func, ast.Name):
        syntax = "direct_name"
        display = func.id
        method_name = func.id
        if func.id not in globals_mapping:
            reason = "global_name_unavailable"
        else:
            value = globals_mapping[func.id]
            if not callable(value):
                reason = "global_value_not_callable"
            else:
                target_data = _target_data(value)
                if target_data is None:
                    status = "unsupported"
                    reason = "target_reference_limit_exceeded"
                else:
                    status = "resolved"
    elif isinstance(func, ast.Attribute):
        flattened = _flatten_name_attributes(func)
        if flattened is None:
            syntax = "other"
            display = "<call-result>." + func.attr if isinstance(func.value, ast.Call) else func.attr
            method_name = func.attr
            status = "unsupported"
            reason = "call_result_receiver" if isinstance(func.value, ast.Call) else "unsupported_receiver_expression"
        else:
            receiver, chain, chain_limit_exceeded = flattened
            method_name = chain[-1] if chain else None
            display = ".".join((receiver, *chain))
            if chain_limit_exceeded:
                syntax = "attribute_chain"
                status = "unsupported"
                reason = "chain_limit_exceeded"
            elif len(chain) != 1:
                syntax = "attribute_chain"
                status = "unsupported"
                reason = "attribute_chain_unsupported"
            elif receiver in annotations:
                syntax = "annotated_receiver_method"
                if receiver in reassigned_names:
                    status = "unresolved"
                    reason = "receiver_reassigned"
                else:
                    annotation = annotations[receiver]
                    target_data, status, reason = _resolve_annotated_method(annotation, method_name)
            elif receiver in parameter_names:
                syntax = "annotated_receiver_method"
                status = "unresolved"
                reason = "missing_annotation"
            else:
                syntax = "attribute_chain"
                status = "unsupported"
                reason = "attribute_chain_unsupported"
    else:
        status = "unsupported"
        reason = "unsupported_callable_expression"
        display = type(func).__name__.lower()

    if any(value is None for value in (bounded_string(display), bounded_string(receiver), bounded_string(method_name)) if value is not None):
        status = "unsupported"
        target_data = None
        reason = "scalar_limit_exceeded"
        display = "<bounded>"
        receiver = None
        method_name = None
    confidence = "exact_static" if status == "resolved" else "conservative_hint"
    relative_line = getattr(call, "lineno", None)
    return StaticCallFact(
        source={"analyzer": "static_calls", "target_kind": target.spec.kind, "filename": filename},
        data={
            "status": status,
            "confidence": confidence,
            "syntax": syntax,
            "display": display,
            "receiver": receiver,
            "method_name": method_name,
            "target": target_data if status == "resolved" else None,
            "reason": None if status == "resolved" else reason,
            "relative_line": relative_line,
            "absolute_line": absolute_line(relative_line, start_line),
            "col_offset": getattr(call, "col_offset", None),
        },
    )


def _flatten_name_attributes(node: ast.Attribute) -> tuple[str, tuple[str, ...], bool] | None:
    chain: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        chain.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    chain.reverse()
    return current.id, tuple(chain[:MAX_CHAIN_COMPONENTS]), len(chain) > MAX_CHAIN_COMPONENTS


def _reassigned_names(tree: ast.AST) -> set[str]:
    """Return names assigned in the target source without inferring their values."""

    names: set[str] = set()
    for node in ast.walk(tree):
        targets: tuple[ast.AST, ...] = ()
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = (node.target,) if not isinstance(node, ast.Assign) else tuple(node.targets)
        for assigned in targets:
            names.update(_target_names(assigned))
    return names


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in node.elts))
    return set()


def _resolve_annotated_method(annotation: Any, method_name: str | None) -> tuple[dict[str, str | None] | None, str, str | None]:
    if method_name is None:
        return None, "unsupported", "method_name_unavailable"
    if isinstance(annotation, str):
        return None, "unresolved", "string_annotation"
    if not inspect.isclass(annotation):
        return None, "ambiguous", "non_concrete_annotation"
    try:
        descriptor = inspect.getattr_static(annotation, method_name)
    except AttributeError:
        return None, "unresolved", "annotated_method_missing"
    if isinstance(descriptor, property):
        return None, "unsupported", "property_descriptor"
    candidate = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor
    if not callable(candidate):
        return None, "unresolved", "annotated_member_not_callable"
    subject = normalize_target(annotation, allow_import=False).spec.import_path
    target_data = _target_data(candidate, method_name=method_name, subject_ref=subject)
    if target_data is None:
        return None, "unsupported", "target_reference_limit_exceeded"
    return target_data, "resolved", None


def _target_data(value: Any, *, method_name: str | None = None, subject_ref: str | None = None) -> dict[str, str | None] | None:
    spec: CodeTargetSpec = normalize_target(value, allow_import=False).spec
    return bounded_target_mapping({
        "kind": spec.kind,
        "import_path": spec.import_path,
        "method_name": method_name if method_name is not None else spec.method_name,
        "subject_ref": subject_ref if subject_ref is not None else spec.subject_ref,
    })


ANALYZER = FunctionAnalyzer("static_calls", analyze_target, can_analyze)


__all__ = [
    "ANALYZER",
    "MAX_AST_NODES",
    "MAX_CALL_SITES",
    "MAX_CHAIN_COMPONENTS",
    "MAX_RESOLUTION_DIAGNOSTICS",
    "MAX_SOURCE_BYTES",
    "MAX_STATIC_SCALAR_CHARS",
    "analyze_target",
]
