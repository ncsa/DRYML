"""Bounded syntactic attribute and method-call hint collection.

This analyzer reports source-level syntax only. It does not resolve or invoke
receivers, attributes, descriptors, or call targets.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, FunctionAnalyzer
from dryml.code.facts import ASTAccessFact, CallSiteFact, DiagnosticFact
from dryml.code.targets import CodeTarget

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
    limit_diagnostic,
    parse_static_source,
)


@dataclass(frozen=True)
class AttrAccess:
    """Static attribute access discovered in Python source."""

    root: str
    chain: tuple[str, ...]
    ctx: str
    lineno: int | None
    col_offset: int | None
    chain_limit_exceeded: bool = False


@dataclass(frozen=True)
class MethodCall:
    """Static method-call-like expression discovered in Python source."""

    root: str
    chain: tuple[str, ...]
    lineno: int | None
    col_offset: int | None
    chain_limit_exceeded: bool = False


def _flatten_attr(node: ast.AST):
    chain: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        chain.reverse()
        exceeded = len(chain) > MAX_CHAIN_COMPONENTS
        return cur.id, tuple(chain[:MAX_CHAIN_COMPONENTS]), exceeded
    return None


class AccessCollector(ast.NodeVisitor):
    """Collect bounded syntactic attribute accesses and method-call-like nodes."""

    def __init__(self):
        self.attr_accesses: list[AttrAccess] = []
        self.method_calls: list[MethodCall] = []
        self.call_sites_seen = 0
        self.call_limit_exhausted = False

    def visit(self, node: ast.AST):
        """Stop traversal at the first method-call fact limit exhaustion."""

        if not self.call_limit_exhausted:
            return super().visit(node)
        return None

    def visit_Attribute(self, node: ast.Attribute):
        """Record a flattened attribute expression and continue traversal."""

        flat = _flatten_attr(node)
        if flat is not None:
            root, chain, chain_limit_exceeded = flat
            if isinstance(node.ctx, ast.Load):
                ctx = "load"
            elif isinstance(node.ctx, ast.Store):
                ctx = "store"
            elif isinstance(node.ctx, ast.Del):
                ctx = "del"
            else:
                ctx = type(node.ctx).__name__.lower()
            self.attr_accesses.append(AttrAccess(
                root,
                chain,
                ctx,
                getattr(node, "lineno", None),
                getattr(node, "col_offset", None),
                chain_limit_exceeded,
            ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Record a flattened call target without resolving its semantics."""

        flat = _flatten_attr(node.func)
        if flat is not None:
            self.call_sites_seen += 1
            if self.call_sites_seen > MAX_CALL_SITES:
                self.call_limit_exhausted = True
                return
            root, chain, chain_limit_exceeded = flat
            self.method_calls.append(MethodCall(
                root,
                chain,
                getattr(node, "lineno", None),
                getattr(node, "col_offset", None),
                chain_limit_exceeded,
            ))
        self.generic_visit(node)


def collect_accesses_from_source(source: str):
    """Parse *source* and return an :class:`AccessCollector` with findings."""

    tree = ast.parse(source)
    coll = AccessCollector()
    coll.visit(tree)
    return coll


def analyze_target(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Produce bounded syntactic AST access and call-site facts for a target."""

    if not context.allow_source:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="info",
            code="dryml.code.source_disabled",
            message="AST access analysis requires source extraction.",
            source={"analyzer": "ast_access", "target_kind": target.spec.kind},
        ),))
    obj = target.unwrapped or target.obj
    info = get_source_info(obj) if obj is not None else None
    if info is None:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="warning",
            code="dryml.code.source_unavailable",
            message="No source is available for AST access analysis.",
            source={"analyzer": "ast_access", "target_kind": target.spec.kind},
        ),))
    parsed, parse_diagnostic = parse_static_source(
        target,
        analyzer="ast_access",
        source=info.source,
        filename=info.filename,
        start_line=info.start_line,
    )
    if parse_diagnostic is not None:
        return CodeAnalysisResult(target=target.spec, diagnostics=(parse_diagnostic,))
    assert parsed is not None

    try:
        # Keep the legacy helper as the collection seam after shared bounds pass.
        collector = collect_accesses_from_source(parsed.source)
    except SyntaxError as exc:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="warning",
            code="dryml.code.ast_parse_failed",
            message="Source could not be parsed for AST access analysis.",
            source={"analyzer": "ast_access", "target_kind": target.spec.kind},
            data={"error": repr(exc)},
        ),))
    attr_details = [_attr_to_data(item, parsed.start_line) for item in collector.attr_accesses]
    call_details = [_call_to_data(item, parsed.start_line) for item in collector.method_calls]
    complete = not collector.call_limit_exhausted
    facts = [ASTAccessFact(
        source={"analyzer": "ast_access", "target_kind": target.spec.kind, "filename": parsed.filename},
        data={
            "attribute_accesses": [item["access"] for item in attr_details],
            "attribute_details": attr_details,
            "method_calls": call_details,
            "complete": complete,
            "limits": STATIC_ANALYSIS_LIMITS,
        },
    )]
    facts.extend(CallSiteFact(
        source={
            "analyzer": "ast_access",
            "target_kind": target.spec.kind,
            "filename": parsed.filename,
            "line": item["lineno"],
        },
        data=item,
    ) for item in call_details)
    diagnostics: tuple[DiagnosticFact, ...] = ()
    if collector.call_limit_exhausted:
        diagnostics = (limit_diagnostic(
            target,
            "ast_access",
            limit_name="call_sites",
            limit=MAX_CALL_SITES,
            observed_lower_bound=collector.call_sites_seen,
        ),)
    return CodeAnalysisResult(target=target.spec, facts=tuple(facts), diagnostics=diagnostics)


def can_analyze(target: CodeTarget, context: CodeAnalysisContext) -> bool:
    """Return true for targets with a live object or unwrapped callable."""

    return target.obj is not None or target.unwrapped is not None or target.spec.source_spec is not None


def _attr_to_data(access: AttrAccess, start_line: int | None) -> dict[str, object]:
    relative_line = access.lineno
    return {
        "root": access.root,
        "chain": list(access.chain),
        "access": ".".join((access.root, *access.chain)),
        "ctx": access.ctx,
        "lineno": relative_line,
        "relative_line": relative_line,
        "absolute_line": absolute_line(relative_line, start_line),
        "col_offset": access.col_offset,
        "chain_limit_exceeded": access.chain_limit_exceeded,
    }


def _call_to_data(call: MethodCall, start_line: int | None) -> dict[str, object]:
    relative_line = call.lineno
    return {
        "receiver": call.root,
        "method": call.chain[-1] if call.chain else None,
        "chain": list(call.chain),
        "access": ".".join((call.root, *call.chain)),
        "lineno": relative_line,
        "relative_line": relative_line,
        "absolute_line": absolute_line(relative_line, start_line),
        "col_offset": call.col_offset,
        "chain_limit_exceeded": call.chain_limit_exceeded,
        "semantic_resolution": "not_attempted",
    }


ANALYZER = FunctionAnalyzer("ast_access", analyze_target, can_analyze)


__all__ = [
    "ANALYZER",
    "AccessCollector",
    "AttrAccess",
    "MAX_AST_NODES",
    "MAX_CALL_SITES",
    "MAX_CHAIN_COMPONENTS",
    "MAX_RESOLUTION_DIAGNOSTICS",
    "MAX_SOURCE_BYTES",
    "MAX_STATIC_SCALAR_CHARS",
    "MethodCall",
    "collect_accesses_from_source",
    "analyze_target",
]
