from __future__ import annotations

import ast
from dataclasses import dataclass

from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult, FunctionAnalyzer
from dryml.code.facts import ASTAccessFact, CallSiteFact, DiagnosticFact
from dryml.code.targets import CodeTarget

from .source import get_source_info


@dataclass(frozen=True)
class AttrAccess:
    """Static attribute access discovered in Python source."""

    root: str
    chain: tuple[str, ...]
    ctx: str
    lineno: int | None
    col_offset: int | None


@dataclass(frozen=True)
class MethodCall:
    """Static method-call-like expression discovered in Python source."""

    root: str
    chain: tuple[str, ...]
    lineno: int | None
    col_offset: int | None


def _flatten_attr(node):
    chain = []
    cur = node
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        chain.reverse()
        return cur.id, tuple(chain)
    return None


class AccessCollector(ast.NodeVisitor):
    """Collect attribute accesses and method-call-like nodes from an AST."""

    def __init__(self):
        self.attr_accesses: list[AttrAccess] = []
        self.method_calls: list[MethodCall] = []

    def visit_Attribute(self, node: ast.Attribute):
        """Record a flattened attribute expression and continue traversal."""

        flat = _flatten_attr(node)
        if flat is not None:
            root, chain = flat
            if isinstance(node.ctx, ast.Load):
                ctx = "load"
            elif isinstance(node.ctx, ast.Store):
                ctx = "store"
            elif isinstance(node.ctx, ast.Del):
                ctx = "del"
            else:
                ctx = type(node.ctx).__name__.lower()

            self.attr_accesses.append(AttrAccess(root, chain, ctx, getattr(node, "lineno", None), getattr(node, "col_offset", None)))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Record a flattened call target and continue traversal."""

        flat = _flatten_attr(node.func)
        if flat is not None:
            root, chain = flat
            self.method_calls.append(MethodCall(root, chain, getattr(node, "lineno", None), getattr(node, "col_offset", None)))
        self.generic_visit(node)


def collect_accesses_from_source(source: str):
    """Parse *source* and return an :class:`AccessCollector` with findings."""

    tree = ast.parse(source)
    coll = AccessCollector()
    coll.visit(tree)
    return coll


def analyze_target(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    """Produce static AST access and call-site facts for a target."""

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
    try:
        collector = collect_accesses_from_source(info.source)
    except SyntaxError as exc:
        return CodeAnalysisResult(target=target.spec, diagnostics=(DiagnosticFact(
            severity="warning",
            code="dryml.code.ast_parse_failed",
            message="Source could not be parsed for AST access analysis.",
            source={"analyzer": "ast_access", "target_kind": target.spec.kind},
            data={"error": repr(exc)},
        ),))

    attr_details = [_attr_to_data(item) for item in collector.attr_accesses]
    call_details = [_call_to_data(item) for item in collector.method_calls]
    facts = [ASTAccessFact(
        source={"analyzer": "ast_access", "target_kind": target.spec.kind, "filename": info.filename},
        data={
            "attribute_accesses": [item["access"] for item in attr_details],
            "attribute_details": attr_details,
            "method_calls": call_details,
        },
    )]
    facts.extend(CallSiteFact(
        source={"analyzer": "ast_access", "target_kind": target.spec.kind, "filename": info.filename, "line": item["lineno"]},
        data=item,
    ) for item in call_details)
    return CodeAnalysisResult(target=target.spec, facts=tuple(facts))


def can_analyze(target: CodeTarget, context: CodeAnalysisContext) -> bool:
    """Return true for targets with a live object or unwrapped callable."""

    return target.obj is not None or target.unwrapped is not None


def _attr_to_data(access: AttrAccess) -> dict[str, object]:
    return {
        "root": access.root,
        "chain": list(access.chain),
        "access": ".".join((access.root, *access.chain)),
        "ctx": access.ctx,
        "lineno": access.lineno,
        "col_offset": access.col_offset,
    }


def _call_to_data(call: MethodCall) -> dict[str, object]:
    return {
        "receiver": call.root,
        "method": call.chain[-1] if call.chain else None,
        "chain": list(call.chain),
        "access": ".".join((call.root, *call.chain)),
        "lineno": call.lineno,
        "col_offset": call.col_offset,
    }


ANALYZER = FunctionAnalyzer("ast_access", analyze_target, can_analyze)


__all__ = [
    "ANALYZER",
    "AccessCollector",
    "AttrAccess",
    "MethodCall",
    "collect_accesses_from_source",
    "analyze_target",
]
