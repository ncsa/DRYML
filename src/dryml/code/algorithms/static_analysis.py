"""Shared bounds and parsing helpers for conservative static analyzers."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dryml.code.facts import DiagnosticFact
from dryml.code.targets import CodeTarget


MAX_SOURCE_BYTES = 1_048_576
MAX_AST_NODES = 100_000
MAX_CALL_SITES = 10_000
MAX_CHAIN_COMPONENTS = 64
MAX_RESOLUTION_DIAGNOSTICS = 1_000
MAX_STATIC_SCALAR_CHARS = 4_096


STATIC_ANALYSIS_LIMITS = {
    "source_bytes": MAX_SOURCE_BYTES,
    "ast_nodes": MAX_AST_NODES,
    "call_sites": MAX_CALL_SITES,
    "chain_components": MAX_CHAIN_COMPONENTS,
    "resolution_diagnostics": MAX_RESOLUTION_DIAGNOSTICS,
    "scalar_chars": MAX_STATIC_SCALAR_CHARS,
}


@dataclass(frozen=True, slots=True)
class ParsedStaticSource:
    """Bounded parsed source and its source-location metadata."""

    tree: ast.AST
    source: str
    filename: str | None
    start_line: int | None


def limit_diagnostic(
    target: CodeTarget,
    analyzer: str,
    *,
    limit_name: str,
    limit: int,
    observed_lower_bound: int,
) -> DiagnosticFact:
    """Create a stable error diagnostic for static-analysis bound exhaustion."""

    return DiagnosticFact(
        severity="error",
        code=f"dryml.code.static_{limit_name}_limit_exceeded",
        message=f"Static analysis exceeded the configured {limit_name} limit.",
        source={"analyzer": analyzer, "target_kind": target.spec.kind},
        data={
            "limit_name": limit_name,
            "limit": limit,
            "observed_lower_bound": observed_lower_bound,
        },
    )


def parse_static_source(
    target: CodeTarget,
    *,
    analyzer: str,
    source: str,
    filename: str | None,
    start_line: int | None,
) -> tuple[ParsedStaticSource | None, DiagnosticFact | None]:
    """Parse source after enforcing the shared source and AST-node bounds."""

    source_bytes = len(source.encode("utf-8"))
    if source_bytes > MAX_SOURCE_BYTES:
        return None, limit_diagnostic(
            target,
            analyzer,
            limit_name="source_bytes",
            limit=MAX_SOURCE_BYTES,
            observed_lower_bound=source_bytes,
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return None, DiagnosticFact(
            severity="warning",
            code="dryml.code.ast_parse_failed",
            message="Source could not be parsed for static analysis.",
            source={"analyzer": analyzer, "target_kind": target.spec.kind},
            data={"error": repr(exc)},
        )

    node_count = 0
    for node_count, _ in enumerate(ast.walk(tree), start=1):
        if node_count > MAX_AST_NODES:
            return None, limit_diagnostic(
                target,
                analyzer,
                limit_name="ast_nodes",
                limit=MAX_AST_NODES,
                observed_lower_bound=node_count,
            )
    return ParsedStaticSource(tree, source, filename, start_line), None


def absolute_line(relative_line: int | None, start_line: int | None) -> int | None:
    """Return a file-absolute line number from a dedented source-relative line."""

    if relative_line is None or start_line is None:
        return None
    return start_line + relative_line - 1


def bounded_string(value: Any) -> str | None:
    """Return a scalar string only when it fits the serialized static-fact bound."""

    if value is None:
        return None
    value = str(value)
    return value if len(value) <= MAX_STATIC_SCALAR_CHARS else None


def bounded_target_mapping(target: Mapping[str, Any]) -> dict[str, str | None] | None:
    """Return the fixed, bounded target-reference schema used by static facts."""

    result: dict[str, str | None] = {}
    for key in ("kind", "import_path", "method_name", "subject_ref"):
        value = bounded_string(target.get(key))
        if target.get(key) is not None and value is None:
            return None
        result[key] = value
    return result


__all__ = [
    "MAX_AST_NODES",
    "MAX_CALL_SITES",
    "MAX_CHAIN_COMPONENTS",
    "MAX_RESOLUTION_DIAGNOSTICS",
    "MAX_SOURCE_BYTES",
    "MAX_STATIC_SCALAR_CHARS",
    "ParsedStaticSource",
    "STATIC_ANALYSIS_LIMITS",
    "absolute_line",
    "bounded_string",
    "bounded_target_mapping",
    "limit_diagnostic",
    "parse_static_source",
]
