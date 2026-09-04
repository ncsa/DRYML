"""Tolerant parsing and deterministic generic attribute-access evidence."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal

from .errors import SourceUnavailableError
from .source import SourceInfo


@dataclass(frozen=True, slots=True)
class AttrAccess:
    """One statically visible attribute expression.

    Args:
        root: Root name of the flattened attribute expression.
        chain: Ordered attribute names after ``root``.
        ctx: Load, store, or delete context.
        lineno: Optional absolute one-based source line.
        col_offset: Optional UTF-8 byte column offset.

    Side Effects:
        None.
    """

    root: str
    chain: tuple[str, ...]
    ctx: Literal["load", "store", "del"]
    lineno: int | None
    col_offset: int | None


@dataclass(frozen=True, slots=True)
class MethodCall:
    """One statically visible call target rooted at a name.

    Args:
        root: Root name of the callee expression.
        chain: Ordered attribute names after ``root``; empty for a direct call.
        lineno: Optional absolute one-based source line.
        col_offset: Optional UTF-8 byte column offset.

    Side Effects:
        None.
    """

    root: str
    chain: tuple[str, ...]
    lineno: int | None
    col_offset: int | None


@dataclass(frozen=True, slots=True)
class AccessCollection:
    """Deterministically ordered generic attribute and call evidence.

    Args:
        attr_accesses: Attribute expressions in source traversal order.
        method_calls: Call targets in source traversal order.

    Raises:
        ValueError: If a collection field is not an exact tuple of the matching
            immutable public value type.

    Side Effects:
        None.
    """

    attr_accesses: tuple[AttrAccess, ...]
    method_calls: tuple[MethodCall, ...]

    def __post_init__(self) -> None:
        """Reject generic containers and subclasses at the public boundary."""

        if (
            type(self.attr_accesses) is not tuple
            or type(self.method_calls) is not tuple
            or any(type(value) is not AttrAccess for value in self.attr_accesses)
            or any(type(value) is not MethodCall for value in self.method_calls)
        ):
            raise ValueError("access collection is invalid")


def _flatten_attr(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    """Return a name-rooted static attribute chain without evaluating it."""

    chain: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        chain.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        chain.reverse()
        return current.id, tuple(chain)
    return None


class _AccessCollector(ast.NodeVisitor):
    """Private visitor so future traversal fusion does not become public API."""

    def __init__(self, line_offset: int) -> None:
        """Initialize ordered evidence lists for one parsed source tree."""

        self.line_offset = line_offset
        self.attr_accesses: list[AttrAccess] = []
        self.method_calls: list[MethodCall] = []

    def _line(self, node: ast.AST) -> int | None:
        """Return the absolute source line for one syntax node when available."""

        line = getattr(node, "lineno", None)
        return line + self.line_offset if type(line) is int else None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Collect one flattened attribute expression before walking children."""

        flattened = _flatten_attr(node)
        if flattened is not None:
            root, chain = flattened
            if isinstance(node.ctx, ast.Load):
                context: Literal["load", "store", "del"] = "load"
            elif isinstance(node.ctx, ast.Store):
                context = "store"
            else:
                context = "del"
            self.attr_accesses.append(
                AttrAccess(root, chain, context, self._line(node), getattr(node, "col_offset", None))
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Collect a name-rooted static callee before walking call children."""

        flattened = _flatten_attr(node.func)
        if flattened is not None:
            root, chain = flattened
            self.method_calls.append(MethodCall(root, chain, self._line(node.func), getattr(node.func, "col_offset", None)))
        self.generic_visit(node)


def parse_source(source: str | SourceInfo) -> ast.Module:
    """Parse source text using the running interpreter's supported syntax.

    Args:
        source: Direct source text or request-local source carrier.

    Returns:
        Parsed module AST with native relative source coordinates.

    Raises:
        SourceUnavailableError: If the source input is invalid or cannot be
            parsed. Parser details and source text are deliberately redacted.

    Side Effects:
        None. Parsing never compiles or executes source.
    """

    text = source.source if type(source) is SourceInfo else source
    if type(text) is not str:
        raise SourceUnavailableError("source is invalid", code="source.invalid")
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError, TypeError):
        raise SourceUnavailableError("source is invalid", code="source.invalid") from None


def collect_accesses_from_source(source: str | SourceInfo) -> AccessCollection:
    """Collect deterministic generic access evidence from static source.

    Args:
        source: Direct source text or request-local source carrier.

    Returns:
        Immutable attribute and call evidence in source traversal order. Lines
        are absolute when ``SourceInfo.start_line`` is available.

    Raises:
        SourceUnavailableError: If the source input is malformed.

    Side Effects:
        None. Source is parsed only and never compiled or executed.
    """

    tree = parse_source(source)
    offset = source.start_line - 1 if type(source) is SourceInfo and source.start_line is not None else 0
    collector = _AccessCollector(offset)
    collector.visit(tree)
    return AccessCollection(tuple(collector.attr_accesses), tuple(collector.method_calls))


__all__ = ["AccessCollection", "AttrAccess", "MethodCall", "collect_accesses_from_source", "parse_source"]
