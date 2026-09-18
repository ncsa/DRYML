"""Static source retrieval that never consults arbitrary loader hooks."""

from __future__ import annotations

import ast
import os
import textwrap
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .callable_info import _function_slot, _module_namespace, _type_slot
from .errors import SourceUnavailableError

if TYPE_CHECKING:
    from .targets import CodeTargetInput


_MAX_SOURCE_BYTES = 1_048_576
_MAX_AST_NODES = 100_000
_MAX_AST_DEPTH = 128


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Request-local source text and its original source-file coordinates.

    Args:
        source: Python source text.
        filename: Original caller or inspected filename, retained only for local
            source handling and never copied into framework provenance.
        start_line: Optional one-based line occupied by the first source line.

    Raises:
        ValueError: If the source carrier fields have invalid built-in types.

    Side Effects:
        None.
    """

    source: str
    filename: str | None
    start_line: int | None

    def __post_init__(self) -> None:
        """Validate the local source carrier without parsing or executing it."""

        if type(self.source) is not str or (self.filename is not None and type(self.filename) is not str):
            raise ValueError("source information is invalid")
        if self.start_line is not None and (type(self.start_line) is not int or self.start_line < 1):
            raise ValueError("source start line is invalid")


def _read_file(filename: object) -> str | None:
    """Read an ordinary existing source file without linecache or loaders."""

    if type(filename) is not str or not filename or not os.path.isfile(filename):
        return None
    try:
        with Path(filename).open("rb") as source_file:
            raw = source_file.read(_MAX_SOURCE_BYTES + 1)
    except OSError:
        return None
    if len(raw) > _MAX_SOURCE_BYTES:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeError:
        return None


def _source_within_bounds(source: str) -> bool:
    """
    Return whether text fits the source byte ceiling without one large
    encoding.
    """

    size = 0
    for offset in range(0, len(source), 8_192):
        size += len(source[offset:offset + 8_192].encode("utf-8"))
        if size > _MAX_SOURCE_BYTES:
            return False
    return True


def _bounded_parse(
        source: str,
        filename: str | None = None) -> tuple[ast.Module | None, bool]:
    """
    Parse bounded source and distinguish syntax failure from resource
    exhaustion.
    """

    if not _source_within_bounds(source):
        return None, True
    try:
        tree = ast.parse(source, filename=filename or "<unknown>")
    except (MemoryError, RecursionError, SyntaxError, ValueError):
        return None, False
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > _MAX_AST_NODES or depth > _MAX_AST_DEPTH:
            return None, True
        stack.extend(
            (child, depth + 1) for child in ast.iter_child_nodes(node))
    return tree, False


def _node_start(node: ast.AST) -> int:
    """Return the first decorator-aware one-based line for an AST definition."""

    decorators = getattr(node, "decorator_list", ())
    return min((getattr(item, "lineno", node.lineno) for item in decorators), default=node.lineno)


def _source_from_file(obj: object) -> SourceInfo | None:
    """Extract a supported function or class directly from its source file."""

    if type(obj) is types.FunctionType:
        code = _function_slot(obj, "__code__")
        filename = code.co_filename  # type: ignore[union-attr]
        name = _function_slot(obj, "__name__")
        first_line = code.co_firstlineno  # type: ignore[union-attr]
        node_types = (ast.Lambda,) if name == "<lambda>" else (ast.FunctionDef, ast.AsyncFunctionDef)
    elif issubclass(type(obj), type):
        filename = _type_slot(obj, "__module__")
        module_name = filename if type(filename) is str else None
        if module_name is None:
            return None
        import sys

        module = sys.modules.get(module_name)
        if module is None or not isinstance(module, types.ModuleType):
            return None
        filename = _module_namespace(module).get("__file__")
        name = _type_slot(obj, "__name__")
        first_line = None
        node_types = (ast.ClassDef,)
    else:
        return None
    text = _read_file(filename)
    if text is None:
        return None
    tree, _ = _bounded_parse(text, filename)
    if tree is None:
        return None
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, node_types)
        and (isinstance(node, ast.Lambda) or getattr(node, "name", None) == name)
        and (first_line is None or node.lineno == first_line or _node_start(node) == first_line)
    ]
    if len(candidates) != 1:
        return None
    node = candidates[0]
    start = _node_start(node)
    end = getattr(node, "end_lineno", None)
    if type(end) is not int:
        return None
    lines = text.splitlines(keepends=True)
    return SourceInfo(textwrap.dedent("".join(lines[start - 1:end])), filename, start)


def get_source_info(obj: object) -> SourceInfo | None:
    """Retrieve file-backed source for a direct Python function or class.

    Args:
        obj: Candidate Python function or class.

    Returns:
        Request-local source information, or ``None`` when the target is not an
        admitted file-backed source subject.

    Raises:
        None.

    Side Effects:
        Reads the ordinary source file only. It never calls arbitrary loader,
        descriptor, or dynamic lookup hooks.
    """

    return _source_from_file(obj)


def extract_source(target: CodeTargetInput) -> SourceInfo:
    """Return static source for a supported target or raise a typed error.

    Args:
        target: Supported target wrapper or live target accepted by target
            normalization.

    Returns:
        Request-local source text and original local source coordinates.

    Raises:
        SourceUnavailableError: If source is unavailable or malformed. Target
            normalization errors propagate unchanged for unsupported targets.

    Side Effects:
        May read a source file or explicitly import an ``ImportTarget`` module;
        it never compiles, reconstructs, or executes source text.
    """

    from .targets import normalize_target
    from .inspection import InspectionTarget

    normalized = normalize_target(target)
    if type(normalized) is InspectionTarget:
        raise SourceUnavailableError()
    if normalized.source is not None:
        return normalized.source
    subject = normalized.callable if normalized.callable is not None else normalized.original
    source = get_source_info(subject)
    if source is None:
        raise SourceUnavailableError()
    return source


__all__ = ["SourceInfo", "extract_source", "get_source_info"]
