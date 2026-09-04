"""Static source retrieval that never consults arbitrary loader hooks."""

from __future__ import annotations

import ast
import os
import textwrap
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import SourceUnavailableError

if TYPE_CHECKING:
    from .targets import CodeTargetInput


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
        return Path(filename).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _node_start(node: ast.AST) -> int:
    """Return the first decorator-aware one-based line for an AST definition."""

    decorators = getattr(node, "decorator_list", ())
    return min((getattr(item, "lineno", node.lineno) for item in decorators), default=node.lineno)


def _source_from_file(obj: object) -> SourceInfo | None:
    """Extract a supported function or class directly from its source file."""

    if type(obj) is types.FunctionType:
        filename = obj.__code__.co_filename
        name = obj.__name__
        first_line = obj.__code__.co_firstlineno
        node_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
    elif issubclass(type(obj), type):
        filename = type.__getattribute__(obj, "__module__")
        module_name = filename if type(filename) is str else None
        if module_name is None:
            return None
        import sys

        module = sys.modules.get(module_name)
        if module is None or not isinstance(module, types.ModuleType):
            return None
        filename = types.ModuleType.__getattribute__(module, "__file__")
        name = type.__getattribute__(obj, "__name__")
        first_line = None
        node_types = (ast.ClassDef,)
    else:
        return None
    text = _read_file(filename)
    if text is None:
        return None
    try:
        tree = ast.parse(text, filename=filename)
    except SyntaxError:
        return None
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, node_types)
        and (getattr(node, "name", None) == name or isinstance(node, ast.Lambda))
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

    normalized = normalize_target(target)
    if normalized.source is not None:
        return normalized.source
    subject = normalized.callable if normalized.callable is not None else normalized.original
    source = get_source_info(subject)
    if source is None:
        raise SourceUnavailableError()
    return source


__all__ = ["SourceInfo", "extract_source", "get_source_info"]
