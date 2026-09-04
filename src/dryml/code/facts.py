"""Immutable generic code-analysis facts and sanitized source provenance."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from .errors import AnalysisErrorCode, _CODES

if TYPE_CHECKING:
    from .kernels import AnalysisKernel


FactScalar: TypeAlias = None | bool | int | float | str | bytes
FactValue: TypeAlias = FactScalar | tuple["FactValue", ...] | tuple[tuple[str, "FactValue"], ...]


def _sanitize_filename(filename: str | None) -> str | None:
    """Return a basename-only filename suitable for framework provenance."""

    if type(filename) is not str or not filename:
        return None
    return os.path.basename(filename.replace("\\", "/")) or None


def _validate_fact_value(value: object) -> None:
    """Validate one closed recursive fact payload without generic introspection."""

    value_type = type(value)
    if value is None or value_type in (bool, int, str, bytes):
        return
    if value_type is float:
        if math.isfinite(value):
            return
        raise ValueError("fact floats must be finite")
    if value_type is not tuple:
        raise ValueError("fact value is not immutable")
    if all(type(item) is tuple and len(item) == 2 and type(item[0]) is str for item in value):
        keys = tuple(item[0] for item in value)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("fact mapping keys must be sorted and unique")
        for _, item_value in value:
            _validate_fact_value(item_value)
        return
    for item in value:
        _validate_fact_value(item)


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Sanitized source coordinates for framework-created evidence.

    Args:
        filename: Logical filename or filesystem path; paths are reduced to a
            basename before storage.
        line: Optional one-based source line.
        column: Optional UTF-8 byte column offset.

    Raises:
        ValueError: If a coordinate is not a non-negative built-in integer.

    Side Effects:
        None. Filesystem directory information is discarded.
    """

    filename: str | None
    line: int | None
    column: int | None

    def __post_init__(self) -> None:
        """Normalize filename provenance and validate source coordinates."""

        if self.filename is not None and type(self.filename) is not str:
            raise ValueError("source filename is invalid")
        if self.line is not None and (type(self.line) is not int or self.line < 1):
            raise ValueError("source line is invalid")
        if self.column is not None and (type(self.column) is not int or self.column < 0):
            raise ValueError("source column is invalid")
        object.__setattr__(self, "filename", _sanitize_filename(self.filename))


@dataclass(frozen=True, slots=True)
class CodeFact:
    """One immutable generic observation with optional sanitized provenance.

    Args:
        kind: Consumer-neutral built-in fact classification string.
        value: Recursively immutable closed fact payload.
        source: Optional sanitized source coordinate.

    Raises:
        ValueError: If ``kind`` or ``value`` violates the closed fact contract.

    Side Effects:
        None.
    """

    kind: str
    value: FactValue
    source: SourceLocation | None = None

    def __post_init__(self) -> None:
        """Validate the closed fact payload without retaining live objects."""

        if type(self.kind) is not str or not self.kind:
            raise ValueError("fact kind is invalid")
        _validate_fact_value(self.value)
        if self.source is not None and type(self.source) is not SourceLocation:
            raise ValueError("fact source is invalid")


@dataclass(frozen=True, slots=True)
class CodeFacts:
    """An exact immutable aggregate of generic facts.

    Args:
        values: Tuple whose members are exactly :class:`CodeFact` instances.

    Raises:
        ValueError: If the aggregate is not an exact tuple of exact facts.

    Side Effects:
        None.
    """

    values: tuple[CodeFact, ...]

    def __post_init__(self) -> None:
        """Reject generic container and subclass introspection at aggregation."""

        if type(self.values) is not tuple or any(type(value) is not CodeFact for value in self.values):
            raise ValueError("code facts require exact CodeFact values")


@dataclass(frozen=True, slots=True)
class FactRecord:
    """Provenance binding for one generic fact produced during analysis.

    Args:
        fact: Exact generic fact value.
        producer: Consumer kernel class that produced the fact.
        graph_digest: Immutable graph identity string.
        origin: Whether the observation was static or trace-derived.

    Raises:
        ValueError: If framework-owned provenance fields are invalid.

    Side Effects:
        None. ``producer`` remains an opaque caller-owned class reference.
    """

    fact: CodeFact
    producer: type[object]
    graph_digest: str
    origin: Literal["static", "trace"]

    def __post_init__(self) -> None:
        """Validate framework-owned provenance without inspecting producer state."""

        if type(self.fact) is not CodeFact or not isinstance(self.producer, type):
            raise ValueError("fact record is invalid")
        if type(self.graph_digest) is not str or not self.graph_digest or self.origin not in ("static", "trace"):
            raise ValueError("fact record provenance is invalid")


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A redacted immutable analysis diagnostic.

    Args:
        code: Stable analysis error category.
        message: Fixed framework-authored explanation.
        severity: Informational, warning, or error severity.
        kernel: Optional opaque consumer kernel class.
        source: Optional sanitized source coordinate.

    Raises:
        ValueError: If a framework-owned diagnostic field is invalid.

    Side Effects:
        None.
    """

    code: AnalysisErrorCode
    message: str
    severity: Literal["info", "warning", "error"] = "error"
    kernel: type[AnalysisKernel[object, object]] | None = None
    source: SourceLocation | None = None

    def __post_init__(self) -> None:
        """Validate stable fields without formatting consumer values."""

        if self.code not in _CODES:
            raise ValueError("diagnostic code is invalid")
        if type(self.message) is not str or self.severity not in ("info", "warning", "error"):
            raise ValueError("diagnostic fields are invalid")
        if self.kernel is not None and not isinstance(self.kernel, type):
            raise ValueError("diagnostic kernel is invalid")
        if self.source is not None and type(self.source) is not SourceLocation:
            raise ValueError("diagnostic source is invalid")


__all__ = [
    "CodeFact",
    "CodeFacts",
    "Diagnostic",
    "FactRecord",
    "FactScalar",
    "FactValue",
    "SourceLocation",
]
