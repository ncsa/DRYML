"""Immutable shared carriers for process-local hard requirement declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Generic, TypeVar

from .errors import RequirementError, _project_text

R = TypeVar("R")
_MAX_SOURCE_LABEL = 256
_MAX_FIELD_TEXT = 512
_MAX_ISSUES = 1024
_MAX_ASSOCIATIONS = 4096
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_ISSUE_CODE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def _validate_source_text(value: str | None, *, label: str) -> str | None:
    """Validate one explicit source field without invoking string subclasses."""

    if value is None:
        return None
    if type(value) is not str or not value or len(value) > _MAX_FIELD_TEXT or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise RequirementError(f"invalid requirement source {label}")
    return _project_text(value, limit=_MAX_FIELD_TEXT)


@dataclass(frozen=True, slots=True)
class RequirementSource:
    """Bounded, non-identifying explanation of one declaration's origin.

    Args:
        label: A nonempty built-in source label of at most 256 characters.
        module: Optional process-local module explanation of at most 512
            characters.
        qualname: Optional process-local qualified-name explanation of at most
            512 characters.

    Raises:
        RequirementError: If an explicit field is malformed, oversized, or
            control-bearing.

    Side Effects:
        None. Source values are process-local diagnostics, not target identity.
    """

    label: str
    module: str | None = field(default=None, kw_only=True)
    qualname: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate and redact bounded source explanations."""

        if type(self.label) is not str or not self.label or len(self.label) > _MAX_SOURCE_LABEL or any(
            ord(char) < 32 or ord(char) == 127 for char in self.label
        ):
            raise RequirementError("invalid requirement source label")
        object.__setattr__(self, "label", _project_text(self.label, limit=_MAX_SOURCE_LABEL))
        object.__setattr__(self, "module", _validate_source_text(self.module, label="module"))
        object.__setattr__(self, "qualname", _validate_source_text(self.qualname, label="qualified name"))


@dataclass(frozen=True, slots=True)
class RequirementDeclaration(Generic[R]):
    """One domain-validated hard value with safely bounded source context.

    Args:
        value: One non-``None`` domain-owned requirement value. The shared layer
            neither copies nor assigns semantics to it.
        source: The exact shared source explaining this declaration.

    Raises:
        RequirementError: If the declaration has no value or source is not an
            exact :class:`RequirementSource`.

    Side Effects:
        None. Declarations do not attach themselves or alter target behavior.
    """

    value: R
    source: RequirementSource = field(kw_only=True)

    def __post_init__(self) -> None:
        """Validate the carrier without interpreting its domain value."""

        if self.value is None or type(self.source) is not RequirementSource:
            raise RequirementError("invalid requirement declaration")


@dataclass(frozen=True, slots=True)
class RequirementIssue:
    """One bounded machine-readable semantic combination conflict.

    Args:
        code: An owner-qualified ASCII issue identifier.
        message: A projected, redacted explanation of at most 512 characters.
        path: Optional projected canonical-path explanation.
        sources: Exact shared source values contributing to this issue.

    Raises:
        RequirementError: If the code, source tuple, or source types are invalid
            or a per-issue source bound is exceeded.

    Side Effects:
        None. Diagnostic text is redacted before public retention.
    """

    code: str
    message: str
    path: str | None = field(default=None, kw_only=True)
    sources: tuple[RequirementSource, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        """Validate and detach one safe, immutable conflict explanation."""

        if type(self.code) is not str or len(self.code) > _MAX_FIELD_TEXT or not self.code.isascii() or _ISSUE_CODE.fullmatch(self.code) is None:
            raise RequirementError("invalid requirement issue code")
        if type(self.message) is not str or self.path is not None and type(self.path) is not str:
            raise RequirementError("invalid requirement issue text")
        if type(self.sources) is not tuple or len(self.sources) > _MAX_ASSOCIATIONS or any(
            type(source) is not RequirementSource for source in self.sources
        ):
            raise RequirementError("invalid requirement issue sources")
        object.__setattr__(self, "message", _project_text(self.message))
        object.__setattr__(self, "path", None if self.path is None else _project_text(self.path))


@dataclass(frozen=True, slots=True)
class RequirementReport:
    """Immutable deterministic collection of shared combination issues.

    Args:
        issues: Exact issue values in deterministic domain-combiner order.

    Raises:
        RequirementError: If issue types, count, source associations, or bounded
            diagnostic serialization exceed shared capacity.

    Side Effects:
        None. The report is diagnostic data only and has no policy effect.
    """

    issues: tuple[RequirementIssue, ...] = ()

    def __post_init__(self) -> None:
        """Validate report shape and aggregate diagnostic capacities."""

        if type(self.issues) is not tuple or len(self.issues) > _MAX_ISSUES or any(type(issue) is not RequirementIssue for issue in self.issues):
            raise RequirementError("invalid requirement report issues")
        associations = sum(len(issue.sources) for issue in self.issues)
        if associations > _MAX_ASSOCIATIONS or _report_bytes(self.issues) > _MAX_REPORT_BYTES:
            raise RequirementError("requirement report exceeds diagnostic capacity")

    @property
    def ok(self) -> bool:
        """Whether the report contains no semantic conflict issues."""

        return not self.issues


def _report_bytes(issues: tuple[RequirementIssue, ...]) -> int:
    """Return the bounded UTF-8 work represented by report diagnostics."""

    return sum(
        len(field.encode("utf-8"))
        for issue in issues
        for field in (
            issue.code,
            issue.message,
            issue.path or "",
            *(source.label for source in issue.sources),
            *(source.module or "" for source in issue.sources),
            *(source.qualname or "" for source in issue.sources),
        )
    )


@dataclass(frozen=True, slots=True)
class RequirementResult(Generic[R]):
    """One legal shared outcome of declaration combination.

    Args:
        value: One domain-owned combined value, or ``None`` for empty or failed
            combination.
        report: The exact shared report accompanying this outcome.

    Raises:
        RequirementError: If the report is not exact or a value is paired with
            conflict issues.

    Side Effects:
        None. A result contains no ambient admission or execution state.
    """

    value: R | None = None
    report: RequirementReport = RequirementReport()

    def __post_init__(self) -> None:
        """Enforce the legal empty, valued, and conflict result shapes."""

        if type(self.report) is not RequirementReport or self.value is not None and not self.report.ok:
            raise RequirementError("invalid requirement result")

    @property
    def ok(self) -> bool:
        """Whether the result's report contains no semantic conflict."""

        return self.report.ok

    @property
    def has_value(self) -> bool:
        """Whether the result contains one usable domain requirement value."""

        return self.value is not None


def _ordinalized_declaration(declaration: RequirementDeclaration[R], ordinal: int) -> RequirementDeclaration[R]:
    """Return a declaration whose source label has deterministic safe ordinal context."""

    source = declaration.source
    label = f"{ordinal}: {source.label}"[:_MAX_SOURCE_LABEL]
    return RequirementDeclaration(
        declaration.value,
        source=RequirementSource(label, module=source.module, qualname=source.qualname),
    )


__all__ = ["RequirementDeclaration", "RequirementIssue", "RequirementReport", "RequirementResult", "RequirementSource"]
