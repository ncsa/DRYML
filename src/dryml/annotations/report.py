"""Structured reports for annotation merge and resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .model import SourceTrace


@dataclass(frozen=True, slots=True)
class AnnotationIssue:
    """One machine-readable annotation finding."""

    severity: Literal["error", "warning", "info"]
    namespace: str
    path: str
    message: str
    expected: Any | None = None
    actual: Any | None = None
    sources: tuple[SourceTrace, ...] = ()


@dataclass(frozen=True, slots=True)
class AnnotationReport:
    """Collection of annotation resolution findings."""

    issues: tuple[AnnotationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether no error-severity issues were found."""

        return all(issue.severity != "error" for issue in self.issues)

    def explain(self) -> str:
        """Return a compact human-readable explanation."""

        return format_report(self)


def format_report(report: AnnotationReport) -> str:
    """Format an annotation report for logs and diagnostics."""

    if report.ok and not report.issues:
        return "Annotation report is ok."
    lines = []
    for issue in report.issues:
        source_bits = []
        for source in issue.sources:
            if source.label:
                source_bits.append(source.label)
            elif source.target is not None and source.target.qualname:
                source_bits.append(f"{source.kind}:{source.target.qualname}")
            else:
                source_bits.append(source.kind)
        suffix = f" sources={', '.join(source_bits)}" if source_bits else ""
        lines.append(f"[{issue.severity}] {issue.namespace}{issue.path}: {issue.message}{suffix}")
    return "\n".join(lines)


__all__ = ["AnnotationIssue", "AnnotationReport", "format_report"]
