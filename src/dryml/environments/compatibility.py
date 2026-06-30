"""Compatibility reports and policy handling for environment checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import EnvironmentCompatibilityError
from .schema import COMPATIBILITY_REPORT_SCHEMA_VERSION

CompatibilityStatus = Literal["compatible", "warning", "incompatible", "unknown"]
IssueSeverity = Literal["error", "warning", "unknown"]
CompatibilityPolicy = Literal["ignore", "warn", "compatible", "strict"]


def coerce_policy(policy: str) -> CompatibilityPolicy:
    """Normalize and validate a compatibility policy string."""

    value = str(policy).strip().lower().replace("-", "_")
    aliases = {
        "default": "compatible",
        "compat": "compatible",
        "compatible_or_warn": "warn",
    }
    value = aliases.get(value, value)
    if value not in {"ignore", "warn", "compatible", "strict"}:
        raise EnvironmentCompatibilityError(
            f"unknown environment compatibility policy {policy!r}",
            context={"policy": policy},
        )
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    """One machine-readable compatibility issue."""

    code: str
    severity: IssueSeverity
    message: str
    requirement_path: str | None = None
    observed_path: str | None = None
    expected: Any = None
    observed: Any = None
    schema_version: int = COMPATIBILITY_REPORT_SCHEMA_VERSION

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible issue data."""

        return {
            "schema_version": self.schema_version,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "requirement_path": self.requirement_path,
            "observed_path": self.observed_path,
            "expected": self.expected,
            "observed": self.observed,
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "CompatibilityIssue":
        """Build an issue from serialized data."""

        return cls(
            code=data["code"],
            severity=data["severity"],
            message=data["message"],
            requirement_path=data.get("requirement_path"),
            observed_path=data.get("observed_path"),
            expected=data.get("expected"),
            observed=data.get("observed"),
            schema_version=data.get("schema_version", COMPATIBILITY_REPORT_SCHEMA_VERSION),
        )


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Structured result of checking a requirement against an environment."""

    status: CompatibilityStatus
    issues: tuple[CompatibilityIssue, ...] = ()
    schema_version: int = COMPATIBILITY_REPORT_SCHEMA_VERSION
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the report permits proceeding under its applied policy."""

        return self.status in {"compatible", "warning"}

    @property
    def is_compatible(self) -> bool:
        """Alias for :attr:`ok` used by callers that prefer explicit naming."""

        return self.ok

    def raise_if_incompatible(self) -> None:
        """Raise when the report status is incompatible or unknown."""

        if not self.ok:
            raise EnvironmentCompatibilityError(
                self.explain(),
                context={"status": self.status, "issues": [issue.to_data() for issue in self.issues]},
            )

    def explain(self) -> str:
        """Return a stable human-readable explanation."""

        if not self.issues:
            return f"Environment {self.status}."
        header = f"Environment {self.status}:"
        lines = [header]
        for issue in self.issues:
            lines.append(f"- {issue.code}: {issue.message}")
        return "\n".join(lines)

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible report data."""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "issues": [issue.to_data() for issue in self.issues],
            "details": dict(self.details),
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "CompatibilityReport":
        """Build a report from serialized data."""

        return cls(
            status=data["status"],
            issues=tuple(CompatibilityIssue.from_data(item) for item in data.get("issues", ())),
            schema_version=data.get("schema_version", COMPATIBILITY_REPORT_SCHEMA_VERSION),
            details=dict(data.get("details", {})),
        )


def report_from_issues(
    issues: tuple[CompatibilityIssue, ...],
    *,
    policy: str = "compatible",
    details: dict[str, Any] | None = None,
) -> CompatibilityReport:
    """Apply a policy at the decision boundary and build a report."""

    coerced = coerce_policy(policy)
    if coerced == "ignore":
        return CompatibilityReport("compatible", (), details={"policy": coerced, **dict(details or {})})

    if coerced == "warn":
        converted = tuple(
            CompatibilityIssue(
                issue.code,
                "warning" if issue.severity == "error" else issue.severity,
                issue.message,
                issue.requirement_path,
                issue.observed_path,
                issue.expected,
                issue.observed,
                issue.schema_version,
            )
            for issue in issues
        )
        status: CompatibilityStatus = "warning" if converted else "compatible"
        if converted and all(issue.severity == "unknown" for issue in converted):
            status = "unknown"
        return CompatibilityReport(status, converted, details={"policy": coerced, **dict(details or {})})

    if any(issue.severity == "error" for issue in issues):
        status = "incompatible"
    elif any(issue.severity == "unknown" for issue in issues):
        status = "unknown"
    elif any(issue.severity == "warning" for issue in issues):
        status = "warning"
    else:
        status = "compatible"
    return CompatibilityReport(status, issues, details={"policy": coerced, **dict(details or {})})


__all__ = [
    "CompatibilityIssue",
    "CompatibilityReport",
    "CompatibilityPolicy",
    "CompatibilityStatus",
    "IssueSeverity",
    "coerce_policy",
    "report_from_issues",
]
