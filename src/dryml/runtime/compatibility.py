"""Compatibility checks for resolved runtime context requirements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .specs import RuntimeContextSpec


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityIssue:
    """One unmet runtime requirement field."""

    path: str
    message: str
    expected: Any
    actual: Any

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready issue data."""

        return {"path": self.path, "message": self.message, "expected": self.expected, "actual": self.actual}


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityReport:
    """Result of checking a runtime context against a merged requirement map."""

    issues: tuple[RuntimeCompatibilityIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether every required runtime field is satisfied."""

        return not self.issues

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready report data."""

        return {"ok": self.ok, "issues": [issue.to_data() for issue in self.issues]}


def check_runtime_spec_satisfies_requirement(spec: RuntimeContextSpec | Mapping[str, Any], requirement: Mapping[str, Any]) -> RuntimeCompatibilityReport:
    """Check that a runtime context contains each resolved requirement value.

    Nested requirement mappings are subset constraints; scalar values require
    equality. Runtime mode/allocation invariants remain the responsibility of
    runtime activation rather than this declarative compatibility adapter.
    """

    candidate = spec.to_data() if isinstance(spec, RuntimeContextSpec) else RuntimeContextSpec.from_data(spec).to_data()
    return RuntimeCompatibilityReport(tuple(_missing(requirement, candidate)))


def _missing(requirement: Mapping[str, Any], candidate: Mapping[str, Any], path: str = ""):
    issues = []
    for key, expected in requirement.items():
        actual = candidate.get(key)
        item_path = f"{path}.{key}" if path else str(key)
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping):
                issues.append(RuntimeCompatibilityIssue(item_path, "runtime requirement mapping is missing", expected, actual))
            else:
                issues.extend(_missing(expected, actual, item_path))
        elif actual != expected:
            issues.append(RuntimeCompatibilityIssue(item_path, "runtime requirement is not satisfied", expected, actual))
    return issues


__all__ = ["RuntimeCompatibilityIssue", "RuntimeCompatibilityReport", "check_runtime_spec_satisfies_requirement"]
