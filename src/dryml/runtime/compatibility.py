"""Pure explicit compatibility checks for runtime declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .specs import RuntimeContextSpec


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityIssue:
    """One missing declarative runtime requirement."""

    path: str
    expected: Any
    actual: Any

    def to_data(self) -> dict[str, Any]:
        """Return a detached diagnostic mapping."""
        return {"path": self.path, "expected": self.expected, "actual": self.actual}


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityReport:
    """Immutable result of an explicit, non-enforcing runtime comparison."""

    issues: tuple[RuntimeCompatibilityIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether every requested declaration matched."""
        return not self.issues

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-compatible report without publishing runtime state."""
        return {"ok": self.ok, "issues": [issue.to_data() for issue in self.issues]}


def check_runtime_spec_satisfies_requirement(spec: RuntimeContextSpec | Mapping[str, Any], requirement: Mapping[str, Any]) -> RuntimeCompatibilityReport:
    """Compare explicit runtime values without effects, imports, or call wrapping.

    Args:
        spec: Runtime declaration to inspect.
        requirement: Nested subset requirement mapping.

    Returns:
        A report listing every missing scalar or nested mapping value.
    """
    value = spec if isinstance(spec, RuntimeContextSpec) else RuntimeContextSpec.from_data(spec)
    candidate = value._identifying_payload()
    aliases = {"visibility": "device_visibility", "framework": "frameworks"}
    normalized = {aliases.get(key, key): item for key, item in requirement.items()}
    return RuntimeCompatibilityReport(tuple(_missing(normalized, candidate)))


def _missing(requirement: Mapping[str, Any], candidate: Mapping[str, Any], path: str = "") -> list[RuntimeCompatibilityIssue]:
    issues: list[RuntimeCompatibilityIssue] = []
    for key, expected in requirement.items():
        actual = candidate.get(key)
        item_path = f"{path}.{key}" if path else str(key)
        if isinstance(expected, Mapping) and isinstance(actual, Mapping):
            issues.extend(_missing(expected, actual, item_path))
        elif actual != expected:
            issues.append(RuntimeCompatibilityIssue(item_path, expected, actual))
    return issues


__all__ = ["RuntimeCompatibilityIssue", "RuntimeCompatibilityReport", "check_runtime_spec_satisfies_requirement"]
