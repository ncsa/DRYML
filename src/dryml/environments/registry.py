"""In-memory registry for named environment specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compatibility import CompatibilityIssue, CompatibilityReport, report_from_issues, unavailable_report
from .errors import EnvironmentRegistryError
from .requirements import EnvironmentRequirement
from .specs import EnvironmentSpec, spec_from_data
from .utils import coerce_tuple


@dataclass(frozen=True, slots=True)
class EnvironmentRegistryEntry:
    """Named environment spec plus selection labels."""

    name: str
    spec: EnvironmentSpec
    provides: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    requirement: EnvironmentRequirement | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provides", tuple(sorted(str(item) for item in coerce_tuple(self.provides))))
        object.__setattr__(self, "tags", tuple(sorted(str(item) for item in coerce_tuple(self.tags))))

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible registry entry data."""

        return {
            "name": self.name,
            "spec": self.spec.to_data(),
            "provides": list(self.provides),
            "tags": list(self.tags),
            "requirement": None if self.requirement is None else self.requirement.to_data(),
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "EnvironmentRegistryEntry":
        """Build a registry entry from serialized data."""

        return cls(
            name=data["name"],
            spec=spec_from_data(data["spec"]),
            provides=tuple(data.get("provides", ())),
            tags=tuple(data.get("tags", ())),
            requirement=(
                None
                if data.get("requirement") is None
                else EnvironmentRequirement.from_data(data["requirement"])
            ),
        )


class EnvironmentRegistry:
    """Deterministic in-memory registry of environment specs.

    The registry is a future dispatch selection aid. It does not persist records
    and does not probe registered environments unless explicitly requested.
    """

    def __init__(self) -> None:
        self._entries: dict[str, EnvironmentRegistryEntry] = {}

    def register(
        self,
        name: str,
        spec: EnvironmentSpec,
        *,
        provides: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        requirement: EnvironmentRequirement | None = None,
    ) -> EnvironmentRegistryEntry:
        """Register a named environment spec.

        Duplicate names are rejected so selection remains deterministic.
        """

        if name in self._entries:
            raise EnvironmentRegistryError(
                f"environment registry entry {name!r} already exists",
                context={"name": name},
            )
        entry = EnvironmentRegistryEntry(name, spec, provides=provides, tags=tags, requirement=requirement)
        self._entries[name] = entry
        return entry

    def get(self, name: str) -> EnvironmentRegistryEntry:
        """Return a registered environment by name."""

        try:
            return self._entries[name]
        except KeyError as exc:
            raise EnvironmentRegistryError(
                f"environment registry entry {name!r} does not exist",
                context={"name": name},
            ) from exc

    def list(self) -> tuple[EnvironmentRegistryEntry, ...]:
        """Return registered entries in deterministic name order."""

        return tuple(self._entries[name] for name in sorted(self._entries))

    def find(
        self,
        requirement: EnvironmentRequirement | None = None,
        *,
        tags: tuple[str, ...] = (),
        provides: tuple[str, ...] = (),
    ) -> EnvironmentRegistryEntry | None:
        """Find the first deterministic entry matching labels and requirement hints."""

        required_tags = set(tags) | set(requirement.tags if requirement else ())
        required_provides = set(provides) | set(requirement.capabilities if requirement else ())
        for entry in self.list():
            if required_tags and not required_tags <= set(entry.tags):
                continue
            if required_provides and not required_provides <= set(entry.provides):
                continue
            return entry
        return None

    def probe_registered(self, name: str, *, timeout: float | None = 30.0) -> Any:
        """Probe a registered environment by name."""

        from .probe import probe

        return probe(self.get(name).spec, timeout=timeout)

    def check_requirement(
        self,
        name: str,
        requirement: EnvironmentRequirement,
        *,
        timeout: float | None = 30.0,
        policy: str = "compatible",
    ) -> CompatibilityReport:
        """Probe a named environment and check a requirement against it."""

        result = self.probe_registered(name, timeout=timeout)
        if not result.ok or result.record is None:
            return result.report or self.no_match_report(requirement, message=f"probe failed for {name!r}")
        return requirement.check(result.record, policy=policy)

    def find_compatible(
        self,
        requirement: EnvironmentRequirement,
        *,
        timeout: float | None = 30.0,
        policy: str = "compatible",
    ) -> tuple[EnvironmentRegistryEntry | None, CompatibilityReport]:
        """Return the first probed compatible entry and its report."""

        first_report: CompatibilityReport | None = None
        for entry in self.list():
            if requirement.tags and not set(requirement.tags) <= set(entry.tags):
                continue
            from .probe import probe

            result = probe(entry.spec, timeout=timeout)
            if not result.ok or result.record is None:
                first_report = first_report or result.report
                continue
            report = requirement.check(result.record, policy=policy)
            first_report = first_report or report
            if report.ok:
                return entry, report
        return None, first_report or self.no_match_report(requirement)

    def no_match_report(
        self,
        requirement: EnvironmentRequirement | None = None,
        *,
        message: str = "no registered environment matched the requirement",
    ) -> CompatibilityReport:
        """Return a structured report explaining registry selection failure."""

        report = unavailable_report(message)
        if requirement is None:
            return report
        issue = CompatibilityIssue("registry_no_match", "error", message, expected=requirement.to_data())
        return CompatibilityReport("unavailable", report.issues + (issue,))

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible registry data."""

        return {"entries": [entry.to_data() for entry in self.list()]}

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "EnvironmentRegistry":
        """Build a registry from serialized data."""

        registry = cls()
        for entry_data in data.get("entries", ()):
            entry = EnvironmentRegistryEntry.from_data(entry_data)
            registry.register(
                entry.name,
                entry.spec,
                provides=entry.provides,
                tags=entry.tags,
                requirement=entry.requirement,
            )
        return registry


__all__ = ["EnvironmentRegistry", "EnvironmentRegistryEntry"]
