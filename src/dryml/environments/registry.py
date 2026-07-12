"""In-memory registry for named environment specs."""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass
from itertools import islice
import json
import math
import time
from typing import Any

from .compatibility import CompatibilityIssue, CompatibilityReport, report_from_issues
from .errors import EnvironmentRegistryError
from .probe import EnvironmentProbeResult, probe
from .requirements import EnvironmentRequirement
from .specs import EnvironmentSpec, spec_from_data
from .utils import coerce_tuple


@dataclass(frozen=True, slots=True)
class EnvironmentRegistryEntry:
    """Named environment spec plus resolver prefilter hints.

    Attributes:
        name: Unique deterministic registry key.
        spec: Environment selected when this entry is resolved.
        provides: Capability hints used to avoid impossible probes.
        tags: Label hints used to avoid impossible probes.
        requirement: Optional declared requirement hint; it is not proof of
            runtime compatibility.
    """

    name: str
    spec: EnvironmentSpec
    provides: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    requirement: EnvironmentRequirement | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise EnvironmentRegistryError("environment registry entry name must be a non-empty string")
        if not isinstance(self.spec, EnvironmentSpec):
            raise EnvironmentRegistryError("environment registry entry spec must be an EnvironmentSpec")
        if self.requirement is not None and not isinstance(self.requirement, EnvironmentRequirement):
            raise EnvironmentRegistryError("environment registry entry requirement must be an EnvironmentRequirement")
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

    The registry is an explicit dispatch selection aid. It does not persist
    records and does not probe registered environments unless explicitly requested.
    """

    def __init__(self) -> None:
        self._entries: dict[str, EnvironmentRegistryEntry] = {}
        self._names: list[str] = []

    def register(
        self,
        name: str,
        spec: EnvironmentSpec,
        *,
        provides: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        requirement: EnvironmentRequirement | None = None,
    ) -> EnvironmentRegistryEntry:
        """Register and return a named environment spec without probing it.

        Args:
            name: Unique non-empty registry key.
            spec: Environment specification to associate with ``name``.
            provides: Optional capability prefilter hints.
            tags: Optional label prefilter hints.
            requirement: Optional requirement prefilter hint.

        Returns:
            The immutable registered entry.

        Raises:
            EnvironmentRegistryError: If the entry is invalid or ``name`` is
                already registered.
        """

        entry = EnvironmentRegistryEntry(name, spec, provides=provides, tags=tags, requirement=requirement)
        if entry.name in self._entries:
            raise EnvironmentRegistryError(
                f"environment registry entry {entry.name!r} already exists",
                context={"name": entry.name},
            )
        # Validate the complete entry before changing either index, preserving
        # deterministic lifecycle behavior after malformed registration input.
        self._entries[entry.name] = entry
        insort(self._names, entry.name)
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

    def unregister(self, name: str) -> EnvironmentRegistryEntry:
        """Remove and return the entry named *name* without probing it.

        Args:
            name: Registered entry name to remove.

        Returns:
            The removed immutable registry entry.

        Raises:
            EnvironmentRegistryError: If *name* is not registered.
        """

        try:
            entry = self._entries.pop(name)
            self._names.remove(name)
            return entry
        except KeyError as exc:
            raise EnvironmentRegistryError(
                f"environment registry entry {name!r} does not exist",
                context={"name": name},
            ) from exc

    def list(self) -> tuple[EnvironmentRegistryEntry, ...]:
        """Return registered entries in deterministic name order."""

        return tuple(self._entries[name] for name in self._names)

    def iter_entries(self, *, limit: int | None = None):
        """Yield name-sorted entries without materializing the full registry."""

        names = self._names if limit is None else islice(self._names, limit)
        yield from (self._entries[name] for name in names)

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

    def probe_registered(self, name: str, *, timeout: float | None = 30.0) -> EnvironmentProbeResult:
        """Probe a registered environment by name."""

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
        max_candidates: int | None = None,
        total_timeout: float | None = None,
    ) -> tuple[EnvironmentRegistryEntry | None, CompatibilityReport]:
        """Return the first bounded, deduplicated compatible registry entry.

        Supplying ``max_candidates`` or ``total_timeout`` opts into bounded
        search. Omitting both preserves this legacy helper's full-registry,
        tag-prefilter behavior; dispatch uses :func:`resolve` for bounded
        deterministic search.

        Args:
            requirement: Hard environment requirement to check.
            timeout: Per-entry probe timeout in seconds, or ``None``.
            policy: Requirement-check policy passed to the requirement.
            max_candidates: Optional positive bound on unique entries checked.
            total_timeout: Optional positive deadline for bounded search.

        Returns:
            The first compatible entry and its report, or ``None`` and the
            no-match report.

        Raises:
            EnvironmentRegistryError: If an optional search bound is invalid.
        """

        if max_candidates is not None and (
            isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0
        ):
            raise EnvironmentRegistryError("max_candidates must be a positive integer or None")
        for name, value in (("timeout", timeout), ("total_timeout", total_timeout)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise EnvironmentRegistryError(f"{name} must be a positive finite number or None")
        first_report: CompatibilityReport | None = None
        bounded_mode = max_candidates is not None or total_timeout is not None
        seen: set[str] = set()
        considered = 0
        started = time.monotonic()
        # Label mismatches and canonical aliases do not consume the candidate
        # budget, so they must not hide a later viable name-sorted entry.
        for entry in self.iter_entries():
            if total_timeout is not None and time.monotonic() - started >= total_timeout:
                break
            if not bounded_mode and requirement.tags and not set(requirement.tags) <= set(entry.tags):
                continue
            if bounded_mode:
                from .resolution import _labels_match

                if not _labels_match(requirement, entry):
                    continue
                identity = json.dumps(entry.spec.to_data(), sort_keys=True, separators=(",", ":"))
                if identity in seen:
                    continue
                seen.add(identity)
                if max_candidates is not None and considered >= max_candidates:
                    break
            considered += 1
            remaining = None if total_timeout is None else total_timeout - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                break
            probe_timeout = timeout if remaining is None else remaining if timeout is None else min(timeout, remaining)
            result = probe(entry.spec, timeout=probe_timeout)
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

        issue = CompatibilityIssue(
            "registry_no_match",
            "error",
            message,
            expected=None if requirement is None else requirement.to_data(),
        )
        return report_from_issues((issue,))

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
