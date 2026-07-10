"""Bounded deterministic environment candidate resolution."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .compatibility import CompatibilityIssue, CompatibilityReport
from .current import current
from .probe import EnvironmentProbeResult, probe
from .records import EnvironmentRecord
from .requirements import EnvironmentRequirement
from .specs import CurrentEnvironmentSpec, EnvironmentSpec, spec_from_data

_MAX_RECORDED_ATTEMPTS = 32


@dataclass(frozen=True, slots=True)
class EnvironmentResolutionAttempt:
    """One ordered candidate considered by environment resolution."""

    source: str
    name: str | None
    spec: EnvironmentSpec
    status: str
    probe: EnvironmentProbeResult | None = None
    compatibility: CompatibilityReport | None = None
    diagnostics: tuple[CompatibilityIssue, ...] = ()

    def to_data(self) -> dict[str, Any]:
        """Return bounded JSON-compatible attempt data."""

        return {
            "source": self.source,
            "name": self.name,
            "spec": self.spec.to_data(),
            "status": self.status,
            "probe": None if self.probe is None else {"ok": self.probe.ok, "returncode": self.probe.returncode, "report": None if self.probe.report is None else self.probe.report.to_data()},
            "compatibility": None if self.compatibility is None else self.compatibility.to_data(),
            "diagnostics": [issue.to_data() for issue in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class EnvironmentResolution:
    """Selected environment plus complete bounded resolution trace."""

    status: str
    requirement: EnvironmentRequirement | None
    selected: EnvironmentSpec | None
    selected_name: str | None
    selected_source: str | None
    selected_record: EnvironmentRecord | None
    selected_probe: EnvironmentProbeResult | None
    attempts: tuple[EnvironmentResolutionAttempt, ...]
    diagnostics: tuple[CompatibilityIssue, ...]
    policy: str

    @property
    def ok(self) -> bool:
        """Return whether a candidate was selected."""

        return self.selected is not None

    def require_selected(self) -> EnvironmentSpec:
        """Return the selected spec or raise a structured compatibility error."""

        if self.selected is None:
            from .errors import EnvironmentCompatibilityError

            raise EnvironmentCompatibilityError("no compatible environment candidate was resolved", context={"attempts": [attempt.to_data() for attempt in self.attempts]})
        return self.selected

    def to_data(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible resolver output."""

        return {
            "status": self.status,
            "requirement": None if self.requirement is None else self.requirement.to_data(),
            "selected": None if self.selected is None else self.selected.to_data(),
            "selected_name": self.selected_name,
            "selected_source": self.selected_source,
            "selected_record": _record_summary(self.selected_record),
            "attempts": [attempt.to_data() for attempt in self.attempts],
            "diagnostics": [issue.to_data() for issue in self.diagnostics],
            "policy": self.policy,
        }


def resolve(
    requirement: EnvironmentRequirement | None,
    *,
    candidates: Iterable[EnvironmentSpec | Mapping[str, Any] | Any] = (),
    registry: Any = None,
    include_current: bool = True,
    policy: str = "first_compatible",
    max_candidates: int = 8,
    probe_timeout: float | None = 10.0,
    total_timeout: float | None = 30.0,
    probe_runner: Callable[..., EnvironmentProbeResult] | None = None,
    clock: Callable[[], float] | None = None,
) -> EnvironmentResolution:
    """Select the first compatible candidate in deterministic bounded order."""

    if policy != "first_compatible":
        raise ValueError(f"unsupported environment resolver policy {policy!r}")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")
    if probe_timeout is not None and (isinstance(probe_timeout, bool) or probe_timeout <= 0):
        raise ValueError("probe_timeout must be positive or None")
    if total_timeout is not None and (isinstance(total_timeout, bool) or total_timeout <= 0):
        raise ValueError("total_timeout must be positive or None")
    runner = probe if probe_runner is None else probe_runner
    now = time.monotonic if clock is None else clock
    attempts: list[EnvironmentResolutionAttempt] = []
    seen: set[str] = set()
    started = now()
    considered = 0
    for source, name, spec, entry in _candidates(candidates, registry, include_current)[:_MAX_RECORDED_ATTEMPTS]:
        identity = json.dumps(spec.to_data(), sort_keys=True, separators=(",", ":"))
        if identity in seen:
            attempts.append(EnvironmentResolutionAttempt(source, name, spec, "duplicate"))
            continue
        seen.add(identity)
        if considered >= max_candidates:
            attempts.append(EnvironmentResolutionAttempt(source, name, spec, "not_considered_limit"))
            continue
        if total_timeout is not None and now() - started >= total_timeout:
            attempts.append(EnvironmentResolutionAttempt(source, name, spec, "not_considered_timeout"))
            continue
        considered += 1
        if entry is not None and not _labels_match(requirement, entry):
            attempts.append(EnvironmentResolutionAttempt(source, name, spec, "label_mismatch"))
            continue
        if requirement is None:
            attempts.append(EnvironmentResolutionAttempt(source, name, spec, "selected"))
            return EnvironmentResolution("selected", None, spec, name, source, None, None, tuple(attempts), (), policy)
        try:
            remaining = None if total_timeout is None else max(0.0, total_timeout - (now() - started))
            timeout = probe_timeout if remaining is None or probe_timeout is None else min(probe_timeout, remaining)
            if timeout is not None and timeout <= 0:
                attempts.append(EnvironmentResolutionAttempt(source, name, spec, "not_considered_timeout"))
                continue
            result = runner(spec, timeout=timeout)
        except Exception as exc:
            issue = CompatibilityIssue("probe_failed", "error", f"environment probe raised {type(exc).__name__}: {exc}")
            attempts.append(EnvironmentResolutionAttempt(source, name, spec, "probe_failed", diagnostics=(issue,)))
            continue
        if not result.ok or result.record is None:
            diagnostics = () if result.report is None else result.report.issues
            attempts.append(EnvironmentResolutionAttempt(source, name, spec, "probe_failed", result, result.report, diagnostics))
            continue
        report = requirement.check(result.record, policy="strict")
        if report.ok:
            attempts.append(EnvironmentResolutionAttempt(source, name, spec, "selected", result, report))
            return EnvironmentResolution("selected", requirement, spec, name, source, result.record, result, tuple(attempts), (), policy)
        attempts.append(EnvironmentResolutionAttempt(source, name, spec, "incompatible", result, report, report.issues))
    issue = CompatibilityIssue("resolver_no_match", "error", "no candidate satisfied the environment requirement", expected=None if requirement is None else requirement.to_data())
    return EnvironmentResolution("no_match", requirement, None, None, None, None, None, tuple(attempts), (issue,), policy)


def _candidates(candidates: Iterable[Any], registry: Any, include_current: bool) -> list[tuple[str, str | None, EnvironmentSpec, Any]]:
    queued: list[tuple[str, str | None, EnvironmentSpec, Any]] = []
    for candidate in candidates:
        queued.append(_normalize_candidate(candidate, "candidate"))
    if registry is not None:
        for entry in registry.list():
            queued.append(("registry", entry.name, entry.spec, entry))
    if include_current:
        selected = current()
        queued.append(("current", None, CurrentEnvironmentSpec() if selected is None else selected, None))
    return queued


def _normalize_candidate(candidate: Any, source: str) -> tuple[str, str | None, EnvironmentSpec, Any]:
    if hasattr(candidate, "spec") and hasattr(candidate, "name"):
        return source, candidate.name, candidate.spec, candidate
    if isinstance(candidate, EnvironmentSpec):
        return source, None, candidate, None
    if isinstance(candidate, Mapping):
        return source, None, spec_from_data(dict(candidate)), None
    raise TypeError(f"environment candidate must be an EnvironmentSpec, registry entry, or mapping, got {type(candidate).__name__}")


def _labels_match(requirement: EnvironmentRequirement | None, entry: Any) -> bool:
    if requirement is None:
        return True
    return set(requirement.tags) <= set(entry.tags) and set(requirement.capabilities) <= set(entry.provides)


def _record_summary(record: EnvironmentRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "id": record.id,
        "python": {"implementation": record.python.implementation, "version": record.python.version},
        "platform": {"system": record.platform.system, "machine": record.platform.machine},
    }


__all__ = ["EnvironmentResolution", "EnvironmentResolutionAttempt", "resolve"]
