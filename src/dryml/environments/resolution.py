"""Bounded deterministic environment candidate resolution."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

from packaging.requirements import Requirement

from .compatibility import CompatibilityIssue, CompatibilityReport, report_from_issues
from .current import current
from .probe import EnvironmentProbeResult, probe
from .records import EnvironmentRecord
from .requirements import EnvironmentRequirement
from .schema import ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION
from .specs import CondaEnvironmentSpec, ContainerEnvironmentSpec, CurrentEnvironmentSpec, EnvironmentSpec, spec_from_data
from .utils import normalize_distribution_name

_MAX_RECORDED_ATTEMPTS = 32
_MAX_SERIALIZED_ITEMS = 64
_MAX_SERIALIZED_STRING = 4096
_MAX_SERIALIZED_DEPTH = 8
_MAX_SERIALIZED_NODES = 1024


@dataclass(frozen=True, slots=True)
class EnvironmentResolutionAttempt:
    """One bounded environment candidate-resolution attempt.

    Attributes:
        source: Candidate origin such as ``"candidate"`` or ``"registry"``.
        name: Optional registry entry name.
        spec: Canonical candidate environment specification.
        status: Selection, skip, probe, or compatibility outcome.
        probe: Optional evidence returned by the candidate probe.
        compatibility: Optional hard-requirement compatibility report.
        diagnostics: Bounded public diagnostic summaries.
        probe_duration_s: Measured cooperative probe duration in seconds.
    """

    source: str
    name: str | None
    spec: EnvironmentSpec
    status: str
    probe: EnvironmentProbeResult | None = None
    compatibility: CompatibilityReport | None = None
    diagnostics: tuple[CompatibilityIssue, ...] = ()
    probe_duration_s: float | None = None

    def to_data(self) -> dict[str, Any]:
        """Return bounded JSON-compatible attempt data."""

        return _bounded_data({
            "source": self.source,
            "name": self.name,
            "spec": _spec_summary(self.spec),
            "status": self.status,
            "probe": _probe_summary(self.probe),
            "probe_duration_s": self.probe_duration_s,
            "compatibility": None if self.compatibility is None else _report_summary(self.compatibility),
            "diagnostics": [_issue_summary(issue) for issue in self.diagnostics],
        })


@dataclass(frozen=True, slots=True)
class EnvironmentResolution:
    """Selected environment and deterministic bounded resolution trace.

    Attributes:
        status: Overall ``selected``, ``no_match``, or ``incomplete`` outcome.
        requirement: Optional hard requirement used for candidate checks.
        selected: Selected environment specification, when available.
        selected_name: Optional selected registry entry name.
        selected_source: Candidate origin for the selected specification.
        selected_record: Reusable environment record from the selected probe.
        selected_probe: Probe result corresponding to ``selected_record``.
        attempts: Ordered, bounded candidate attempt records.
        attempt_count: Total candidate attempts, including records omitted from
            the bounded public trace.
        probe_count: Total probes started during resolution.
        probe_duration_s: Aggregate measured duration of started probes.
        fallback_spec: Implicit-current candidate retained for dispatch fallback.
        fallback_record: Reusable record for ``fallback_spec``.
        fallback_probe: Probe evidence that produced ``fallback_record``.
        diagnostics: Bounded public resolver diagnostics.
        policy: Resolver selection policy.
    """

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
    attempt_count: int = 0
    probe_count: int = 0
    probe_duration_s: float = 0.0
    fallback_spec: EnvironmentSpec | None = None
    fallback_record: EnvironmentRecord | None = None
    fallback_probe: EnvironmentProbeResult | None = None

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

        return _bounded_data({
            "status": self.status,
            "requirement": None if self.requirement is None else self.requirement.to_data(),
            "selected": None if self.selected is None else _spec_summary(self.selected),
            "selected_name": self.selected_name,
            "selected_source": self.selected_source,
            "selected_record": _record_summary(self.selected_record),
            "selected_probe": _probe_summary(self.selected_probe),
            "attempts": [attempt.to_data() for attempt in self.attempts],
            "attempt_count": self.attempt_count,
            "probe_count": self.probe_count,
            "probe_duration_s": self.probe_duration_s,
            "fallback_probe": _probe_summary(self.fallback_probe),
            "diagnostics": [_issue_summary(issue) for issue in self.diagnostics],
            "policy": self.policy,
        })


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
    """Select the first compatible candidate in deterministic bounded order.

    Args:
        requirement: Optional hard environment requirement.
        candidates: Caller candidates, considered before registry entries.
        registry: Optional explicit registry, considered in name order.
        include_current: Include the current environment after other sources.
        policy: Supported resolver policy, currently ``"first_compatible"``.
        max_candidates: Positive bound on unique candidates considered, up to
            the bounded public attempt-trace limit.
        probe_timeout: Per-probe deadline for DRYML-managed probes.
        total_timeout: Total deadline observed between cooperative callbacks.
        probe_runner: Optional probe hook. In-process runners and candidate
            iterators are cooperative and must enforce their own hard deadline.
        clock: Optional monotonic clock injection for deterministic tests.

    Returns:
        A deterministic, bounded resolution report. Input or search truncation
        returns ``incomplete`` rather than allowing a lower-precedence fallback.
    """

    if policy != "first_compatible":
        raise ValueError(f"unsupported environment resolver policy {policy!r}")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")
    if max_candidates > _MAX_RECORDED_ATTEMPTS:
        raise ValueError(f"max_candidates must not exceed {_MAX_RECORDED_ATTEMPTS}")
    if probe_timeout is not None and (isinstance(probe_timeout, bool) or not isinstance(probe_timeout, (int, float)) or not math.isfinite(probe_timeout) or probe_timeout <= 0):
        raise ValueError("probe_timeout must be positive or None")
    if total_timeout is not None and (isinstance(total_timeout, bool) or not isinstance(total_timeout, (int, float)) or not math.isfinite(total_timeout) or total_timeout <= 0):
        raise ValueError("total_timeout must be positive or None")
    if probe_timeout is None and total_timeout is None:
        raise ValueError("probe_timeout or total_timeout must provide a finite deadline")
    runner = probe if probe_runner is None else probe_runner
    now = time.monotonic if clock is None else clock
    started = now()
    deadline = None if total_timeout is None else started + total_timeout
    # The normalized raw prefix and public attempt trace share one bound. This
    # keeps every considered candidate visible while duplicate-heavy input still
    # cannot make intake unbounded.
    raw_candidate_limit = _MAX_RECORDED_ATTEMPTS
    raw_candidates, candidates_truncated, candidates_timed_out = _normalize_candidates(
        candidates,
        max_items=raw_candidate_limit,
        select_first=requirement is None,
        now=now,
        deadline=deadline,
    )
    attempts: list[EnvironmentResolutionAttempt] = []
    seen: set[str] = set()
    considered = 0
    truncated = False
    search_incomplete = False
    attempt_count = 0
    probe_count = 0
    total_probe_duration_s = 0.0
    fallback_spec: EnvironmentSpec | None = None
    fallback_record: EnvironmentRecord | None = None
    fallback_probe: EnvironmentProbeResult | None = None
    registry_entries, registry_state = _registry_entries(
        registry,
        max_items=raw_candidate_limit,
        select_first=requirement is None,
        now=now,
        deadline=deadline,
    )

    def record(attempt: EnvironmentResolutionAttempt) -> None:
        nonlocal attempt_count, fallback_probe, fallback_record, fallback_spec, truncated
        attempt_count += 1
        # Dispatch may use the implicit current environment as a relaxed-policy
        # fallback. Preserve its evidence even when the public trace is full.
        if attempt.source == "current" and attempt.probe is not None:
            fallback_spec = attempt.spec
            fallback_probe = attempt.probe
            fallback_record = attempt.probe.record
        if len(attempts) < _MAX_RECORDED_ATTEMPTS:
            attempts.append(attempt)
        else:
            truncated = True

    def resolution(
        status: str,
        requirement: EnvironmentRequirement | None,
        selected: EnvironmentSpec | None,
        selected_name: str | None,
        selected_source: str | None,
        selected_record: EnvironmentRecord | None,
        selected_probe: EnvironmentProbeResult | None,
        diagnostics: tuple[CompatibilityIssue, ...],
    ) -> EnvironmentResolution:
        return EnvironmentResolution(
            status=status,
            requirement=requirement,
            selected=selected,
            selected_name=selected_name,
            selected_source=selected_source,
            selected_record=selected_record,
            selected_probe=selected_probe,
            attempts=tuple(attempts),
            diagnostics=diagnostics,
            policy=policy,
            attempt_count=attempt_count,
            probe_count=probe_count,
            probe_duration_s=total_probe_duration_s,
            fallback_spec=fallback_spec,
            fallback_record=fallback_record,
            fallback_probe=fallback_probe,
        )

    def result_diagnostics() -> tuple[CompatibilityIssue, ...]:
        diagnostics = []
        if truncated:
            diagnostics.append(CompatibilityIssue("resolver_trace_truncated", "warning", "resolver attempt metadata was truncated", expected=_MAX_RECORDED_ATTEMPTS))
        if candidates_truncated:
            diagnostics.append(CompatibilityIssue("resolver_candidates_truncated", "warning", "resolver candidate input was truncated", expected=raw_candidate_limit))
        if candidates_timed_out:
            diagnostics.append(CompatibilityIssue("resolver_candidate_input_timeout", "warning", "resolver candidate input exceeded the total timeout"))
        if registry_state["truncated"]:
            diagnostics.append(CompatibilityIssue("resolver_registry_truncated", "warning", "registry candidate input was truncated before a compatible candidate was found"))
        if registry_state["timed_out"]:
            diagnostics.append(CompatibilityIssue("resolver_candidate_input_timeout", "warning", "registry candidate input exceeded the total timeout"))
        return tuple(diagnostics)

    for source, name, spec, entry in _candidates(
        raw_candidates,
        registry_entries,
        include_current,
        candidates_truncated=candidates_truncated,
        candidates_timed_out=candidates_timed_out,
        registry_state=registry_state,
    ):
        if total_timeout is not None and now() - started >= total_timeout:
            record(EnvironmentResolutionAttempt(source, name, spec, "not_considered_timeout"))
            search_incomplete = True
            break
        attempt_probe_duration_s: float | None = None
        try:
            identity = _identity(spec)
        except Exception as exc:
            record(EnvironmentResolutionAttempt(source, name, spec, "probe_failed", diagnostics=(CompatibilityIssue("candidate_invalid", "error", f"environment candidate could not be serialized: {type(exc).__name__}"),)))
            continue
        if identity in seen:
            record(EnvironmentResolutionAttempt(source, name, spec, "duplicate"))
            continue
        seen.add(identity)
        if considered >= max_candidates:
            record(EnvironmentResolutionAttempt(source, name, spec, "not_considered_limit"))
            search_incomplete = True
            break
        if total_timeout is not None and now() - started >= total_timeout:
            record(EnvironmentResolutionAttempt(source, name, spec, "not_considered_timeout"))
            search_incomplete = True
            break
        considered += 1
        if entry is not None and not _labels_match(requirement, entry):
            record(EnvironmentResolutionAttempt(source, name, spec, "label_mismatch"))
            continue
        structural_issue = _structural_candidate_issue(spec)
        if structural_issue is not None:
            record(EnvironmentResolutionAttempt(
                source,
                name,
                spec,
                "unsupported",
                diagnostics=(structural_issue,),
            ))
            continue
        if requirement is None:
            record(EnvironmentResolutionAttempt(source, name, spec, "selected"))
            return resolution("selected", None, spec, name, source, None, None, result_diagnostics())
        try:
            remaining = None if total_timeout is None else max(0.0, total_timeout - (now() - started))
            timeout = remaining if probe_timeout is None else probe_timeout if remaining is None else min(probe_timeout, remaining)
            if timeout is not None and timeout <= 0:
                record(EnvironmentResolutionAttempt(source, name, spec, "not_considered_timeout"))
                search_incomplete = True
                break
            probe_started = now()
            probe_count += 1
            result = runner(spec, timeout=timeout)
            attempt_probe_duration_s = max(0.0, now() - probe_started)
            total_probe_duration_s += attempt_probe_duration_s
            _validate_probe_result(result, identity)
            if _identity(result.spec) != identity:
                issue = CompatibilityIssue("probe_spec_mismatch", "error", "environment probe result did not match the requested candidate")
                record(EnvironmentResolutionAttempt(source, name, spec, "probe_failed", result, diagnostics=(issue,), probe_duration_s=attempt_probe_duration_s))
                continue
        except Exception as exc:
            issue = CompatibilityIssue("probe_failed", "error", f"environment probe raised {type(exc).__name__}")
            duration = attempt_probe_duration_s
            if duration is None:
                duration = max(0.0, now() - probe_started)
                total_probe_duration_s += duration
            record(EnvironmentResolutionAttempt(source, name, spec, "probe_failed", diagnostics=(issue,), probe_duration_s=duration))
            if total_timeout is not None and now() - started >= total_timeout:
                # A malformed result or runner exception can consume the final
                # budget just as a normal result can. Later candidates are
                # unsearched, so this must not become a relaxed-policy fallback.
                search_incomplete = True
                break
            continue
        if total_timeout is not None and now() - started >= total_timeout:
            timeout_issue = CompatibilityIssue(
                "resolver_total_timeout",
                "error",
                "environment probe exhausted the total resolver timeout",
            )
            timeout_report = report_from_issues(
                (*(() if result.report is None else result.report.issues), timeout_issue)
            )
            timed_out_result = replace(result, ok=False, report=timeout_report)
            record(EnvironmentResolutionAttempt(
                source,
                name,
                spec,
                "probe_failed",
                timed_out_result,
                diagnostics=timeout_report.issues,
                probe_duration_s=attempt_probe_duration_s,
            ))
            # The completed probe consumed the resolver deadline, so later
            # ordered candidates remain unsearched and cannot be bypassed.
            search_incomplete = True
            break
        if not result.ok or result.record is None:
            diagnostics = () if result.report is None else result.report.issues
            record(EnvironmentResolutionAttempt(source, name, spec, "probe_failed", result, diagnostics=diagnostics, probe_duration_s=attempt_probe_duration_s))
            continue
        report = requirement.check(result.record, policy="strict")
        if report.ok:
            record(EnvironmentResolutionAttempt(source, name, spec, "selected", result, report, probe_duration_s=attempt_probe_duration_s))
            return resolution("selected", requirement, spec, name, source, result.record, result, result_diagnostics())
        record(EnvironmentResolutionAttempt(source, name, spec, "incompatible", result, report, report.issues, attempt_probe_duration_s))
    if search_incomplete or candidates_truncated or candidates_timed_out or registry_state["truncated"] or registry_state["timed_out"]:
        diagnostics = [CompatibilityIssue("resolver_input_truncated", "error", "environment candidate input was incomplete before compatibility could be determined")]
        diagnostics.extend(result_diagnostics())
        return resolution("incomplete", requirement, None, None, None, None, None, tuple(diagnostics))
    diagnostics = [CompatibilityIssue("resolver_no_match", "error", "no candidate satisfied the environment requirement", expected=None if requirement is None else requirement.to_data())]
    diagnostics.extend(result_diagnostics())
    return resolution("no_match", requirement, None, None, None, None, None, tuple(diagnostics))


def _candidates(
    candidates: Iterable[tuple[str, str | None, EnvironmentSpec, Any]],
    registry_entries: Iterable[tuple[str, str | None, EnvironmentSpec, Any]],
    include_current: bool,
    *,
    candidates_truncated: bool,
    candidates_timed_out: bool,
    registry_state: Mapping[str, bool],
) -> Iterator[tuple[str, str | None, EnvironmentSpec, Any]]:
    for candidate in candidates:
        yield candidate
    # An omitted caller candidate could take precedence over every registry or
    # current candidate. Do not silently select a lower-precedence source.
    if candidates_truncated or candidates_timed_out:
        return
    for entry in registry_entries:
        yield entry
    # Registry entries likewise precede the implicit current environment.
    if registry_state["truncated"] or registry_state["timed_out"]:
        return
    if include_current:
        yield "current", None, CurrentEnvironmentSpec(), None


def _registry_entries(
    registry: Any,
    *,
    max_items: int,
    select_first: bool,
    now: Callable[[], float],
    deadline: float | None,
) -> tuple[Iterable[tuple[str, str | None, EnvironmentSpec, Any]], dict[str, bool]]:
    """Lazily normalize a finite, deterministic registry entry prefix."""

    state = {"truncated": False, "timed_out": False}
    if registry is None:
        return (), state

    def entries() -> Iterator[tuple[str, str | None, EnvironmentSpec, Any]]:
        if deadline is not None and now() >= deadline:
            state["timed_out"] = True
            return
        raw_entries = registry.iter_entries() if hasattr(registry, "iter_entries") else iter(registry.list())
        normalized, state["truncated"], state["timed_out"] = _normalize_candidates(
            raw_entries,
            max_items=max_items,
            select_first=select_first,
            now=now,
            deadline=deadline,
            source="registry",
        )
        yield from normalized

    return entries(), state


def _normalize_candidate(candidate: Any, source: str) -> tuple[str, str | None, EnvironmentSpec, Any]:
    from .registry import EnvironmentRegistryEntry

    if isinstance(candidate, EnvironmentRegistryEntry):
        return source, candidate.name, candidate.spec, candidate
    if isinstance(candidate, EnvironmentSpec):
        return source, None, candidate, None
    if isinstance(candidate, Mapping):
        return source, None, spec_from_data(dict(candidate)), None
    raise TypeError(f"environment candidate must be an EnvironmentSpec, registry entry, or mapping, got {type(candidate).__name__}")


def _structural_candidate_issue(spec: EnvironmentSpec) -> CompatibilityIssue | None:
    """Return an environment-owned local-launch issue without probing it."""

    if isinstance(spec, ContainerEnvironmentSpec):
        return CompatibilityIssue("unsupported_environment_spec", "error", "container execution is not implemented")
    if isinstance(spec, CondaEnvironmentSpec):
        if spec.launch_mode == "direct" and not spec.prefix:
            return CompatibilityIssue("unsupported_environment_spec", "error", "direct Conda launch requires a prefix")
        if spec.launch_mode == "conda-run" and not (spec.prefix or spec.name):
            return CompatibilityIssue("unsupported_environment_spec", "error", "conda-run launch requires a prefix or name")
    return None


def _normalize_candidates(
    candidates: Iterable[Any],
    *,
    max_items: int | None,
    select_first: bool,
    now: Callable[[], float],
    deadline: float | None,
    source: str = "candidate",
) -> tuple[tuple[tuple[str, str | None, EnvironmentSpec, Any], ...], bool, bool]:
    """Normalize the bounded caller candidate prefix before any probe starts."""

    iterator = iter(candidates)
    values = []
    timed_out = False
    while max_items is None or len(values) <= max_items:
        if deadline is not None and now() >= deadline:
            timed_out = True
            break
        try:
            value = next(iterator)
        except StopIteration:
            break
        if deadline is not None and now() >= deadline:
            timed_out = True
            break
        values.append(value)
        if select_first:
            try:
                _source, _name, spec, _entry = _normalize_candidate(value, source)
            except Exception as exc:
                raise ValueError(f"invalid environment resolver candidate: {type(exc).__name__}") from exc
            if _structural_candidate_issue(spec) is None:
                break
    truncated = max_items is not None and len(values) > max_items
    if max_items is not None:
        values = values[:max_items]
    try:
        normalized = []
        for candidate in values:
            if deadline is not None and now() >= deadline:
                timed_out = True
                break
            normalized.append(_normalize_candidate(candidate, source))
            if deadline is not None and now() >= deadline:
                timed_out = True
                break
        return tuple(normalized), truncated, timed_out
    except Exception as exc:
        raise ValueError(f"invalid environment resolver candidate: {type(exc).__name__}") from exc


def _validate_probe_result(result: Any, identity: str) -> None:
    """Reject malformed injected probe evidence before it enters resolution."""

    if not isinstance(result, EnvironmentProbeResult):
        raise TypeError("probe runner returned an invalid result")
    if not isinstance(result.ok, bool):
        raise TypeError("probe runner result ok must be a boolean")
    if result.schema_version != ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION:
        raise TypeError("probe runner returned an unsupported result schema")
    if _identity(result.spec) != identity:
        raise TypeError("probe runner returned evidence for a different candidate")
    if result.ok and not isinstance(result.record, EnvironmentRecord):
        raise TypeError("successful probe result requires an environment record")
    if result.report is not None:
        if not isinstance(result.report, CompatibilityReport):
            raise TypeError("probe runner result report must be a CompatibilityReport")
        if result.report.status not in {"compatible", "warning", "incompatible", "unknown"}:
            raise TypeError("probe runner report has an invalid status")
        if not isinstance(result.report.issues, tuple) or not all(
            isinstance(issue, CompatibilityIssue) for issue in result.report.issues
        ):
            raise TypeError("probe runner report issues must be CompatibilityIssue values")
        if not isinstance(result.report.details, Mapping):
            raise TypeError("probe runner report details must be a mapping")


def _labels_match(requirement: EnvironmentRequirement | None, entry: Any) -> bool:
    if requirement is None:
        return True
    tags = set(entry.tags)
    provides = set(entry.provides)
    return (
        (not tags or set(requirement.tags) <= tags)
        and (not provides or set(requirement.capabilities) <= provides)
        and not _requirement_hints_conflict(requirement, entry.requirement)
    )


def _requirement_hints_conflict(requested: EnvironmentRequirement, hint: EnvironmentRequirement | None) -> bool:
    """Return only contradictions that registry requirement hints can prove."""

    if hint is None:
        return False
    requested_packages = {normalize_distribution_name(Requirement(item).name): item for item in requested.requirements}
    hinted_packages = {normalize_distribution_name(Requirement(item).name): item for item in hint.requirements}
    if set(requested.excludes) & set(hinted_packages):
        return True
    if set(hint.excludes) & set(requested_packages):
        return True
    return False


def _identity(spec: EnvironmentSpec) -> str:
    return json.dumps(spec.to_data(), sort_keys=True, separators=(",", ":"))


def _spec_summary(spec: EnvironmentSpec) -> dict[str, Any]:
    """Return redacted candidate metadata with a safe canonical identifier."""

    data = spec.to_data()
    data.pop("env", None)
    # The content ID preserves distinct candidate identities without exposing
    # environment override names or values in a public resolver trace.
    data["id"] = spec.id
    if "extra_pythonpath" in data:
        data["extra_pythonpath"] = list(data["extra_pythonpath"][:16])
    return _bounded_data(data)


def _record_summary(record: EnvironmentRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return _bounded_data({
        "id": record.id,
        "python": {"implementation": record.python.implementation, "version": record.python.version},
        "platform": {"system": record.platform.system, "machine": record.platform.machine},
    })


def _probe_summary(result: EnvironmentProbeResult | None) -> dict[str, Any] | None:
    """Return bounded probe evidence without duplicating selected record data."""

    if result is None:
        return None
    return _bounded_data({
        "ok": result.ok,
        "returncode": result.returncode,
        "report": None if result.report is None else _report_summary(result.report),
    })


def _report_summary(report: CompatibilityReport) -> dict[str, Any]:
    """Return public compatibility metadata without injected detail values."""

    return {
        "schema_version": report.schema_version,
        "status": report.status,
        "issues": [_issue_summary(issue) for issue in report.issues],
        "details": {"redacted": True},
    }


def _issue_summary(issue: CompatibilityIssue) -> dict[str, Any]:
    """Return safe public issue metadata from trusted or injected evidence."""

    # Probe runners are injectable, so messages and observed values can contain
    # environment overrides or credentials. Keep stable diagnosis identifiers
    # and paths without serializing arbitrary runner-controlled values.
    return {
        "schema_version": issue.schema_version,
        "code": issue.code,
        "severity": issue.severity,
        "message": f"environment compatibility issue: {issue.code}",
        "requirement_path": issue.requirement_path,
        "observed_path": issue.observed_path,
        "expected": {"redacted": True} if issue.expected is not None else None,
        "observed": {"redacted": True} if issue.observed is not None else None,
    }


def _bounded_data(value: Any, *, depth: int = 0, budget: list[int] | None = None) -> Any:
    """Bound public resolver serialization without retaining secret overrides."""

    budget = [_MAX_SERIALIZED_NODES] if budget is None else budget
    if budget[0] <= 0 or depth > _MAX_SERIALIZED_DEPTH:
        return {"__dryml_truncated__": "depth_or_size"}
    budget[0] -= 1
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return value[:_MAX_SERIALIZED_STRING]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key)[:_MAX_SERIALIZED_STRING]: _bounded_data(item, depth=depth + 1, budget=budget)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))[:_MAX_SERIALIZED_ITEMS]
            if str(key) != "env"
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_data(item, depth=depth + 1, budget=budget) for item in value[:_MAX_SERIALIZED_ITEMS]]
    # Probe runners are injectable. Never allow arbitrary objects supplied by a
    # runner to make public resolver metadata non-JSON-serializable or expose
    # their representation.
    return {"__dryml_unsupported_type__": type(value).__name__}


__all__ = ["EnvironmentResolution", "EnvironmentResolutionAttempt", "resolve"]
