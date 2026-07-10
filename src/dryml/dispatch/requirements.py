"""Requirement-aware dispatch planning over normalized operation targets.

This module orchestrates existing code analysis, annotation, environment, world,
and runtime APIs.  It does not normalize user targets, merge annotations, solve
environments, or synthesize worlds.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dryml import annotations, environments, runtime, worlds
from dryml.code import analyze, probe_target
from dryml.code.analysis import CodeAnalysisContext, CodeAnalysisResult
from dryml.code.facts import AnnotationFact, DiagnosticFact
from dryml.code.probe import CodeProbeResult
from dryml.code.targets import CodeTargetSpec
from dryml.environments.records import EnvironmentRecord
from dryml.environments.specs import EnvironmentSpec, PythonExecutableSpec, spec_from_data
from dryml.runtime import RuntimeEnforcement, RuntimeMode
from dryml.runtime.specs import RuntimeContextSpec
from dryml.worlds.specs import WorldSpec

from .errors import DispatchPlanningError
from .normalize import NormalizedDispatchTarget


PLANNING_METADATA_VERSION = 1
DEFAULT_PROBE_TIMEOUT_S = 30.0
_RESERVED_PLANNING_KEYS = frozenset(
    {
        "dryml.dispatch.planning_version",
        "dryml.code_analysis",
        "dryml.code_probe",
        "dryml.requirements",
        "dryml.requirement_sources",
        "dryml.environment_selection",
        "dryml.environment_probe",
        "dryml.environment_check",
        "dryml.world_selection",
        "dryml.world_check",
        "dryml.runtime_selection",
        "dryml.runtime_check",
        "dryml.requirement_policy",
        "dryml.runtime_enforcement",
        "dryml.dispatch.launchable",
        "dryml.dispatch.diagnostics",
    }
)


class RequirementPolicy(str, Enum):
    """How dispatch handles discovered hard requirement incompatibilities."""

    STRICT = "strict"
    WARN = "warn"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class CandidateConsideration:
    """One deterministic candidate-precedence slot considered by planning."""

    slot: str
    status: str
    candidate: Mapping[str, Any] | None = None

    def to_data(self) -> dict[str, Any]:
        """Return bounded JSON-ready consideration data."""

        return {"slot": self.slot, "status": self.status, "candidate": None if self.candidate is None else dict(self.candidate)}


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """A selected environment, world, or runtime and its precedence trace."""

    kind: str
    candidate: Mapping[str, Any]
    source: str
    considered: tuple[CandidateConsideration, ...]
    diagnostics: tuple[DiagnosticFact, ...] = ()

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready candidate selection data."""

        return {
            "kind": self.kind,
            "candidate": dict(self.candidate),
            "source": self.source,
            "considered": [item.to_data() for item in self.considered],
            "diagnostics": [item.to_data() for item in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class CandidateCheckReport:
    """Normalized result of checking one selected candidate."""

    kind: str
    status: str
    compatible: bool | None
    requirement: Mapping[str, Any] | None
    candidate: Mapping[str, Any] | None
    details: tuple[Mapping[str, Any], ...] = ()
    diagnostics: tuple[DiagnosticFact, ...] = ()

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready compatibility report data."""

        return {
            "kind": self.kind,
            "status": self.status,
            "compatible": self.compatible,
            "requirement": None if self.requirement is None else dict(self.requirement),
            "candidate": None if self.candidate is None else dict(self.candidate),
            "details": [dict(item) for item in self.details],
            "diagnostics": [item.to_data() for item in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class DispatchPlanningResolution:
    """Complete serializable decisions for a normalized dispatch target."""

    normalized_target: Mapping[str, Any]
    code_analysis: CodeAnalysisResult | None
    code_probe: CodeProbeResult | None
    bootstrap_environment: Mapping[str, Any] | None
    bootstrap_code_probe: CodeProbeResult | None
    final_code_probe: CodeProbeResult | None
    requirements: annotations.RequirementResolution
    environment_selection: CandidateSelection
    environment_record: EnvironmentRecord | None
    environment_check: CandidateCheckReport
    world_selection: CandidateSelection
    world_check: CandidateCheckReport
    runtime_selection: CandidateSelection
    runtime_check: CandidateCheckReport
    requirement_policy: RequirementPolicy
    runtime_enforcement: RuntimeEnforcement
    diagnostics: tuple[DiagnosticFact, ...]
    launchable: bool

    def to_data(self) -> dict[str, Any]:
        """Return bounded JSON-ready planning decisions without live targets."""

        return _bounded_data({
            "normalized_target": dict(self.normalized_target),
            "code_analysis": _analysis_summary(self.code_analysis),
            "code_probe": _probe_summary(self.code_probe),
            "bootstrap_environment": None if self.bootstrap_environment is None else dict(self.bootstrap_environment),
            "bootstrap_code_probe": _probe_summary(self.bootstrap_code_probe),
            "final_code_probe": _probe_summary(self.final_code_probe),
            "requirements": self.requirements.to_data(),
            "environment_selection": self.environment_selection.to_data(),
            "environment_record": _environment_record_summary(self.environment_record),
            "environment_check": self.environment_check.to_data(),
            "world_selection": self.world_selection.to_data(),
            "world_check": self.world_check.to_data(),
            "runtime_selection": self.runtime_selection.to_data(),
            "runtime_check": self.runtime_check.to_data(),
            "requirement_policy": self.requirement_policy.value,
            "runtime_enforcement": self.runtime_enforcement.value,
            "diagnostics": [item.to_data() for item in self.diagnostics],
            "launchable": self.launchable,
        })

    def metadata(self) -> dict[str, Any]:
        """Return authoritative dispatch metadata for this resolution."""

        data = self.to_data()
        return {
            "dryml.dispatch.planning_version": PLANNING_METADATA_VERSION,
            "dryml.code_analysis": data["code_analysis"],
            "dryml.code_probe": {
                "bootstrap_environment": data["bootstrap_environment"],
                "bootstrap_probe": data["bootstrap_code_probe"],
                "final_probe": data["final_code_probe"],
            },
            "dryml.requirements": data["requirements"],
            "dryml.requirement_sources": data["requirements"]["source_traces"],
            "dryml.environment_selection": data["environment_selection"],
            "dryml.environment_probe": _environment_record_summary(self.environment_record),
            "dryml.environment_check": data["environment_check"],
            "dryml.world_selection": data["world_selection"],
            "dryml.world_check": data["world_check"],
            "dryml.runtime_selection": data["runtime_selection"],
            "dryml.runtime_check": data["runtime_check"],
            "dryml.requirement_policy": self.requirement_policy.value,
            "dryml.runtime_enforcement": self.runtime_enforcement.value,
            "dryml.dispatch.launchable": self.launchable,
            "dryml.dispatch.diagnostics": data["diagnostics"],
        }


@dataclass(frozen=True, slots=True)
class DispatchExplanation:
    """Non-launching view of dispatch planning decisions."""

    resolution: DispatchPlanningResolution
    operation_preview: Mapping[str, Any]
    blocking_diagnostics: tuple[DiagnosticFact, ...]

    @property
    def launchable(self) -> bool:
        """Return whether the equivalent dispatch plan may launch."""

        return self.resolution.launchable

    @property
    def requirements(self):
        """Return the resolved annotation requirements/defaults."""

        return self.resolution.requirements

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready explanation data."""

        return _bounded_data({
            "resolution": self.resolution.to_data(),
            "operation_preview": dict(self.operation_preview),
            "blocking_diagnostics": [item.to_data() for item in self.blocking_diagnostics],
            "launchable": self.launchable,
        })

    def __str__(self) -> str:
        """Format a concise human-readable planning summary."""

        return (
            f"dispatch target={self.operation_preview.get('kind')} policy={self.resolution.requirement_policy.value} "
            f"environment={self.resolution.environment_selection.source} "
            f"world={self.resolution.world_selection.source} "
            f"runtime={self.resolution.runtime_selection.source} launchable={self.launchable}"
        )


def effective_requirement_policy(explicit: RequirementPolicy | str | None, enforcement: RuntimeEnforcement | str | None = None) -> RequirementPolicy:
    """Return an explicit policy or the context-local enforcement-derived default."""

    if explicit is not None:
        try:
            return RequirementPolicy(explicit)
        except ValueError as exc:
            raise DispatchPlanningError("invalid requirement_policy; expected strict, warn, or ignore", context={"requirement_policy": explicit}) from exc
    current = runtime.enforcement() if enforcement is None else RuntimeEnforcement(enforcement)
    return {
        RuntimeEnforcement.STRICT: RequirementPolicy.STRICT,
        RuntimeEnforcement.WARN: RequirementPolicy.WARN,
        RuntimeEnforcement.OFF: RequirementPolicy.IGNORE,
    }[current]


def resolve_dispatch_plan(
    normalized: NormalizedDispatchTarget,
    *,
    environment: Any | None = None,
    world: Any | None = None,
    runtime_spec: Any | None = None,
    requirement_policy: RequirementPolicy | str | None = None,
    analysis_policy: Any | None = None,
    emit_warnings: bool = False,
    single_worker_only: bool = False,
) -> DispatchPlanningResolution:
    """Resolve requirements and candidate checks for one normalized target.

    ``normalized`` is intentionally the only target input.  The resolver never
    calls normalization and therefore retains live annotation targets supplied by
    the public normalization boundary.
    """

    enforcement = runtime.enforcement()
    policy = effective_requirement_policy(requirement_policy, enforcement)
    analysis_context, probe_timeout_s = _analysis_options(analysis_policy)
    fragments, analysis, bootstrap_probe, bootstrap_environment, discovery_diagnostics, complete = _discover(
        normalized,
        environment,
        analysis_context=analysis_context,
        probe_timeout_s=probe_timeout_s,
    )
    resolution = annotations.resolve_fragments(fragments, source="dryml.dispatch")
    diagnostics = list(discovery_diagnostics)
    diagnostics.extend(_annotation_diagnostics(resolution))

    env_selection, env_spec = _select_environment(environment, resolution.environment_default)
    world_selection, world_spec = _select_world(world, resolution.world_default)
    runtime_selection, selected_runtime = _select_runtime(runtime_spec, resolution.runtime_default)
    structural_safe = True
    if normalized.launch.get("same_environment_only") and not _same_python_environment(env_spec):
        structural_safe = False
        diagnostics.append(_diagnostic(
            "dryml.dispatch.pickle_environment_restriction",
            "Pickled callable transport requires the current Python executable.",
            data={"candidate": env_spec, "restriction": "same_environment_only"},
        ))
    if single_worker_only and _is_multi_worker_world(world_spec):
        structural_safe = False
        diagnostics.append(_diagnostic(
            "dryml.dispatch.single_subprocess_world_unsupported",
            "The local subprocess planner supports one worker only; use plan_world() or run_world() for this world.",
            data={"world": world_spec},
        ))

    final_probe = None
    if _needs_final_probe(normalized, env_spec, bootstrap_environment):
        final_probe = probe_target(
            normalized.code_target,
            environment=spec_from_data(env_spec),
            include_environment_record=True,
            timeout=probe_timeout_s,
        )
        diagnostics.extend(final_probe.diagnostics)
        if not final_probe.ok:
            diagnostics.append(_diagnostic(
                "dryml.dispatch.final_environment_probe_failed",
                "Final selected environment could not validate the dispatch target.",
                severity="error",
                data={"environment": env_spec, "timeout_s": probe_timeout_s},
            ))

    env_record, env_probe_diagnostics = _environment_record(
        env_spec,
        resolution.environment_requirement,
        final_probe or bootstrap_probe,
        bootstrap_environment,
        environment,
        policy,
        validate_candidate=_requires_environment_validation(env_spec, normalized),
    )
    diagnostics.extend(env_probe_diagnostics)
    environment_check = _check_environment(
        resolution.environment_requirement,
        env_spec,
        env_record,
        policy,
        env_probe_diagnostics,
        validate_candidate=_requires_environment_validation(env_spec, normalized),
    )
    world_check = _check_world(resolution.world_requirement, world_spec, policy)
    runtime_check = _check_runtime(resolution.runtime_requirement, selected_runtime, policy)
    diagnostics.extend(_check_diagnostics((environment_check, world_check, runtime_check), policy))

    if not complete:
        diagnostics.append(_diagnostic("dryml.dispatch.discovery_incomplete", "Requirement discovery is incomplete.", severity="error" if policy is RequirementPolicy.STRICT else "warning", data={"policy": policy.value, "action": "use an importable target or call dispatch.explain(...)"}))
    checks = (environment_check, world_check, runtime_check)
    merge_safe = not _has_annotation_errors(resolution)
    if policy is RequirementPolicy.STRICT:
        launchable = structural_safe and merge_safe and complete and (final_probe is None or final_probe.ok) and all(report.compatible is not False and report.status != "error" for report in checks)
    else:
        launchable = structural_safe and merge_safe and (final_probe is None or final_probe.ok)
    if emit_warnings and policy is RequirementPolicy.WARN:
        warning_items = [item for item in diagnostics if item.severity in {"warning", "error"}]
        if warning_items:
            warnings.warn("; ".join(item.message for item in warning_items), RuntimeWarning, stacklevel=3)

    return DispatchPlanningResolution(
        normalized_target=_normalized_target_data(normalized),
        code_analysis=analysis,
        code_probe=final_probe or bootstrap_probe,
        bootstrap_environment=bootstrap_environment,
        bootstrap_code_probe=bootstrap_probe,
        final_code_probe=final_probe,
        requirements=resolution,
        environment_selection=env_selection,
        environment_record=env_record,
        environment_check=environment_check,
        world_selection=world_selection,
        world_check=world_check,
        runtime_selection=runtime_selection,
        runtime_check=runtime_check,
        requirement_policy=policy,
        runtime_enforcement=enforcement,
        diagnostics=tuple(diagnostics),
        launchable=launchable,
    )


def explanation_for(normalized: NormalizedDispatchTarget, **kwargs: Any) -> DispatchExplanation:
    """Resolve a normalized target without launching or emitting warnings."""

    result = resolve_dispatch_plan(normalized, emit_warnings=False, **kwargs)
    blocking = tuple(item for item in result.diagnostics if item.severity == "error")
    return DispatchExplanation(result, dict(normalized.operation_spec), blocking)


def _discover(
    normalized: NormalizedDispatchTarget,
    explicit_environment: Any | None,
    *,
    analysis_context: CodeAnalysisContext,
    probe_timeout_s: float,
):
    diagnostics: list[DiagnosticFact] = []
    analysis: CodeAnalysisResult | None = None
    probe: CodeProbeResult | None = None
    bootstrap_environment: Mapping[str, Any] | None = None
    fragments = []
    complete = False
    if normalized.definition_target is not None and normalized.method_name:
        try:
            fragments.extend(annotations.fragments_for_definition_method(normalized.definition_target, normalized.method_name))
            complete = True
        except Exception as exc:
            diagnostics.append(_diagnostic("dryml.dispatch.definition_method_annotation_collection_failed", "Definition-method annotation collection failed.", data={"error": type(exc).__name__, "method": normalized.method_name}))
    if not complete and normalized.method_name and normalized.subject_class is not None:
        try:
            fragments.extend(annotations.fragments_for_method(normalized.subject_class, normalized.method_name))
            complete = True
        except Exception as exc:
            diagnostics.append(_diagnostic("dryml.dispatch.method_annotation_collection_failed", "Method annotation collection failed.", data={"error": type(exc).__name__, "method": normalized.method_name}))
    elif not complete and normalized.live_annotation_targets:
        for target in normalized.live_annotation_targets:
            try:
                fragments.extend(annotations.fragments_for(target))
                complete = True
            except Exception as exc:
                diagnostics.append(_diagnostic("dryml.dispatch.annotation_collection_failed", "Live annotation collection failed.", data={"error": type(exc).__name__}))
    if normalized.code_target is not None:
        try:
            analysis = analyze(normalized.code_target, context=analysis_context)
            fragments.extend(_fragments_from_analysis(analysis))
            if not complete:
                complete = _analysis_is_complete(analysis)
            diagnostics.extend(analysis.diagnostics)
        except Exception as exc:
            diagnostics.append(_diagnostic("dryml.dispatch.code_analysis_failed", "Local code analysis failed.", data={"error": type(exc).__name__}))
    if not complete and normalized.code_target is not None:
        bootstrap = _bootstrap_environment(explicit_environment)
        bootstrap_environment = bootstrap.to_data()
        probe = probe_target(normalized.code_target, environment=bootstrap, include_environment_record=True, timeout=probe_timeout_s)
        diagnostics.extend(probe.diagnostics)
        if probe.analysis is not None:
            fragments.extend(_fragments_from_analysis(probe.analysis))
        complete = probe.ok and probe.analysis is not None
    return _dedupe_fragments(fragments), analysis, probe, bootstrap_environment, tuple(diagnostics), complete


def _analysis_options(policy: Any | None) -> tuple[CodeAnalysisContext, float]:
    """Validate bounded dispatch analysis/probe options before discovery starts."""

    if policy is None:
        return CodeAnalysisContext(), DEFAULT_PROBE_TIMEOUT_S
    if isinstance(policy, CodeAnalysisContext):
        return policy, DEFAULT_PROBE_TIMEOUT_S
    if not isinstance(policy, Mapping):
        raise DispatchPlanningError("analysis_policy must be a CodeAnalysisContext or mapping")
    unknown = set(policy) - {"context", "probe_timeout_s"}
    if unknown:
        raise DispatchPlanningError("analysis_policy contains unsupported fields", context={"fields": sorted(unknown)})
    context = policy.get("context")
    if context is None:
        context = CodeAnalysisContext()
    if not isinstance(context, CodeAnalysisContext):
        raise DispatchPlanningError("analysis_policy.context must be a CodeAnalysisContext")
    timeout = policy.get("probe_timeout_s", DEFAULT_PROBE_TIMEOUT_S)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise DispatchPlanningError("analysis_policy.probe_timeout_s must be a positive number")
    return context, float(timeout)


def _analysis_is_complete(analysis: CodeAnalysisResult) -> bool:
    if any(item.severity == "error" for item in analysis.diagnostics):
        return False
    incomplete_codes = {"dryml.code.algorithm_not_applicable", "dryml.code.annotations_unsupported_target"}
    return not any(item.code in incomplete_codes for item in analysis.diagnostics)


def _fragments_from_analysis(analysis: CodeAnalysisResult) -> list[annotations.AnnotationFragment]:
    result = []
    for fact in analysis.facts:
        if isinstance(fact, AnnotationFact):
            try:
                result.append(annotations.AnnotationFragment.from_data(fact.data))
            except Exception:
                continue
    return result


def _dedupe_fragments(fragments):
    result = []
    seen = set()
    for fragment in fragments:
        key = json.dumps(fragment.to_data(), sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            result.append(fragment)
    return tuple(result)


def _select_environment(explicit: Any | None, annotation_default: Any | None):
    return _select("environment", (("explicit", explicit), ("annotation_default", annotation_default), ("current", environments.current(default=None)), ("fallback", environments.CurrentEnvironmentSpec())), _environment_data)


def _select_world(explicit: Any | None, annotation_default: Any | None):
    fallback = {"roles": {"main": {"replicas": 1, "process": {}}}, "backend": {"kind": "local", "parameters": {}}}
    return _select("world", (("explicit", explicit), ("annotation_default", annotation_default), ("current", worlds.current(default=None)), ("fallback", fallback)), _world_data)


def _select_runtime(explicit: Any | None, annotation_default: Any | None):
    fallback = RuntimeContextSpec(mode=RuntimeMode.WORKER, device_visibility={"policy": "assigned"}, metadata={"source": "dryml.dispatch"})
    return _select("runtime", (("explicit", explicit), ("annotation_default", annotation_default), ("fallback", fallback)), _runtime_data)


def _select(kind: str, choices, serializer):
    considered = []
    selected = None
    source = None
    for slot, value in choices:
        if value is None:
            considered.append(CandidateConsideration(slot, "absent"))
            continue
        try:
            data = serializer(value)
        except Exception as exc:
            raise DispatchPlanningError(f"invalid {kind} candidate", context={"source": slot, "error": str(exc)}) from exc
        if selected is None:
            selected, source = data, slot
            considered.append(CandidateConsideration(slot, "selected", data))
        else:
            considered.append(CandidateConsideration(slot, "not_selected", data))
    assert selected is not None and source is not None
    return CandidateSelection(kind, selected, source, tuple(considered)), selected


def _environment_data(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and "spec" in value:
        value = value["spec"]
    if isinstance(value, str):
        return PythonExecutableSpec(value).to_data()
    if isinstance(value, Mapping):
        return spec_from_data(value).to_data()
    if isinstance(value, EnvironmentSpec):
        return value.to_data()
    if hasattr(value, "to_data"):
        return value.to_data()
    raise TypeError(type(value).__name__)


def _world_data(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and "spec" in value:
        value = value["spec"]
    if isinstance(value, WorldSpec):
        return value.to_data()
    # Sprint 6 accepted opaque local-backend world policy mappings. Preserve
    # them for no-requirement compatibility; a hard requirement will report
    # their non-checkable shape instead of inventing a replacement world.
    if isinstance(value, Mapping) and "roles" not in value:
        return dict(value)
    return WorldSpec.from_data(value).to_data()


def _runtime_data(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and "spec" in value:
        value = value["spec"]
    data = value.to_data() if isinstance(value, RuntimeContextSpec) else dict(value)
    # A dispatch worker always receives a worker runtime unless explicitly set.
    data.setdefault("mode", RuntimeMode.WORKER.value)
    if data.get("mode") == RuntimeMode.ORCHESTRATOR.value:
        data["mode"] = RuntimeMode.WORKER.value
    return RuntimeContextSpec.from_data(data).to_data()


def _bootstrap_environment(explicit: Any | None) -> EnvironmentSpec:
    if explicit is not None:
        return spec_from_data(_environment_data(explicit))
    current = environments.current(default=None)
    return spec_from_data(_environment_data(current)) if current is not None else environments.CurrentEnvironmentSpec()


def _same_python_environment(env_data: Mapping[str, Any]) -> bool:
    if env_data.get("kind") == "current":
        return True
    if env_data.get("kind") == "python":
        return os.path.abspath(os.fspath(env_data.get("executable", ""))) == os.path.abspath(sys.executable)
    return False


def _needs_final_probe(normalized: NormalizedDispatchTarget, final_environment: Mapping[str, Any], bootstrap_environment: Mapping[str, Any] | None) -> bool:
    if normalized.code_target is None or normalized.code_target.kind != "import_path" or normalized.transport == "pickle_small":
        return False
    # Local discovery proves only the orchestrator environment. A non-current
    # selected candidate must prove it can import/analyze the serialized target.
    return bootstrap_environment != final_environment and final_environment.get("kind") != "current"


def _requires_environment_validation(env_data: Mapping[str, Any], normalized: NormalizedDispatchTarget) -> bool:
    del normalized
    return env_data.get("kind") != "current"


def _is_multi_worker_world(candidate: Mapping[str, Any]) -> bool:
    try:
        world = WorldSpec.from_data(candidate)
    except Exception:
        return False
    return len(world.roles) > 1 or sum(role.replicas for role in world.roles.values()) > 1


def _environment_record(env_data, requirement, code_probe, bootstrap_environment, explicit_environment, policy, *, validate_candidate: bool):
    if requirement is None and not validate_candidate:
        return None, ()
    spec = spec_from_data(env_data)
    attached = _attached_environment_record(explicit_environment, env_data)
    if attached is not None:
        return attached, ()
    if code_probe is not None and code_probe.environment_record is not None and bootstrap_environment == env_data:
        return code_probe.environment_record, ()
    probe = environments.probe(spec)
    if probe.ok and probe.record is not None:
        return probe.record, ()
    return None, (_diagnostic("dryml.dispatch.environment_probe_failed", "Environment probe failed for the selected candidate.", severity="error", data={"candidate": env_data, "probe": _bounded_probe_data(probe.to_data())}),)


def _attached_environment_record(candidate, selected_data):
    if not isinstance(candidate, Mapping) or "record" not in candidate:
        return None
    spec_value = candidate.get("spec", candidate)
    try:
        if _environment_data(spec_value) != selected_data:
            return None
        record = candidate["record"]
        return record if isinstance(record, EnvironmentRecord) else EnvironmentRecord.from_data(record)
    except Exception:
        return None


def _check_environment(requirement, candidate, record, policy, probe_diagnostics, *, validate_candidate: bool):
    requirement_data = None if requirement is None else requirement.to_data()
    if requirement is None and record is None and not validate_candidate:
        return CandidateCheckReport("environment", "not_required", None, None, candidate)
    if requirement is None and record is None:
        return CandidateCheckReport("environment", "error", False, None, candidate, ({"reason": "environment_record_unavailable"},), probe_diagnostics)
    if requirement is None:
        return CandidateCheckReport("environment", "satisfied", True, None, candidate)
    if policy is RequirementPolicy.IGNORE:
        return CandidateCheckReport("environment", "skipped", None, requirement_data, candidate, ({"reason": "requirement_policy_ignore"},))
    if record is None:
        return CandidateCheckReport("environment", "error", False, requirement_data, candidate, ({"reason": "environment_record_unavailable"},), probe_diagnostics)
    report = requirement.check(record, policy="strict" if policy is RequirementPolicy.STRICT else "warn")
    details = tuple(report.to_data().get("issues") or ())
    return CandidateCheckReport("environment", "satisfied" if report.ok else "incompatible", report.ok, requirement_data, candidate, details)


def _check_world(requirement, candidate, policy):
    requirement_data = None if requirement is None else requirement.to_data()
    if requirement is None:
        return CandidateCheckReport("world", "not_required", None, None, candidate)
    if policy is RequirementPolicy.IGNORE:
        return CandidateCheckReport("world", "skipped", None, requirement_data, candidate, ({"reason": "requirement_policy_ignore"},))
    try:
        world = WorldSpec.from_data(candidate)
    except Exception as exc:
        detail = {"path": "world", "message": "selected world cannot be checked against a hard requirement", "expected": requirement_data, "actual": candidate, "error": type(exc).__name__}
        return CandidateCheckReport("world", "error", False, requirement_data, candidate, (detail,))
    report = worlds.check_world_spec_satisfies_requirement(world, requirement)
    details = tuple({"severity": issue.severity, "path": issue.path, "message": issue.message, "expected": issue.expected, "actual": issue.actual, "source": issue.source} for issue in report.issues)
    return CandidateCheckReport("world", "satisfied" if report.ok else "incompatible", report.ok, requirement_data, candidate, details)


def _check_runtime(requirement, candidate, policy):
    if requirement is None:
        return CandidateCheckReport("runtime", "not_required", None, None, candidate)
    if policy is RequirementPolicy.IGNORE:
        return CandidateCheckReport("runtime", "skipped", None, dict(requirement), candidate, ({"reason": "requirement_policy_ignore"},))
    report = runtime.check_runtime_spec_satisfies_requirement(candidate, requirement)
    details = tuple(item.to_data() for item in report.issues)
    return CandidateCheckReport("runtime", "satisfied" if report.ok else "incompatible", report.ok, dict(requirement), candidate, details)


def _annotation_diagnostics(resolution):
    return tuple(_diagnostic(item.code, item.message, severity=item.level, data=item.data) for item in resolution.diagnostics)


def _has_annotation_errors(resolution) -> bool:
    return any(item.level == "error" for item in resolution.diagnostics)


def _check_diagnostics(reports, policy):
    diagnostics = []
    for report in reports:
        if report.status in {"not_required", "skipped", "satisfied"}:
            continue
        severity = "error" if policy is RequirementPolicy.STRICT else "warning"
        details = report.details or ({"reason": report.status},)
        diagnostics.append(_diagnostic(
            f"dryml.dispatch.{report.kind}_check_{report.status}",
            f"Selected {report.kind} candidate does not satisfy dispatch requirements.",
            severity=severity,
            data={"check": report.to_data(), "details": details, "policy": policy.value},
        ))
    return diagnostics


def _diagnostic(code, message, *, severity="error", data=None):
    return DiagnosticFact(severity=severity, code=code, message=message, source={"component": "dryml.dispatch.requirements"}, data=data or {})


def _normalized_target_data(normalized):
    return {
        "operation_id": normalized.operation_spec.get("id"),
        "operation_kind": normalized.operation_spec.get("kind"),
        "code_target": None if normalized.code_target is None else normalized.code_target.to_data(),
        "method_name": normalized.method_name,
        "transport": normalized.transport,
        "restrictions": list(normalized.launch.get("transport_restrictions") or ()),
    }


def _bounded_probe_data(data):
    return _bounded_data(data)


def _probe_summary(probe):
    return None if probe is None else _bounded_probe_data(probe.to_data())


def _analysis_summary(analysis):
    return None if analysis is None else _bounded_data(analysis.to_data())


def _environment_record_summary(record):
    if record is None:
        return None
    return {
        "id": record.id,
        "python": {"implementation": record.python.implementation, "version": record.python.version},
        "platform": {"system": record.platform.system, "machine": record.platform.machine},
    }


def _bounded_data(value):
    if isinstance(value, str):
        return value[:4096]
    if isinstance(value, Mapping):
        return {str(key): _bounded_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded_data(item) for item in value]
    return value


__all__ = [
    "PLANNING_METADATA_VERSION",
    "CandidateCheckReport",
    "CandidateConsideration",
    "CandidateSelection",
    "DispatchExplanation",
    "DispatchPlanningResolution",
    "RequirementPolicy",
    "effective_requirement_policy",
    "explanation_for",
    "resolve_dispatch_plan",
]
