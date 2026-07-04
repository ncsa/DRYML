"""Merge and resolution APIs for DRYML annotation fragments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from dryml.environments.requirements import EnvironmentRequirement
from dryml.runtime.specs import RuntimeContextSpec
from dryml.worlds.compatibility import check_world_spec_satisfies_requirement
from dryml.worlds.specs import WorldRequirement, WorldSpec

from .collect import collect_fragments
from .model import AnnotationFragment, SourceTrace
from .namespaces import ENVIRONMENT, RUNTIME, WORLD
from .report import AnnotationIssue, AnnotationReport


@dataclass(frozen=True, slots=True)
class ResolvedRequirements:
    """Resolved hard planning requirements by namespace."""

    environment: EnvironmentRequirement | None = None
    world: WorldRequirement | None = None
    runtime: Mapping[str, Any] | None = None
    fragments: tuple[AnnotationFragment, ...] = ()
    report: AnnotationReport = AnnotationReport()


@dataclass(frozen=True, slots=True)
class ResolvedDefaults:
    """Resolved overrideable defaults by namespace."""

    world: WorldSpec | None = None
    runtime: RuntimeContextSpec | None = None
    fragments: tuple[AnnotationFragment, ...] = ()
    report: AnnotationReport = AnnotationReport()


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Complete annotation resolution result."""

    requirements: ResolvedRequirements
    defaults: ResolvedDefaults
    report: AnnotationReport


def resolve_requirements(target: Any, *, provider_fragments: Iterable[AnnotationFragment] = ()) -> ResolvedRequirements:
    """Collect and merge hard requirement fragments for *target*."""

    fragments = collect_fragments((target,), provider_fragments=provider_fragments, kind="requirement")
    issues: list[AnnotationIssue] = []
    env_req = _merge_environment_requirements(_namespace_fragments(fragments, ENVIRONMENT), issues)
    world_req = _merge_world_requirements(_namespace_fragments(fragments, WORLD), issues)
    runtime_req = _merge_mapping_fragments(_namespace_fragments(fragments, RUNTIME)) or None
    return ResolvedRequirements(env_req, world_req, runtime_req, fragments, AnnotationReport(tuple(issues)))


def resolve_defaults(target: Any, *, provider_fragments: Iterable[AnnotationFragment] = (), overrides: Mapping[str, Any] | None = None) -> ResolvedDefaults:
    """Collect and merge default fragments for *target*, then apply overrides."""

    fragments = collect_fragments((target,), provider_fragments=provider_fragments, kind="default")
    issues: list[AnnotationIssue] = []
    overrides = overrides or {}
    world_data = _merge_mapping_fragments(_namespace_fragments(fragments, WORLD))
    if WORLD in overrides:
        world_data = _deep_merge(world_data, dict(overrides[WORLD]))
    runtime_data = _merge_mapping_fragments(_namespace_fragments(fragments, RUNTIME))
    if RUNTIME in overrides:
        runtime_data = _deep_merge(runtime_data, dict(overrides[RUNTIME]))
    world = _world_spec_or_issue(world_data, issues, _sources(_namespace_fragments(fragments, WORLD))) if world_data else None
    runtime = _runtime_spec_or_issue(runtime_data, issues, _sources(_namespace_fragments(fragments, RUNTIME))) if runtime_data else None
    return ResolvedDefaults(world, runtime, fragments, AnnotationReport(tuple(issues)))


def resolve(target: Any, *, provider_fragments: Iterable[AnnotationFragment] = (), overrides: Mapping[str, Any] | None = None) -> ResolutionResult:
    """Resolve requirements and defaults, validating defaults against hard requirements."""

    provider_fragments = tuple(provider_fragments)
    requirements = resolve_requirements(target, provider_fragments=provider_fragments)
    defaults = resolve_defaults(target, provider_fragments=provider_fragments, overrides=overrides)
    issues = list(requirements.report.issues) + list(defaults.report.issues)
    if requirements.world is not None and defaults.world is not None:
        report = check_world_spec_satisfies_requirement(defaults.world, requirements.world)
        if not report.ok:
            sources = _sources(_namespace_fragments(requirements.fragments + defaults.fragments, WORLD)) + (_override_source(WORLD),)
            for issue in report.issues:
                issues.append(AnnotationIssue(issue.severity, WORLD, issue.path, issue.message, issue.expected, issue.actual, sources))
    combined = AnnotationReport(tuple(issues))
    return ResolutionResult(requirements, defaults, combined)


def resolve_environment_requirement(target: Any, *, provider_fragments: Iterable[AnnotationFragment] = ()) -> EnvironmentRequirement:
    """Resolve environment requirements for *target*."""

    return resolve_requirements(target, provider_fragments=provider_fragments).environment or EnvironmentRequirement()


def resolve_world_requirement(target: Any, *, provider_fragments: Iterable[AnnotationFragment] = ()) -> WorldRequirement | None:
    """Resolve hard world requirements for *target*."""

    return resolve_requirements(target, provider_fragments=provider_fragments).world


def resolve_world_default(target: Any, *, overrides: Mapping[str, Any] | None = None) -> WorldSpec | None:
    """Resolve world defaults for *target*."""

    if overrides is not None and WORLD not in overrides:
        overrides = {WORLD: overrides}
    return resolve_defaults(target, overrides=overrides).world


def resolve_runtime_default(target: Any, *, overrides: Mapping[str, Any] | None = None) -> RuntimeContextSpec | None:
    """Resolve runtime defaults for *target*."""

    if overrides is not None and RUNTIME not in overrides:
        overrides = {RUNTIME: overrides}
    return resolve_defaults(target, overrides=overrides).runtime


def _merge_environment_requirements(fragments: tuple[AnnotationFragment, ...], issues: list[AnnotationIssue]) -> EnvironmentRequirement | None:
    if not fragments:
        return None
    data: dict[str, Any] = {"requirements": [], "excludes": [], "capabilities": [], "tags": [], "schema_versions": {}, "details": {"sources": [f.source.to_data() for f in fragments]}}
    for fragment in fragments:
        payload = EnvironmentRequirement.from_data(fragment.fragment).to_data()
        for key in ("requirements", "excludes", "capabilities", "tags"):
            data[key].extend(payload.get(key, ()))
        for key in ("python", "dryml_protocol"):
            value = payload.get(key)
            if value is not None and data.get(key) not in (None, value):
                issues.append(AnnotationIssue("error", ENVIRONMENT, f"/{key}", "conflicting environment requirement", data.get(key), value, _sources(fragments)))
            if value is not None:
                data[key] = value
        for key, value in payload.get("schema_versions", {}).items():
            if key in data["schema_versions"] and data["schema_versions"][key] != value:
                issues.append(AnnotationIssue("error", ENVIRONMENT, f"/schema_versions/{key}", "conflicting schema requirement", data["schema_versions"][key], value, _sources(fragments)))
            data["schema_versions"][key] = value
    return EnvironmentRequirement(**data)


def _merge_world_requirements(fragments: tuple[AnnotationFragment, ...], issues: list[AnnotationIssue]) -> WorldRequirement | None:
    req: WorldRequirement | None = None
    for fragment in fragments:
        try:
            current = WorldRequirement.from_data(fragment.fragment)
            req = current if req is None else req.merge(current)
        except Exception as exc:
            path = getattr(exc, "context", {}).get("path", "/")
            issues.append(AnnotationIssue("error", WORLD, f"/{str(path).replace('.', '/')}", str(exc), sources=(fragment.source,)))
    return req


def _merge_mapping_fragments(fragments: tuple[AnnotationFragment, ...]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for fragment in sorted(fragments, key=lambda f: f.priority):
        data = _deep_merge(data, dict(fragment.fragment))
    return data


def _world_spec_or_issue(data: Mapping[str, Any], issues: list[AnnotationIssue], sources: tuple[SourceTrace, ...]) -> WorldSpec | None:
    try:
        return WorldSpec.from_data(data)
    except Exception as exc:
        issues.append(AnnotationIssue("error", WORLD, "/", str(exc), sources=sources))
        return None


def _runtime_spec_or_issue(data: Mapping[str, Any], issues: list[AnnotationIssue], sources: tuple[SourceTrace, ...]) -> RuntimeContextSpec | None:
    try:
        return RuntimeContextSpec.from_data(data)
    except Exception as exc:
        issues.append(AnnotationIssue("error", RUNTIME, "/", str(exc), sources=sources))
        return None


def _deep_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _namespace_fragments(fragments: Iterable[AnnotationFragment], namespace: str) -> tuple[AnnotationFragment, ...]:
    return tuple(fragment for fragment in fragments if fragment.namespace == namespace)


def _sources(fragments: Iterable[AnnotationFragment]) -> tuple[SourceTrace, ...]:
    return tuple(fragment.source for fragment in fragments)


def _override_source(namespace: str) -> SourceTrace:
    return SourceTrace("override", namespace=namespace, label="user override")


__all__ = [
    "ResolvedDefaults",
    "ResolvedRequirements",
    "ResolutionResult",
    "resolve",
    "resolve_defaults",
    "resolve_environment_requirement",
    "resolve_requirements",
    "resolve_runtime_default",
    "resolve_world_default",
    "resolve_world_requirement",
]
