"""Merge and resolution APIs for DRYML annotation fragments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from dryml.environments.requirements import EnvironmentRequirement
from dryml.environments.utils import normalize_distribution_name
from dryml.runtime.specs import RuntimeContextSpec
from dryml.worlds.compatibility import check_world_spec_satisfies_requirement
from dryml.worlds.specs import WorldRequirement, WorldSpec
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from .collect import collect_fragments
from .model import AnnotationFragment, SourceTrace
from .namespaces import ENVIRONMENT, RUNTIME, WORLD
from .report import AnnotationIssue, AnnotationReport

_DELETE = object()


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
    runtime_req = _merge_mapping_fragments(_namespace_fragments(fragments, RUNTIME), issues=issues, namespace=RUNTIME) or None
    return ResolvedRequirements(env_req, world_req, runtime_req, fragments, AnnotationReport(tuple(issues)))


def resolve_defaults(target: Any, *, provider_fragments: Iterable[AnnotationFragment] = (), overrides: Mapping[str, Any] | None = None) -> ResolvedDefaults:
    """Collect and merge default fragments for *target*, then apply overrides."""

    fragments = collect_fragments((target,), provider_fragments=provider_fragments, kind="default")
    issues: list[AnnotationIssue] = []
    overrides = overrides or {}
    world_data = _merge_mapping_fragments(_namespace_fragments(fragments, WORLD), issues=issues, namespace=WORLD)
    if WORLD in overrides:
        world_data = _deep_merge(world_data, dict(overrides[WORLD]), empty_mapping_replaces=True)
    runtime_data = _merge_mapping_fragments(_namespace_fragments(fragments, RUNTIME), issues=issues, namespace=RUNTIME)
    if RUNTIME in overrides:
        runtime_data = _deep_merge(runtime_data, dict(overrides[RUNTIME]), empty_mapping_replaces=True)
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
    saw_payload = False
    for fragment in fragments:
        try:
            payload = EnvironmentRequirement.from_data(fragment.fragment).to_data()
        except Exception as exc:
            issues.append(AnnotationIssue("error", ENVIRONMENT, "/", str(exc), sources=(fragment.source,)))
            continue
        saw_payload = True
        mode = fragment.merge_policy or fragment.source.metadata.get("legacy_environment_fragment_mode") or "add"
        if mode == "override":
            for key in ("requirements", "excludes", "capabilities", "tags"):
                if payload.get(key):
                    data[key] = list(payload[key])
            for key in ("python", "dryml_protocol"):
                if payload.get(key) is not None:
                    data[key] = payload[key]
            if payload.get("schema_versions"):
                data["schema_versions"].update(payload["schema_versions"])
            continue
        for key in ("requirements", "excludes", "capabilities", "tags"):
            data[key].extend(payload.get(key, ()))
        for key in ("python", "dryml_protocol"):
            value = payload.get(key)
            if value is not None and mode != "base" and data.get(key) not in (None, value):
                issues.append(AnnotationIssue("error", ENVIRONMENT, f"/{key}", "conflicting environment requirement", data.get(key), value, _sources(fragments)))
            if value is not None:
                data[key] = value
        for key, value in payload.get("schema_versions", {}).items():
            if mode != "base" and key in data["schema_versions"] and data["schema_versions"][key] != value:
                issues.append(AnnotationIssue("error", ENVIRONMENT, f"/schema_versions/{key}", "conflicting schema requirement", data["schema_versions"][key], value, _sources(fragments)))
            data["schema_versions"][key] = value
    if not saw_payload:
        return None
    issues.extend(_package_conflict_issues(tuple(data["requirements"]), _sources(fragments)))
    try:
        return EnvironmentRequirement(**data)
    except Exception as exc:
        issues.append(AnnotationIssue("error", ENVIRONMENT, "/", str(exc), sources=_sources(fragments)))
        return None


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


def _merge_mapping_fragments(fragments: tuple[AnnotationFragment, ...], *, issues: list[AnnotationIssue] | None = None, namespace: str = "") -> dict[str, Any]:
    data: dict[str, Any] = {}
    for fragment in sorted(fragments, key=lambda f: f.priority):
        policy = fragment.merge_policy or "merge"
        if policy in {"merge", "add", "base"}:
            data = _deep_merge(data, dict(fragment.fragment))
        elif policy in {"replace", "override"}:
            data = dict(fragment.fragment)
        elif policy == "append":
            data = _append_merge(data, dict(fragment.fragment))
        elif policy == "error_on_conflict":
            conflicts = _conflict_issues(data, fragment.fragment, namespace=namespace, sources=(fragment.source,))
            if issues is not None:
                issues.extend(conflicts)
            if not conflicts:
                data = _deep_merge(data, dict(fragment.fragment))
        else:
            if issues is not None:
                issues.append(AnnotationIssue("error", namespace or fragment.namespace, "/merge_policy", "unknown annotation merge policy", expected="merge|replace|append|error_on_conflict", actual=policy, sources=(fragment.source,)))
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


def _deep_merge(left: Mapping[str, Any], right: Mapping[str, Any], *, empty_mapping_replaces: bool = False) -> dict[str, Any]:
    if "$replace" in right:
        replacement = right["$replace"]
        return dict(replacement) if isinstance(replacement, Mapping) else replacement
    result = dict(left)
    for key, value in right.items():
        if key == "$delete":
            continue
        merged = _merge_value(result.get(key), value, empty_mapping_replaces=empty_mapping_replaces)
        if merged is _DELETE:
            result.pop(key, None)
        else:
            result[key] = merged
    return result


def _merge_value(left: Any, right: Any, *, empty_mapping_replaces: bool) -> Any:
    if isinstance(right, Mapping):
        if right.get("$delete") is True:
            return _DELETE
        if "$replace" in right:
            return right["$replace"]
        if isinstance(left, Mapping) and (right or not empty_mapping_replaces):
            return _deep_merge(left, right, empty_mapping_replaces=empty_mapping_replaces)
        return dict(right)
    return right


def _append_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        current = result.get(key)
        if isinstance(current, list) and isinstance(value, list):
            result[key] = current + value
        elif isinstance(current, tuple) and isinstance(value, tuple):
            result[key] = current + value
        elif isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _append_merge(current, value)
        else:
            result[key] = value
    return result


def _conflict_issues(left: Mapping[str, Any], right: Mapping[str, Any], *, namespace: str, sources: tuple[SourceTrace, ...], path: str = "") -> list[AnnotationIssue]:
    issues: list[AnnotationIssue] = []
    for key, value in right.items():
        child_path = f"{path}/{key}"
        if key not in left:
            continue
        current = left[key]
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            issues.extend(_conflict_issues(current, value, namespace=namespace, sources=sources, path=child_path))
        elif current != value:
            issues.append(AnnotationIssue("error", namespace, child_path, "annotation merge conflict", expected=current, actual=value, sources=sources))
    return issues


def _package_conflict_issues(requirements: tuple[str, ...], sources: tuple[SourceTrace, ...]) -> list[AnnotationIssue]:
    grouped: dict[str, list[Requirement]] = {}
    for text in requirements:
        try:
            req = Requirement(text)
        except Exception as exc:
            return [AnnotationIssue("error", ENVIRONMENT, "/requirements", str(exc), actual=text, sources=sources)]
        if req.marker is not None or req.url is not None:
            continue
        grouped.setdefault(normalize_distribution_name(req.name), []).append(req)
    issues: list[AnnotationIssue] = []
    for name, reqs in grouped.items():
        spec_texts = [str(req.specifier) for req in reqs if str(req.specifier)]
        if len(spec_texts) < 2:
            continue
        combined = SpecifierSet(",".join(spec_texts))
        if not _specifier_has_candidate(combined):
            issues.append(AnnotationIssue("error", ENVIRONMENT, f"/requirements/{name}", "conflicting package requirement specifiers", expected="satisfiable specifier set", actual=", ".join(spec_texts), sources=sources))
    return issues


def _specifier_has_candidate(specifier: SpecifierSet) -> bool:
    candidates = {"0", "0.0.1", "0.1", "0.9", "1", "1.0", "1.5", "2", "2.0", "3", "10", "100"}
    for spec in specifier:
        candidates.add(spec.version)
        try:
            version = Version(spec.version)
        except InvalidVersion:
            continue
        release = list(version.release)
        if release:
            bumped = release[:]
            bumped[-1] += 1
            candidates.add(".".join(str(part) for part in bumped))
            candidates.add(str(release[0] + 1))
            if release[-1] > 0:
                lowered = release[:]
                lowered[-1] -= 1
                candidates.add(".".join(str(part) for part in lowered))
    for candidate in candidates:
        try:
            if Version(candidate) in specifier:
                return True
        except InvalidVersion:
            continue
    return False


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
