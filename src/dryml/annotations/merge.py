"""Merge and resolution APIs for DRYML annotation fragments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dryml.environments.requirements import EnvironmentRequirement
from dryml.environments.utils import normalize_distribution_name
from dryml.formats import json_ready
from dryml import reporting
from dryml.runtime.specs import RuntimeContextSpec
from dryml.worlds.compatibility import check_world_spec_satisfies_requirement
from dryml.worlds.specs import WorldRequirement, WorldSpec
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from .collect import collect_fragments, fragments_for_definition_method, fragments_for_method
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


@dataclass(frozen=True, slots=True)
class RequirementSourceTrace:
    """Small JSON-ready explanation of where one requirement fragment came from.

    Args:
        namespace: Annotation namespace such as ``"environment"`` or
            ``"world"``.
        kind: Fragment kind, usually ``"requirement"`` or ``"default"``.
        label: Human-readable source label.
        target_label: Optional label for the decorated target.
        module: Optional Python module for the decorated target.
        qualname: Optional Python qualified name for the decorated target.
        priority: Fragment priority metadata.
        merge_policy: Fragment merge-policy metadata.
        fragment_index: Stable index in the resolved fragment sequence.
        data: Extra JSON-compatible trace metadata.
    """

    namespace: str
    kind: str
    label: str
    target_label: str | None = None
    module: str | None = None
    qualname: str | None = None
    priority: int = 0
    merge_policy: str | None = None
    fragment_index: int | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", json_ready(self.data))

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-compatible trace mapping."""

        return {
            "namespace": self.namespace,
            "kind": self.kind,
            "label": self.label,
            "target_label": self.target_label,
            "module": self.module,
            "qualname": self.qualname,
            "priority": self.priority,
            "merge_policy": self.merge_policy,
            "fragment_index": self.fragment_index,
            "data": json_ready(self.data),
        }


@dataclass(frozen=True, slots=True)
class RequirementDiagnostic:
    """JSON-ready diagnostic emitted while resolving annotation requirements."""

    level: Literal["debug", "info", "warning", "error"]
    code: str
    message: str
    target_label: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", json_ready(self.data))

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic mapping."""

        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "target_label": self.target_label,
            "data": json_ready(self.data),
        }


@dataclass(frozen=True, slots=True)
class RequirementResolution:
    """Authoritative merged requirement/default result for annotation APIs.

    Args:
        environment_requirement: Merged hard environment requirement, if any.
        world_requirement: Merged hard world requirement, if any.
        runtime_requirement: Merged hard runtime requirement mapping, if any.
        environment_default: Reserved for future environment defaults.
        world_default: Merged default world specification, if any.
        runtime_default: Merged default runtime context, if any.
        fragments: Raw fragments used for this resolution, including provider
            fragments after target fragments.
        source_traces: JSON-ready source traces corresponding to fragments.
        diagnostics: JSON-ready resolution diagnostics.
        merge_report: Underlying annotation report containing merge issues.
    """

    environment_requirement: EnvironmentRequirement | None = None
    world_requirement: WorldRequirement | None = None
    runtime_requirement: Mapping[str, Any] | None = None
    environment_default: Any | None = None
    world_default: WorldSpec | None = None
    runtime_default: RuntimeContextSpec | None = None
    fragments: tuple[AnnotationFragment, ...] = ()
    source_traces: tuple[RequirementSourceTrace, ...] = ()
    diagnostics: tuple[RequirementDiagnostic, ...] = ()
    merge_report: AnnotationReport | None = None

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this resolution."""

        return {
            "environment_requirement": _to_data_or_none(self.environment_requirement),
            "world_requirement": _to_data_or_none(self.world_requirement),
            "runtime_requirement": json_ready(self.runtime_requirement) if self.runtime_requirement is not None else None,
            "environment_default": json_ready(self.environment_default) if self.environment_default is not None else None,
            "world_default": _to_data_or_none(self.world_default),
            "runtime_default": _to_data_or_none(self.runtime_default),
            "fragments": [fragment.to_data() for fragment in self.fragments],
            "source_traces": [trace.to_data() for trace in self.source_traces],
            "diagnostics": [diagnostic.to_data() for diagnostic in self.diagnostics],
            "merge_report": _report_to_data(self.merge_report),
        }


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
    world_sources = _sources(_namespace_fragments(fragments, WORLD))
    runtime_sources = _sources(_namespace_fragments(fragments, RUNTIME))
    world_data = _merge_mapping_fragments(_namespace_fragments(fragments, WORLD), issues=issues, namespace=WORLD)
    if WORLD in overrides:
        world_data = _deep_merge(world_data, dict(overrides[WORLD]), empty_mapping_replaces=True)
        world_sources = world_sources + (_override_source(WORLD),)
    runtime_data = _merge_mapping_fragments(_namespace_fragments(fragments, RUNTIME), issues=issues, namespace=RUNTIME)
    if RUNTIME in overrides:
        runtime_data = _deep_merge(runtime_data, dict(overrides[RUNTIME]), empty_mapping_replaces=True)
        runtime_sources = runtime_sources + (_override_source(RUNTIME),)
    world = _world_spec_or_issue(world_data, issues, world_sources) if world_data else None
    runtime = _runtime_spec_or_issue(runtime_data, issues, runtime_sources) if runtime_data else None
    return ResolvedDefaults(world, runtime, fragments, AnnotationReport(tuple(issues)))


def resolve(target: Any, *, provider_fragments: Iterable[AnnotationFragment] = (), overrides: Mapping[str, Any] | None = None) -> ResolutionResult:
    """Resolve requirements and defaults, validating defaults against hard requirements."""

    provider_fragments = tuple(provider_fragments)
    reporting.step(
        "dryml.annotations.resolve",
        "Gathering environment/world/runtime requirements",
        data={"provider_fragments": len(provider_fragments), "has_overrides": overrides is not None},
    )
    requirements = resolve_requirements(target, provider_fragments=provider_fragments)
    defaults = resolve_defaults(target, provider_fragments=provider_fragments, overrides=overrides)
    issues = list(requirements.report.issues) + list(defaults.report.issues)
    overrides = overrides or {}
    if requirements.world is not None and defaults.world is not None:
        report = check_world_spec_satisfies_requirement(defaults.world, requirements.world)
        if not report.ok:
            sources = _sources(_namespace_fragments(requirements.fragments + defaults.fragments, WORLD))
            if WORLD in overrides:
                sources = sources + (_override_source(WORLD),)
            for issue in report.issues:
                issues.append(AnnotationIssue(issue.severity, WORLD, issue.path, issue.message, issue.expected, issue.actual, sources))
    combined = AnnotationReport(tuple(issues))
    reporting.detail(
        "dryml.annotations.resolve.result",
        "Merged requirements and defaults",
        data={
            "requirement_fragments": len(requirements.fragments),
            "default_fragments": len(defaults.fragments),
            "issues": len(combined.issues),
            "has_environment_requirement": requirements.environment is not None,
            "has_world_requirement": requirements.world is not None,
            "has_world_default": defaults.world is not None,
            "has_runtime_default": defaults.runtime is not None,
        },
    )
    return ResolutionResult(requirements, defaults, combined)


def resolve_fragments(
    fragments: Iterable[AnnotationFragment],
    *,
    provider_fragments: Iterable[AnnotationFragment] = (),
    source: str | None = None,
) -> RequirementResolution:
    """Merge already-collected annotation fragments into one resolution.

    Args:
        fragments: Target fragments in collection order.
        provider_fragments: Additional provider fragments appended after target
            fragments.
        source: Optional caller label recorded in trace metadata.

    Returns:
        A :class:`RequirementResolution` containing merged hard requirements,
        defaults, raw fragments, source traces, diagnostics, and merge report.
    """

    ordered = tuple(fragments) + tuple(provider_fragments)
    issues: list[AnnotationIssue] = []
    requirement_fragments = tuple(fragment for fragment in ordered if fragment.kind == "requirement")
    default_fragments = tuple(fragment for fragment in ordered if fragment.kind == "default")

    environment_requirement = _merge_environment_requirements(_namespace_fragments(requirement_fragments, ENVIRONMENT), issues)
    world_requirement = _merge_world_requirements(_namespace_fragments(requirement_fragments, WORLD), issues)
    runtime_requirement = _merge_mapping_fragments(_namespace_fragments(requirement_fragments, RUNTIME), issues=issues, namespace=RUNTIME) or None

    world_default_data = _merge_mapping_fragments(_namespace_fragments(default_fragments, WORLD), issues=issues, namespace=WORLD)
    runtime_default_data = _merge_mapping_fragments(_namespace_fragments(default_fragments, RUNTIME), issues=issues, namespace=RUNTIME)
    world_default = _world_spec_or_issue(world_default_data, issues, _sources(_namespace_fragments(default_fragments, WORLD))) if world_default_data else None
    runtime_default = _runtime_spec_or_issue(runtime_default_data, issues, _sources(_namespace_fragments(default_fragments, RUNTIME))) if runtime_default_data else None

    if world_requirement is not None and world_default is not None:
        world_report = check_world_spec_satisfies_requirement(world_default, world_requirement)
        if not world_report.ok:
            sources = _sources(_namespace_fragments(requirement_fragments + default_fragments, WORLD))
            for issue in world_report.issues:
                issues.append(AnnotationIssue(issue.severity, WORLD, issue.path, issue.message, issue.expected, issue.actual, sources))

    report = AnnotationReport(tuple(issues))
    return RequirementResolution(
        environment_requirement=environment_requirement,
        world_requirement=world_requirement,
        runtime_requirement=runtime_requirement,
        world_default=world_default,
        runtime_default=runtime_default,
        fragments=ordered,
        source_traces=_source_traces_for_fragments(ordered, source=source),
        diagnostics=_diagnostics_from_report(report),
        merge_report=report,
    )


def resolve_target_requirements(
    target: Any,
    *,
    provider_fragments: Iterable[AnnotationFragment] = (),
    namespace: str | None = None,
    kind: str | None = None,
) -> RequirementResolution:
    """Collect and resolve requirements/defaults for a live target."""

    fragments = collect_fragments(target, namespace=namespace, kind=kind)
    return resolve_fragments(fragments, provider_fragments=provider_fragments)


def resolve_method_requirements(
    cls: type,
    method_name: str,
    *,
    provider_fragments: Iterable[AnnotationFragment] = (),
) -> RequirementResolution:
    """Collect and resolve class plus method requirements for ``cls.method``."""

    return resolve_fragments(fragments_for_method(cls, method_name), provider_fragments=provider_fragments)


def resolve_definition_method_requirements(
    defn: Any,
    method_name: str,
    *,
    provider_fragments: Iterable[AnnotationFragment] = (),
) -> RequirementResolution:
    """Collect and resolve requirements for a Definition/CDef method target."""

    return resolve_fragments(fragments_for_definition_method(defn, method_name), provider_fragments=provider_fragments)


def resolve_environment_requirement(target: Any, *, provider_fragments: Iterable[AnnotationFragment] = ()) -> EnvironmentRequirement:
    """Resolve environment requirements for *target*."""

    return resolve_requirements(target, provider_fragments=provider_fragments).environment or EnvironmentRequirement()


def resolve_world_requirement(target: Any, *, provider_fragments: Iterable[AnnotationFragment] = ()) -> WorldRequirement | None:
    """Resolve hard world requirements for *target*."""

    return resolve_requirements(target, provider_fragments=provider_fragments).world


def resolve_world_default(target: Any, *, provider_fragments: Iterable[AnnotationFragment] = (), overrides: Mapping[str, Any] | None = None) -> WorldSpec | None:
    """Resolve world defaults for *target*."""

    if overrides is not None and WORLD not in overrides:
        overrides = {WORLD: overrides}
    return resolve_defaults(target, provider_fragments=provider_fragments, overrides=overrides).world


def resolve_runtime_default(target: Any, *, provider_fragments: Iterable[AnnotationFragment] = (), overrides: Mapping[str, Any] | None = None) -> RuntimeContextSpec | None:
    """Resolve runtime defaults for *target*."""

    if overrides is not None and RUNTIME not in overrides:
        overrides = {RUNTIME: overrides}
    return resolve_defaults(target, provider_fragments=provider_fragments, overrides=overrides).runtime


def _merge_environment_requirements(fragments: tuple[AnnotationFragment, ...], issues: list[AnnotationIssue]) -> EnvironmentRequirement | None:
    if not fragments:
        return None
    data: dict[str, Any] = {"requirements": [], "excludes": [], "capabilities": [], "tags": [], "schema_versions": {}, "details": {"sources": [f.source.to_data() for f in fragments]}}
    saw_payload = False
    for fragment in _merge_order(fragments):
        try:
            payload = EnvironmentRequirement.from_data(fragment.fragment).to_data()
        except Exception as exc:
            issues.append(AnnotationIssue("error", ENVIRONMENT, "/", str(exc), sources=(fragment.source,)))
            continue
        saw_payload = True
        mode = fragment.merge_policy or fragment.source.metadata.get("legacy_environment_fragment_mode") or "add"
        if mode not in {"base", "add", "override"}:
            issues.append(AnnotationIssue("error", ENVIRONMENT, "/merge_policy", "unsupported environment requirement merge policy", expected="base|add|override", actual=mode, sources=(fragment.source,)))
            continue
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
    for fragment in _merge_order(fragments):
        try:
            current = WorldRequirement.from_data(fragment.fragment)
            policy = fragment.merge_policy or "merge"
            if policy in {"merge", "add", "base"}:
                req = current if req is None else req.merge(current)
            elif policy in {"replace", "override"}:
                req = current
            else:
                issues.append(AnnotationIssue("error", WORLD, "/merge_policy", "unsupported world requirement merge policy", expected="merge|add|base|replace|override", actual=policy, sources=(fragment.source,)))
        except Exception as exc:
            path = getattr(exc, "context", {}).get("path", "/")
            issues.append(AnnotationIssue("error", WORLD, f"/{str(path).replace('.', '/')}", str(exc), sources=(fragment.source,)))
    return req


def _merge_mapping_fragments(fragments: tuple[AnnotationFragment, ...], *, issues: list[AnnotationIssue] | None = None, namespace: str = "") -> dict[str, Any]:
    data: dict[str, Any] = {}
    for fragment in _merge_order(fragments):
        policy = fragment.merge_policy or "merge"
        if policy in {"merge", "add", "base"}:
            data = _deep_merge(data, dict(fragment.fragment))
        elif policy in {"replace", "override"}:
            data = dict(fragment.fragment)
        elif policy == "append":
            data = _append_merge(data, dict(fragment.fragment))
        elif policy == "error_on_conflict":
            conflicts = _conflict_issues(data, fragment.fragment, namespace=namespace, sources=_sources(fragments))
            if issues is not None:
                issues.extend(conflicts)
            if not conflicts:
                data = _deep_merge(data, dict(fragment.fragment))
        else:
            if issues is not None:
                issues.append(AnnotationIssue("error", namespace or fragment.namespace, "/merge_policy", "unknown annotation merge policy", expected="merge|replace|append|error_on_conflict", actual=policy, sources=(fragment.source,)))
    return data


def _merge_order(fragments: tuple[AnnotationFragment, ...]) -> tuple[AnnotationFragment, ...]:
    """Return fragments in stable priority order for merge decisions."""

    return tuple(sorted(fragments, key=lambda fragment: fragment.priority))


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
        if _specifier_has_obvious_conflict(SpecifierSet(",".join(spec_texts))):
            issues.append(AnnotationIssue("error", ENVIRONMENT, f"/requirements/{name}", "conflicting package requirement specifiers", expected="satisfiable specifier set", actual=", ".join(spec_texts), sources=sources))
    return issues


def _specifier_has_obvious_conflict(specifier: SpecifierSet) -> bool:
    exact_versions: set[Version] = set()
    lower: tuple[Version, bool] | None = None
    upper: tuple[Version, bool] | None = None
    for spec in specifier:
        try:
            version = Version(spec.version)
        except InvalidVersion:
            return False
        if spec.operator == "==":
            exact_versions.add(version)
        elif spec.operator in {">", ">="}:
            inclusive = spec.operator == ">="
            if lower is None or version > lower[0] or (version == lower[0] and not inclusive and lower[1]):
                lower = (version, inclusive)
        elif spec.operator in {"<", "<="}:
            inclusive = spec.operator == "<="
            if upper is None or version < upper[0] or (version == upper[0] and not inclusive and upper[1]):
                upper = (version, inclusive)
    if len(exact_versions) > 1:
        return True
    if exact_versions:
        exact = next(iter(exact_versions))
        return exact not in specifier
    if lower is not None and upper is not None:
        if lower[0] > upper[0]:
            return True
        if lower[0] == upper[0] and (not lower[1] or not upper[1]):
            return True
    return False


def _namespace_fragments(fragments: Iterable[AnnotationFragment], namespace: str) -> tuple[AnnotationFragment, ...]:
    return tuple(fragment for fragment in fragments if fragment.namespace == namespace)


def _sources(fragments: Iterable[AnnotationFragment]) -> tuple[SourceTrace, ...]:
    return tuple(fragment.source for fragment in fragments)


def _override_source(namespace: str) -> SourceTrace:
    return SourceTrace("override", namespace=namespace, label="user override")


def _source_traces_for_fragments(fragments: tuple[AnnotationFragment, ...], *, source: str | None) -> tuple[RequirementSourceTrace, ...]:
    return tuple(_source_trace_for_fragment(fragment, index, source=source) for index, fragment in enumerate(fragments))


def _source_trace_for_fragment(fragment: AnnotationFragment, index: int, *, source: str | None) -> RequirementSourceTrace:
    trace = fragment.source
    target = trace.target
    qualname = target.qualname if target is not None else None
    module = target.module if target is not None else None
    target_label = f"{module}:{qualname}" if module and qualname else qualname
    label = trace.label or target_label or trace.kind
    return RequirementSourceTrace(
        namespace=fragment.namespace,
        kind=fragment.kind,
        label=label,
        target_label=target_label,
        module=module,
        qualname=qualname,
        priority=fragment.priority,
        merge_policy=fragment.merge_policy,
        fragment_index=index,
        data={"source": trace.to_data(), **({"resolution_source": source} if source is not None else {})},
    )


def _diagnostics_from_report(report: AnnotationReport) -> tuple[RequirementDiagnostic, ...]:
    return tuple(_diagnostic_from_issue(issue) for issue in report.issues)


def _diagnostic_from_issue(issue: AnnotationIssue) -> RequirementDiagnostic:
    target_label = None
    if issue.sources:
        source = issue.sources[0]
        if source.target is not None:
            module = source.target.module
            qualname = source.target.qualname
            target_label = f"{module}:{qualname}" if module and qualname else qualname
        target_label = target_label or source.label
    return RequirementDiagnostic(
        level=issue.severity,
        code="dryml.annotations.merge_issue",
        message=issue.message,
        target_label=target_label,
        data={
            "namespace": issue.namespace,
            "path": issue.path,
            "expected": issue.expected,
            "actual": issue.actual,
            "sources": [source.to_data() for source in issue.sources],
        },
    )


def _to_data_or_none(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_data") and callable(value.to_data):
        return value.to_data()
    return json_ready(value)


def _report_to_data(report: AnnotationReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "ok": report.ok,
        "issues": [
            {
                "severity": issue.severity,
                "namespace": issue.namespace,
                "path": issue.path,
                "message": issue.message,
                "expected": issue.expected,
                "actual": issue.actual,
                "sources": [source.to_data() for source in issue.sources],
            }
            for issue in report.issues
        ],
    }


__all__ = [
    "ResolvedDefaults",
    "ResolvedRequirements",
    "RequirementDiagnostic",
    "RequirementResolution",
    "RequirementSourceTrace",
    "ResolutionResult",
    "resolve",
    "resolve_defaults",
    "resolve_definition_method_requirements",
    "resolve_environment_requirement",
    "resolve_fragments",
    "resolve_method_requirements",
    "resolve_requirements",
    "resolve_runtime_default",
    "resolve_target_requirements",
    "resolve_world_default",
    "resolve_world_requirement",
]
