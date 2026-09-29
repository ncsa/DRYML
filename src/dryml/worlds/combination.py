"""Diagnostic-first combination of passive world requirement declarations."""

from __future__ import annotations

from collections.abc import Mapping
from collections import defaultdict
import re
from typing import Any

from dryml.requirements import RequirementDeclaration, RequirementIssue, RequirementReport, RequirementResult, RequirementSource, combine_requirements
from dryml.requirements.collection import collect_declarations

from ._diagnostics import WorldPath as _Path
from ._diagnostics import path as _path
from ._diagnostics import render_path as _render_path
from .declarations import WORLD_REQUIREMENT_KEY
from .errors import ResourceValidationError, WorldRequirementError, WorldSpecValidationError
from .resources import CountConstraint, ResourceRequirement, _is_constrained, maximum, minimum
from .specs import RoleRequirement, WorldRequirement

_MAX_PATHS = 1024
_MAX_OCCURRENCES = 4096
_MAX_BYTES = 1024 * 1024
_MAX_TEXT = 4096
_SENSITIVE_SEGMENT = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|credential)")
_RESOURCE_FIELDS = ("cpus", "memory", "accelerators", "accelerator_memory", "devices", "named")


def _paths(value: WorldRequirement) -> tuple[_Path, ...]:
    """Return strict canonical paths that contribute hard world semantics."""

    if type(value) is not WorldRequirement:
        raise WorldRequirementError("world declaration value is invalid")
    paths: list[_Path] = []
    for role_name, role in value.roles.items():
        if _is_constrained(role.replicas):
            paths.append(_path("roles", role_name, "replicas"))
        for name in ("cpus", "memory"):
            if _is_constrained(getattr(role.resources, name)):
                paths.append(_path("roles", role_name, "resources", name))
        for family in _RESOURCE_FIELDS[2:]:
            for name, constraint in getattr(role.resources, family).items():
                if _is_constrained(constraint):
                    paths.append(_path("roles", role_name, "resources", family, name))
        paths.extend(_path("roles", role_name, "topology", name) for name in role.topology)
    return tuple(paths)


def _text_work(value: Any) -> int:
    """Validate and count bounded built-in text in an already-normalized value."""

    if type(value) is str:
        if len(value) > _MAX_TEXT:
            raise WorldRequirementError("world requirement text exceeds declaration limit")
        return len(value.encode("utf-8"))
    if value is None or type(value) in (bool, int, float):
        return 0
    if isinstance(value, Mapping):
        return sum(_text_work(key) + _text_work(item) for key, item in value.items())
    if type(value) in (tuple, list):
        return sum(_text_work(item) for item in value)
    raise WorldRequirementError("world requirement contains unsupported normalized data")


def _preflight(declarations: tuple[RequirementDeclaration[WorldRequirement], ...]) -> dict[_Path, tuple[RequirementSource, ...]]:
    """Bound diagnostic work and index each structured path's contributors."""

    paths: list[_Path] = []
    sources_by_path: dict[_Path, list[RequirementSource]] = defaultdict(list)
    for declaration in declarations:
        declaration_paths = _paths(declaration.value)
        paths.extend(declaration_paths)
        for path in set(declaration_paths):
            sources_by_path[path].append(declaration.source)
    if len(set(paths)) > _MAX_PATHS or len(paths) > _MAX_OCCURRENCES:
        raise WorldRequirementError("world requirement combination exceeds path capacity")
    work = sum(len(_render_path(path).encode("utf-8")) for path in paths)
    for declaration in declarations:
        work += _text_work(declaration.value.to_payload())
        work += sum(
            _text_work(item)
            for item in (declaration.source.label, declaration.source.module, declaration.source.qualname)
            if item is not None
        )
    if work > _MAX_BYTES:
        raise WorldRequirementError("world requirement combination exceeds byte capacity")
    return {path: tuple(sources) for path, sources in sources_by_path.items()}


def _intersect(values: tuple[CountConstraint, ...]) -> CountConstraint | None:
    """Return the hard range intersection or ``None`` for a contradiction."""

    lower = maximum(tuple(value.min for value in values))
    upper = minimum(tuple(value.max for value in values))
    if lower is not None and upper is not None and lower > upper:
        return None
    return CountConstraint(lower, upper)


def _safe_path(path: _Path, ordinal: int) -> str:
    """Redact sensitive caller-derived path segments without merging diagnostics."""

    if any(
        type(segment) is str and (_SENSITIVE_SEGMENT.search(segment) or "/" in segment or "\\" in segment)
        for segment in path
    ):
        return f"roles.<redacted-{ordinal}>"
    return _render_path(path)


def _combine_values(values: tuple[WorldRequirement, ...]) -> tuple[WorldRequirement | None, tuple[_Path, ...]]:
    """Combine all role paths only after discovering every semantic conflict."""

    conflicts: set[_Path] = set()
    combined: dict[str, tuple[CountConstraint | None, dict[str, Any], dict[str, Any]]] = {}
    role_names = sorted({name for value in values for name in value.roles})
    for name in role_names:
        roles = tuple(value.roles[name] for value in values if name in value.roles)
        replicas = _intersect(tuple(role.replicas for role in roles))
        if replicas is None:
            conflicts.add(_path("roles", name, "replicas"))

        resources: dict[str, Any] = {}
        for field in ("cpus", "memory"):
            constraint = _intersect(tuple(getattr(role.resources, field) for role in roles))
            if constraint is None:
                conflicts.add(_path("roles", name, "resources", field))
            resources[field] = constraint
        for family in _RESOURCE_FIELDS[2:]:
            result: dict[str, CountConstraint] = {}
            names = sorted({key for role in roles for key in getattr(role.resources, family)})
            for resource_name in names:
                constraints = tuple(
                    getattr(role.resources, family)[resource_name]
                    for role in roles
                    if resource_name in getattr(role.resources, family)
                )
                constraint = _intersect(constraints)
                if constraint is None:
                    conflicts.add(_path("roles", name, "resources", family, resource_name))
                result[resource_name] = constraint
            resources[family] = result

        topology: dict[str, Any] = {}
        topology_names = sorted({key for role in roles for key in role.topology})
        for topology_name in topology_names:
            declarations = tuple(role.topology[topology_name] for role in roles if topology_name in role.topology)
            if any(item != declarations[0] for item in declarations[1:]):
                conflicts.add(_path("roles", name, "topology", topology_name))
            else:
                topology[topology_name] = declarations[0]
        combined[name] = (replicas, resources, topology)
    if conflicts:
        return None, tuple(sorted(conflicts))
    return WorldRequirement(
        {
            name: RoleRequirement(replicas, ResourceRequirement(**resources), topology)
            for name, (replicas, resources, topology) in combined.items()
        }
    ), ()


class _WorldCombiner:
    """Domain-owned shared-combiner adapter for world requirements."""

    def combine(self, declarations: tuple[RequirementDeclaration[WorldRequirement], ...]) -> RequirementResult[WorldRequirement]:
        """Return one complete conflict report or a compatible world value."""

        if len(declarations) == 1:
            if type(declarations[0].value) is not WorldRequirement:
                raise WorldRequirementError("world declaration value is invalid")
            return RequirementResult(declarations[0].value)
        sources_by_path = _preflight(declarations)
        value, conflicts = _combine_values(tuple(declaration.value for declaration in declarations))
        if conflicts:
            return RequirementResult(
                report=RequirementReport(
                    tuple(
                        RequirementIssue(
                            "dryml.worlds.requirement_conflict",
                            "conflicting world requirement constraints",
                            path=_safe_path(path, ordinal),
                            sources=sources_by_path[path],
                        )
                        for ordinal, path in enumerate(conflicts, start=1)
                    )
                )
            )
        return RequirementResult(value)


def requirements_for(target: object) -> RequirementResult[WorldRequirement]:
    """Combine passive world declarations directly attached to a target.

    Args:
        target: A class, function, or supported descriptor inspected statically.

    Returns:
        An empty, valued, or conflict result for only world declarations.

    Raises:
        WorldRequirementError: If attached metadata or declaration values are
            malformed or exceed world combination bounds.

    Side Effects:
        None. The target is neither bound, invoked, nor modified.
    """

    try:
        declarations = collect_declarations(target, key=WORLD_REQUIREMENT_KEY, value_type=WorldRequirement)
        return combine_requirements(declarations, combiner=_WorldCombiner())
    except Exception:
        raise WorldRequirementError("world requirement collection or combination failed") from None


def requirements_for_method(owner: type | object, method_name: str) -> RequirementResult[WorldRequirement]:
    """Combine class declarations and one statically selected method's declarations.

    Args:
        owner: A class or instance whose exact class supplies the method.
        method_name: Exact method name resolved through static MRO inspection.

    Returns:
        An empty, valued, or conflict world requirement result.

    Raises:
        WorldRequirementError: If static selection, metadata, or values are
            malformed or exceed world combination bounds.

    Side Effects:
        None. Instance state and dynamic attribute hooks are never consulted.
    """

    try:
        declarations = collect_declarations(owner, key=WORLD_REQUIREMENT_KEY, value_type=WorldRequirement, method_name=method_name)
        return combine_requirements(declarations, combiner=_WorldCombiner())
    except Exception:
        raise WorldRequirementError("world method requirement collection or combination failed") from None


def merge_count_constraints(left: CountConstraint, right: CountConstraint) -> CountConstraint:
    """Intersect two count constraints while retaining resource validation errors."""

    if not isinstance(right, CountConstraint):
        raise ResourceValidationError("count constraint merge requires a CountConstraint")
    value = _intersect((left, right))
    if value is None:
        raise ResourceValidationError("count constraint min exceeds max")
    return value


def merge_resource_requirements(left: ResourceRequirement, right: ResourceRequirement) -> ResourceRequirement:
    """Intersect two resource requirements using the world combination primitive."""

    if not isinstance(right, ResourceRequirement):
        raise ResourceValidationError("resource requirement merge requires a ResourceRequirement")
    fields: dict[str, Any] = {}
    for field in ("cpus", "memory"):
        fields[field] = merge_count_constraints(getattr(left, field), getattr(right, field))
    for family in _RESOURCE_FIELDS[2:]:
        left_map = getattr(left, family)
        right_map = getattr(right, family)
        values = dict(left_map)
        for name, value in right_map.items():
            values[name] = merge_count_constraints(values[name], value) if name in values else value
        fields[family] = values
    return ResourceRequirement(**fields)


def merge_role_requirements(left: RoleRequirement, right: RoleRequirement) -> RoleRequirement:
    """Intersect two role requirements using the diagnostic combination primitive."""

    if not isinstance(right, RoleRequirement):
        raise WorldSpecValidationError("role requirement merge requires a RoleRequirement")
    topology = dict(left.topology)
    for key, value in right.topology.items():
        if key in topology and topology[key] != value:
            raise WorldSpecValidationError("topology declarations conflict", context={"path": f"topology.{key}"})
        topology[key] = value
    return RoleRequirement(merge_count_constraints(left.replicas, right.replicas), merge_resource_requirements(left.resources, right.resources), topology)


def merge_world_requirements(left: WorldRequirement, right: WorldRequirement) -> WorldRequirement:
    """Intersect two world values while retaining the public exception surface."""

    if not isinstance(right, WorldRequirement):
        raise WorldSpecValidationError("world requirement merge requires a WorldRequirement")
    values = dict(left.roles)
    for name, role in right.roles.items():
        values[name] = merge_role_requirements(values[name], role) if name in values else role
    return WorldRequirement(values)


__all__ = ["requirements_for", "requirements_for_method"]
