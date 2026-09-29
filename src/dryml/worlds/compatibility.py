"""Explicit checks between world requirements, shapes, and assignments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._diagnostics import WorldPath, path, render_path
from .allocation import ProcessAllocation, WorldAllocation
from .resources import CountConstraint
from .specs import WorldRequirement, WorldSpec


@dataclass(frozen=True, slots=True)
class WorldCompatibilityIssue:
    """One deterministic compatibility failure or unsupported declaration."""

    code: str
    path: str
    message: str
    expected: Any = None
    observed: Any = None
    severity: str = "error"


@dataclass(frozen=True, slots=True)
class WorldCompatibilityReport:
    """Immutable compatibility result; unsupported assertions are never OK."""

    issues: tuple[WorldCompatibilityIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Return true only when every asserted constraint is satisfied."""
        return not self.issues

    @property
    def admission_ok(self) -> bool:
        """Return the unchanged all-constraints-satisfied admission decision."""

        return self.ok


def check_world_spec_satisfies_requirement(world: WorldSpec, requirement: WorldRequirement) -> WorldCompatibilityReport:
    """Check a requested shape against all declared hard supported constraints."""
    issues: list[WorldCompatibilityIssue] = []
    for name, role in requirement.roles.items():
        spec = world.roles.get(name)
        if spec is None:
            issues.append(_issue("missing_role", path("roles", name), "required role is absent"))
            continue
        _constraint(issues, role.replicas, spec.replicas, path("roles", name, "replicas"))
        resources = spec.process.resources
        _constraint(issues, role.resources.cpus, resources.cpus, path("roles", name, "resources", "cpus"))
        _constraint(issues, role.resources.memory, resources.memory or 0, path("roles", name, "resources", "memory"))
        _resource_constraints(issues, role, resources, path("roles", name, "resources"))
        _topology(issues, role.topology, path("roles", name, "topology"))
    return WorldCompatibilityReport(tuple(issues))


def check_allocation_satisfies_requirement(allocation: WorldAllocation, requirement: WorldRequirement) -> WorldCompatibilityReport:
    """Check exact local assignments against all declared hard constraints."""
    issues: list[WorldCompatibilityIssue] = []
    for name, role in requirement.roles.items():
        processes = allocation.roles.get(name)
        if processes is None:
            issues.append(_issue("missing_role", path("roles", name), "required role is absent"))
            continue
        _constraint(issues, role.replicas, len(processes), path("roles", name, "replicas"))
        for process in processes:
            base = path("roles", name, process.replica, "resources")
            _constraint(issues, role.resources.cpus, len(process.cpus), path(*base, "cpus"))
            _constraint(issues, role.resources.memory, process.memory or 0, path(*base, "memory"))
            for kind, constraint in role.resources.accelerators.items():
                _constraint(issues, constraint, len(process.accelerators.get(kind, ())), path(*base, "accelerators", kind))
            for kind, constraint in role.resources.accelerator_memory.items():
                assigned = process.accelerators.get(kind, ())
                values = process.accelerator_memory.get(kind, {})
                for device in assigned:
                    _constraint(issues, constraint, values.get(device, 0), path(*base, "accelerator_memory", kind, str(device)))
            _unsupported_resources(issues, role, base)
        _topology(issues, role.topology, path("roles", name, "topology"))
    return WorldCompatibilityReport(tuple(issues))


def check_selected_process_satisfies_requirement(
        role: str | None, process: ProcessAllocation | None,
        requirement: WorldRequirement | None,
) -> WorldCompatibilityReport:
    """
    Check one selected current-process allocation against a hard requirement.

        Args:
            role: Selected role name, or ``None`` when no process is selected.
            process: Exact selected-process allocation, or ``None`` when
            absent.
            requirement: A valued effective world requirement, or ``None`` when
            no
                declaration was collected.

        Returns:
            A compatibility report based solely on the selected role and exact
            process
            evidence.  An absent requirement is compatible; a valued
            requirement with
            no allocation receives an ``allocation_missing`` issue.

        Raises:
            TypeError: If role/process evidence is malformed or only one is
            absent.

        Side Effects:
            None. This checker does not import Session, inspect host inventory,
            create
            a synthetic full-world allocation, reserve resources, or change
            runtime.
    """

    if requirement is not None and not isinstance(requirement,
                                                  WorldRequirement):
        raise TypeError("requirement must be a WorldRequirement or None")
    if role is not None and (not isinstance(role, str) or not role):
        raise TypeError("role must be a non-empty string or None")
    if process is not None and not isinstance(process, ProcessAllocation):
        raise TypeError("process must be a ProcessAllocation or None")
    if (role is None) != (process is None):
        raise TypeError("role and process must be provided together")
    if requirement is None:
        return WorldCompatibilityReport()
    if role is None:
        return WorldCompatibilityReport((_issue(
            "allocation_missing", path("allocation"),
            "a valued world requirement requires selected-process evidence"),
                                         ))
    issues: list[WorldCompatibilityIssue] = []
    selected_role = role
    selected = process
    role = requirement.roles.get(selected_role)
    if role is None:
        issues.append(
            _issue("selected_role_unexpected", path("roles", selected_role),
                   "selected process role is not declared by the requirement"))
    for name in requirement.roles:
        if name != selected_role:
            issues.append(
                _issue("missing_role", path("roles", name),
                       "required role has no selected-process evidence"))
    if role is None:
        return WorldCompatibilityReport(tuple(issues))
    _constraint(issues, role.replicas, 1,
                path("roles", selected_role, "replicas"))
    base = path("roles", selected_role, "resources")
    _constraint(issues, role.resources.cpus, len(selected.cpus),
                path(*base, "cpus"))
    if selected.memory is None and _is_constrained(role.resources.memory):
        issues.append(
            _issue("memory_missing", path(*base, "memory"),
                   "memory evidence is required for a hard constraint"))
    else:
        _constraint(issues, role.resources.memory, selected.memory or 0,
                    path(*base, "memory"))
    for kind, constraint in role.resources.accelerators.items():
        assigned = selected.accelerators.get(kind, ())
        _constraint(issues, constraint, len(assigned),
                    path(*base, "accelerators", kind))
    for kind, constraint in role.resources.accelerator_memory.items():
        assigned = selected.accelerators.get(kind, ())
        values = selected.accelerator_memory.get(kind)
        if values is None or not assigned:
            issues.append(
                _issue(
                    "accelerator_memory_missing",
                    path(*base, "accelerator_memory", kind),
                    "accelerator memory requires exact assigned-device "
                    "evidence"
                ))
            continue
        for device in assigned:
            if device not in values:
                issues.append(
                    _issue(
                        "accelerator_memory_missing",
                        path(*base, "accelerator_memory", kind, str(device)),
                        "assigned accelerator has no memory evidence"))
            else:
                _constraint(
                    issues, constraint, values[device],
                    path(*base, "accelerator_memory", kind, str(device)))
    _unsupported_resources(issues, role, base)
    _topology(issues, role.topology, path("roles", selected_role, "topology"))
    return WorldCompatibilityReport(tuple(issues))


def _resource_constraints(issues: list[WorldCompatibilityIssue], role: Any, resources: Any, path_value: WorldPath) -> None:
    for kind, constraint in role.resources.accelerators.items():
        _constraint(issues, constraint, resources.accelerators.get(kind, 0), path(*path_value, "accelerators", kind))
    for kind, constraint in role.resources.accelerator_memory.items():
        values = resources.accelerator_memory.get(kind, ())
        for index, value in enumerate(values):
            _constraint(issues, constraint, value, path(*path_value, "accelerator_memory", kind, index))
        if resources.accelerators.get(kind, 0) and len(values) != resources.accelerators[kind]:
            issues.append(_issue("accelerator_memory_missing", path(*path_value, "accelerator_memory", kind), "every assigned accelerator requires memory evidence"))
    _unsupported_resources(issues, role, path_value)


def _unsupported_resources(issues: list[WorldCompatibilityIssue], role: Any, path_value: WorldPath) -> None:
    for family in ("devices", "named"):
        for name, constraint in getattr(role.resources, family).items():
            if constraint.min is not None or constraint.max is not None:
                issues.append(_issue("unsupported_resource", path(*path_value, family, name), "resource semantic is unsupported by local planning"))


def _topology(issues: list[WorldCompatibilityIssue], topology: Any, path_value: WorldPath) -> None:
    if topology:
        issues.append(_issue("unsupported_topology", path_value, "non-empty topology is representable but unsupported"))


def _constraint(issues: list[WorldCompatibilityIssue], constraint: CountConstraint, actual: int, path_value: WorldPath) -> None:
    if not constraint.satisfied_by(actual):
        issues.append(_issue("constraint_unsatisfied", path_value, "hard resource constraint is not satisfied", constraint.to_data(), actual))


def _is_constrained(constraint: CountConstraint) -> bool:
    """Return whether a hard count range needs concrete evidence."""

    return constraint.min is not None or constraint.max is not None


def _issue(code: str, path_value: WorldPath, message: str, expected: Any = None, observed: Any = None) -> WorldCompatibilityIssue:
    return WorldCompatibilityIssue(code, render_path(path_value), message, expected, observed)
