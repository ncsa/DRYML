"""Explicit checks between world requirements, shapes, and assignments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .allocation import WorldAllocation
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
            issues.append(_issue("missing_role", f"roles.{name}", "required role is absent"))
            continue
        _constraint(issues, role.replicas, spec.replicas, f"roles.{name}.replicas")
        resources = spec.process.resources
        _constraint(issues, role.resources.cpus, resources.cpus, f"roles.{name}.resources.cpus")
        _constraint(issues, role.resources.memory, resources.memory or 0, f"roles.{name}.resources.memory")
        _resource_constraints(issues, role, resources, f"roles.{name}.resources")
        _topology(issues, role.topology, f"roles.{name}.topology")
    return WorldCompatibilityReport(tuple(issues))


def check_allocation_satisfies_requirement(allocation: WorldAllocation, requirement: WorldRequirement) -> WorldCompatibilityReport:
    """Check exact local assignments against all declared hard constraints."""
    issues: list[WorldCompatibilityIssue] = []
    for name, role in requirement.roles.items():
        processes = allocation.roles.get(name)
        if processes is None:
            issues.append(_issue("missing_role", f"roles.{name}", "required role is absent"))
            continue
        _constraint(issues, role.replicas, len(processes), f"roles.{name}.replicas")
        for process in processes:
            base = f"roles.{name}[{process.replica}].resources"
            _constraint(issues, role.resources.cpus, len(process.cpus), f"{base}.cpus")
            _constraint(issues, role.resources.memory, process.memory or 0, f"{base}.memory")
            for kind, constraint in role.resources.accelerators.items():
                _constraint(issues, constraint, len(process.accelerators.get(kind, ())), f"{base}.accelerators.{kind}")
            for kind, constraint in role.resources.accelerator_memory.items():
                assigned = process.accelerators.get(kind, ())
                values = process.accelerator_memory.get(kind, {})
                for device in assigned:
                    _constraint(issues, constraint, values.get(device, 0), f"{base}.accelerator_memory.{kind}.{device}")
            _unsupported_resources(issues, role, base)
        _topology(issues, role.topology, f"roles.{name}.topology")
    return WorldCompatibilityReport(tuple(issues))


def _resource_constraints(issues: list[WorldCompatibilityIssue], role: Any, resources: Any, path: str) -> None:
    for kind, constraint in role.resources.accelerators.items():
        _constraint(issues, constraint, resources.accelerators.get(kind, 0), f"{path}.accelerators.{kind}")
    for kind, constraint in role.resources.accelerator_memory.items():
        values = resources.accelerator_memory.get(kind, ())
        for index, value in enumerate(values):
            _constraint(issues, constraint, value, f"{path}.accelerator_memory.{kind}[{index}]")
        if resources.accelerators.get(kind, 0) and len(values) != resources.accelerators[kind]:
            issues.append(_issue("accelerator_memory_missing", f"{path}.accelerator_memory.{kind}", "every assigned accelerator requires memory evidence"))
    _unsupported_resources(issues, role, path)


def _unsupported_resources(issues: list[WorldCompatibilityIssue], role: Any, path: str) -> None:
    for family in ("devices", "named"):
        for name, constraint in getattr(role.resources, family).items():
            if constraint.min is not None or constraint.max is not None:
                issues.append(_issue("unsupported_resource", f"{path}.{family}.{name}", "resource semantic is unsupported by local planning"))


def _topology(issues: list[WorldCompatibilityIssue], topology: Any, path: str) -> None:
    if topology:
        issues.append(_issue("unsupported_topology", path, "non-empty topology is representable but unsupported"))


def _constraint(issues: list[WorldCompatibilityIssue], constraint: CountConstraint, actual: int, path: str) -> None:
    if not constraint.satisfied_by(actual):
        issues.append(_issue("constraint_unsatisfied", path, "hard resource constraint is not satisfied", constraint.to_data(), actual))


def _issue(code: str, path: str, message: str, expected: Any = None, observed: Any = None) -> WorldCompatibilityIssue:
    return WorldCompatibilityIssue(code, path, message, expected, observed)
