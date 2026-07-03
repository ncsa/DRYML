"""Structured compatibility checks for world requirements, specs, and allocations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .allocation import WorldAllocation
from .resources import CountConstraint
from .specs import WorldRequirement, WorldSpec


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    """One structured world compatibility finding."""

    severity: Literal["error", "warning"]
    path: str
    message: str
    expected: Any | None = None
    actual: Any | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Result of checking world compatibility."""

    issues: tuple[CompatibilityIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether no error-severity issues were found."""

        return all(issue.severity != "error" for issue in self.issues)


def check_world_spec_satisfies_requirement(world: WorldSpec, requirement: WorldRequirement) -> CompatibilityReport:
    """Check a requested world spec against a hard world requirement."""

    issues: list[CompatibilityIssue] = []
    for role_name, role_req in requirement.roles.items():
        role_spec = world.roles.get(role_name)
        if role_spec is None:
            issues.append(_issue(f"/roles/{role_name}", "required role is missing", expected="present", actual="missing"))
            continue
        _check_count(role_req.replicas, role_spec.replicas, f"/roles/{role_name}/replicas", issues)
        _check_count(role_req.resources.cpus, role_spec.process.resources.cpus, f"/roles/{role_name}/process/resources/cpus", issues)
        memory = role_spec.process.resources.memory or 0
        _check_count(role_req.resources.memory, memory, f"/roles/{role_name}/process/resources/memory", issues)
        for accel, constraint in role_req.resources.accelerators.items():
            _check_count(constraint, role_spec.process.resources.accelerators.get(accel, 0), f"/roles/{role_name}/process/resources/accelerators/{accel}", issues)
        for device, constraint in role_req.resources.devices.items():
            _check_count(constraint, _count_resource_value(role_spec.process.resources.devices.get(device)), f"/roles/{role_name}/process/resources/devices/{device}", issues)
        for name, constraint in role_req.resources.named.items():
            _check_count(constraint, _count_resource_value(role_spec.process.resources.named.get(name)), f"/roles/{role_name}/process/resources/named/{name}", issues)
        _check_topology(role_req.topology, role_spec.replicas, f"/roles/{role_name}/topology", issues)
    return CompatibilityReport(tuple(issues))


def check_allocation_satisfies_requirement(allocation: WorldAllocation, requirement: WorldRequirement) -> CompatibilityReport:
    """Check an actual world allocation against a hard world requirement."""

    issues: list[CompatibilityIssue] = []
    for role_name, role_req in requirement.roles.items():
        allocations = allocation.roles.get(role_name)
        if allocations is None:
            issues.append(_issue(f"/roles/{role_name}", "required role is missing", expected="present", actual="missing"))
            continue
        _check_count(role_req.replicas, len(allocations), f"/roles/{role_name}", issues)
        for index, process in enumerate(allocations):
            base = f"/roles/{role_name}/{index}/resources"
            _check_count(role_req.resources.cpus, len(process.cpus), f"{base}/cpus", issues)
            _check_count(role_req.resources.memory, process.memory or 0, f"{base}/memory", issues)
            for accel, constraint in role_req.resources.accelerators.items():
                _check_count(constraint, len(process.accelerators.get(accel, ())), f"{base}/accelerators/{accel}", issues)
            for device, constraint in role_req.resources.devices.items():
                _check_count(constraint, _count_resource_value(process.devices.get(device)), f"{base}/devices/{device}", issues)
            for name, constraint in role_req.resources.named.items():
                _check_count(constraint, _count_resource_value(process.metadata.get(name)), f"{base}/named/{name}", issues)
        _check_topology(role_req.topology, len(allocations), f"/roles/{role_name}/topology", issues)
    return CompatibilityReport(tuple(issues))


def _check_count(constraint: CountConstraint, actual: int, path: str, issues: list[CompatibilityIssue]) -> None:
    if not constraint.satisfied_by(actual):
        issues.append(_issue(path, "count constraint is not satisfied", expected=constraint.to_data(), actual=actual))


def _check_topology(topology: dict[str, Any] | Any, replicas: int, path: str, issues: list[CompatibilityIssue]) -> None:
    if not topology:
        return
    if topology.get("single_process") is True and replicas != 1:
        issues.append(_issue(f"{path}/single_process", "single_process requires exactly one replica", expected=True, actual=replicas))
    for key in sorted(set(topology) - {"single_process"}):
        value = topology[key]
        if key in {"collectives", "shared_filesystem"} and value not in (None, False):
            issues.append(_issue(f"{path}/{key}", "topology field is not enforced by this compatibility checker", expected="unsupported/false", actual=value))
        elif key not in {"collectives", "shared_filesystem"}:
            issues.append(CompatibilityIssue("warning", f"{path}/{key}", "unsupported topology field was ignored", actual=value))


def _issue(path: str, message: str, *, expected: Any = None, actual: Any = None) -> CompatibilityIssue:
    return CompatibilityIssue("error", path, message, expected=expected, actual=actual, source="world_requirement")


def _count_resource_value(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (str, bytes)):
        return 1
    if isinstance(value, Mapping):
        return len(value)
    try:
        return len(value)
    except TypeError:
        return 1


__all__ = [
    "CompatibilityIssue",
    "CompatibilityReport",
    "check_allocation_satisfies_requirement",
    "check_world_spec_satisfies_requirement",
]
