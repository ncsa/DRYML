"""Deterministic smallest feasible local requested-world synthesis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .compatibility import WorldCompatibilityReport, check_world_spec_satisfies_requirement
from .errors import WorldCompatibilityError
from .inventory import LocalResourceInventory, local_inventory
from .local_allocation import assign_local_world
from .resources import CountConstraint
from .specs import WorldRequirement, WorldSpec


@dataclass(frozen=True, slots=True)
class WorldSynthesisDiagnostic:
    """One bounded local synthesis shortfall or unsupported declaration."""

    code: str
    message: str
    path: str | None = None
    expected: Any = None
    observed: Any = None


@dataclass(frozen=True, slots=True)
class WorldSynthesisResult:
    """Effect-free local synthesis outcome and explanatory diagnostics."""

    status: str
    requirement: WorldRequirement | None
    inventory: LocalResourceInventory | None
    world: WorldSpec | None
    compatibility: WorldCompatibilityReport | None
    diagnostics: tuple[WorldSynthesisDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether a compatible requested local world was produced."""
        return self.status == "synthesized" and self.world is not None


def synthesize(requirement: WorldRequirement | Mapping[str, Any] | None, *, inventory: LocalResourceInventory | None = None, policy: str = "local") -> WorldSynthesisResult:
    """Build the smallest feasible local requested shape without allocation.

    Args:
        requirement: Explicit hard requirements, payload, or ``None`` for a
            one-process CPU default.
        inventory: Authoritative injected inventory, or omitted for lightweight
            inherited discovery.
        policy: Only ``"local"`` is supported.

    Returns:
        A requested world or structured shortfall/unsupported diagnostics.
    """
    if policy != "local":
        return _failure("unsupported_policy", "only local synthesis is supported")
    try:
        req = _requirement(requirement)
        inv = inventory if inventory is not None else local_inventory()
    except Exception as exc:
        return _failure("invalid_requirement", str(exc))
    if not isinstance(inv, LocalResourceInventory):
        return _failure("invalid_inventory", "inventory must be a LocalResourceInventory", requirement=req)
    roles, cpus, memory, accelerators = {}, 0, 0, {}
    for name, role in req.roles.items():
        if role.topology:
            return _failure("unsupported_topology", "non-empty topology is unsupported", f"roles.{name}.topology", requirement=req, inventory=inv)
        if any(value.min is not None or value.max is not None for value in role.resources.devices.values()) or any(value.min is not None or value.max is not None for value in role.resources.named.values()):
            return _failure("unsupported_resource", "named device/resource semantics are unsupported", f"roles.{name}.resources", requirement=req, inventory=inv)
        try:
            replicas = _choose(role.replicas, 1)
            role_cpus = _choose(role.resources.cpus, 1)
            role_memory = _choose(role.resources.memory, 0) if role.resources.memory.min is not None or role.resources.memory.max is not None else None
            role_accelerators = {kind: _choose(value, 0) for kind, value in role.resources.accelerators.items()}
        except ValueError:
            return _failure("unexecutable_constraint", "constraint cannot produce the smallest executable local process", f"roles.{name}", requirement=req, inventory=inv)
        cpus += replicas * role_cpus
        memory += replicas * (role_memory or 0)
        for kind, count in role_accelerators.items():
            accelerators[kind] = accelerators.get(kind, 0) + replicas * count
        limits = {kind: tuple(_choose(value, 0) for _ in range(role_accelerators[kind])) for kind, value in role.resources.accelerator_memory.items() if role_accelerators.get(kind, 0)}
        roles[name] = {"replicas": replicas, "process": {"resources": {"cpus": role_cpus, **({"memory": role_memory} if role_memory is not None else {}), **({"accelerators": {kind: count for kind, count in role_accelerators.items() if count}} if any(role_accelerators.values()) else {}), **({"accelerator_memory": limits} if limits else {})}}}
    if cpus > len(inv.cpus):
        return _capacity("insufficient_cpus", cpus, len(inv.cpus), req, inv)
    if memory and inv.memory is None:
        return _failure("memory_unknown", "positive process memory cannot be proven", "resources.memory", memory, None, req, inv)
    if inv.memory is not None and memory > inv.memory:
        return _capacity("insufficient_memory", memory, inv.memory, req, inv)
    for kind, count in accelerators.items():
        if count > len(inv.accelerators.get(kind, ())):
            return _capacity("insufficient_accelerators", count, len(inv.accelerators.get(kind, ())), req, inv, f"resources.accelerators.{kind}")
    world = WorldSpec.from_payload({"roles": roles})
    report = check_world_spec_satisfies_requirement(world, req)
    if not report.ok:
        return WorldSynthesisResult("incompatible", req, inv, None, report, tuple(WorldSynthesisDiagnostic(issue.code, issue.message, issue.path, issue.expected, issue.observed) for issue in report.issues))
    try:
        assign_local_world(world, inventory=inv)
    except WorldCompatibilityError as exc:
        code = "accelerator_memory_unknown" if "memory" in str(exc) else "insufficient_accelerators"
        return _failure(code, str(exc), "resources.accelerators", requirement=req, inventory=inv)
    return WorldSynthesisResult("synthesized", req, inv, world, report)


def _requirement(value: WorldRequirement | Mapping[str, Any] | None) -> WorldRequirement:
    if value is None:
        return WorldRequirement.from_payload({"roles": {"main": {"resources": {"cpus": {"min": 1, "max": 1}}}}})
    if isinstance(value, WorldRequirement):
        return value
    return WorldRequirement.from_payload(value if "roles" in value else {"roles": value})


def _choose(value: CountConstraint, floor: int) -> int:
    selected = max(floor, value.min or 0)
    if value.max is not None and selected > value.max:
        raise ValueError
    return selected


def _failure(code: str, message: str, path: str | None = None, expected: Any = None, observed: Any = None, requirement: WorldRequirement | None = None, inventory: LocalResourceInventory | None = None) -> WorldSynthesisResult:
    return WorldSynthesisResult("unsupported_requirement" if code.startswith("unsupported") else "insufficient_inventory" if code in {"memory_unknown", "insufficient_cpus", "insufficient_memory", "insufficient_accelerators", "accelerator_memory_unknown"} else "error", requirement, inventory, None, None, (WorldSynthesisDiagnostic(code, message, path, expected, observed),))


def _capacity(code: str, required: int, available: int, requirement: WorldRequirement, inventory: LocalResourceInventory, path: str = "resources") -> WorldSynthesisResult:
    return _failure(code, "local inventory is insufficient", path, required, available, requirement, inventory)
