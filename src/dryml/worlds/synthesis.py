"""Deterministic local world synthesis from hard requirements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .compatibility import CompatibilityReport, check_world_spec_satisfies_requirement
from .errors import WorldCompatibilityError, WorldSpecValidationError
from .inventory import LocalResourceInventory, local_inventory
from .resources import CountConstraint
from .specs import WorldRequirement, WorldSpec


@dataclass(frozen=True, slots=True)
class WorldSynthesisDiagnostic:
    """One bounded machine-readable local synthesis finding."""

    code: str
    severity: str
    message: str
    path: str | None = None
    expected: Any | None = None
    observed: Any | None = None

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostic data."""

        return {"code": self.code, "severity": self.severity, "message": self.message, "path": self.path, "expected": self.expected, "observed": self.observed}


@dataclass(frozen=True, slots=True)
class WorldSynthesisResult:
    """Result and diagnostics from deterministic local world synthesis."""

    status: str
    requirement: WorldRequirement | None
    inventory: Mapping[str, Any]
    world: WorldSpec | None
    compatibility: CompatibilityReport | None
    diagnostics: tuple[WorldSynthesisDiagnostic, ...]
    policy: str
    resource_inventory: LocalResourceInventory | None = None

    @property
    def ok(self) -> bool:
        """Return whether synthesis produced a compatible world."""

        return self.status == "synthesized" and self.world is not None

    def require_world(self) -> WorldSpec:
        """Return the synthesized world or raise a structured error."""

        if self.world is None:
            raise WorldCompatibilityError("world synthesis did not produce a compatible local world", context={"status": self.status, "diagnostics": [item.to_data() for item in self.diagnostics]})
        return self.world

    def to_data(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible result data."""

        return {
            "status": self.status,
            "requirement": None if self.requirement is None else self.requirement.to_data(),
            "inventory": dict(self.inventory),
            "world": None if self.world is None else self.world.to_data(),
            "compatibility": None
            if self.compatibility is None
            else {
                "ok": self.compatibility.ok,
                "issues": [
                    {
                        "severity": issue.severity,
                        "path": issue.path,
                        "message": issue.message,
                        "expected": issue.expected,
                        "actual": issue.actual,
                        "source": issue.source,
                    }
                    for issue in self.compatibility.issues
                ],
            },
            "diagnostics": [item.to_data() for item in self.diagnostics],
            "policy": self.policy,
        }


def synthesize(requirement: WorldRequirement | Mapping[str, Any] | None, *, inventory: LocalResourceInventory | None = None, policy: str = "local") -> WorldSynthesisResult:
    """Build the smallest disjoint local ``WorldSpec`` satisfying *requirement*.

    The result remains a requested world.  No allocation or runtime activation
    occurs; callers pass it to a backend allocator separately.
    """

    if policy != "local":
        raise WorldSpecValidationError("unsupported world synthesis policy", context={"policy": policy})
    try:
        req = _coerce_requirement(requirement)
    except Exception as exc:
        return _failure("invalid_requirement", None, inventory, policy, "invalid_requirement", str(exc))
    inv = inventory or local_inventory()
    if req is None:
        world = WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}, "backend": {"kind": "local", "parameters": {}}})
        return WorldSynthesisResult("synthesized", None, inv.summary(), world, None, (), policy, inv)
    try:
        roles: dict[str, Any] = {}
        required_cpus = required_memory = 0
        required_accelerators: dict[str, int] = {}
        for name in sorted(req.roles):
            role = req.roles[name]
            _validate_topology(role.topology, name)
            _reject_unsupported(role.resources.devices, "devices", name)
            _reject_unsupported(role.resources.named, "named", name)
            replicas = _choose(role.replicas, minimum=1, path=f"roles.{name}.replicas")
            cpus = _choose(role.resources.cpus, minimum=1, path=f"roles.{name}.resources.cpus")
            memory = _choose(role.resources.memory, minimum=0, path=f"roles.{name}.resources.memory") if role.resources.memory.to_data() else None
            accelerators = {kind: _choose(constraint, minimum=0, path=f"roles.{name}.resources.accelerators.{kind}") for kind, constraint in sorted(role.resources.accelerators.items())}
            required_cpus += replicas * cpus
            required_memory += replicas * (memory or 0)
            for kind, count in accelerators.items():
                required_accelerators[kind] = required_accelerators.get(kind, 0) + replicas * count
            resources: dict[str, Any] = {"cpus": cpus}
            if memory is not None:
                resources["memory"] = memory
            if accelerators:
                resources["accelerators"] = accelerators
            roles[name] = {"replicas": replicas, "process": {"resources": resources}}
        _check_capacity(inv, required_cpus, required_memory, required_accelerators)
        world = WorldSpec.from_data({"roles": roles, "backend": {"kind": "local", "parameters": {}}})
        report = check_world_spec_satisfies_requirement(world, req)
        if not report.ok:
            return WorldSynthesisResult("error", req, inv.summary(), None, report, tuple(WorldSynthesisDiagnostic("authoritative_check_failed", "error", issue.message, issue.path, issue.expected, issue.actual) for issue in report.issues), policy, inv)
        return WorldSynthesisResult("synthesized", req, inv.summary(), world, report, (), policy, inv)
    except _SynthesisFailure as exc:
        return _failure(exc.status, req, inv, policy, exc.code, str(exc), exc.path, exc.expected, exc.observed)
    except Exception as exc:
        return _failure("error", req, inv, policy, "synthesis_error", str(exc))


@dataclass(frozen=True, slots=True)
class _SynthesisFailure(Exception):
    status: str
    code: str
    message: str
    path: str | None = None
    expected: Any | None = None
    observed: Any | None = None

    def __str__(self) -> str:
        return self.message


def _coerce_requirement(requirement: WorldRequirement | Mapping[str, Any] | None) -> WorldRequirement | None:
    if requirement is None or isinstance(requirement, WorldRequirement):
        return requirement
    if not isinstance(requirement, Mapping):
        raise WorldSpecValidationError("world requirement must be a mapping")
    return WorldRequirement.from_data(requirement if "roles" in requirement else {"roles": requirement})


def _choose(constraint: CountConstraint, *, minimum: int, path: str) -> int:
    value = max(minimum, constraint.min or 0)
    if constraint.max is not None and value > constraint.max:
        raise _SynthesisFailure("invalid_requirement", "unexecutable_constraint", "constraint cannot produce an executable local worker", path, constraint.to_data(), value)
    return value


def _check_capacity(inventory: LocalResourceInventory, cpus: int, memory: int, accelerators: Mapping[str, int]) -> None:
    if cpus > len(inventory.cpus):
        raise _SynthesisFailure("insufficient_inventory", "insufficient_cpus", "local CPU inventory is insufficient", "resources.cpus", cpus, len(inventory.cpus))
    if memory and inventory.memory is None:
        raise _SynthesisFailure("insufficient_inventory", "memory_unknown", "local memory inventory is unknown", "resources.memory", memory, None)
    if inventory.memory is not None and memory > inventory.memory:
        raise _SynthesisFailure("insufficient_inventory", "insufficient_memory", "local memory inventory is insufficient", "resources.memory", memory, inventory.memory)
    for kind, count in accelerators.items():
        available = len(inventory.accelerators.get(kind, ()))
        if count > available:
            raise _SynthesisFailure("insufficient_inventory", "insufficient_accelerators", "local accelerator inventory is insufficient", f"resources.accelerators.{kind}", count, available)


def _reject_unsupported(resources: Mapping[str, CountConstraint], kind: str, role: str) -> None:
    if any((constraint.min or 0) > 0 for constraint in resources.values()):
        raise _SynthesisFailure("unsupported_requirement", f"unsupported_{kind}", f"local synthesis cannot allocate named {kind}", f"roles.{role}.resources.{kind}")


def _validate_topology(topology: Mapping[str, Any], role: str) -> None:
    for key in ("collectives", "shared_filesystem"):
        if topology.get(key) not in (None, False):
            raise _SynthesisFailure("unsupported_requirement", "unsupported_topology", "local synthesis cannot enforce requested topology", f"roles.{role}.topology.{key}")


def _failure(status: str, requirement: WorldRequirement | None, inventory: LocalResourceInventory | None, policy: str, code: str, message: str, path: str | None = None, expected: Any | None = None, observed: Any | None = None) -> WorldSynthesisResult:
    return WorldSynthesisResult(status, requirement, {} if inventory is None else inventory.summary(), None, None, (WorldSynthesisDiagnostic(code, "error", message, path, expected, observed),), policy, inventory)


__all__ = ["WorldSynthesisDiagnostic", "WorldSynthesisResult", "synthesize"]
