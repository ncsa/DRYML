"""Deterministic local world synthesis from hard requirements."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .compatibility import CompatibilityReport, check_world_spec_satisfies_requirement
from .errors import WorldCompatibilityError, WorldSpecValidationError
from .inventory import LocalResourceInventory, local_inventory
from .resources import CountConstraint
from .specs import WorldRequirement, WorldSpec

_MAX_SERIALIZATION_DEPTH = 8
_MAX_SERIALIZATION_ITEMS = 64
_MAX_SERIALIZATION_STRING = 4096
_MAX_SERIALIZATION_NODES = 1024
_MAX_LOCAL_WORLD_WORKERS = 4096


@dataclass(frozen=True, slots=True)
class WorldSynthesisDiagnostic:
    """One bounded machine-readable local synthesis finding.

    Attributes:
        code: Stable diagnostic identifier.
        severity: Finding severity.
        message: Human-readable finding summary.
        path: Optional canonical requirement path.
        expected: Optional requested value.
        observed: Optional available or observed value.
        data: Optional bounded structured context.
    """

    code: str
    severity: str
    message: str
    path: str | None = None
    expected: Any | None = None
    observed: Any | None = None
    data: Mapping[str, Any] | None = None

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostic data."""

        return _bounded_data({"code": self.code, "severity": self.severity, "message": self.message, "path": self.path, "expected": self.expected, "observed": self.observed, "data": self.data})


@dataclass(frozen=True, slots=True)
class WorldSynthesisResult:
    """Result and diagnostics from deterministic local world synthesis.

    Attributes:
        status: Synthesis outcome such as ``"synthesized"`` or
            ``"insufficient_inventory"``.
        requirement: Canonical requested requirement, when supplied.
        inventory: Bounded inventory summary used for synthesis.
        world: Synthesized requested world, when successful.
        compatibility: Authoritative post-synthesis compatibility report.
        diagnostics: Ordered structured synthesis findings.
        policy: Applied synthesis policy.
        resource_inventory: Internal inventory reused by dispatch allocation.
    """

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

        data = {
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
        return _bounded_data(data)


def synthesize(
    requirement: WorldRequirement | Mapping[str, Any] | None,
    *,
    inventory: LocalResourceInventory | None = None,
    policy: str = "local",
    inventory_policy: str = "lightweight",
) -> WorldSynthesisResult:
    """Build the smallest disjoint local ``WorldSpec`` satisfying *requirement*.

    Args:
        requirement: A world requirement, its canonical data, or ``None`` for
            the direct one-worker local default.
        inventory: Optional immutable capacity facts; supplied inventory avoids
            host discovery.
        policy: Supported synthesis policy, currently ``"local"`` only.
        inventory_policy: Discovery policy used only when inventory is omitted.

    Returns:
        A report containing a requested world or structured diagnostics. This
        function never allocates resources or activates a runtime.
    """

    if policy != "local":
        raise WorldSpecValidationError("unsupported world synthesis policy", context={"policy": policy})
    if inventory_policy not in {"lightweight", "external"}:
        raise WorldSpecValidationError("unsupported local inventory policy", context={"inventory_policy": inventory_policy})
    if inventory is not None and not isinstance(inventory, LocalResourceInventory):
        return _failure("error", None, None, policy, "invalid_inventory", "inventory must be a LocalResourceInventory")
    try:
        req = _coerce_requirement(requirement)
    except Exception as exc:
        return _failure("invalid_requirement", None, inventory, policy, "invalid_requirement", str(exc))
    try:
        inv = inventory or local_inventory(policy=inventory_policy)
    except Exception as exc:
        return _failure("error", req, inventory, policy, "inventory_discovery_failed", str(exc))
    if req is None:
        world = WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}, "backend": {"kind": "local", "parameters": {}}})
        return WorldSynthesisResult("synthesized", None, inv.summary(), world, None, (), policy, inv)
    try:
        if len(req.roles) > _MAX_LOCAL_WORLD_WORKERS:
            raise _SynthesisFailure(
                "invalid_requirement",
                "role_count_exceeds_local_limit",
                "local synthesis role count exceeds the worker limit",
                "roles",
                _MAX_LOCAL_WORLD_WORKERS,
                len(req.roles),
                {"limit": _MAX_LOCAL_WORLD_WORKERS, "roles": len(req.roles)},
            )
        roles: dict[str, Any] = {}
        required_cpus = required_memory = 0
        required_accelerators: dict[str, int] = {}
        worker_count = 0
        for name in sorted(req.roles):
            role = req.roles[name]
            _validate_topology(role.topology, name)
            _reject_unsupported(role.resources.devices, "devices", name)
            _reject_unsupported(role.resources.named, "named", name)
            replicas = _choose(role.replicas, minimum=1, path=f"roles.{name}.replicas")
            worker_count += replicas
            if worker_count > _MAX_LOCAL_WORLD_WORKERS:
                raise _SynthesisFailure(
                    "invalid_requirement",
                    "worker_count_exceeds_local_limit",
                    "local synthesis worker count exceeds the worker limit",
                    "roles",
                    _MAX_LOCAL_WORLD_WORKERS,
                    worker_count,
                    {"limit": _MAX_LOCAL_WORLD_WORKERS, "workers": worker_count},
                )
            cpus = _choose(role.resources.cpus, minimum=1, path=f"roles.{name}.resources.cpus")
            memory = _choose(role.resources.memory, minimum=0, path=f"roles.{name}.resources.memory") if role.resources.memory.to_data() else None
            accelerators = {
                kind: count
                for kind, constraint in sorted(role.resources.accelerators.items())
                if (count := _choose(constraint, minimum=0, path=f"roles.{name}.resources.accelerators.{kind}")) > 0
            }
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
            return WorldSynthesisResult(
                "error",
                req,
                inv.summary(),
                None,
                report,
                tuple(
                    WorldSynthesisDiagnostic(
                        "authoritative_check_failed",
                        issue.severity,
                        issue.message,
                        issue.path,
                        issue.expected,
                        issue.actual,
                    )
                    for issue in report.issues
                ),
                policy,
                inv,
            )
        return WorldSynthesisResult("synthesized", req, inv.summary(), world, report, (), policy, inv)
    except _SynthesisFailure as exc:
        return _failure(exc.status, req, inv, policy, exc.code, str(exc), exc.path, exc.expected, exc.observed, exc.data)
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
    data: Mapping[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


def _coerce_requirement(requirement: WorldRequirement | Mapping[str, Any] | None) -> WorldRequirement | None:
    if requirement is None:
        return None
    if isinstance(requirement, WorldRequirement):
        # Direct dataclass construction can bypass ``from_data`` validation.
        # Round-trip through canonical data before synthesis uses its fields.
        return WorldRequirement.from_data(requirement.to_data())
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
        raise _SynthesisFailure("insufficient_inventory", "insufficient_cpus", "local CPU inventory is insufficient", "resources.cpus", cpus, len(inventory.cpus), _capacity_data(cpus, len(inventory.cpus)))
    if memory and inventory.memory is None:
        raise _SynthesisFailure("insufficient_inventory", "memory_unknown", "local memory inventory is unknown", "resources.memory", memory, None, {"required": memory, "available": None, "shortfall": None})
    if inventory.memory is not None and memory > inventory.memory:
        raise _SynthesisFailure("insufficient_inventory", "insufficient_memory", "local memory inventory is insufficient", "resources.memory", memory, inventory.memory, _capacity_data(memory, inventory.memory))
    for kind, count in accelerators.items():
        available = len(inventory.accelerators.get(kind, ()))
        if count > available:
            raise _SynthesisFailure("insufficient_inventory", "insufficient_accelerators", "local accelerator inventory is insufficient", f"resources.accelerators.{kind}", count, available, _capacity_data(count, available))


def _capacity_data(required: int, available: int) -> dict[str, int]:
    """Return explicit aggregate capacity evidence for synthesis diagnostics."""

    return {"required": required, "available": available, "shortfall": required - available}


def _reject_unsupported(resources: Mapping[str, CountConstraint], kind: str, role: str) -> None:
    if any((constraint.min or 0) > 0 for constraint in resources.values()):
        raise _SynthesisFailure("unsupported_requirement", f"unsupported_{kind}", f"local synthesis cannot allocate named {kind}", f"roles.{role}.resources.{kind}")


def _validate_topology(topology: Mapping[str, Any], role: str) -> None:
    for key in ("collectives", "shared_filesystem"):
        if topology.get(key) not in (None, False):
            raise _SynthesisFailure("unsupported_requirement", "unsupported_topology", "local synthesis cannot enforce requested topology", f"roles.{role}.topology.{key}")


def _failure(status: str, requirement: WorldRequirement | None, inventory: LocalResourceInventory | None, policy: str, code: str, message: str, path: str | None = None, expected: Any | None = None, observed: Any | None = None, data: Mapping[str, Any] | None = None) -> WorldSynthesisResult:
    return WorldSynthesisResult(status, requirement, {} if inventory is None else inventory.summary(), None, None, (WorldSynthesisDiagnostic(code, "error", message, path, expected, observed, data),), policy, inventory)


def _bounded_data(value: Any, *, depth: int = 0, budget: list[int] | None = None) -> Any:
    """Return deterministic JSON-compatible data within public size limits."""

    budget = [_MAX_SERIALIZATION_NODES] if budget is None else budget
    if budget[0] <= 0 or depth > _MAX_SERIALIZATION_DEPTH:
        return {"__dryml_truncated__": "depth_or_size"}
    budget[0] -= 1
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return value[:_MAX_SERIALIZATION_STRING]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        result = {
            str(key)[:_MAX_SERIALIZATION_STRING]: _bounded_data(item, depth=depth + 1, budget=budget)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))[:_MAX_SERIALIZATION_ITEMS]
        }
        if len(value) > _MAX_SERIALIZATION_ITEMS:
            result["__dryml_truncated__"] = "items"
        return result
    if isinstance(value, (list, tuple)):
        result = [_bounded_data(item, depth=depth + 1, budget=budget) for item in value[:_MAX_SERIALIZATION_ITEMS]]
        if len(value) > _MAX_SERIALIZATION_ITEMS:
            result.append({"__dryml_truncated__": "items"})
        return result
    return str(value)[:_MAX_SERIALIZATION_STRING]


__all__ = ["WorldSynthesisDiagnostic", "WorldSynthesisResult", "synthesize"]
