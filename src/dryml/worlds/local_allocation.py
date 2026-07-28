"""Pure deterministic local resource assignment shared by sessions and dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .errors import WorldSpecValidationError
from .inventory import LocalResourceInventory
from .specs import WorldSpec

_MAX_LOCAL_WORKERS = 4096
_MAX_CPU_ASSIGNMENTS = 4096


@dataclass(frozen=True, slots=True)
class LocalWorldAssignment:
    """Concrete, effect-free slots assigned from one local inventory."""

    roles: Mapping[str, tuple[Mapping[str, Any], ...]]
    worker_keys: tuple[tuple[str, int, int, int], ...]


def assign_local_world(world: WorldSpec, *, inventory: LocalResourceInventory, oversubscribe: bool = False) -> LocalWorldAssignment:
    """Assign CPU, memory, accelerator, and accelerator-memory slots deterministically."""

    if not isinstance(world, WorldSpec) or not isinstance(inventory, LocalResourceInventory):
        raise WorldSpecValidationError("local assignment requires WorldSpec and LocalResourceInventory")
    worker_count = sum(role.replicas for role in world.roles.values())
    if not worker_count or worker_count > _MAX_LOCAL_WORKERS:
        raise WorldSpecValidationError("local world worker count exceeds the bounded limit")
    cpu_assignments = sum(role.replicas * (role.process.resources.cpus or 1) for role in world.roles.values())
    if cpu_assignments > _MAX_CPU_ASSIGNMENTS:
        raise WorldSpecValidationError("local world CPU assignments exceed the bounded limit")
    cpu_cursor = memory_cursor = rank = 0
    accelerator_cursors = {kind: 0 for kind in inventory.accelerators}
    roles: dict[str, tuple[Mapping[str, Any], ...]] = {}
    keys: list[tuple[str, int, int, int]] = []
    for role_name in sorted(world.roles):
        role = world.roles[role_name]
        allocations: list[Mapping[str, Any]] = []
        for replica in range(role.replicas):
            resources = role.process.resources
            if resources.devices or resources.named:
                raise WorldSpecValidationError("local world allocation does not support named devices or resources", context={"role": role_name})
            requested_cpus = resources.cpus or 1
            if not oversubscribe and cpu_cursor + requested_cpus > len(inventory.cpus):
                raise WorldSpecValidationError("local world CPU requests exceed disjoint inventory", context={"role": role_name, "replica": replica})
            cpus = tuple(inventory.cpus[(cpu_cursor + index) % len(inventory.cpus)] for index in range(requested_cpus)) if oversubscribe else tuple(inventory.cpus[cpu_cursor : cpu_cursor + requested_cpus])
            cpu_cursor = (cpu_cursor + requested_cpus) % len(inventory.cpus) if oversubscribe else cpu_cursor + requested_cpus
            if resources.memory is not None:
                if resources.memory > 0 and inventory.memory is None:
                    raise WorldSpecValidationError("local world memory request cannot be proven against unknown inventory")
                if not oversubscribe and inventory.memory is not None and memory_cursor + resources.memory > inventory.memory:
                    raise WorldSpecValidationError("local world memory requests exceed disjoint inventory")
                memory_cursor += resources.memory
            accelerators: dict[str, tuple[str | int, ...]] = {}
            accelerator_memory: dict[str, list[dict[str, Any]]] = {}
            for kind, count in resources.accelerators.items():
                available = inventory.accelerators.get(kind, ())
                cursor = accelerator_cursors.get(kind, 0)
                devices = tuple(available[cursor : cursor + count])
                if len(devices) != count:
                    raise WorldSpecValidationError("local world accelerator request exceeds explicit inventory", context={"role": role_name, "replica": replica, "accelerator": kind})
                accelerator_cursors[kind] = cursor + count
                accelerators[kind] = devices
                limits = resources.accelerator_memory.get(kind, ())
                if limits:
                    known = inventory.accelerator_memory.get(kind, {})
                    for device, limit in zip(devices, limits, strict=True):
                        if device in known and limit > known[device]:
                            raise WorldSpecValidationError("accelerator-memory request exceeds known device capacity", context={"accelerator": kind, "device": device})
                    accelerator_memory[kind] = [
                        {"device": device, "memory": _canonical_memory(limit)}
                        for device, limit in zip(devices, limits, strict=True)
                    ]
            assigned: dict[str, Any] = {"cpus": list(cpus), "accelerators": {kind: list(values) for kind, values in sorted(accelerators.items())}}
            if resources.memory is not None:
                assigned["memory"] = _canonical_memory(resources.memory)
            if accelerator_memory:
                assigned["accelerator_memory"] = accelerator_memory
            allocations.append(MappingProxyType({"replica": replica, "rank": rank, "local_rank": rank, "resources": MappingProxyType(assigned)}))
            keys.append((role_name, replica, rank, rank))
            rank += 1
        roles[role_name] = tuple(allocations)
    return LocalWorldAssignment(MappingProxyType(roles), tuple(keys))


def _canonical_memory(value: int) -> str:
    from .resources import canonical_byte_size

    return canonical_byte_size(value)  # type: ignore[return-value]


__all__ = ["LocalWorldAssignment", "assign_local_world"]
