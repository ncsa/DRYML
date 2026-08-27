"""Deterministic disjoint assignment of a requested world from local inventory."""

from __future__ import annotations

from .allocation import ProcessAllocation, WorldAllocation
from .errors import WorldCompatibilityError, WorldSpecValidationError
from .inventory import LocalResourceInventory
from .specs import WorldSpec


def assign_local_world(world: WorldSpec, *, inventory: LocalResourceInventory, oversubscribe: bool = False) -> WorldAllocation:
    """Assign deterministic disjoint local resources without reserving them.

    Args:
        world: Requested launch shape to bind exactly.
        inventory: Authoritative explicit local resource evidence.
        oversubscribe: Must remain false; public oversubscription is unsupported.

    Returns:
        An exact allocation with sorted role, replica, CPU, and accelerator IDs.

    Raises:
        WorldCompatibilityError: If oversubscription or capacity invention is
            requested.
    """
    if oversubscribe:
        raise WorldCompatibilityError("oversubscribe=True is unsupported before allocation")
    if not isinstance(world, WorldSpec) or not isinstance(inventory, LocalResourceInventory):
        raise WorldSpecValidationError("local assignment requires WorldSpec and LocalResourceInventory")
    cpu_index = memory_total = rank = 0
    accelerator_plan = _plan_accelerators(world, inventory)
    roles = {}
    for name, role in world.roles.items():
        values = []
        for replica in range(role.replicas):
            resources = role.process.resources
            if any(resources.devices.values()) or any(resources.named.values()):
                raise WorldCompatibilityError("local assignment cannot allocate named devices or resources")
            requested_cpus = resources.cpus
            if cpu_index + requested_cpus > len(inventory.cpus):
                raise WorldCompatibilityError("local assignment exceeds disjoint CPU capacity")
            cpus = inventory.cpus[cpu_index : cpu_index + requested_cpus]
            cpu_index += requested_cpus
            if resources.memory:
                if inventory.memory is None:
                    raise WorldCompatibilityError("positive process memory cannot be proven against unknown inventory")
                memory_total += resources.memory
                if memory_total > inventory.memory:
                    raise WorldCompatibilityError("local assignment exceeds aggregate process memory")
            accelerators, limits = {}, {}
            for kind, count in resources.accelerators.items():
                devices = accelerator_plan[(name, replica, kind)]
                accelerators[kind] = devices
                required = resources.accelerator_memory.get(kind, ())
                if required:
                    limits[kind] = {device: amount for device, amount in zip(devices, required, strict=True)}
            values.append(ProcessAllocation(
                replica=replica,
                rank=rank,
                local_rank=replica,
                cpus=cpus,
                memory=resources.memory,
                accelerators=accelerators,
                accelerator_memory=limits,
                devices=resources.devices,
                named=resources.named,
                environment=role.process.environment,
                env=role.process.env,
            ))
            rank += 1
        roles[name] = tuple(values)
    return WorldAllocation(roles)


def _plan_accelerators(world: WorldSpec, inventory: LocalResourceInventory) -> dict[tuple[str, int, str], tuple[str | int, ...]]:
    """Match harder per-device requests first while preserving output order."""

    requests: dict[str, list[tuple[int, str, int, int]]] = {}
    for role_name, role in world.roles.items():
        for replica in range(role.replicas):
            resources = role.process.resources
            for kind, count in resources.accelerators.items():
                limits = resources.accelerator_memory.get(kind, ())
                for slot in range(count):
                    requests.setdefault(kind, []).append((limits[slot] if limits else 0, role_name, replica, slot))

    assignments: dict[tuple[str, int, str], dict[int, str | int]] = {}
    for kind, kind_requests in requests.items():
        available = list(inventory.accelerators.get(kind, ()))
        capacities = inventory.accelerator_memory.get(kind, {})
        for required, role_name, replica, slot in sorted(kind_requests, key=lambda item: (-item[0], item[1], item[2], item[3])):
            selected_index = next(
                (
                    index
                    for index, device in enumerate(available)
                    if required == 0 or capacities.get(device, 0) >= required
                ),
                None,
            )
            if selected_index is None:
                if required:
                    raise WorldCompatibilityError("assigned accelerator lacks required per-device memory")
                raise WorldCompatibilityError("local assignment exceeds disjoint accelerator capacity")
            device = available.pop(selected_index)
            assignments.setdefault((role_name, replica, kind), {})[slot] = device

    return {
        key: tuple(values[index] for index in sorted(values))
        for key, values in assignments.items()
    }
