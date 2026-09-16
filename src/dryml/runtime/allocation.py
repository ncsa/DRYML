"""Immutable current-process views over U4 exact world allocations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dryml.formats import deep_freeze_json
from dryml.worlds import ProcessAllocation, WorldAllocation, parse_byte_size

from .errors import RuntimeTransitionError

_WORLD_ID = re.compile(r"^worldalloc-v1\.1-[0-9a-f]{64}$")


class _NoAllocation:
    """Sentinel used by modes that must not hold workload resources."""

    def __repr__(self) -> str:
        """Return a stable diagnostic representation."""
        return "NoAllocation"


NoAllocation = _NoAllocation()


@dataclass(frozen=True, slots=True)
class RuntimeAllocationView:
    """One role-qualified current-process backend grant projection.

    Exact grants retain CPU IDs and a world allocation association. Logical
    backend grants retain only reported capacity and accelerator evidence; they
    never become affinity or allocation identity claims. A baseline grant has
    no resource controls because its backend supplied no allocation evidence.
    This value does not reserve resources or start a process.

    Args:
        role: Selected role, or ``"main"`` for a non-exact backend grant.
        replica: Selected role replica, or ``0`` for a non-exact grant.
        rank: Exact global rank when supplied by an exact allocation.
        local_rank: Exact local rank when supplied by an exact allocation.
        cpus: Exact physical CPU IDs; empty for logical or baseline grants.
        memory: Verified byte quantity when the backend reports one.
        accelerators: Verified accelerator IDs grouped by accelerator kind.
        accelerator_memory: Verified per-accelerator byte limits.
        env: Backend-provided worker environment controls.
        world_allocation_id: Exact world-allocation identity, if available.
        metadata: Detached diagnostic allocation metadata.
        logical_cpus: Verified logical CPU capacity, if the backend reports one.
        grant_provenance: ``"exact"``, ``"logical"``, or ``"baseline"``.

    Raises:
        RuntimeTransitionError: If resource evidence is malformed or a logical or
            baseline view claims exact CPU/allocation authority.

    Side Effects:
        Construction validates and freezes the supplied detached evidence only.
    """

    role: str | None = None
    replica: int | None = None
    rank: int | None = None
    local_rank: int | None = None
    cpus: tuple[int, ...] = ()
    memory: int | None = None
    accelerators: Mapping[str, tuple[str | int, ...]] = field(default_factory=dict)
    accelerator_memory: Mapping[str, Mapping[str | int, int]] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    world_allocation_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    logical_cpus: int | None = None
    grant_provenance: str = "exact"

    def __post_init__(self) -> None:
        """Freeze and validate an exact, role-qualified process projection."""
        if not isinstance(self.role, str) or not self.role:
            raise RuntimeTransitionError("inline allocation requires a non-empty role")
        if self.replica is None or isinstance(self.replica, bool) or not isinstance(self.replica, int) or self.replica < 0:
            raise RuntimeTransitionError("inline allocation requires a non-negative replica")
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0) for value in (self.rank, self.local_rank)):
            raise RuntimeTransitionError("runtime allocation rank and local_rank must be non-negative when supplied")
        cpus = tuple(sorted(self.cpus))
        if len(cpus) != len(set(cpus)) or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in cpus):
            raise RuntimeTransitionError("runtime allocation CPUs must be unique non-negative integers")
        if not isinstance(self.accelerators, Mapping) or not isinstance(self.accelerator_memory, Mapping):
            raise RuntimeTransitionError("runtime allocation accelerator fields must be mappings")
        accelerators = {kind: tuple(values) for kind, values in self.accelerators.items()}
        if any(not isinstance(kind, str) or not kind or len(values) != len(set(values)) for kind, values in accelerators.items()):
            raise RuntimeTransitionError("runtime allocation accelerator assignments are invalid")
        try:
            limits = {
                kind: MappingProxyType({device: parse_byte_size(value) for device, value in values.items()})
                for kind, values in self.accelerator_memory.items()
            }
        except Exception as exc:
            raise RuntimeTransitionError("runtime accelerator memory must contain positive byte counts") from exc
        if set(limits) - set(accelerators):
            raise RuntimeTransitionError("accelerator memory must reference assigned accelerator kinds")
        for kind, values in limits.items():
            if set(values) - set(accelerators[kind]) or any(value in {None, 0} for value in values.values()):
                raise RuntimeTransitionError("accelerator memory must be positive and reference assigned devices")
        if not isinstance(self.env, Mapping) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.env.items()):
            raise RuntimeTransitionError("runtime allocation environment must be a string mapping")
        if self.logical_cpus is not None and (isinstance(self.logical_cpus, bool) or not isinstance(self.logical_cpus, int) or self.logical_cpus < 0):
            raise RuntimeTransitionError("runtime logical CPU capacity must be a non-negative integer")
        if self.grant_provenance not in {"exact", "logical", "baseline"}:
            raise RuntimeTransitionError("runtime allocation grant provenance must be exact, logical, or baseline")
        if self.grant_provenance in {"logical", "baseline"} and (cpus or self.world_allocation_id is not None):
            raise RuntimeTransitionError("logical or baseline runtime grants cannot claim CPU IDs or world allocation identity")
        object.__setattr__(self, "cpus", cpus)
        try:
            object.__setattr__(self, "memory", parse_byte_size(self.memory))
        except Exception as exc:
            raise RuntimeTransitionError("runtime allocation memory must be a non-negative byte count") from exc
        if self.world_allocation_id is not None and (not isinstance(self.world_allocation_id, str) or _WORLD_ID.fullmatch(self.world_allocation_id) is None):
            raise RuntimeTransitionError("runtime allocation association must be a worldalloc-v1.1 ID")
        object.__setattr__(self, "accelerators", MappingProxyType({key: accelerators[key] for key in sorted(accelerators)}))
        object.__setattr__(self, "accelerator_memory", MappingProxyType({key: limits[key] for key in sorted(limits)}))
        object.__setattr__(self, "env", MappingProxyType({key: self.env[key] for key in sorted(self.env)}))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))

    @classmethod
    def from_world_allocation(cls, value: WorldAllocation, *, role: str, replica: int = 0) -> "RuntimeAllocationView":
        """Select one exact process from a U4 allocation.

        Args:
            value: Immutable U4 world allocation.
            role: Required role name.
            replica: Required role replica number.

        Returns:
            The selected current-process view.

        Raises:
            RuntimeTransitionError: If the role/replica selection is absent or ambiguous.
        """
        processes = tuple(item for item in value.roles.get(role, ()) if item.replica == replica)
        if len(processes) != 1:
            raise RuntimeTransitionError("runtime allocation selection must identify exactly one process")
        process: ProcessAllocation = processes[0]
        return cls(role, process.replica, process.rank, process.local_rank, process.cpus, process.memory, process.accelerators, process.accelerator_memory, process.env, value.semantic_id, process.metadata)


def is_no_allocation(value: object) -> bool:
    """Return whether ``value`` is the no-workload sentinel."""
    return value is NoAllocation


__all__ = ["NoAllocation", "RuntimeAllocationView", "is_no_allocation"]
