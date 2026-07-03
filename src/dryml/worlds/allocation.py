"""Actual backend allocations for DRYML worlds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dryml.records import attach_spec_id, compute_spec_id, make_spec, validate_spec
from dryml.records.errors import SpecValidationError

from .errors import ResourceValidationError, WorldSpecValidationError
from .resources import canonical_byte_size, parse_byte_size
from .specs import _iter_valid_roles


@dataclass(frozen=True, slots=True)
class RuntimeResourceView:
    """Process-local resource view derived from a world allocation."""

    world_allocation_id: str | None
    role: str | None
    replica: int | None
    rank: int | None
    local_rank: int | None
    cpus: tuple[int, ...] = ()
    memory: int | None = None
    accelerators: Mapping[str, tuple[str | int, ...]] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_runtime_allocation_view(self):
        """Return the matching :mod:`dryml.runtime` allocation view."""

        from dryml.runtime import RuntimeAllocationView

        return RuntimeAllocationView(
            world_allocation_id=self.world_allocation_id,
            role=self.role,
            replica=self.replica,
            rank=self.rank,
            local_rank=self.local_rank,
            cpus=self.cpus,
            memory=self.memory,
            accelerators=self.accelerators,
            env=self.env,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class ProcessAllocation:
    """Actual backend-assigned resources for one process."""

    replica: int
    rank: int
    local_rank: int
    cpus: tuple[int, ...] = ()
    memory: int | None = None
    accelerators: Mapping[str, tuple[str | int, ...]] = field(default_factory=dict)
    devices: Mapping[str, Any] = field(default_factory=dict)
    environment: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProcessAllocation":
        """Build a process allocation from JSON-ready data."""

        if not isinstance(data, Mapping):
            raise WorldSpecValidationError("process allocation must be a mapping")
        unknown = set(data) - {"replica", "rank", "local_rank", "resources", "environment", "env", "metadata"}
        if unknown:
            raise WorldSpecValidationError("process allocation has unknown fields", context={"fields": sorted(unknown)})
        resources = data.get("resources") or {}
        if not isinstance(resources, Mapping):
            raise WorldSpecValidationError("allocation resources must be a mapping")
        accelerators = resources.get("accelerators") or {}
        if not isinstance(accelerators, Mapping):
            raise WorldSpecValidationError("allocation accelerators must be a mapping")
        env = data.get("env") or {}
        return cls(
            replica=_as_nonneg_int("replica", data.get("replica")),
            rank=_as_nonneg_int("rank", data.get("rank")),
            local_rank=_as_nonneg_int("local_rank", data.get("local_rank")),
            cpus=tuple(_as_nonneg_int("cpu", cpu) for cpu in resources.get("cpus", ())),
            memory=parse_byte_size(resources.get("memory")),
            accelerators={str(key): tuple(value) for key, value in accelerators.items()},
            devices=dict(resources.get("devices") or {}),
            environment=data.get("environment"),
            env={str(key): str(value) for key, value in env.items()},
            metadata=dict(data.get("metadata") or {}),
        )

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready process allocation data."""

        resources: dict[str, Any] = {"cpus": list(self.cpus), "accelerators": {key: list(self.accelerators[key]) for key in sorted(self.accelerators)}}
        if self.memory is not None:
            resources["memory"] = canonical_byte_size(self.memory)
        if self.devices:
            resources["devices"] = dict(sorted(self.devices.items()))
        return {
            "replica": self.replica,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "resources": resources,
            "environment": self.environment,
            "env": {key: self.env[key] for key in sorted(self.env)},
            "metadata": dict(sorted(self.metadata.items())),
        }

    def to_runtime_resource_view(self, *, world_allocation_id: str | None = None, role: str | None = None) -> RuntimeResourceView:
        """Return this process allocation as a runtime allocation view input."""

        return RuntimeResourceView(
            world_allocation_id=world_allocation_id,
            role=role,
            replica=self.replica,
            rank=self.rank,
            local_rank=self.local_rank,
            cpus=self.cpus,
            memory=self.memory,
            accelerators=self.accelerators,
            env=self.env,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class WorldAllocation:
    """Actual backend assignment for a requested world."""

    roles: Mapping[str, tuple[ProcessAllocation, ...]]
    backend: Mapping[str, Any] = field(default_factory=lambda: {"kind": "local"})

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "WorldAllocation":
        """Build a world allocation from its spec payload."""

        if not isinstance(data, Mapping):
            raise WorldSpecValidationError("world allocation payload must be a mapping")
        unknown = set(data) - {"backend", "roles"}
        if unknown:
            raise WorldSpecValidationError("world allocation payload has unknown fields", context={"fields": sorted(unknown)})
        roles = data.get("roles")
        if not isinstance(roles, Mapping) or not roles:
            raise WorldSpecValidationError("world allocation roles must be a non-empty mapping")
        backend = data.get("backend") or {"kind": "local"}
        if not isinstance(backend, Mapping):
            raise WorldSpecValidationError("world allocation backend must be a mapping")
        return cls({str(name): tuple(ProcessAllocation.from_data(item) for item in value) for name, value in _iter_role_allocations(roles)}, dict(backend))

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready world allocation payload data."""

        return {"backend": dict(sorted(self.backend.items())), "roles": {name: [item.to_data() for item in self.roles[name]] for name in sorted(self.roles)}}

    def runtime_view(self, role: str, replica: int, *, world_allocation_id: str | None = None):
        """Return a :mod:`dryml.runtime` allocation view for one process."""

        for allocation in self.roles.get(role, ()):
            if allocation.replica == replica:
                return allocation.to_runtime_resource_view(world_allocation_id=world_allocation_id, role=role).to_runtime_allocation_view()
        raise WorldSpecValidationError("allocation role/replica not found", context={"role": role, "replica": replica})


def make_world_allocation_spec(roles: Mapping[str, Any] | WorldAllocation, *, backend: Mapping[str, Any] | None = None, kind: str = "local_allocation", metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a canonical ``world_allocation`` spec envelope."""

    allocation = roles if isinstance(roles, WorldAllocation) else WorldAllocation.from_data({"roles": roles, "backend": backend or {"kind": "local"}})
    return make_spec(family="world_allocation", kind=kind, payload=allocation.to_data(), metadata=metadata)


def validate_world_allocation_spec(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a ``world_allocation`` spec and semantic payload."""

    try:
        validate_spec(spec, family="world_allocation")
        WorldAllocation.from_data(spec["payload"])
    except (SpecValidationError, WorldSpecValidationError, ResourceValidationError) as exc:
        context = getattr(exc, "context", {})
        raise WorldSpecValidationError(str(exc), context=context) from exc
    return spec


def compute_world_allocation_id(spec: Mapping[str, Any]) -> str:
    """Compute the stable ``worldalloc-v1-*`` ID for an allocation spec."""

    validate_world_allocation_spec(spec)
    return compute_spec_id(spec, family="world_allocation")


def attach_world_allocation_id(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *spec* with its canonical allocation ID attached."""

    attached = attach_spec_id(spec, family="world_allocation")
    validate_world_allocation_spec(attached)
    return attached


def _iter_role_allocations(roles: Mapping[str, Any]):
    for name, value in _iter_valid_roles(roles):
        if not isinstance(value, list) or not value:
            raise WorldSpecValidationError("role allocations must be a non-empty list", context={"role": name})
        yield name, value


def _as_nonneg_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorldSpecValidationError(f"{name} must be an integer >= 0", context={"value": value})
    return value


__all__ = [
    "ProcessAllocation",
    "RuntimeResourceView",
    "WorldAllocation",
    "attach_world_allocation_id",
    "compute_world_allocation_id",
    "make_world_allocation_spec",
    "validate_world_allocation_spec",
]
