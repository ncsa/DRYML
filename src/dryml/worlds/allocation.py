"""Actual backend allocations for DRYML worlds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dryml.records import attach_spec_id, compute_spec_id, make_spec, validate_spec
from dryml.records.errors import SpecValidationError
from dryml.formats import deep_freeze_json

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
    accelerator_memory: Mapping[str, Mapping[str | int, int]] = field(default_factory=dict, kw_only=True)
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
            accelerator_memory=self.accelerator_memory,
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
    accelerator_memory: Mapping[str, Mapping[str | int, int]] = field(default_factory=dict, kw_only=True)
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
        unknown_resources = set(resources) - {"cpus", "memory", "accelerators", "accelerator_memory", "devices"}
        if unknown_resources:
            raise WorldSpecValidationError("allocation resources has unknown fields", context={"fields": sorted(unknown_resources)})
        accelerators = resources.get("accelerators") or {}
        if not isinstance(accelerators, Mapping):
            raise WorldSpecValidationError("allocation accelerators must be a mapping")
        env = data.get("env") or {}
        devices = resources.get("devices") or {}
        metadata = data.get("metadata") or {}
        for name, value in (("env", env), ("devices", devices), ("metadata", metadata)):
            if not isinstance(value, Mapping):
                raise WorldSpecValidationError(f"process allocation {name} must be a mapping")
        try:
            normalized_accelerators = {str(key): tuple(_require_sequence(f"accelerators.{key}", value)) for key, value in accelerators.items()}
            return cls(
                replica=_as_nonneg_int("replica", data.get("replica")),
                rank=_as_nonneg_int("rank", data.get("rank")),
                local_rank=_as_nonneg_int("local_rank", data.get("local_rank")),
                cpus=tuple(_as_nonneg_int("cpu", cpu) for cpu in resources.get("cpus", ())),
                memory=parse_byte_size(resources.get("memory")),
                accelerators=normalized_accelerators,
                accelerator_memory=_accelerator_memory_from_data(resources.get("accelerator_memory") or {}, normalized_accelerators),
                devices=dict(devices),
                environment=data.get("environment"),
                env={str(key): str(value) for key, value in env.items()},
                metadata=dict(metadata),
            )
        except ResourceValidationError as exc:
            raise WorldSpecValidationError(str(exc), context=exc.context) from exc

    def __post_init__(self) -> None:
        object.__setattr__(self, "accelerators", MappingProxyType({key: tuple(value) for key, value in sorted(self.accelerators.items())}))
        object.__setattr__(self, "accelerator_memory", MappingProxyType({key: MappingProxyType(dict(value)) for key, value in sorted(self.accelerator_memory.items())}))
        object.__setattr__(self, "devices", deep_freeze_json(self.devices))
        object.__setattr__(self, "env", MappingProxyType(dict(sorted(self.env.items()))))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready process allocation data."""

        resources: dict[str, Any] = {"cpus": list(self.cpus), "accelerators": {key: list(self.accelerators[key]) for key in sorted(self.accelerators)}}
        if self.memory is not None:
            resources["memory"] = canonical_byte_size(self.memory)
        if self.accelerator_memory:
            resources["accelerator_memory"] = {
                key: [
                    {"device": device, "memory": canonical_byte_size(self.accelerator_memory[key][device])}
                    for device in self.accelerators[key]
                    if device in self.accelerator_memory[key]
                ]
                for key in sorted(self.accelerator_memory)
            }
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
            accelerator_memory=self.accelerator_memory,
            env=self.env,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class WorldAllocation:
    """Actual backend assignment for a requested world."""

    roles: Mapping[str, tuple[ProcessAllocation, ...]]
    backend: Mapping[str, Any] = field(default_factory=lambda: {"kind": "local"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", MappingProxyType({name: tuple(items) for name, items in sorted(self.roles.items())}))
        object.__setattr__(self, "backend", deep_freeze_json(self.backend))

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


def _require_sequence(path: str, value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise WorldSpecValidationError("allocation sequence must be a non-string sequence", context={"path": path, "type": type(value).__name__})
    return tuple(value)


def _accelerator_memory_from_data(data: Any, accelerators: Mapping[str, tuple[str | int, ...]]) -> dict[str, Mapping[str | int, int]]:
    """Validate exact per-device accelerator-memory allocation limits."""

    if not isinstance(data, Mapping):
        raise WorldSpecValidationError("allocation accelerator_memory must be a mapping")
    result: dict[str, Mapping[str | int, int]] = {}
    for kind, entries in data.items():
        if kind not in accelerators:
            raise WorldSpecValidationError("accelerator_memory names an unassigned accelerator kind", context={"accelerator": kind})
        if isinstance(entries, (str, bytes)) or not hasattr(entries, "__iter__"):
            raise WorldSpecValidationError("allocation accelerator_memory must be a non-string sequence", context={"accelerator": kind})
        limits: dict[str | int, int] = {}
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {"device", "memory"}:
                raise WorldSpecValidationError("allocation accelerator_memory entries require device and memory")
            device = entry["device"]
            if device not in accelerators[kind] or device in limits:
                raise WorldSpecValidationError("allocation accelerator_memory must refer to unique assigned devices", context={"accelerator": kind, "device": device})
            memory = parse_byte_size(entry["memory"])
            if memory is None or memory <= 0:
                raise WorldSpecValidationError("allocation accelerator_memory limits must be positive", context={"accelerator": kind})
            limits[device] = memory
        result[str(kind)] = limits
    return result


__all__ = [
    "ProcessAllocation",
    "RuntimeResourceView",
    "WorldAllocation",
    "attach_world_allocation_id",
    "compute_world_allocation_id",
    "make_world_allocation_spec",
    "validate_world_allocation_spec",
]
