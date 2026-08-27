"""Exact immutable process-local v1.1 world assignments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dryml.formats import deep_freeze_json, json_ready, make_envelope, semantic_id, validate_envelope

from .errors import WorldSpecValidationError
from .resources import canonical_byte_size, parse_byte_size

_BOUNDS = {"max_depth": 8, "max_nodes": 65536, "max_entries": 4096}


def _nonnegative(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorldSpecValidationError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ProcessAllocation:
    """Exact process-local resource assignment for one role replica."""

    replica: int
    rank: int
    local_rank: int
    cpus: tuple[int, ...] = ()
    memory: int | None = None
    accelerators: Mapping[str, tuple[str | int, ...]] = field(default_factory=dict)
    accelerator_memory: Mapping[str, Mapping[str | int, int]] = field(default_factory=dict)
    devices: Mapping[str, Any] = field(default_factory=dict)
    named: Mapping[str, Any] = field(default_factory=dict)
    environment: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for name, value in (("replica", self.replica), ("rank", self.rank), ("local_rank", self.local_rank)):
            _nonnegative(name, value)
        cpus = tuple(_nonnegative("CPU ID", value) for value in self.cpus)
        if len(cpus) != len(set(cpus)):
            raise WorldSpecValidationError("process allocation CPU IDs must be unique")
        accelerators = {}
        for kind, values in self.accelerators.items():
            if not isinstance(kind, str) or not kind or isinstance(values, str | bytes):
                raise WorldSpecValidationError("accelerator assignments must be string-keyed sequences")
            assigned = tuple(values)
            if len(assigned) != len(set(assigned)):
                raise WorldSpecValidationError("process allocation accelerator IDs must be unique")
            accelerators[kind] = assigned
        memory = parse_byte_size(self.memory)
        accel_memory = {}
        for kind, values in self.accelerator_memory.items():
            if kind not in accelerators or not isinstance(values, Mapping):
                raise WorldSpecValidationError("accelerator memory must refer to assigned accelerator IDs")
            parsed = {device: parse_byte_size(amount) for device, amount in values.items()}
            if set(parsed) - set(accelerators[kind]) or any(value is None or value <= 0 for value in parsed.values()):
                raise WorldSpecValidationError("accelerator memory must be positive for assigned devices")
            accel_memory[kind] = MappingProxyType({device: parsed[device] for device in accelerators[kind] if device in parsed})
        if not isinstance(self.env, Mapping) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.env.items()):
            raise WorldSpecValidationError("allocation environment must be a string mapping")
        object.__setattr__(self, "cpus", cpus)
        object.__setattr__(self, "memory", memory)
        object.__setattr__(self, "accelerators", MappingProxyType({key: accelerators[key] for key in sorted(accelerators)}))
        object.__setattr__(self, "accelerator_memory", MappingProxyType({key: accel_memory[key] for key in sorted(accel_memory)}))
        object.__setattr__(self, "devices", deep_freeze_json(self.devices))
        object.__setattr__(self, "named", deep_freeze_json(self.named))
        object.__setattr__(self, "env", MappingProxyType({key: self.env[key] for key in sorted(self.env)}))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "ProcessAllocation":
        """Decode one closed exact-process payload fragment."""
        fields = {"replica", "rank", "local_rank", "resources", "environment", "env", "metadata"}
        if not isinstance(data, Mapping) or set(data) - fields:
            raise WorldSpecValidationError("process allocation fields are closed")
        resources = data.get("resources", {})
        if not isinstance(resources, Mapping) or set(resources) - {"cpus", "memory", "accelerators", "accelerator_memory", "devices", "named"}:
            raise WorldSpecValidationError("process allocation resources fields are closed")
        accelerators = resources.get("accelerators", {})
        if not isinstance(accelerators, Mapping):
            raise WorldSpecValidationError("allocation accelerators must be a mapping")
        limits = resources.get("accelerator_memory", {})
        parsed_limits = {}
        if not isinstance(limits, Mapping):
            raise WorldSpecValidationError("allocation accelerator_memory must be a mapping")
        for kind, entries in limits.items():
            if isinstance(entries, str | bytes) or not isinstance(entries, Sequence):
                raise WorldSpecValidationError("allocation accelerator_memory entries must be sequences")
            parsed_limits[kind] = {item["device"]: item["memory"] for item in entries if isinstance(item, Mapping) and set(item) == {"device", "memory"}}
            if len(parsed_limits[kind]) != len(entries):
                raise WorldSpecValidationError("allocation accelerator_memory entries require device and memory")
        return cls(
            replica=data.get("replica"),
            rank=data.get("rank"),
            local_rank=data.get("local_rank"),
            cpus=tuple(resources.get("cpus", ())),
            memory=resources.get("memory"),
            accelerators={key: tuple(value) for key, value in accelerators.items()},
            accelerator_memory=parsed_limits,
            devices=resources.get("devices", {}),
            named=resources.get("named", {}),
            environment=data.get("environment"),
            env=data.get("env", {}),
            metadata=data.get("metadata", {}),
        )

    def to_payload(self, *, identifying: bool = False) -> dict[str, Any]:
        """Return exact process payload, omitting non-identifying metadata if requested."""
        resources: dict[str, Any] = {"cpus": list(self.cpus), "accelerators": {key: list(value) for key, value in self.accelerators.items()}}
        if self.memory is not None:
            resources["memory"] = canonical_byte_size(self.memory)
        if self.accelerator_memory:
            resources["accelerator_memory"] = {kind: [{"device": device, "memory": canonical_byte_size(memory)} for device, memory in values.items()] for kind, values in self.accelerator_memory.items()}
        if self.devices:
            resources["devices"] = json_ready(self.devices)
        if self.named:
            resources["named"] = json_ready(self.named)
        result = {"replica": self.replica, "rank": self.rank, "local_rank": self.local_rank, "resources": resources, "environment": self.environment, "env": dict(self.env), "metadata": json_ready(self.metadata)}
        if identifying:
            result.pop("metadata")
        return result


@dataclass(frozen=True, slots=True)
class WorldAllocation:
    """Exact v1.1 local assignments, distinct from requested world shape."""

    roles: Mapping[str, tuple[ProcessAllocation, ...]]
    backend: Mapping[str, Any] = field(default_factory=lambda: {"kind": "local"})
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.roles, Mapping) or not self.roles or len(self.roles) > 4096:
            raise WorldSpecValidationError("allocation roles must be a non-empty bounded mapping")
        roles = {}
        all_ranks: set[int] = set()
        all_cpus: set[int] = set()
        all_accelerators: dict[str, set[str | int]] = {}
        for name in sorted(self.roles):
            values = tuple(self.roles[name])
            if not isinstance(name, str) or not name or not values:
                raise WorldSpecValidationError("allocation roles require non-empty named process sequences")
            processes = tuple(value if isinstance(value, ProcessAllocation) else ProcessAllocation.from_payload(value) for value in values)
            replicas = [item.replica for item in processes]
            if len(replicas) != len(set(replicas)) or any(item.rank in all_ranks for item in processes):
                raise WorldSpecValidationError("allocation replica and rank IDs must be unique")
            for process in processes:
                if all_cpus.intersection(process.cpus):
                    raise WorldSpecValidationError("allocation CPU assignments must be globally disjoint")
                all_cpus.update(process.cpus)
                for kind, device_ids in process.accelerators.items():
                    assigned = all_accelerators.setdefault(kind, set())
                    if assigned.intersection(device_ids):
                        raise WorldSpecValidationError("allocation accelerator IDs cannot be assigned more than once")
                    assigned.update(device_ids)
            all_ranks.update(item.rank for item in processes)
            roles[name] = processes
        if sum(len(value) for value in roles.values()) > 4096:
            raise WorldSpecValidationError("allocation process count exceeds the bounded limit")
        object.__setattr__(self, "roles", MappingProxyType(roles))
        object.__setattr__(self, "backend", deep_freeze_json(self.backend))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))

    @property
    def semantic_id(self) -> str:
        """Return the ID over exact assignments, excluding diagnostic metadata."""
        return semantic_id("worldalloc", "dryml.world_allocation.v1.1", "local_allocation", self._identifying_payload(), **_BOUNDS)

    @property
    def id(self) -> str:
        """Alias for :attr:`semantic_id`."""
        return self.semantic_id

    @classmethod
    def from_payload(cls, data: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> "WorldAllocation":
        """Build exact assignments from a closed allocation payload."""
        if not isinstance(data, Mapping) or set(data) - {"roles", "backend"} or "roles" not in data or not isinstance(data["roles"], Mapping):
            raise WorldSpecValidationError("world allocation payload fields are closed")
        return cls({name: tuple(ProcessAllocation.from_payload(value) for value in values) for name, values in data["roles"].items()}, data.get("backend", {"kind": "local"}), metadata or {})

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "WorldAllocation":
        """Validate and decode a closed exact-allocation v1.1 envelope."""
        raw = dict(data)
        attached = raw.pop("id", None)
        envelope = validate_envelope(raw, schema="dryml.world_allocation.v1.1", kind="local_allocation", prefix="worldalloc", identifying_payload=raw.get("payload", {}), max_bytes=16_777_216, **_BOUNDS)
        value = cls.from_payload(envelope["payload"], metadata=envelope.get("metadata", {}))
        if attached is not None and attached != value.semantic_id:
            raise WorldSpecValidationError("world allocation attached ID does not match payload")
        return value

    def _identifying_payload(self) -> dict[str, Any]:
        return {"roles": {name: [value.to_payload(identifying=True) for value in values] for name, values in self.roles.items()}, "backend": json_ready(self.backend)}

    def to_payload(self) -> dict[str, Any]:
        """Return the complete allocation payload including non-identifying metadata."""
        return {"roles": {name: [value.to_payload() for value in values] for name, values in self.roles.items()}, "backend": json_ready(self.backend)}

    def to_data(self) -> dict[str, Any]:
        """Return this allocation as a closed v1.1 envelope."""
        return make_envelope(schema="dryml.world_allocation.v1.1", kind="local_allocation", prefix="worldalloc", payload=self.to_payload(), semantic_id=self.semantic_id, identifying_payload=self._identifying_payload(), metadata=self.metadata, max_bytes=16_777_216, **_BOUNDS)
