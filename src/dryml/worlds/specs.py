"""Immutable v1.1 world requirements and requested launch shapes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dryml.formats import deep_freeze_json, json_ready, make_envelope, semantic_id, validate_envelope

from .errors import WorldSpecValidationError
from .resources import CountConstraint, ResourceRequirement, ResourceSpec

_ROLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_MAX_ROLES = _MAX_PROCESSES = 4096
_BOUNDS = {"max_depth": 8, "max_nodes": 65536, "max_entries": 4096}


def _roles(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(data, Mapping) or not data or len(data) > _MAX_ROLES:
        raise WorldSpecValidationError("world roles must be a non-empty bounded mapping")
    if any(not isinstance(name, str) or not _ROLE.match(name) for name in data):
        raise WorldSpecValidationError("world role name is invalid")
    return MappingProxyType({name: data[name] for name in sorted(data)})


@dataclass(frozen=True, slots=True)
class RoleRequirement:
    """Hard replica, resource, and topology declarations for a named role."""

    replicas: CountConstraint = field(default_factory=lambda: CountConstraint(1, 1))
    resources: ResourceRequirement = field(default_factory=ResourceRequirement)
    topology: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.topology, Mapping):
            raise WorldSpecValidationError("role topology must be a mapping")
        object.__setattr__(self, "topology", deep_freeze_json(self.topology))

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "RoleRequirement":
        """Decode one closed role-requirement payload fragment."""

        fields = {"replicas", "resources", "topology"}
        if not isinstance(data, Mapping) or set(data) - fields:
            raise WorldSpecValidationError("role requirement fields are closed")
        replicas = data.get("replicas", {"min": 1, "max": 1})
        return cls(replicas if isinstance(replicas, CountConstraint) else CountConstraint.from_data(replicas), ResourceRequirement.from_data(data.get("resources")), data.get("topology", {}))

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible role requirement fragment."""

        return {"replicas": self.replicas.to_data(), "resources": self.resources.to_data(), "topology": json_ready(self.topology)}

    def merge(self, other: "RoleRequirement") -> "RoleRequirement":
        """Intersect resources and replicas while rejecting contradictory topology."""

        topology = dict(self.topology)
        for key, value in other.topology.items():
            if key in topology and topology[key] != value:
                raise WorldSpecValidationError("topology declarations conflict", context={"path": f"topology.{key}"})
            topology[key] = value
        return RoleRequirement(self.replicas.merge(other.replicas), self.resources.merge(other.resources), topology)


@dataclass(frozen=True, slots=True)
class WorldRequirement:
    """Immutable hard v1.1 world requirement graph independent of allocation."""

    roles: Mapping[str, RoleRequirement]
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        raw = _roles(self.roles)
        object.__setattr__(self, "roles", MappingProxyType({name: raw[name] if isinstance(raw[name], RoleRequirement) else RoleRequirement.from_payload(raw[name]) for name in raw}))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))

    @property
    def semantic_id(self) -> str:
        """Return the content-addressed v1.1 requirement ID."""
        return semantic_id("worldreq", "dryml.world_requirement.v1.1", "world_requirement", self.to_payload(), **_BOUNDS)

    @property
    def id(self) -> str:
        """Alias for :attr:`semantic_id`."""
        return self.semantic_id

    @classmethod
    def from_payload(cls, data: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> "WorldRequirement":
        """Build a requirement from its closed identifying payload."""
        if not isinstance(data, Mapping) or set(data) != {"roles"}:
            raise WorldSpecValidationError("world requirement payload fields are closed")
        return cls({name: RoleRequirement.from_payload(value) for name, value in _roles(data["roles"]).items()}, metadata or {})

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "WorldRequirement":
        """Validate and decode a closed world-requirement v1.1 envelope."""
        raw = dict(data)
        attached = raw.pop("id", None)
        envelope = validate_envelope(raw, schema="dryml.world_requirement.v1.1", kind="world_requirement", prefix="worldreq", identifying_payload=raw.get("payload", {}), max_bytes=16_777_216, **_BOUNDS)
        value = cls.from_payload(envelope["payload"], metadata=envelope.get("metadata", {}))
        if attached is not None and attached != value.semantic_id:
            raise WorldSpecValidationError("world requirement attached ID does not match payload")
        return value

    def to_payload(self) -> dict[str, Any]:
        """Return the closed identifying requirement payload."""
        return {"roles": {name: role.to_payload() for name, role in self.roles.items()}}

    def to_data(self) -> dict[str, Any]:
        """Return this requirement as a closed v1.1 envelope."""
        return make_envelope(schema="dryml.world_requirement.v1.1", kind="world_requirement", prefix="worldreq", payload=self.to_payload(), semantic_id=self.semantic_id, identifying_payload=self.to_payload(), metadata=self.metadata, max_bytes=16_777_216, **_BOUNDS)

    def merge(self, other: "WorldRequirement") -> "WorldRequirement":
        """Return the hard-constraint union/intersection by role name."""
        values = dict(self.roles)
        for name, role in other.roles.items():
            values[name] = values[name].merge(role) if name in values else role
        return WorldRequirement(values)


@dataclass(frozen=True, slots=True)
class ProcessSpec:
    """One requested process launch shape; it neither launches nor allocates."""

    resources: ResourceSpec = field(default_factory=ResourceSpec)
    environment: str | None = None
    runtime: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.env, Mapping) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.env.items()):
            raise WorldSpecValidationError("process environment must be a string mapping")
        object.__setattr__(self, "env", MappingProxyType({key: self.env[key] for key in sorted(self.env)}))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "ProcessSpec":
        """Decode one closed requested-process payload fragment."""
        fields = {"resources", "environment", "runtime", "env", "metadata"}
        if not isinstance(data, Mapping) or set(data) - fields:
            raise WorldSpecValidationError("process spec fields are closed")
        return cls(ResourceSpec.from_data(data.get("resources")), data.get("environment"), data.get("runtime"), data.get("env", {}), data.get("metadata", {}))

    def to_payload(self) -> dict[str, Any]:
        """Return the identifying requested-process payload."""
        return {"resources": self.resources.to_data(), "environment": self.environment, "runtime": self.runtime, "env": dict(self.env), "metadata": json_ready(self.metadata)}


@dataclass(frozen=True, slots=True)
class RoleSpec:
    """Requested replica count and process launch shape for one role."""

    replicas: int = 1
    process: ProcessSpec = field(default_factory=ProcessSpec)

    def __post_init__(self) -> None:
        if isinstance(self.replicas, bool) or not isinstance(self.replicas, int) or not 1 <= self.replicas <= _MAX_PROCESSES:
            raise WorldSpecValidationError("role replicas must be a bounded positive integer")

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "RoleSpec":
        """Decode one closed requested-role payload fragment."""
        if not isinstance(data, Mapping) or set(data) - {"replicas", "process"}:
            raise WorldSpecValidationError("role spec fields are closed")
        return cls(data.get("replicas", 1), ProcessSpec.from_payload(data.get("process", {})))

    def to_payload(self) -> dict[str, Any]:
        """Return a canonical requested-role payload fragment."""
        return {"replicas": self.replicas, "process": self.process.to_payload()}


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """Immutable v1.1 requested launch shape, never an exact assignment."""

    roles: Mapping[str, RoleSpec]
    backend: Mapping[str, Any] = field(default_factory=lambda: {"kind": "local", "parameters": {}})
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        raw = _roles(self.roles)
        role_values = {name: raw[name] if isinstance(raw[name], RoleSpec) else RoleSpec.from_payload(raw[name]) for name in raw}
        if sum(role.replicas for role in role_values.values()) > _MAX_PROCESSES:
            raise WorldSpecValidationError("world process count exceeds the bounded limit")
        if not isinstance(self.backend, Mapping):
            raise WorldSpecValidationError("world backend must be a mapping")
        object.__setattr__(self, "roles", MappingProxyType(role_values))
        object.__setattr__(self, "backend", deep_freeze_json(self.backend))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))

    @property
    def semantic_id(self) -> str:
        """Return the content-addressed v1.1 requested-world ID."""
        return semantic_id("world", "dryml.world.v1.1", "world_spec", self.to_payload(), **_BOUNDS)

    @property
    def id(self) -> str:
        """Alias for :attr:`semantic_id`."""
        return self.semantic_id

    @classmethod
    def from_payload(cls, data: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> "WorldSpec":
        """Build a requested world from a closed identifying payload."""
        if not isinstance(data, Mapping) or set(data) - {"roles", "backend"} or "roles" not in data:
            raise WorldSpecValidationError("world spec payload fields are closed")
        return cls({name: RoleSpec.from_payload(value) for name, value in _roles(data["roles"]).items()}, data.get("backend", {"kind": "local", "parameters": {}}), metadata or {})

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "WorldSpec":
        """Validate and decode a closed requested-world v1.1 envelope."""
        raw = dict(data)
        attached = raw.pop("id", None)
        envelope = validate_envelope(raw, schema="dryml.world.v1.1", kind="world_spec", prefix="world", identifying_payload=raw.get("payload", {}), max_bytes=16_777_216, **_BOUNDS)
        value = cls.from_payload(envelope["payload"], metadata=envelope.get("metadata", {}))
        if attached is not None and attached != value.semantic_id:
            raise WorldSpecValidationError("world spec attached ID does not match payload")
        return value

    def to_payload(self) -> dict[str, Any]:
        """Return the closed identifying requested-world payload."""
        return {"roles": {name: role.to_payload() for name, role in self.roles.items()}, "backend": json_ready(self.backend)}

    def to_data(self) -> dict[str, Any]:
        """Return this requested world as a closed v1.1 envelope."""
        return make_envelope(schema="dryml.world.v1.1", kind="world_spec", prefix="world", payload=self.to_payload(), semantic_id=self.semantic_id, identifying_payload=self.to_payload(), metadata=self.metadata, max_bytes=16_777_216, **_BOUNDS)
