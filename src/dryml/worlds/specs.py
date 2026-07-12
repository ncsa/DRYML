"""Canonical world requirement and requested world specs."""

from __future__ import annotations

import re
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dryml.records import attach_spec_id, compute_spec_id, make_spec, validate_spec
from dryml.records.errors import SpecValidationError

from .errors import ResourceValidationError, WorldSpecValidationError
from .resources import CountConstraint, ResourceRequirement, ResourceSpec

_ROLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_MAX_ROLE_NAME = 4096
_MAX_TOPOLOGY_DEPTH = 8
_MAX_TOPOLOGY_ITEMS = 64
_MAX_TOPOLOGY_STRING = 4096
_MAX_TOPOLOGY_NODES = 1024
_MAX_TOPOLOGY_INTEGER_BITS = 4096


def _validated_process_env(value: Any) -> dict[str, str]:
    """Return OS-launchable process environment overrides without coercion."""

    if not isinstance(value, Mapping):
        raise WorldSpecValidationError("process env must be a mapping")
    if any(
        not isinstance(key, str)
        or not key
        or "=" in key
        or "\x00" in key
        or not isinstance(item, str)
        or "\x00" in item
        for key, item in value.items()
    ):
        raise WorldSpecValidationError(
            "process env keys must be non-empty strings and values must be strings"
        )
    return dict(value)


@dataclass(frozen=True, slots=True)
class RoleRequirement:
    """Hard replica, resource, and topology constraints for a world role."""

    replicas: CountConstraint = field(default_factory=lambda: CountConstraint(min=1, max=1))
    resources: ResourceRequirement = field(default_factory=ResourceRequirement)
    topology: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze topology as deterministic JSON-safe data for direct construction."""

        if not isinstance(self.topology, Mapping):
            raise WorldSpecValidationError("role topology must be a mapping", context={"type": type(self.topology).__name__})
        object.__setattr__(self, "topology", _normalize_topology(self.topology))

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RoleRequirement":
        """Build a role requirement from JSON-ready data."""

        if not isinstance(data, Mapping):
            raise WorldSpecValidationError("role requirement must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"replicas", "resources", "topology"}
        if unknown:
            raise WorldSpecValidationError("role requirement has unknown fields", context={"fields": sorted(unknown)})
        topology = data["topology"] if "topology" in data else {}
        if not isinstance(topology, Mapping):
            raise WorldSpecValidationError("role topology must be a mapping", context={"type": type(topology).__name__})
        try:
            requirement = cls(
                replicas=CountConstraint.from_data(data.get("replicas", {"exact": 1}), path="replicas"),
                resources=ResourceRequirement.from_data(data.get("resources") or {}),
                topology=topology,
            )
        except ResourceValidationError as exc:
            raise WorldSpecValidationError(str(exc), context=exc.context) from exc
        if requirement.topology.get("single_process") is not None and not isinstance(requirement.topology["single_process"], bool):
            raise WorldSpecValidationError("topology single_process must be a boolean or null")
        return requirement

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready role requirement data."""

        return {
            "replicas": self.replicas.to_data(),
            "resources": self.resources.to_data(),
            "topology": _topology_to_data(self.topology),
        }

    def merge(self, other: "RoleRequirement", *, path: str) -> "RoleRequirement":
        """Merge this role requirement with another requirement."""

        topology = dict(self.topology)
        for key, value in other.topology.items():
            if key in topology and topology[key] != value and topology[key] is not None and value is not None:
                raise WorldSpecValidationError("topology merge conflict", context={"path": f"{path}.topology.{key}", "left": topology[key], "right": value})
            topology[key] = value if value is not None else topology.get(key)
        try:
            return RoleRequirement(
                replicas=self.replicas.merge(other.replicas, path=f"{path}.replicas"),
                resources=self.resources.merge(other.resources, path=f"{path}.resources"),
                topology=topology,
            )
        except ResourceValidationError as exc:
            raise WorldSpecValidationError(str(exc), context=exc.context) from exc


@dataclass(frozen=True, slots=True)
class WorldRequirement:
    """Hard topology/resource requirement for an execution world."""

    roles: Mapping[str, RoleRequirement]

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "WorldRequirement":
        """Build a world requirement from its spec payload."""

        if not isinstance(data, Mapping):
            raise WorldSpecValidationError("world requirement payload must be a mapping")
        if set(data) != {"roles"}:
            raise WorldSpecValidationError("world requirement payload must contain only roles", context={"fields": sorted(data)})
        roles = data.get("roles")
        if not isinstance(roles, Mapping) or not roles:
            raise WorldSpecValidationError("world requirement roles must be a non-empty mapping")
        return cls({str(name): RoleRequirement.from_data(value) for name, value in _iter_valid_roles(roles)})

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready world requirement payload data."""

        return {"roles": {name: self.roles[name].to_data() for name in sorted(self.roles)}}

    def merge(self, other: "WorldRequirement") -> "WorldRequirement":
        """Merge this hard requirement with another hard requirement."""

        roles = dict(self.roles)
        for name, role in other.roles.items():
            roles[name] = roles[name].merge(role, path=f"roles.{name}") if name in roles else role
        return WorldRequirement(roles)


@dataclass(frozen=True, slots=True)
class ProcessSpec:
    """Requested/default process shape for one role replica."""

    resources: ResourceSpec = field(default_factory=ResourceSpec)
    environment: str | None = None
    runtime: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Apply launch-safe environment validation to direct construction."""

        object.__setattr__(self, "env", _validated_process_env(self.env))

    @classmethod
    def from_data(cls, data: Mapping[str, Any] | None) -> "ProcessSpec":
        """Build a requested process spec from JSON-ready data."""

        data = data or {}
        if not isinstance(data, Mapping):
            raise WorldSpecValidationError("process spec must be a mapping")
        unknown = set(data) - {"resources", "environment", "runtime", "env", "metadata"}
        if unknown:
            raise WorldSpecValidationError("process spec has unknown fields", context={"fields": sorted(unknown)})
        env = data.get("env") or {}
        return cls(
            resources=ResourceSpec.from_data(data.get("resources") or {}),
            environment=data.get("environment"),
            runtime=data.get("runtime"),
            env=_validated_process_env(env),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready process spec data."""

        return {
            "resources": self.resources.to_data(),
            "environment": self.environment,
            "runtime": self.runtime,
            "env": {key: self.env[key] for key in sorted(self.env)},
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True, slots=True)
class RoleSpec:
    """Requested/default launch shape for a world role."""

    replicas: int
    process: ProcessSpec = field(default_factory=ProcessSpec)

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RoleSpec":
        """Build a role spec from JSON-ready data."""

        if not isinstance(data, Mapping):
            raise WorldSpecValidationError("role spec must be a mapping")
        unknown = set(data) - {"replicas", "process"}
        if unknown:
            raise WorldSpecValidationError("role spec has unknown fields", context={"fields": sorted(unknown)})
        replicas = data.get("replicas", 1)
        if isinstance(replicas, int) and not isinstance(replicas, bool) and replicas.bit_length() > 4096:
            raise WorldSpecValidationError("role replicas exceed the bounded integer limit")
        if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 0:
            raise WorldSpecValidationError("role replicas must be an integer >= 0", context={"replicas": replicas})
        return cls(replicas=replicas, process=ProcessSpec.from_data(data.get("process") or {}))

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready role spec data."""

        return {"replicas": self.replicas, "process": self.process.to_data()}


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """Requested/default execution shape, separate from actual allocation."""

    roles: Mapping[str, RoleSpec]
    backend: Mapping[str, Any] = field(default_factory=lambda: {"kind": "local", "parameters": {}})

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "WorldSpec":
        """Build a requested world spec from its spec payload."""

        if not isinstance(data, Mapping):
            raise WorldSpecValidationError("world spec payload must be a mapping")
        unknown = set(data) - {"roles", "backend"}
        if unknown:
            raise WorldSpecValidationError("world spec payload has unknown fields", context={"fields": sorted(unknown)})
        roles = data.get("roles")
        if not isinstance(roles, Mapping) or not roles:
            raise WorldSpecValidationError("world spec roles must be a non-empty mapping")
        backend = data.get("backend") or {"kind": "local", "parameters": {}}
        if not isinstance(backend, Mapping):
            raise WorldSpecValidationError("world backend must be a mapping")
        return cls({str(name): RoleSpec.from_data(value) for name, value in _iter_valid_roles(roles)}, dict(backend))

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready requested world payload data."""

        return {"roles": {name: self.roles[name].to_data() for name in sorted(self.roles)}, "backend": dict(sorted(self.backend.items()))}


def make_world_requirement_spec(roles: Mapping[str, Any] | WorldRequirement, *, kind: str = "world_requirement", metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a canonical ``world_requirement`` spec envelope."""

    req = roles if isinstance(roles, WorldRequirement) else WorldRequirement.from_data({"roles": roles})
    return make_spec(family="world_requirement", kind=kind, payload=req.to_data(), metadata=metadata)


def validate_world_requirement_spec(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a ``world_requirement`` spec and its semantic payload."""

    try:
        validate_spec(spec, family="world_requirement")
        WorldRequirement.from_data(spec["payload"])
    except (SpecValidationError, WorldSpecValidationError, ResourceValidationError) as exc:
        context = getattr(exc, "context", {})
        raise WorldSpecValidationError(str(exc), context=context) from exc
    return spec


def compute_world_requirement_id(spec: Mapping[str, Any]) -> str:
    """Compute the stable ``worldreq-v1-*`` ID for a requirement spec."""

    validate_world_requirement_spec(spec)
    return compute_spec_id(spec, family="world_requirement")


def attach_world_requirement_id(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *spec* with its canonical requirement ID attached."""

    attached = attach_spec_id(spec, family="world_requirement")
    validate_world_requirement_spec(attached)
    return attached


def make_world_spec(roles: Mapping[str, Any] | WorldSpec, *, backend: Mapping[str, Any] | None = None, kind: str = "world_spec", metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a canonical requested ``world`` spec envelope."""

    world = roles if isinstance(roles, WorldSpec) else WorldSpec.from_data({"roles": roles, "backend": backend or {"kind": "local", "parameters": {}}})
    return make_spec(family="world", kind=kind, payload=world.to_data(), metadata=metadata)


def validate_world_spec(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a requested ``world`` spec and its semantic payload."""

    try:
        validate_spec(spec, family="world")
        WorldSpec.from_data(spec["payload"])
    except (SpecValidationError, WorldSpecValidationError, ResourceValidationError) as exc:
        context = getattr(exc, "context", {})
        raise WorldSpecValidationError(str(exc), context=context) from exc
    return spec


def compute_world_id(spec: Mapping[str, Any]) -> str:
    """Compute the stable ``world-v1-*`` ID for a requested world spec."""

    validate_world_spec(spec)
    return compute_spec_id(spec, family="world")


def attach_world_id(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *spec* with its canonical world ID attached."""

    attached = attach_spec_id(spec, family="world")
    validate_world_spec(attached)
    return attached


def _iter_valid_roles(roles: Mapping[str, Any]):
    for name, value in roles.items():
        if not isinstance(name, str) or not _ROLE_RE.match(name):
            raise WorldSpecValidationError("invalid role name", context={"role": name})
        if len(name) > _MAX_ROLE_NAME:
            raise WorldSpecValidationError("role name exceeds the bounded limit", context={"role": name[:64]})
        yield name, value


def _normalize_topology(
    value: Any,
    *,
    path: str = "topology",
    ancestors: set[int] | None = None,
    depth: int = 0,
    budget: list[int] | None = None,
) -> Any:
    """Return a bounded frozen JSON-safe topology without arbitrary coercion."""

    if depth > _MAX_TOPOLOGY_DEPTH:
        raise WorldSpecValidationError(
            "role topology nesting exceeds the bounded limit",
            context={"path": path, "limit": _MAX_TOPOLOGY_DEPTH},
        )
    budget = [_MAX_TOPOLOGY_NODES] if budget is None else budget
    if budget[0] <= 0:
        raise WorldSpecValidationError(
            "role topology exceeds the aggregate bounded limit",
            context={"path": path, "limit": _MAX_TOPOLOGY_NODES},
        )
    budget[0] -= 1
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > _MAX_TOPOLOGY_INTEGER_BITS:
            raise WorldSpecValidationError("role topology integer exceeds the bounded limit", context={"path": path})
        if isinstance(value, str) and len(value) > _MAX_TOPOLOGY_STRING:
            raise WorldSpecValidationError("role topology string exceeds the bounded limit", context={"path": path})
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorldSpecValidationError("role topology floats must be finite", context={"path": path})
        return value
    ancestors = set() if ancestors is None else ancestors
    if isinstance(value, Mapping):
        if len(value) > _MAX_TOPOLOGY_ITEMS:
            raise WorldSpecValidationError(
                "role topology mapping exceeds the bounded limit",
                context={"path": path, "limit": _MAX_TOPOLOGY_ITEMS},
            )
        if path == "topology" and any(not isinstance(key, str) or not key for key in value):
            raise WorldSpecValidationError("role topology names must be non-empty strings")
        if any(not isinstance(key, str) for key in value):
            raise WorldSpecValidationError("role topology object keys must be strings", context={"path": path})
        if any(len(key) > _MAX_TOPOLOGY_STRING for key in value):
            raise WorldSpecValidationError("role topology key exceeds the bounded limit", context={"path": path})
        identity = id(value)
        if identity in ancestors:
            raise WorldSpecValidationError("role topology must not contain cycles", context={"path": path})
        ancestors.add(identity)
        try:
            return MappingProxyType(
                {
                    key: _normalize_topology(
                        item,
                        path=f"{path}.{key}",
                        ancestors=ancestors,
                        depth=depth + 1,
                        budget=budget,
                    )
                    for key, item in sorted(value.items())
                }
            )
        finally:
            ancestors.remove(identity)
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_TOPOLOGY_ITEMS:
            raise WorldSpecValidationError(
                "role topology sequence exceeds the bounded limit",
                context={"path": path, "limit": _MAX_TOPOLOGY_ITEMS},
            )
        identity = id(value)
        if identity in ancestors:
            raise WorldSpecValidationError("role topology must not contain cycles", context={"path": path})
        ancestors.add(identity)
        try:
            return tuple(
                _normalize_topology(
                    item,
                    path=f"{path}[{index}]",
                    ancestors=ancestors,
                    depth=depth + 1,
                    budget=budget,
                )
                for index, item in enumerate(value)
            )
        finally:
            ancestors.remove(identity)
    raise WorldSpecValidationError(
        "role topology must be JSON-compatible",
        context={"path": path, "type": type(value).__name__},
    )


def _topology_to_data(value: Any) -> Any:
    """Thaw a normalized topology into standard JSON-ready containers."""

    if isinstance(value, Mapping):
        return {key: _topology_to_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_topology_to_data(item) for item in value]
    return value


__all__ = [
    "ProcessSpec",
    "RoleRequirement",
    "RoleSpec",
    "WorldRequirement",
    "WorldSpec",
    "attach_world_id",
    "attach_world_requirement_id",
    "compute_world_id",
    "compute_world_requirement_id",
    "make_world_requirement_spec",
    "make_world_spec",
    "validate_world_requirement_spec",
    "validate_world_spec",
]
