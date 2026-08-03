"""Canonical runtime context specs and stable IDs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dryml.records import attach_spec_id, compute_spec_id, make_spec, validate_spec
from dryml.records.errors import SpecValidationError

from .errors import RuntimeSpecError
from .modes import RuntimeMode


@dataclass(frozen=True, slots=True)
class RuntimeContextSpec:
    """Process-local runtime setup, separate from resource allocation.

    ``RuntimeMode.NONE`` serializes explicitly as ``"none"``. Omitted legacy
    mode data remains ``orchestrator`` so older payloads are never reinterpreted
    as the new no-role state.
    """

    mode: RuntimeMode = RuntimeMode.ORCHESTRATOR
    device_visibility: Mapping[str, Any] = field(default_factory=dict)
    frameworks: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    limits: Mapping[str, Any] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    world_allocation_id: str | None = None

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RuntimeContextSpec":
        """Build a runtime context spec from JSON-ready payload data."""

        if not isinstance(data, Mapping):
            raise RuntimeSpecError("runtime payload must be a mapping")
        unknown = set(data) - {"mode", "device_visibility", "frameworks", "limits", "env", "metadata", "world_allocation_id"}
        if unknown:
            raise RuntimeSpecError("runtime payload has unknown fields", context={"fields": sorted(unknown)})
        device_visibility = data.get("device_visibility") or {}
        frameworks = data.get("frameworks") or {}
        limits = data.get("limits") or {}
        env = data.get("env") or {}
        for name, value in (("device_visibility", device_visibility), ("frameworks", frameworks), ("limits", limits), ("env", env)):
            if not isinstance(value, Mapping):
                raise RuntimeSpecError(f"{name} must be a mapping")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise RuntimeSpecError("metadata must be a mapping")
        for key, value in frameworks.items():
            if not isinstance(value, Mapping):
                raise RuntimeSpecError("framework config must be a mapping", context={"framework": key, "type": type(value).__name__})
        world_allocation_id = data.get("world_allocation_id")
        if world_allocation_id is not None and not isinstance(world_allocation_id, str):
            raise RuntimeSpecError("world_allocation_id must be a string")
        return cls(
            mode=RuntimeMode.coerce(data.get("mode", RuntimeMode.ORCHESTRATOR.value)),
            device_visibility=dict(device_visibility),
            frameworks={str(key): dict(value or {}) for key, value in frameworks.items()},
            limits=dict(limits),
            env={str(key): str(value) for key, value in env.items()},
            metadata=dict(metadata),
            world_allocation_id=world_allocation_id,
        )

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready runtime payload data."""

        data = {
            "mode": self.mode.value,
            "device_visibility": dict(sorted(self.device_visibility.items())),
            "frameworks": {key: dict(sorted(self.frameworks[key].items())) for key in sorted(self.frameworks)},
            "limits": dict(sorted(self.limits.items())),
            "env": {key: self.env[key] for key in sorted(self.env)},
            "metadata": dict(sorted(self.metadata.items())),
        }
        if self.world_allocation_id is not None:
            data["world_allocation_id"] = self.world_allocation_id
        return data


def make_runtime_spec(*, mode: RuntimeMode | str = RuntimeMode.ORCHESTRATOR, device_visibility: Mapping[str, Any] | None = None, frameworks: Mapping[str, Mapping[str, Any]] | None = None, limits: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None, metadata: Mapping[str, Any] | None = None, world_allocation_id: str | None = None, kind: str = "runtime_context") -> dict[str, Any]:
    """Build a canonical ``runtime`` spec envelope."""

    payload = {
        "mode": RuntimeMode.coerce(mode).value,
        "device_visibility": device_visibility or {},
        "frameworks": frameworks or {},
        "limits": limits or {},
        "env": env or {},
        "metadata": metadata or {},
    }
    if world_allocation_id is not None:
        payload["world_allocation_id"] = world_allocation_id
    runtime_spec = RuntimeContextSpec.from_data(payload)
    return make_spec(family="runtime", kind=kind, payload=runtime_spec.to_data())


def validate_runtime_spec(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a runtime spec envelope and semantic payload."""

    try:
        validate_spec(spec, family="runtime")
        RuntimeContextSpec.from_data(spec["payload"])
    except (SpecValidationError, RuntimeSpecError, ValueError) as exc:
        context = getattr(exc, "context", {})
        raise RuntimeSpecError(str(exc), context=context) from exc
    return spec


def compute_runtime_id(spec: Mapping[str, Any]) -> str:
    """Compute the stable ``runtime-v1-*`` ID for a runtime spec."""

    validate_runtime_spec(spec)
    return compute_spec_id(spec, family="runtime")


def attach_runtime_id(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *spec* with its canonical runtime ID attached."""

    attached = attach_spec_id(spec, family="runtime")
    validate_runtime_spec(attached)
    return attached


__all__ = ["RuntimeContextSpec", "attach_runtime_id", "compute_runtime_id", "make_runtime_spec", "validate_runtime_spec"]
