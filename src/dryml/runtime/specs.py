"""Closed canonical v1.1 runtime declarations and semantic identity."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dryml.formats import deep_freeze_json, json_ready, make_envelope, semantic_id, validate_envelope

from .errors import RuntimeSpecError
from .modes import RuntimeMode

_BOUNDS = {"max_depth": 8, "max_nodes": 1024, "max_entries": 64}
_WORLD_ID = re.compile(r"^worldalloc-v1\.1-[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RuntimeContextSpec:
    """Immutable process-control declaration independent of publication.

    Metadata is diagnostic-only and excluded from the runtime semantic ID.
    """

    mode: RuntimeMode = RuntimeMode.NONE
    visibility: Mapping[str, Any] = field(default_factory=dict)
    framework: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    limits: Mapping[str, Any] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    world_allocation_id: str | None = None

    def __post_init__(self) -> None:
        """Normalize mappings and reject malformed allocation associations."""
        object.__setattr__(self, "mode", RuntimeMode.coerce(self.mode))
        if not all(isinstance(value, Mapping) for value in (self.visibility, self.framework, self.limits, self.env, self.metadata)):
            raise RuntimeSpecError("runtime visibility, framework, limits, env, and metadata must be mappings")
        if any(not isinstance(name, str) or not isinstance(value, Mapping) for name, value in self.framework.items()):
            raise RuntimeSpecError("runtime framework controls must be named mappings")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.env.items()):
            raise RuntimeSpecError("runtime environment controls must be strings")
        if self.world_allocation_id is not None and (not isinstance(self.world_allocation_id, str) or _WORLD_ID.fullmatch(self.world_allocation_id) is None):
            raise RuntimeSpecError("runtime world allocation association must be a worldalloc-v1.1 ID")
        object.__setattr__(self, "visibility", deep_freeze_json(self.visibility, **_BOUNDS))
        object.__setattr__(self, "framework", MappingProxyType({key: deep_freeze_json(self.framework[key], **_BOUNDS) for key in sorted(self.framework)}))
        object.__setattr__(self, "limits", deep_freeze_json(self.limits, **_BOUNDS))
        object.__setattr__(self, "env", MappingProxyType({key: self.env[key] for key in sorted(self.env)}))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata, **_BOUNDS))

    @property
    def frameworks(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the framework mapping under the historical plural spelling."""
        return self.framework

    @property
    def semantic_id(self) -> str:
        """Return the ``runtime-v1.1`` ID excluding envelope metadata."""
        return semantic_id("runtime", "dryml.runtime.v1.1", "runtime_context", self._identifying_payload(), **_BOUNDS)

    @property
    def id(self) -> str:
        """Return the semantic ID alias used by other v1.1 values."""
        return self.semantic_id

    def _identifying_payload(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "device_visibility": json_ready(self.visibility), "frameworks": json_ready(self.framework), "limits": json_ready(self.limits), "env": dict(self.env), "world_allocation_id": self.world_allocation_id}

    def to_payload(self) -> dict[str, Any]:
        """Return the complete closed payload including non-identifying metadata."""
        return {**self._identifying_payload(), "metadata": json_ready(self.metadata)}

    def to_data(self) -> dict[str, Any]:
        """Return a self-validating ``dryml.runtime.v1.1`` envelope."""
        return make_envelope(schema="dryml.runtime.v1.1", kind="runtime_context", prefix="runtime", payload=self.to_payload(), identifying_payload=self._identifying_payload(), semantic_id=self.semantic_id, metadata=self.metadata, **_BOUNDS)

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RuntimeContextSpec":
        """Decode one closed self-validating runtime v1.1 envelope."""
        if not isinstance(data, Mapping):
            raise RuntimeSpecError("runtime data must be a mapping")
        if "contract_version" not in data:
            raise RuntimeSpecError("runtime from_data requires a v1.1 envelope")
        raw = dict(data)
        attached = raw.pop("id", None)
        raw_payload = raw.get("payload", {})
        identifying = {key: value for key, value in raw_payload.items() if key != "metadata"}
        envelope = validate_envelope(raw, schema="dryml.runtime.v1.1", kind="runtime_context", prefix="runtime", identifying_payload=identifying, **_BOUNDS)
        payload = envelope["payload"]
        fields = {"mode", "device_visibility", "frameworks", "limits", "env", "metadata", "world_allocation_id"}
        if set(payload) != fields:
            raise RuntimeSpecError("runtime payload fields are closed")
        value = cls(payload["mode"], payload["device_visibility"], payload["frameworks"], payload["limits"], payload["env"], payload["metadata"], payload["world_allocation_id"])
        if attached is not None and attached != value.semantic_id:
            raise RuntimeSpecError("runtime attached ID does not match payload")
        return value


__all__ = ["RuntimeContextSpec"]
