"""Immutable public projections for normalized session configuration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dryml.environments import EnvironmentRequirement
from dryml.formats import canonical_json_bytes, deep_freeze_json, json_ready
from dryml.worlds import ProcessAllocation, ResourceSpec, WorldSpec


@dataclass(frozen=True, slots=True)
class SelectedWorldAllocation:
    """One unambiguously selected process from an exact world allocation."""

    role: str
    process: ProcessAllocation

    def to_data(self) -> dict[str, Any]:
        """Return a canonical, role-qualified exact allocation projection."""

        return {"role": self.role, "process": self.process.to_data()}


@dataclass(frozen=True, slots=True)
class SessionConfiguration:
    """Deeply immutable, effect-free session candidate and display projection."""

    mode: str
    resources: ResourceSpec | None = None
    allocation: SelectedWorldAllocation | None = None
    requested_world: WorldSpec | None = None
    environment: EnvironmentRequirement = field(default_factory=EnvironmentRequirement)
    controls: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "controls", deep_freeze_json(self.controls))
        object.__setattr__(self, "fingerprint", "session-config-v1-" + hashlib.sha256(canonical_json_bytes(self.to_data(include_fingerprint=False))).hexdigest())

    def to_data(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        """Return canonical JSON-ready session data without mutable references."""

        data = {
            "mode": self.mode,
            "resources": None if self.resources is None else self.resources.to_data(),
            "allocation": None if self.allocation is None else self.allocation.to_data(),
            "requested_world": None if self.requested_world is None else self.requested_world.to_data(),
            "environment": self.environment.to_data(),
            "controls": json_ready(self.controls),
        }
        if include_fingerprint:
            data["fingerprint"] = self.fingerprint
        return data


SessionSnapshot = SessionConfiguration


__all__ = ["SelectedWorldAllocation", "SessionConfiguration", "SessionSnapshot"]
