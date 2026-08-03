"""Immutable public projections for normalized session configuration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dryml.environments import EnvironmentRequirement, EnvironmentSpec
from dryml.formats import canonical_json_bytes, deep_freeze_json, json_ready
from dryml.runtime import RequirementAxes
from dryml.worlds import LocalResourceInventory, ProcessAllocation, ResourceSpec, WorldSpec


def _default_requirement_axes(mode: str) -> RequirementAxes:
    """Return the requirement-axis default for one public session mode."""

    return RequirementAxes.all() if mode in {"managed", "orchestrator"} else RequirementAxes()


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
    """Deeply immutable session candidate and display projection.

    The requirement-axis mask affects compatibility collection only. Omitted
    masks default to empty for Python and all axes for managed/orchestrator.
    """

    mode: str
    resources: ResourceSpec | None = None
    allocation: SelectedWorldAllocation | None = None
    requested_environment: EnvironmentSpec | None = None
    requested_world: WorldSpec | None = None
    environment: EnvironmentRequirement = field(default_factory=EnvironmentRequirement)
    controls: Mapping[str, Any] = field(default_factory=dict)
    requirement_axes: RequirementAxes | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        """Freeze display controls and derive this candidate's semantic fingerprint.

        Returns:
            None.
        """
        axes = self.requirement_axes
        if axes is None:
            axes = _default_requirement_axes(self.mode)
        if not isinstance(axes, RequirementAxes):
            raise TypeError("session requirement_axes must be a RequirementAxes value")
        object.__setattr__(self, "requirement_axes", axes)
        object.__setattr__(self, "controls", deep_freeze_json(self.controls))
        object.__setattr__(self, "fingerprint", "session-config-v1-" + hashlib.sha256(canonical_json_bytes(self.to_data(include_fingerprint=False))).hexdigest())

    def to_data(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        """Return canonical JSON-ready session data without mutable references."""

        data = {
            "mode": self.mode,
            "resources": None if self.resources is None else self.resources.to_data(),
            "allocation": None if self.allocation is None else self.allocation.to_data(),
            "requested_environment": None if self.requested_environment is None else self.requested_environment.to_data(),
            "requested_world": None if self.requested_world is None else self.requested_world.to_data(),
            "environment": self.environment.to_data(),
            "controls": json_ready(self.controls),
            "requirement_axes": self.requirement_axes.to_data(),
        }
        if include_fingerprint:
            data["fingerprint"] = self.fingerprint
        return data


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Immutable public projection of one published process generation.

    ``requirement_axes`` is the effective runtime compatibility mask and remains
    separate from lifecycle status and allocation controls.
    """

    mode: str
    resources: ResourceSpec | None
    allocation: SelectedWorldAllocation | None
    requested_environment: EnvironmentSpec | None
    requested_world: WorldSpec | None
    environment: EnvironmentRequirement
    controls: Mapping[str, Any]
    statuses: Mapping[str, str]
    runtime: Any
    generation: int
    health: str = "healthy"
    inventory: LocalResourceInventory | None = None
    requirement_axes: RequirementAxes = field(default_factory=RequirementAxes)
    selected_environment: EnvironmentSpec | None = None
    selected_world: WorldSpec | None = None
    selected_runtime: Any | None = None
    compatibility_policy: str | None = None
    compatibility_axes: RequirementAxes | None = None

    def __post_init__(self) -> None:
        """Freeze nested public control and status projections.

        Returns:
            None.
        """
        object.__setattr__(self, "controls", deep_freeze_json(self.controls))
        object.__setattr__(self, "statuses", deep_freeze_json(self.statuses))
        if not isinstance(self.requirement_axes, RequirementAxes):
            raise TypeError("snapshot requirement_axes must be a RequirementAxes value")
        if self.compatibility_axes is not None and not isinstance(self.compatibility_axes, RequirementAxes):
            raise TypeError("snapshot compatibility_axes must be a RequirementAxes value")

    def to_data(self) -> dict[str, Any]:
        """Return a bounded JSON-ready display projection."""

        return {
            "mode": self.mode,
            "resources": None if self.resources is None else self.resources.to_data(),
            "allocation": None if self.allocation is None else self.allocation.to_data(),
            "requested_environment": None if self.requested_environment is None else self.requested_environment.to_data(),
            "requested_world": None if self.requested_world is None else self.requested_world.to_data(),
            "environment": self.environment.to_data(),
            "controls": json_ready(self.controls),
            "statuses": json_ready(self.statuses),
            "requirement_axes": self.requirement_axes.to_data(),
            "generation": self.generation,
            "health": self.health,
            "inventory": None if self.inventory is None else self.inventory.summary(),
            "selected_environment": None if self.selected_environment is None else self.selected_environment.to_data(),
            "selected_world": None if self.selected_world is None else self.selected_world.to_data(),
            "selected_runtime": None if self.selected_runtime is None else self.selected_runtime.to_data(),
            "compatibility_policy": self.compatibility_policy,
            "compatibility_axes": None if self.compatibility_axes is None else self.compatibility_axes.to_data(),
        }


__all__ = ["SelectedWorldAllocation", "SessionConfiguration", "SessionSnapshot"]
