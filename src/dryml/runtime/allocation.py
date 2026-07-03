"""Process-local allocation views for active runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class _NoAllocation:
    """Sentinel indicating no workload allocation is active."""

    def __repr__(self) -> str:
        return "NoAllocation"

    @property
    def is_no_allocation(self) -> bool:
        """Return ``True`` for the sentinel."""

        return True


NoAllocation = _NoAllocation()


@dataclass(frozen=True, slots=True)
class RuntimeAllocationView:
    """Process-local resources derived from a ``WorldAllocation``."""

    world_allocation_id: str | None = None
    role: str | None = None
    replica: int | None = None
    rank: int | None = None
    local_rank: int | None = None
    cpus: tuple[int, ...] = ()
    memory: int | None = None
    accelerators: Mapping[str, tuple[str | int, ...]] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpus", tuple(self.cpus))
        object.__setattr__(self, "accelerators", {str(key): tuple(value) for key, value in self.accelerators.items()})
        object.__setattr__(self, "env", {str(key): str(value) for key, value in self.env.items()})
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_no_allocation(self) -> bool:
        """Return ``False``; CPU-only workers are still real allocations."""

        return False


def is_no_allocation(value: Any) -> bool:
    """Return whether *value* is the ``NoAllocation`` sentinel."""

    return value is NoAllocation or getattr(value, "is_no_allocation", False) is True


__all__ = ["NoAllocation", "RuntimeAllocationView", "is_no_allocation"]
