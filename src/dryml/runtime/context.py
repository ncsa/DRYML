"""Read-only process-global runtime context projections."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .allocation import NoAllocation, RuntimeAllocationView, is_no_allocation
from .modes import RuntimeMode
from .specs import RuntimeContextSpec


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """Immutable runtime generation payload validated before publication."""

    mode: RuntimeMode = RuntimeMode.NONE
    allocation: RuntimeAllocationView | object = NoAllocation
    spec: RuntimeContextSpec | None = None
    controls: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce the closed mode/allocation contract without side effects."""
        mode = RuntimeMode.coerce(self.mode)
        if mode is RuntimeMode.INLINE and is_no_allocation(self.allocation):
            from .errors import RuntimeTransitionError
            raise RuntimeTransitionError("INLINE runtime requires one exact current-process allocation")
        if mode in {RuntimeMode.NONE, RuntimeMode.ORCHESTRATOR} and not is_no_allocation(self.allocation):
            from .errors import RuntimeTransitionError
            raise RuntimeTransitionError(f"{mode.value} runtime cannot hold a workload allocation")
        if self.spec is not None and self.spec.mode is not mode:
            from .errors import RuntimeTransitionError
            raise RuntimeTransitionError("runtime state mode must match its runtime context spec")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "controls", MappingProxyType(dict(self.controls)))


from .publication import PublicationService

publication = PublicationService()
publication.initialize(RuntimeState())
sys.modules[PublicationService.__module__].publication = publication


def active_runtime() -> RuntimeState:
    """Return the published runtime state without activating or mutating it."""
    return publication.current().runtime


def active_runtime_mode() -> RuntimeMode:
    """Return the published closed runtime mode without process effects."""
    return active_runtime().mode


__all__ = ["RuntimeState", "active_runtime", "active_runtime_mode", "publication"]
