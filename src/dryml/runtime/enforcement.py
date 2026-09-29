"""Independent runtime-control statuses and dependency-light control plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from .modes import RuntimeMode


class ControlStatus(str, Enum):
    """Closed truthful status vocabulary for each independent control family."""

    UNDECLARED = "undeclared"
    NOT_APPLICABLE = "not-applicable"
    PENDING_IMPORT = "pending-import"
    VISIBILITY_ENFORCED = "visibility-enforced"
    FRAMEWORK_CONFIGURED = "framework-configured"
    ENFORCED = "enforced"
    DECLARATIVE = "declarative"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class RuntimeEnforcement(str, Enum):
    """Closed materialization-guard action for trusted explicit scopes."""

    STRICT = "strict"
    WARN = "warn"
    OFF = "off"

    @classmethod
    def coerce(cls, value: "RuntimeEnforcement | str") -> "RuntimeEnforcement":
        """Return a normalized action or raise for an unsupported value."""

        return value if isinstance(value, cls) else cls(value)


CONTROL_CATEGORIES = ("affinity", "process_memory", "visibility", "threading", "allocator", "accelerator_memory")


@dataclass(frozen=True, slots=True)
class ControlPlan:
    """Pre-import control status plan; applying effects remains publication-owned."""

    statuses: Mapping[str, ControlStatus] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Close and freeze all categories so one status cannot imply another."""
        values = dict(self.statuses)
        if set(values) - set(CONTROL_CATEGORIES):
            raise ValueError("runtime control status categories are closed")
        object.__setattr__(self, "statuses", MappingProxyType({name: ControlStatus(values.get(name, ControlStatus.UNDECLARED)) for name in CONTROL_CATEGORIES}))


def build_control_plan(mode: RuntimeMode | str, *, affinity: bool = False, process_memory: bool = False, framework_controls: bool = False, accelerator_memory: bool = False) -> ControlPlan:
    """Build independent status declarations without mutating the process.

    Args:
        mode: Declared runtime role.
        affinity: Whether CPU affinity has an explicit requested plan.
        process_memory: Whether a process-memory plan was declared.
        framework_controls: Whether framework-only controls were declared.
        accelerator_memory: Whether per-device limits were declared.

    Returns:
        An immutable pre-import status projection.
    """
    resolved = RuntimeMode.coerce(mode)
    if resolved is RuntimeMode.NONE:
        return ControlPlan({name: ControlStatus.NOT_APPLICABLE for name in CONTROL_CATEGORIES})
    return ControlPlan({
        "affinity": ControlStatus.DECLARATIVE if affinity else ControlStatus.NOT_APPLICABLE,
        "process_memory": ControlStatus.DECLARATIVE if process_memory else ControlStatus.NOT_APPLICABLE,
        "visibility": ControlStatus.PENDING_IMPORT,
        "threading": ControlStatus.PENDING_IMPORT if framework_controls else ControlStatus.UNDECLARED,
        "allocator": ControlStatus.PENDING_IMPORT if framework_controls else ControlStatus.UNDECLARED,
        "accelerator_memory": ControlStatus.PENDING_IMPORT if accelerator_memory else ControlStatus.UNDECLARED,
    })


__all__ = ["CONTROL_CATEGORIES", "ControlPlan", "ControlStatus", "RuntimeEnforcement", "build_control_plan"]
