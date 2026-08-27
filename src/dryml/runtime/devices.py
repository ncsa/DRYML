"""Visibility planning without framework imports or process mutation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from .allocation import NoAllocation, RuntimeAllocationView, is_no_allocation
from .errors import DeviceVisibilityError
from .modes import RuntimeMode


class DeviceVisibilityPolicy(str, Enum):
    """Closed environment visibility policies for the supported runtime modes."""

    NONE = "none"
    ASSIGNED = "assigned"
    INHERIT = "inherit"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class DeviceVisibilityPlan:
    """Reversible environment updates staged before a controlled import."""

    policy: DeviceVisibilityPolicy
    env_updates: Mapping[str, str] = field(default_factory=dict)
    visible_devices: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze exact environment and visible-device mappings."""
        object.__setattr__(self, "env_updates", MappingProxyType(dict(self.env_updates)))
        object.__setattr__(self, "visible_devices", MappingProxyType({key: tuple(value) for key, value in self.visible_devices.items()}))


def build_device_visibility_plan(*, mode: RuntimeMode | str, allocation: RuntimeAllocationView | object = NoAllocation, policy: DeviceVisibilityPolicy | str | None = None, explicit_devices: Mapping[str, Any] | None = None) -> DeviceVisibilityPlan:
    """Plan visibility before framework imports.

    Args:
        mode: Runtime role determining the default policy.
        allocation: Exact inline allocation for assigned visibility.
        policy: Optional explicit closed visibility policy.
        explicit_devices: Device mapping for explicit visibility.

    Returns:
        A plan that publication may apply transactionally.

    Raises:
        DeviceVisibilityError: If assigned visibility lacks an exact allocation.
    """
    resolved = RuntimeMode.coerce(mode)
    selected = DeviceVisibilityPolicy(policy) if policy is not None else (DeviceVisibilityPolicy.INHERIT if resolved is RuntimeMode.NONE else DeviceVisibilityPolicy.NONE if resolved is RuntimeMode.ORCHESTRATOR else DeviceVisibilityPolicy.ASSIGNED)
    required = {
        RuntimeMode.NONE: {DeviceVisibilityPolicy.INHERIT},
        RuntimeMode.ORCHESTRATOR: {DeviceVisibilityPolicy.NONE},
        RuntimeMode.INLINE: {DeviceVisibilityPolicy.ASSIGNED, DeviceVisibilityPolicy.EXPLICIT},
    }
    if selected not in required[resolved]:
        raise DeviceVisibilityError(f"{resolved.value} runtime does not permit {selected.value} visibility")
    if selected is DeviceVisibilityPolicy.INHERIT:
        return DeviceVisibilityPlan(selected)
    if selected is DeviceVisibilityPolicy.NONE:
        updates = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "XLA_VISIBLE_DEVICES": ""}
        return DeviceVisibilityPlan(selected, updates, {"gpu": (), "rocm": (), "xla": ()})
    if selected is DeviceVisibilityPolicy.ASSIGNED:
        if is_no_allocation(allocation):
            raise DeviceVisibilityError("assigned visibility requires one exact inline allocation")
        devices = allocation.accelerators
    else:
        if not isinstance(explicit_devices, Mapping):
            raise DeviceVisibilityError("explicit visibility requires a device mapping")
        devices = explicit_devices
    gpu = tuple(str(value) for value in devices.get("gpu", devices.get("cuda", ())))
    rocm = tuple(str(value) for value in devices.get("rocm", devices.get("hip", ())))
    xla = tuple(str(value) for value in devices.get("xla", ()))
    return DeviceVisibilityPlan(selected, {"CUDA_VISIBLE_DEVICES": ",".join(gpu), "HIP_VISIBLE_DEVICES": ",".join(rocm), "ROCR_VISIBLE_DEVICES": ",".join(rocm), "XLA_VISIBLE_DEVICES": ",".join(xla)}, {"gpu": gpu, "rocm": rocm, "xla": xla})


__all__ = ["DeviceVisibilityPlan", "DeviceVisibilityPolicy", "build_device_visibility_plan"]
