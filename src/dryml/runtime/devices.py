"""Device visibility plans for runtime setup before framework import."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .allocation import NoAllocation, RuntimeAllocationView, is_no_allocation
from .errors import DeviceVisibilityError
from .modes import RuntimeMode
from .specs import RuntimeContextSpec


class DeviceVisibilityPolicy(str, Enum):
    """Policy for process environment device visibility."""

    NONE = "none"
    ASSIGNED = "assigned"
    INHERIT = "inherit"
    EXPLICIT = "explicit"

    @classmethod
    def coerce(cls, value: "DeviceVisibilityPolicy | str") -> "DeviceVisibilityPolicy":
        """Return *value* as a visibility policy."""

        if isinstance(value, DeviceVisibilityPolicy):
            return value
        return cls(str(value))


@dataclass(frozen=True, slots=True)
class DeviceVisibilityPlan:
    """Environment updates required to enforce device visibility."""

    policy: DeviceVisibilityPolicy
    env_updates: Mapping[str, str] = field(default_factory=dict)
    visible_devices: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    remap_assigned: bool = True


def build_device_visibility_plan(runtime_spec: RuntimeContextSpec | Mapping[str, Any] | None = None, allocation_view: RuntimeAllocationView | Any = NoAllocation, *, mode: RuntimeMode | str | None = None, policy: DeviceVisibilityPolicy | str | None = None, explicit_devices: Mapping[str, Any] | None = None, allow_inherit: bool = False) -> DeviceVisibilityPlan:
    """Build a visibility plan without importing any framework modules."""

    spec = RuntimeContextSpec.from_data(runtime_spec) if isinstance(runtime_spec, Mapping) else runtime_spec
    resolved_mode = RuntimeMode.coerce(mode or (spec.mode if spec else RuntimeMode.ORCHESTRATOR))
    visibility = dict(spec.device_visibility) if spec else {}
    resolved_policy = DeviceVisibilityPolicy.coerce(policy or visibility.get("policy") or _default_policy(resolved_mode))
    explicit = explicit_devices or visibility.get("devices") or visibility.get("explicit") or {}
    if resolved_policy is DeviceVisibilityPolicy.INHERIT:
        if resolved_mode is not RuntimeMode.NONE and not allow_inherit and not visibility.get("allow_inherit", False):
            raise DeviceVisibilityError("inherit visibility requires explicit opt-in", context={"mode": resolved_mode.value})
        return DeviceVisibilityPlan(resolved_policy, {}, {}, remap_assigned=False)
    if resolved_policy is DeviceVisibilityPolicy.NONE:
        return DeviceVisibilityPlan(resolved_policy, _hidden_env(), {"gpu": (), "rocm": (), "xla": ()})
    if resolved_policy is DeviceVisibilityPolicy.ASSIGNED:
        if is_no_allocation(allocation_view):
            raise DeviceVisibilityError("assigned visibility requires an allocation", context={"mode": resolved_mode.value, "allocation": repr(allocation_view)})
        visible = _visible_from_mapping(allocation_view.accelerators)
        return DeviceVisibilityPlan(resolved_policy, _visible_env(visible), visible)
    if resolved_policy is DeviceVisibilityPolicy.EXPLICIT:
        if not isinstance(explicit, Mapping):
            raise DeviceVisibilityError("explicit visibility requires a device map")
        visible = _visible_from_mapping(explicit)
        return DeviceVisibilityPlan(resolved_policy, _visible_env(visible), visible, remap_assigned=False)
    raise DeviceVisibilityError("unknown device visibility policy", context={"policy": resolved_policy})


def apply_device_visibility_plan(plan: DeviceVisibilityPlan, *, environ: dict[str, str] | None = None) -> None:
    """Apply a visibility plan to *environ* or ``os.environ``."""

    target = os.environ if environ is None else environ
    for key, value in plan.env_updates.items():
        target[key] = value


def _default_policy(mode: RuntimeMode) -> DeviceVisibilityPolicy:
    if mode is RuntimeMode.NONE:
        return DeviceVisibilityPolicy.INHERIT
    if mode in {RuntimeMode.ORCHESTRATOR, RuntimeMode.PROBE}:
        return DeviceVisibilityPolicy.NONE
    if mode is RuntimeMode.WORKER:
        return DeviceVisibilityPolicy.ASSIGNED
    return DeviceVisibilityPolicy.EXPLICIT


def _hidden_env() -> dict[str, str]:
    return {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "XLA_VISIBLE_DEVICES": ""}


def _visible_from_mapping(devices: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    gpu_ids = tuple(str(item) for item in devices.get("gpu", devices.get("cuda", ())))
    rocm_ids = tuple(str(item) for item in devices.get("rocm", devices.get("hip", devices.get("amd", ()))) )
    xla_ids = tuple(str(item) for item in devices.get("xla", ()))
    return {"gpu": gpu_ids, "rocm": rocm_ids, "xla": xla_ids}


def _visible_env(devices: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(devices.get("gpu", ())),
        "HIP_VISIBLE_DEVICES": ",".join(devices.get("rocm", ())),
        "ROCR_VISIBLE_DEVICES": ",".join(devices.get("rocm", ())),
        "XLA_VISIBLE_DEVICES": ",".join(devices.get("xla", ())),
    }


__all__ = ["DeviceVisibilityPlan", "DeviceVisibilityPolicy", "apply_device_visibility_plan", "build_device_visibility_plan"]
