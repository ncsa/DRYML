"""Runtime bootstrap planning and application."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .allocation import NoAllocation, RuntimeAllocationView
from .devices import DeviceVisibilityPlan, apply_device_visibility_plan, build_device_visibility_plan
from .frameworks import FrameworkBootstrapAdapter, FrameworkBootstrapResult, default_adapters
from .specs import RuntimeContextSpec


@dataclass(frozen=True, slots=True)
class FrameworkBootstrapPolicy:
    """Controls which lightweight framework adapters participate in bootstrap."""

    frameworks: tuple[str, ...] = ("plain", "torch", "tensorflow", "jax")


@dataclass(frozen=True, slots=True)
class RuntimeBootstrapPlan:
    """Full pre/post-import runtime bootstrap plan."""

    runtime_spec: RuntimeContextSpec
    allocation_view: RuntimeAllocationView | Any
    visibility_plan: DeviceVisibilityPlan
    framework_results: Mapping[str, FrameworkBootstrapResult] = field(default_factory=dict)
    env_updates: Mapping[str, str] = field(default_factory=dict)


def build_runtime_bootstrap_plan(runtime_spec: RuntimeContextSpec | Mapping[str, Any] | None = None, allocation_view: RuntimeAllocationView | Any = NoAllocation, *, env: Mapping[str, str] | None = None, policy: FrameworkBootstrapPolicy | None = None, adapters: Mapping[str, FrameworkBootstrapAdapter] | None = None) -> RuntimeBootstrapPlan:
    """Build a full runtime bootstrap plan without importing heavy frameworks."""

    spec = RuntimeContextSpec.from_data(runtime_spec) if isinstance(runtime_spec, Mapping) else (runtime_spec or RuntimeContextSpec())
    visibility_plan = build_device_visibility_plan(spec, allocation_view)
    adapter_map = dict(default_adapters())
    if adapters:
        adapter_map.update(adapters)
    selected = policy.frameworks if policy else tuple(dict.fromkeys(("plain", *spec.frameworks.keys())))
    framework_results = {name: adapter_map[name].build_plan(spec, allocation_view, visibility_plan) for name in selected if name in adapter_map}
    env_updates: dict[str, str] = dict(visibility_plan.env_updates)
    env_updates.update({str(key): str(value) for key, value in (env or {}).items()})
    for result in framework_results.values():
        env_updates.update(result.env_updates)
    return RuntimeBootstrapPlan(spec, allocation_view, visibility_plan, framework_results, env_updates)


def apply_runtime_bootstrap_plan(plan: RuntimeBootstrapPlan, *, phase: str = "pre_import", environ: dict[str, str] | None = None, adapters: Mapping[str, FrameworkBootstrapAdapter] | None = None) -> None:
    """Apply a runtime bootstrap plan for ``pre_import`` or ``post_import``."""

    adapter_map = dict(default_adapters())
    if adapters:
        adapter_map.update(adapters)
    if phase == "pre_import":
        for name, result in plan.framework_results.items():
            adapter_map[name].validate_before_import(result)
        apply_device_visibility_plan(plan.visibility_plan, environ=environ)
        for name, result in plan.framework_results.items():
            adapter_map[name].apply_pre_import(result, environ=environ)
        target = os.environ if environ is None else environ
        target.update(plan.env_updates)
        return
    if phase == "post_import":
        for name, result in plan.framework_results.items():
            adapter_map[name].apply_post_import(result)
        return
    raise ValueError("phase must be 'pre_import' or 'post_import'")


__all__ = ["FrameworkBootstrapPolicy", "RuntimeBootstrapPlan", "apply_runtime_bootstrap_plan", "build_runtime_bootstrap_plan"]
