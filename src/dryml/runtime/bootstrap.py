"""Import-light planning and validation for managed framework controls."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .devices import DeviceVisibilityPlan, DeviceVisibilityPolicy, build_device_visibility_plan
from .errors import RuntimeTransitionError
from .frameworks import FrameworkImportPlan, FrameworkRegistration, framework_registry
from .modes import RuntimeMode


@dataclass(frozen=True, slots=True)
class FrameworkBootstrapPlan:
    """Visibility and adapter facts for one active runtime generation."""

    visibility: DeviceVisibilityPlan
    adapter_plan: Any


def build_framework_bootstrap_plan(registration: FrameworkRegistration, runtime: Any, adapter: Any) -> FrameworkBootstrapPlan:
    """Build one adapter plan without importing a watched framework.

    Args:
        registration: Group whose root entered the loader lifecycle.
        runtime: Published immutable runtime state.
        adapter: Resolved dependency-light adapter.

    Returns:
        A visibility plan and adapter-owned immutable plan.

    Raises:
        RuntimeTransitionError: If active controls cannot produce visibility.
    """
    if getattr(runtime, "mode", RuntimeMode.NONE) is RuntimeMode.NONE:
        raise RuntimeTransitionError("Python runtime has no managed framework import plan")
    spec = getattr(runtime, "spec", None)
    declared = getattr(spec, "visibility", {}) if spec is not None else {}
    policy = declared.get("policy") if isinstance(declared, Mapping) else None
    explicit = declared.get("devices") if isinstance(declared, Mapping) else None
    visibility = build_device_visibility_plan(mode=runtime.mode, allocation=runtime.allocation, policy=DeviceVisibilityPolicy(policy) if policy else None, explicit_devices=explicit)
    method = getattr(adapter, "plan", None)
    value = method(runtime, visibility) if method is not None else FrameworkImportPlan(visibility.env_updates, visibility.visible_devices)
    if isinstance(value, Mapping):
        value = FrameworkImportPlan(**dict(value))
    if not isinstance(value, FrameworkImportPlan):
        raise RuntimeTransitionError("framework adapter plan must be FrameworkImportPlan or a compatible mapping")
    updates = dict(visibility.env_updates)
    updates.update(getattr(runtime.allocation, "env", {}))
    updates.update(getattr(spec, "env", {}) if spec is not None else {})
    updates.update(value.env_updates)
    value = replace(value, env_updates=updates)
    return FrameworkBootstrapPlan(visibility, value)


def validate_framework_transition(runtime: Any) -> None:
    """Reject a visibility-changing managed transition after late root import.

    This function is intentionally separate from publication so the U7 facade
    can call it before staging effects.  It never unloads or reloads modules.
    """
    if getattr(runtime, "mode", RuntimeMode.NONE) is RuntimeMode.NONE:
        return
    for registration in framework_registry.registrations().values():
        if any(root in sys.modules for root in registration.roots):
            raise RuntimeTransitionError("framework was imported before managed visibility control; restart the process")


__all__ = ["FrameworkBootstrapPlan", "build_framework_bootstrap_plan", "validate_framework_transition"]
