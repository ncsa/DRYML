"""Runtime bootstrap planning and application."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .allocation import NoAllocation, RuntimeAllocationView
from .context import RuntimeBootstrapState, reset_runtime_bootstrap, set_runtime_bootstrap
from .devices import DeviceVisibilityPlan, apply_device_visibility_plan, build_device_visibility_plan
from .frameworks import FrameworkBootstrapAdapter, FrameworkBootstrapResult, default_adapters
from .guards import BOOTSTRAP_MARKER_ENV
from .specs import RuntimeContextSpec


@dataclass(frozen=True, slots=True)
class FrameworkBootstrapPolicy:
    """Controls which lightweight framework adapters participate in bootstrap."""

    frameworks: tuple[str, ...] = ("plain",)
    strict_preimport: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeBootstrapPlan:
    """Full pre/post-import runtime bootstrap plan."""

    runtime_spec: RuntimeContextSpec
    allocation_view: RuntimeAllocationView | Any
    visibility_plan: DeviceVisibilityPlan
    framework_results: Mapping[str, FrameworkBootstrapResult] = field(default_factory=dict)
    env_updates: Mapping[str, str] = field(default_factory=dict)
    strict_preimport: bool = False


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
    return RuntimeBootstrapPlan(spec, allocation_view, visibility_plan, framework_results, env_updates, strict_preimport=bool(policy.strict_preimport) if policy else False)


def apply_runtime_bootstrap_plan(plan: RuntimeBootstrapPlan, *, phase: str = "pre_import", environ: dict[str, str] | None = None, adapters: Mapping[str, FrameworkBootstrapAdapter] | None = None) -> None:
    """Apply a runtime bootstrap plan for ``pre_import`` or ``post_import``."""

    adapter_map = dict(default_adapters())
    if adapters:
        adapter_map.update(adapters)
    if phase == "pre_import":
        if plan.strict_preimport:
            for name, result in plan.framework_results.items():
                adapter_map[name].validate_before_import(result)
        apply_device_visibility_plan(plan.visibility_plan, environ=environ)
        for name, result in plan.framework_results.items():
            adapter_map[name].apply_pre_import(result, environ=environ)
        target = os.environ if environ is None else environ
        target.update(plan.env_updates)
        target[BOOTSTRAP_MARKER_ENV] = "1"
        return
    if phase == "post_import":
        for name, result in plan.framework_results.items():
            adapter_map[name].apply_post_import(result)
        return
    raise ValueError("phase must be 'pre_import' or 'post_import'")


@contextmanager
def activate_runtime_bootstrap(plan: RuntimeBootstrapPlan, *, restore_environ: bool = True, adapters: Mapping[str, FrameworkBootstrapAdapter] | None = None) -> Iterator[RuntimeBootstrapState]:
    """Activate *plan* in the current context and optionally restore env vars."""

    snapshot = _snapshot_environ(plan) if restore_environ else None
    state = _state_from_plan(plan)
    token = set_runtime_bootstrap(state)
    try:
        apply_runtime_bootstrap_plan(plan, adapters=adapters)
        yield state
    finally:
        reset_runtime_bootstrap(token)
        if snapshot is not None:
            _restore_environ(snapshot)


def _state_from_plan(plan: RuntimeBootstrapPlan) -> RuntimeBootstrapState:
    env_updates = {str(key): str(value) for key, value in plan.env_updates.items()}
    env_updates[BOOTSTRAP_MARKER_ENV] = "1"
    return RuntimeBootstrapState(
        plan_id=f"runtime-bootstrap-{id(plan):x}",
        mode=plan.runtime_spec.mode,
        frameworks=frozenset(plan.framework_results),
        env_updates=env_updates,
        allocation_fingerprint=repr(plan.allocation_view),
        strict_preimport=plan.strict_preimport,
    )


def _snapshot_environ(plan: RuntimeBootstrapPlan) -> dict[str, str | None]:
    keys = set(plan.env_updates) | set(plan.visibility_plan.env_updates) | {BOOTSTRAP_MARKER_ENV}
    for result in plan.framework_results.values():
        keys.update(result.env_updates)
    return {key: os.environ.get(key) for key in keys}


def _restore_environ(snapshot: Mapping[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


__all__ = ["FrameworkBootstrapPolicy", "RuntimeBootstrapPlan", "activate_runtime_bootstrap", "apply_runtime_bootstrap_plan", "build_runtime_bootstrap_plan"]
