"""Runtime bootstrap planning and application."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

from .allocation import NoAllocation, RuntimeAllocationView, is_no_allocation
from .context import RuntimeBootstrapState, active_runtime, enter_runtime, reset_runtime_bootstrap, set_runtime_bootstrap
from .devices import DeviceVisibilityPlan, apply_device_visibility_plan, build_device_visibility_plan
from .frameworks import FrameworkBootstrapAdapter, FrameworkBootstrapResult, default_adapters, framework_registry
from .errors import RuntimeTransitionError
from .guards import BOOTSTRAP_MARKER_ENV
from .enforcement import RuntimeEnforcement
from .modes import RuntimeMode
from .specs import RuntimeContextSpec


@dataclass(frozen=True, slots=True)
class FrameworkBootstrapPolicy:
    """Controls which lightweight framework adapters participate in bootstrap."""

    frameworks: tuple[str, ...] | None = None
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
    selected = policy.frameworks if policy and policy.frameworks is not None else tuple(dict.fromkeys(("plain", *spec.frameworks.keys())))
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
                adapter = adapter_map[name]
                getattr(adapter, "validate_before_activation", adapter.validate_before_import)(result)
        apply_device_visibility_plan(plan.visibility_plan, environ=environ)
        for name, result in plan.framework_results.items():
            adapter_map[name].apply_pre_import(result, environ=environ)
        target = os.environ if environ is None else environ
        target.update(plan.env_updates)
        target[BOOTSTRAP_MARKER_ENV] = "1"
        return
    if phase == "post_import":
        for name, result in plan.framework_results.items():
            from .imports import finalize_helper

            if not finalize_helper(name):
                adapter_map[name].apply_post_import(result)
        return
    raise ValueError("phase must be 'pre_import' or 'post_import'")


@contextmanager
def activate_runtime_bootstrap(plan: RuntimeBootstrapPlan, *, restore_environ: bool = True, allow_process_controls: bool = False, adapters: Mapping[str, FrameworkBootstrapAdapter] | None = None) -> Iterator[RuntimeBootstrapState]:
    """Activate *plan* in the current context and optionally restore env vars."""

    _validate_plan_matches_active_runtime(plan)
    if restore_environ and not allow_process_controls:
        _reject_process_controls(plan)
    # A selected framework plan is the first controlled import lifecycle.  The
    # writer-protected freeze makes later registration unable to race its first
    # wrapped callback; plain-only activation and passive Python imports remain
    # registry-effect-free.
    if any(name != "plain" for name in plan.framework_results):
        framework_registry.freeze()
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


@contextmanager
def activate(
    *,
    mode: RuntimeMode | str | None = None,
    allocation: RuntimeAllocationView | Any = NoAllocation,
    spec: RuntimeContextSpec | Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    policy: FrameworkBootstrapPolicy | None = None,
    restore_environ: bool = True,
    allow_process_controls: bool = False,
    adapters: Mapping[str, FrameworkBootstrapAdapter] | None = None,
    enforcement: RuntimeEnforcement | str | None = None,
) -> Iterator[RuntimeBootstrapState]:
    """Enter runtime mode and activate bootstrap in one scoped barrier.

    This is a convenience API for explicit worker/probe/backend setup. It keeps
    the same validation as the lower-level primitives: the generated bootstrap
    plan must match the active runtime state before any effects are applied.
    """

    runtime_spec = _activation_spec(spec, mode)
    plan = build_runtime_bootstrap_plan(runtime_spec, allocation, env=env, policy=policy, adapters=adapters)
    with enter_runtime(runtime_spec.mode, allocation, runtime_spec, enforcement=enforcement):
        with activate_runtime_bootstrap(
            plan,
            restore_environ=restore_environ,
            allow_process_controls=allow_process_controls,
            adapters=adapters,
        ) as state:
            yield state


def _activation_spec(spec: RuntimeContextSpec | Mapping[str, Any] | None, mode: RuntimeMode | str | None) -> RuntimeContextSpec:
    if isinstance(spec, Mapping):
        data = dict(spec)
        if mode is not None:
            data["mode"] = RuntimeMode.coerce(mode).value
        return RuntimeContextSpec.from_data(data)
    if spec is None:
        if mode is None:
            return RuntimeContextSpec()
        return replace(RuntimeContextSpec(), mode=RuntimeMode.coerce(mode))
    if mode is None:
        return spec
    return replace(spec, mode=RuntimeMode.coerce(mode))


def _validate_plan_matches_active_runtime(plan: RuntimeBootstrapPlan) -> None:
    runtime = active_runtime()
    if runtime.mode is not plan.runtime_spec.mode:
        raise RuntimeTransitionError(
            "runtime bootstrap plan mode does not match active runtime",
            context={"active_mode": runtime.mode.value, "plan_mode": plan.runtime_spec.mode.value, "fix": "enter the matching runtime mode before activating bootstrap"},
        )
    if is_no_allocation(runtime.allocation) != is_no_allocation(plan.allocation_view) or (not is_no_allocation(runtime.allocation) and runtime.allocation != plan.allocation_view):
        raise RuntimeTransitionError(
            "runtime bootstrap plan allocation does not match active runtime",
            context={"active_allocation": repr(runtime.allocation), "plan_allocation": repr(plan.allocation_view), "fix": "build bootstrap plan from the active runtime allocation"},
        )


def _reject_process_controls(plan: RuntimeBootstrapPlan) -> None:
    controls = {
        name: {"cpu_affinity": result.cpu_affinity, "memory_limit": result.memory_limit}
        for name, result in plan.framework_results.items()
        if result.cpu_affinity is not None or result.memory_limit is not None
    }
    if controls:
        raise RuntimeTransitionError(
            "runtime bootstrap process controls require explicit opt-in in reusable processes",
            context={"controls": controls, "fix": "pass allow_process_controls=True or use a dedicated worker process"},
        )


def _state_from_plan(plan: RuntimeBootstrapPlan) -> RuntimeBootstrapState:
    env_updates = {str(key): str(value) for key, value in plan.env_updates.items()}
    env_updates[BOOTSTRAP_MARKER_ENV] = "1"
    return RuntimeBootstrapState(
        plan_id=f"runtime-bootstrap-{id(plan):x}",
        mode=plan.runtime_spec.mode,
        frameworks=frozenset(plan.framework_results),
        env_updates=env_updates,
        framework_results=plan.framework_results,
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


__all__ = ["FrameworkBootstrapPolicy", "RuntimeBootstrapPlan", "activate", "activate_runtime_bootstrap", "apply_runtime_bootstrap_plan", "build_runtime_bootstrap_plan"]
