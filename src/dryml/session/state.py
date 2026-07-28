"""Publication-backed persistent state for the public session facade.

This module deliberately stores no current session pointer.  The immutable
configuration and retained inventory are metadata on the sole runtime
publication generation; this module only stages candidates and projects them.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from dryml.environments import EnvironmentRequirement
from dryml.environments import inspect_current
from dryml.runtime import (
    EffectPlan,
    FrameworkBootstrapPolicy,
    NoAllocation,
    RuntimeAllocationView,
    RuntimeContextSpec,
    RuntimeEnforcement,
    RuntimeMode,
    RuntimeState,
    build_runtime_bootstrap_plan,
)
from dryml.runtime.frameworks import FrameworkBootstrapResult, framework_registry
from dryml.runtime.errors import PublicationError
from dryml.runtime.publication import SessionGeneration, publication
from dryml.worlds import LocalResourceInventory, ProcessAllocation, ResourceSpec, WorldAllocation, WorldSpec, assign_local_world, local_inventory

from .configuration import _normalize_environment, _normalize_resources, normalize_configuration, select_world_allocation
from .errors import SessionConfigurationError
from .model import SelectedWorldAllocation, SessionConfiguration, SessionSnapshot

_VISIBILITY_KEYS = frozenset({"CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "XLA_VISIBLE_DEVICES"})


def current() -> SessionSnapshot:
    """Project the current published generation without probing host state."""

    return _snapshot(publication.current())


def mode() -> str:
    """Return the stable facade mode without process or package probing."""

    return current().mode


def set_mode(mode: str) -> SessionSnapshot:
    """Replace the mode while preserving worker intent and requirements."""

    if not isinstance(mode, str) or mode not in {"python", "managed", "orchestrator"}:
        raise SessionConfigurationError("session mode must be python, managed, or orchestrator")
    before = _configuration(publication.current())
    allocation = before.allocation if mode == "managed" else None
    resources = before.resources if mode == "managed" and allocation is not None else None
    return _publish(mode, resources, allocation, before.requested_world, before.environment, default_managed=mode == "managed" and allocation is None)


def manage(*, cpus: int | None = None, memory: str | int | None = None, gpus: int | None = None, accelerator_memory: Any = None) -> SessionSnapshot:
    """Enter managed mode with a replacement current-process allowance."""

    before = _configuration(publication.current())
    supplied = {key: value for key, value in {"cpus": cpus, "memory": memory, "gpus": gpus, "accelerator_memory": accelerator_memory}.items() if value is not None}
    return _publish("managed", None, None, before.requested_world, before.environment, synthesize=True, simple_resources=supplied)


def request_world(*, cpus: int | None = None, memory: str | int | None = None, gpus: int | None = None, accelerator_memory: Any = None) -> SessionSnapshot:
    """Replace only the default requested worker world."""

    supplied = {key: value for key, value in {"cpus": cpus, "memory": memory, "gpus": gpus, "accelerator_memory": accelerator_memory}.items() if value is not None}
    if not supplied:
        raise SessionConfigurationError("request_world requires at least one resource field")
    before = _configuration(publication.current())
    resources = _normalize_resources(supplied)
    world = _world_for_resources(resources, role="worker")
    return _publish(before.mode, before.resources, before.allocation, world, before.environment)


def allocate_world(value: WorldAllocation | Mapping[str, Any], /, *, role: str | None = None, replica: int | None = None) -> SessionSnapshot:
    """Enter managed mode using one unambiguously selected exact allocation."""

    before = _configuration(publication.current())
    return _publish("managed", None, (value, role, replica), before.requested_world, before.environment, exact=True)


def require_env(*requirements: str, python: str | None = None, excludes: Any = (), capabilities: Any = ()) -> SessionSnapshot:
    """Atomically merge package and concise software constraints."""

    if not requirements and python is None and not excludes and not capabilities:
        raise SessionConfigurationError("require_env requires at least one constraint")
    if any(not isinstance(requirement, str) for requirement in requirements):
        raise SessionConfigurationError("package requirements must be PEP 508 strings")
    if python is not None and not isinstance(python, str):
        raise SessionConfigurationError("python requirement must be a string")
    for name, value in (("excludes", excludes), ("capabilities", capabilities)):
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__") or any(not isinstance(item, str) for item in value):
            raise SessionConfigurationError(f"{name} must be a non-string sequence of strings")
    before = _configuration(publication.current())
    try:
        addition = _normalize_environment(
            {"requirements": requirements, "python": python, "excludes": excludes, "capabilities": capabilities}
        )
        environment = before.environment.merge(addition, sources=("session",))
    except Exception as exc:
        raise SessionConfigurationError(str(exc), context=getattr(exc, "context", {})) from exc
    return _publish(before.mode, before.resources, before.allocation, before.requested_world, environment)


def configure(*, mode: str, resources: Mapping[str, Any] | None = None, allocation: Mapping[str, Any] | None = None, requested_world: Mapping[str, Any] | None = None, environment: Mapping[str, Any] | None = None) -> SessionSnapshot:
    """Atomically replace every facade category from one closed declaration."""

    candidate = normalize_configuration(
        mode=mode,
        resources=resources,
        allocation=allocation,
        requested_world=requested_world,
        environment=environment,
    )
    default_managed = candidate.mode == "managed" and resources is None and allocation is None
    exact_allocation = None
    if allocation is not None:
        assert isinstance(allocation, Mapping)
        exact_allocation = (allocation["value"], allocation.get("role"), allocation.get("replica"))
    return _publish(
        candidate.mode,
        None if default_managed else candidate.resources,
        exact_allocation if exact_allocation is not None else candidate.allocation,
        candidate.requested_world,
        candidate.environment,
        default_managed=default_managed,
        synthesize=candidate.mode == "managed" and resources is not None,
        exact=exact_allocation is not None,
    )


def reset() -> SessionSnapshot:
    """Restore the ordinary unchecked Python baseline and clear all categories."""

    return _publish("python", None, None, None, EnvironmentRequirement())


def _configuration(generation: SessionGeneration) -> SessionConfiguration:
    value = generation.metadata.get("session_configuration")
    if isinstance(value, SessionConfiguration):
        return value
    return SessionConfiguration("python")


def _snapshot(generation: SessionGeneration) -> SessionSnapshot:
    configuration = _configuration(generation)
    statuses = {
        "visibility": "visibility-enforced" if configuration.mode in {"managed", "orchestrator"} else "pending-import",
        "memory": configuration.controls.get("memory", "undeclared"),
        "accelerator_memory": configuration.controls.get("accelerator_memory", "undeclared"),
    }
    statuses.update(generation.metadata.get("framework_statuses", {}))
    return SessionSnapshot(
        configuration.mode,
        configuration.resources,
        configuration.allocation,
        configuration.requested_world,
        configuration.environment,
        configuration.controls,
        statuses,
        generation.runtime,
        generation.number,
        generation.health,
        generation.visibility_epoch if isinstance(generation.visibility_epoch, LocalResourceInventory) else None,
    )


def _managed_resources(supplied: Mapping[str, Any], inventory: LocalResourceInventory | None = None) -> ResourceSpec:
    if inventory is None:
        # Defaults are completed from one observed inherited-bound inventory.
        inventory = local_inventory()
    data = dict(supplied)
    data.setdefault("cpus", len(inventory.cpus))
    data.setdefault("gpus", 0)
    if "memory" not in data and inventory.memory is not None:
        data["memory"] = inventory.memory
    return _normalize_resources(data)


def _world_for_resources(resources: ResourceSpec, *, role: str) -> WorldSpec:
    return WorldSpec.from_data({"roles": {role: {"replicas": 1, "process": {"resources": resources.to_data()}}}})


def _synthesize(resources: ResourceSpec, inventory: LocalResourceInventory) -> SelectedWorldAllocation:
    assignment = assign_local_world(_world_for_resources(resources, role="main"), inventory=inventory)
    allocation = WorldAllocation.from_data({"roles": {name: list(processes) for name, processes in assignment.roles.items()}})
    return select_world_allocation(allocation, inventory=inventory)


def _runtime_for(mode: str, allocation: SelectedWorldAllocation | None) -> RuntimeState:
    if mode == "managed":
        assert allocation is not None
        view = allocation.process.to_runtime_resource_view(role=allocation.role).to_runtime_allocation_view()
        return RuntimeState(RuntimeMode.INLINE, view, enforcement=RuntimeEnforcement.STRICT)
    return RuntimeState(RuntimeMode.ORCHESTRATOR, NoAllocation, enforcement=RuntimeEnforcement.OFF if mode == "python" else RuntimeEnforcement.STRICT)


def _loaded_framework_roots() -> tuple[str, ...]:
    return tuple(root for root in ("tensorflow", "torch", "jax", "jaxlib") if root in sys.modules)


def _publish(
    mode: str,
    resources: ResourceSpec | None,
    allocation: SelectedWorldAllocation | tuple[Any, str | None, int | None] | None,
    requested_world: WorldSpec | None,
    environment: EnvironmentRequirement,
    *,
    default_managed: bool = False,
    synthesize: bool = False,
    exact: bool = False,
    simple_resources: Mapping[str, Any] | None = None,
) -> SessionSnapshot:
    """Stage, revalidate, and publish one complete immutable session generation."""

    for _attempt in range(3):
        before = publication.current()
        previous = _configuration(before)
        retained = before.visibility_epoch if isinstance(before.visibility_epoch, LocalResourceInventory) and before.metadata.get("session_active") else None
        needs_inventory = mode in {"managed", "orchestrator"} and (retained is None or default_managed or synthesize or exact)
        observed_from_probe = retained is None and needs_inventory
        observed = local_inventory() if observed_from_probe else retained
        if observed is not None and not isinstance(observed, LocalResourceInventory):
            raise SessionConfigurationError("retained session inventory is invalid")
        candidate_resources = _managed_resources({}, observed) if default_managed else resources
        if simple_resources is not None:
            assert observed is not None
            candidate_resources = _managed_resources(simple_resources, observed)
        candidate_allocation = allocation
        if exact:
            assert isinstance(candidate_allocation, tuple)
            candidate_allocation = select_world_allocation(candidate_allocation[0], role=candidate_allocation[1], replica=candidate_allocation[2], inventory=observed)
        elif mode == "managed" and (synthesize or default_managed):
            assert candidate_resources is not None and observed is not None
            candidate_allocation = _synthesize(candidate_resources, observed)
        elif mode != "managed":
            candidate_allocation = None
            candidate_resources = None
        elif candidate_allocation is None:
            # A managed candidate always has an exact process allocation.
            assert candidate_resources is not None and observed is not None
            candidate_allocation = _synthesize(candidate_resources, observed)
        assert candidate_allocation is None or isinstance(candidate_allocation, SelectedWorldAllocation)
        candidate = SessionConfiguration(mode, candidate_resources, candidate_allocation, requested_world, environment, _controls(candidate_resources, candidate_allocation))
        _check_current_environment(mode, environment)
        runtime = _runtime_for(mode, candidate_allocation)
        bootstrap = _bootstrap_plan(runtime)
        framework_results = dict(bootstrap.framework_results)
        framework_statuses = _pending_framework_statuses(framework_results)
        target_active = mode in {"managed", "orchestrator"}
        adapter_plan_changes = dict(before.metadata.get("framework_results", {})) != framework_results
        if not adapter_plan_changes:
            framework_statuses.update(before.metadata.get("framework_statuses", {}))
        if (
            candidate == previous
            and before.health == "healthy"
            and before.runtime == runtime
            and bool(before.metadata.get("session_active")) == target_active
            and dict(before.metadata.get("framework_results", {})) == framework_results
        ):
            return _snapshot(before)
        observed_roots = _loaded_framework_roots()
        registry_revision = framework_registry.revision
        plan = _effect_plan(runtime, before, bootstrap)
        generation = SessionGeneration(
            before.number + 1,
            runtime,
            visibility_epoch=observed if observed is not None else before.visibility_epoch,
            metadata={
                "session_configuration": candidate,
                "session_active": target_active,
                "framework_results": framework_results,
                "frameworks": tuple(framework_results),
                "framework_statuses": framework_statuses,
                "framework_registry_revision": registry_revision,
                "control_epoch": before.number + 1,
            },
        )

        def validate() -> None:
            if observed_from_probe and local_inventory() != observed:
                raise PublicationError("session inventory changed while staging; restage before applying effects")
            if framework_registry.revision != registry_revision:
                raise PublicationError("framework adapter registry changed while staging; restage before applying effects")
            if _loaded_framework_roots() != observed_roots:
                raise PublicationError("framework import changed while staging; restage before applying effects")
            if observed_roots and adapter_plan_changes:
                raise PublicationError("restart the process; a framework was imported before its adapter plan could change")
            if target_active:
                framework_registry.freeze()

        try:
            committed = publication.commit(publication.stage(before, generation), plan, validator=validate)
        except PublicationError as exc:
            if "inventory changed" in str(exc) or "stale publication candidate" in str(exc):
                continue
            raise
        return _snapshot(committed)
    raise SessionConfigurationError("session transition could not obtain a stable inventory")


def _controls(resources: ResourceSpec | None, allocation: SelectedWorldAllocation | None) -> dict[str, str]:
    process = None if allocation is None else allocation.process
    return {
        "memory": "declarative" if (resources and resources.memory is not None) or (process and process.memory is not None) else "undeclared",
        "accelerator_memory": "declarative" if (resources and resources.accelerator_memory) or (process and process.accelerator_memory) else "undeclared",
    }


def _check_current_environment(mode: str, environment: EnvironmentRequirement) -> None:
    """Reject an incompatible managed interpreter before process effects begin."""

    if mode != "managed" or not any((environment.requirements, environment.python, environment.excludes, environment.capabilities)):
        return
    report = environment.check(inspect_current(), policy="strict")
    if not report.ok:
        raise SessionConfigurationError("current environment does not satisfy managed session requirements", context={"issues": [issue.code for issue in report.issues]})


def _effect_plan(runtime: RuntimeState, before: SessionGeneration, bootstrap: Any) -> EffectPlan:
    if before.runtime == runtime and before.metadata.get("session_active") == (runtime.enforcement is RuntimeEnforcement.STRICT):
        # Declarative requirements and worker intent may advance independently
        # without invalidating a leased direct invocation.
        return EffectPlan()
    if runtime.mode is RuntimeMode.INLINE:
        visibility = dict(bootstrap.env_updates)
        visibility.update({key: "" for key in _VISIBILITY_KEYS if key not in visibility})
        allocation = runtime.allocation
        if getattr(allocation, "accelerators", {}):
            from dryml.runtime import build_device_visibility_plan

            visibility.update(build_device_visibility_plan(mode=RuntimeMode.INLINE, allocation_view=allocation, policy="assigned").env_updates)
        return EffectPlan(environment=visibility, cpu_affinity=tuple(allocation.cpus))
    if runtime.enforcement is RuntimeEnforcement.STRICT:
        environment = dict(bootstrap.env_updates)
        environment.update({key: "" for key in _VISIBILITY_KEYS})
        return EffectPlan(environment=environment)
    restore_environment: dict[str, str | None] = {}
    affinity = None
    for record in publication.effect_journal():
        if record.kind == "environment":
            restore_environment[record.key] = record.previous
        elif record.kind == "cpu_affinity":
            affinity = record.previous
    return EffectPlan(environment=restore_environment, cpu_affinity=affinity)


def _bootstrap_plan(runtime: RuntimeState):
    spec = RuntimeContextSpec(mode=runtime.mode, device_visibility={"policy": "assigned" if runtime.mode is RuntimeMode.INLINE else "none"})
    frameworks = ("plain", "tensorflow", "torch", "jax") if runtime.enforcement is RuntimeEnforcement.STRICT else ("plain",)
    return build_runtime_bootstrap_plan(spec, runtime.allocation, policy=FrameworkBootstrapPolicy(frameworks))


def _pending_framework_statuses(results: Mapping[str, FrameworkBootstrapResult]) -> dict[str, str]:
    """Publish an honest pending control outcome for each planned root."""

    statuses: dict[str, str] = {}
    for name in results:
        registration = framework_registry.registrations().get(name)
        if registration is None:
            continue
        for root in registration.roots:
            for control in ("visibility", "threads", "process_memory", "accelerator_memory", "allocator"):
                statuses[f"{name}:{root}:{control}"] = "pending-import"
    return statuses


def _visibility_changes(previous: SessionConfiguration, candidate: SessionConfiguration) -> bool:
    return previous.mode != candidate.mode or previous.allocation != candidate.allocation


__all__ = ["allocate_world", "configure", "current", "manage", "mode", "request_world", "require_env", "reset", "set_mode"]
