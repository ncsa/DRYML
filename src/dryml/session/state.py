"""Publication-backed persistent state for the public session facade.

This module deliberately stores no current session pointer.  The immutable
configuration and retained inventory are metadata on the sole runtime
publication generation; this module only stages candidates and projects them.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from typing import Any

from dryml.environments import EnvironmentRequirement, EnvironmentSpec, spec_from_data
from dryml.environments import inspect_current
from dryml.runtime import (
    EffectPlan,
    FrameworkBootstrapPolicy,
    NoAllocation,
    RuntimeContextSpec,
    RuntimeEnforcement,
    RuntimeMode,
    RuntimeState,
    RequirementAxes,
    build_runtime_bootstrap_plan,
)
from dryml.runtime.frameworks import FrameworkBootstrapResult, framework_registry
from dryml.runtime.errors import PublicationBusyError, PublicationError
from dryml.runtime.publication import SessionGeneration, publication
from dryml.worlds import LocalResourceInventory, ProcessAllocation, ResourceSpec, WorldAllocation, WorldSpec, assign_local_world, local_inventory

from .configuration import _normalize_environment, _normalize_requirement_axes, _normalize_requested_environment, _normalize_resources, normalize_configuration, select_world_allocation
from .errors import SessionConfigurationError
from .model import _default_requirement_axes, SelectedWorldAllocation, SessionConfiguration, SessionSnapshot

_VISIBILITY_KEYS = frozenset({"CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "XLA_VISIBLE_DEVICES"})


def current() -> SessionSnapshot:
    """Project the current published generation without probing host state.

    Returns:
        Immutable snapshot of the active process session.
    """

    return _snapshot(publication.current())


def mode() -> str:
    """Return the stable facade mode without process or package probing.

    Returns:
        One of ``python``, ``managed``, or ``orchestrator``.
    """

    return _configuration(publication.current()).mode


def set_mode(mode: str) -> SessionSnapshot:
    """Replace the mode while preserving worker intent and requirements.

    Args:
        mode: Requested stable facade mode.

    Returns:
        Snapshot of the successfully published session.
    """

    if not isinstance(mode, str) or mode not in {"python", "managed", "orchestrator"}:
        raise SessionConfigurationError("session mode must be python, managed, or orchestrator")
    before = _configuration(publication.current())
    allocation = before.allocation if mode == "managed" else None
    resources = before.resources if mode == "managed" and allocation is not None else None
    return _publish(
        mode,
        resources,
        allocation,
        before.requested_environment,
        before.requested_world,
        before.environment,
        _default_requirement_axes(mode),
        default_managed=mode == "managed" and allocation is None,
    )


def manage(*, cpus: int | None = None, memory: str | int | None = None, gpus: int | None = None, accelerator_memory: Any = None) -> SessionSnapshot:
    """Enter managed mode with a replacement current-process allowance.

    Args:
        cpus: Optional inherited CPU count to allocate.
        memory: Optional declarative process-memory allowance.
        gpus: Optional accelerator count to assign.
        accelerator_memory: Optional per-accelerator memory allowance.

    Returns:
        Snapshot of the successfully published managed session.
    """

    before = _configuration(publication.current())
    supplied = {key: value for key, value in {"cpus": cpus, "memory": memory, "gpus": gpus, "accelerator_memory": accelerator_memory}.items() if value is not None}
    return _publish(
        "managed",
        None,
        None,
        before.requested_environment,
        before.requested_world,
        before.environment,
        RequirementAxes.all(),
        synthesize=True,
        simple_resources=supplied,
    )


def worker_world_request(*, cpus: int | None = None, memory: str | int | None = None, gpus: int | None = None, accelerator_memory: Any = None) -> SessionSnapshot:
    """Replace the default world request used by later dispatched workers.

    Args:
        cpus: Optional worker CPU count.
        memory: Optional worker process-memory allowance.
        gpus: Optional worker accelerator count.
        accelerator_memory: Optional worker per-accelerator memory allowance.

    Returns:
        Snapshot containing the replacement default worker world.
    """

    supplied = {key: value for key, value in {"cpus": cpus, "memory": memory, "gpus": gpus, "accelerator_memory": accelerator_memory}.items() if value is not None}
    if not supplied:
        raise SessionConfigurationError("worker_world_request requires at least one resource field")
    before = _configuration(publication.current())
    resources = _normalize_resources(supplied)
    world = _world_for_resources(resources, role="worker")
    return _publish(before.mode, before.resources, before.allocation, before.requested_environment, world, before.environment, before.requirement_axes)


def worker_env_request(value: EnvironmentSpec | Mapping[str, Any], /) -> SessionSnapshot:
    """Replace the inert concrete environment candidate for future workers.

    Args:
        value: An ``EnvironmentSpec`` or its bounded canonical mapping.

    Returns:
        Snapshot with the replacement request. The request is consumed only by
        an explicit dispatch operation and never describes this process.
    """

    try:
        candidate = value if isinstance(value, EnvironmentSpec) else _normalize_requested_environment(value)
        if candidate is None:
            raise SessionConfigurationError("worker_env_request requires an environment spec")
        candidate = spec_from_data(candidate.to_data())
    except SessionConfigurationError:
        raise
    except Exception as exc:
        raise SessionConfigurationError(str(exc), context=getattr(exc, "context", {})) from exc
    before = _configuration(publication.current())
    return _publish(before.mode, before.resources, before.allocation, candidate, before.requested_world, before.environment, before.requirement_axes)


def allocate_world(value: WorldAllocation | Mapping[str, Any], /, *, role: str | None = None, replica: int | None = None) -> SessionSnapshot:
    """Enter managed mode using one unambiguously selected exact allocation.

    Args:
        value: Typed allocation or canonical allocation envelope.
        role: Role selector required for multi-process allocations.
        replica: Replica selector paired with ``role``.

    Returns:
        Snapshot of the managed session using the selected process.
    """

    before = _configuration(publication.current())
    return _publish("managed", None, (value, role, replica), before.requested_environment, before.requested_world, before.environment, RequirementAxes.all(), exact=True)


def require_env(*requirements: str, python: str | None = None, excludes: Any = (), capabilities: Any = ()) -> SessionSnapshot:
    """Atomically merge package and concise software constraints.

    Args:
        requirements: PEP 508 package constraints to merge.
        python: Optional Python-version constraint.
        excludes: Optional excluded environment capabilities.
        capabilities: Optional required environment capabilities.

    Returns:
        Snapshot containing the merged requirements.
    """

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
    return _publish(before.mode, before.resources, before.allocation, before.requested_environment, before.requested_world, environment, before.requirement_axes)


def enforce_requirements(*, environment: bool, world: bool, runtime: bool) -> SessionSnapshot:
    """Atomically replace the session requirement-axis mask.

    Args:
        environment: Whether environment compatibility is enabled.
        world: Whether world compatibility is enabled.
        runtime: Whether runtime compatibility is enabled.

    Returns:
        The published snapshot with the canonical replacement mask.

    Raises:
        SessionConfigurationError: If any value is not an exact boolean.

    Side Effects:
        Publishes a generation only after all values validate. This does not
        change role, allocation, framework visibility, or lifecycle controls.
    """

    before = _configuration(publication.current())
    axes = _normalize_requirement_axes(
        {"environment": environment, "world": world, "runtime": runtime},
        mode=before.mode,
    )
    return _publish(before.mode, before.resources, before.allocation, before.requested_environment, before.requested_world, before.environment, axes)


def configure(*, mode: str, resources: Mapping[str, Any] | None = None, allocation: Mapping[str, Any] | None = None, requested_environment: Mapping[str, Any] | None = None, requested_world: Mapping[str, Any] | None = None, environment: Mapping[str, Any] | None = None, requirement_axes: Mapping[str, bool] | None = None) -> SessionSnapshot:
    """Atomically replace every facade category from one closed declaration.

    Args:
        mode: Mandatory target facade mode.
        resources: Optional simple current-process resource declaration.
        allocation: Optional exact current-process allocation declaration.
        requested_environment: Optional concrete default worker environment.
        requested_world: Optional default worker world declaration.
        environment: Optional software requirement declaration.
        requirement_axes: Optional complete exact-boolean compatibility-axis
            replacement. Omission selects the target mode's default mask.

    Returns:
        Snapshot of the complete replacement session.

    Raises:
        SessionConfigurationError: If any category is malformed, incompatible,
            or cannot be published atomically.

    Side Effects:
        Publishes one complete process generation and applies any required
        visibility or affinity controls only after all categories validate.
    """

    candidate = normalize_configuration(
        mode=mode,
        resources=resources,
        allocation=allocation,
        requested_environment=requested_environment,
        requested_world=requested_world,
        environment=environment,
        requirement_axes=requirement_axes,
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
        candidate.requested_environment,
        candidate.requested_world,
        candidate.environment,
        candidate.requirement_axes,
        default_managed=default_managed,
        synthesize=candidate.mode == "managed" and resources is not None,
        exact=exact_allocation is not None,
    )


def reset() -> SessionSnapshot:
    """Restore the ordinary unchecked Python baseline and clear all categories.

    Returns:
        Snapshot of the restored Python-mode session.
    """

    return _publish("python", None, None, None, None, EnvironmentRequirement(), RequirementAxes())


def publish_worker_session(
    *,
    environment: EnvironmentSpec,
    world: WorldSpec,
    runtime_spec: RuntimeContextSpec,
    allocation: Any,
    requirement_policy: str,
    requirement_axes: RequirementAxes,
) -> SessionSnapshot:
    """Publish the internal strict worker session before worker setup begins.

    Args:
        environment: Canonical selected worker environment.
        world: Canonical selected worker world.
        runtime_spec: Canonical selected worker runtime configuration.
        allocation: Exact allocated worker runtime view.
        requirement_policy: Dispatch compatibility action retained for inspection.
        requirement_axes: Dispatch compatibility axes retained for inspection.

    Returns:
        The managed-vocabulary snapshot carrying the internal worker runtime.

    Raises:
        SessionConfigurationError: If the worker selection is not an exact
            strict allocated worker session.

    Side Effects:
        Replaces the process-local publication before handshake, Store access,
        or workload setup. It never treats future-worker requests as active
        selected values.
    """

    if runtime_spec.mode is not RuntimeMode.WORKER:
        raise SessionConfigurationError("worker session runtime selection must use worker mode")
    if requirement_policy not in {"strict", "warn", "ignore"}:
        raise SessionConfigurationError("worker compatibility policy is invalid")
    if not isinstance(requirement_axes, RequirementAxes):
        raise SessionConfigurationError("worker compatibility axes are invalid")
    if not getattr(allocation, "world_allocation_id", None) or not getattr(allocation, "role", None):
        raise SessionConfigurationError("worker session requires an exact allocation identity")
    if runtime_spec.world_allocation_id != allocation.world_allocation_id:
        raise SessionConfigurationError("worker runtime and allocation identities must match")
    if allocation.role not in world.roles:
        raise SessionConfigurationError("worker allocation role is absent from selected world")
    process = ProcessAllocation(
        replica=allocation.replica,
        rank=allocation.rank,
        local_rank=allocation.local_rank,
        cpus=tuple(allocation.cpus),
        memory=allocation.memory,
        accelerators=allocation.accelerators,
        accelerator_memory=allocation.accelerator_memory,
        env=allocation.env,
        metadata=allocation.metadata,
    )
    selected = SelectedWorldAllocation(allocation.role, process)
    worker_runtime = RuntimeState(
        RuntimeMode.WORKER,
        allocation,
        spec=runtime_spec,
        enforcement=RuntimeEnforcement.STRICT,
        requirement_axes=requirement_axes,
    )
    before = publication.current()
    bootstrap = _bootstrap_plan(worker_runtime)
    framework_results = dict(bootstrap.framework_results)
    framework_statuses = _pending_framework_statuses(framework_results)
    configuration = SessionConfiguration(
        "managed",
        None,
        selected,
        None,
        None,
        EnvironmentRequirement(),
        _controls(None, selected),
        requirement_axes,
    )
    registry_revision, registry_frozen = framework_registry.state()
    observed_roots = _loaded_framework_roots()
    plan = _effect_plan(worker_runtime, before, bootstrap)
    control_epoch = int(before.metadata.get("control_epoch", before.number))
    if publication._changes_process_effects(plan):
        control_epoch = before.number + 1
    generation = SessionGeneration(
        before.number + 1,
        worker_runtime,
        visibility_epoch=before.visibility_epoch,
        metadata={
            "session_configuration": configuration,
            "session_active": True,
            "selected_environment": environment,
            "selected_world": world,
            "selected_runtime": runtime_spec,
            "compatibility_policy": requirement_policy,
            "worker_session": True,
            "framework_results": framework_results,
            "frameworks": tuple(framework_results),
            "framework_statuses": framework_statuses,
            "framework_registry_revision": registry_revision,
            "control_epoch": control_epoch,
        },
    )
    freeze_needed = not registry_frozen
    freeze_started = False

    def rollback_validator() -> None:
        """Undo this publication's provisional registry freeze on failure."""

        if freeze_started:
            framework_registry.unfreeze()

    def validate() -> None:
        """Revalidate framework state before publishing worker controls."""

        nonlocal freeze_started
        if framework_registry.state() != (registry_revision, registry_frozen):
            raise PublicationError("framework adapter registry changed while staging worker session")
        if _loaded_framework_roots() != observed_roots:
            raise PublicationError("framework import changed while staging worker session")
        if observed_roots:
            raise PublicationError("worker session must be published before framework imports")
        if freeze_needed:
            freeze_started = True
            framework_registry.freeze()

    try:
        committed = publication.commit(
            publication.stage(before, generation),
            plan,
            validator=validate,
            validator_rollback=rollback_validator if freeze_needed else None,
        )
    except Exception as exc:
        raise SessionConfigurationError("worker session publication failed", context={"error": type(exc).__name__}) from exc
    return _snapshot(committed)


def _configuration(generation: SessionGeneration) -> SessionConfiguration:
    if generation.health == "failed":
        return SessionConfiguration("orchestrator")
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
        configuration.requested_environment,
        configuration.requested_world,
        configuration.environment,
        configuration.controls,
        statuses,
        generation.runtime,
        generation.number,
        generation.health,
        generation.visibility_epoch if isinstance(generation.visibility_epoch, LocalResourceInventory) else None,
        requirement_axes=generation.runtime.requirement_axes,
        selected_environment=generation.metadata.get("selected_environment"),
        selected_world=generation.metadata.get("selected_world"),
        selected_runtime=generation.metadata.get("selected_runtime"),
        compatibility_policy=generation.metadata.get("compatibility_policy"),
        compatibility_axes=(
            generation.runtime.requirement_axes
            if generation.metadata.get("worker_session")
            else None
        ),
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


def _runtime_for(mode: str, allocation: SelectedWorldAllocation | None, requirement_axes: RequirementAxes) -> RuntimeState:
    """Project one public session mode into its low-level runtime state."""

    if mode == "managed":
        assert allocation is not None
        view = allocation.process.to_runtime_resource_view(role=allocation.role).to_runtime_allocation_view()
        return RuntimeState(RuntimeMode.INLINE, view, enforcement=RuntimeEnforcement.STRICT, requirement_axes=requirement_axes)
    runtime_mode = RuntimeMode.NONE if mode == "python" else RuntimeMode.ORCHESTRATOR
    enforcement = RuntimeEnforcement.OFF if mode == "python" else RuntimeEnforcement.STRICT
    return RuntimeState(runtime_mode, NoAllocation, enforcement=enforcement, requirement_axes=requirement_axes)


def _loaded_framework_roots() -> tuple[str, ...]:
    roots = {
        root
        for registration in framework_registry.registrations().values()
        for root in registration.roots
    }
    loaded = tuple(sys.modules)
    return tuple(sorted(root for root in roots if any(name == root or name.startswith(root + ".") for name in loaded)))


def _publish(
    mode: str,
    resources: ResourceSpec | None,
    allocation: SelectedWorldAllocation | tuple[Any, str | None, int | None] | None,
    requested_environment: EnvironmentSpec | None,
    requested_world: WorldSpec | None,
    environment: EnvironmentRequirement,
    requirement_axes: RequirementAxes,
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
        candidate = SessionConfiguration(
            mode,
            candidate_resources,
            candidate_allocation,
            requested_environment,
            requested_world,
            environment,
            _controls(candidate_resources, candidate_allocation),
            requirement_axes,
        )
        _check_current_environment(mode, environment, candidate.requirement_axes)
        runtime = _runtime_for(mode, candidate_allocation, candidate.requirement_axes)
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
        registry_revision, registry_frozen = framework_registry.state()
        plan = _effect_plan(runtime, before, bootstrap)
        control_epoch = int(before.metadata.get("control_epoch", before.number))
        if publication._changes_process_effects(plan):
            control_epoch = before.number + 1
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
                "control_epoch": control_epoch,
                # The core ContextVar projects this epoch as definition mode
                # without absorbing unrelated repository/cache configuration.
                "object_mode_floor": control_epoch if mode == "orchestrator" else None,
            },
        )

        freeze_needed = target_active and not registry_frozen
        freeze_started = False

        def rollback_validator() -> None:
            """Undo this transition's provisional registry freeze on failure.

            Returns:
                None.
            """
            if freeze_started:
                framework_registry.unfreeze()

        def validate() -> None:
            """Revalidate staged inventory, registry, imports, and freeze state.

            Returns:
                None.
            """
            nonlocal freeze_started
            if observed_from_probe and not _same_visibility_inventory(local_inventory(), observed):
                raise PublicationError(
                    "session inventory drifted while staging; restage before applying effects",
                    context={"reason": "inventory_changed"},
                )
            if framework_registry.state() != (registry_revision, registry_frozen):
                raise PublicationError("framework adapter registry changed while staging; restage before applying effects")
            if _loaded_framework_roots() != observed_roots:
                raise PublicationError("framework import changed while staging; restage before applying effects")
            if observed_roots and adapter_plan_changes:
                raise PublicationError("restart the process; a framework was imported before its adapter plan could change")
            if freeze_needed:
                # Arm rollback before calling the mutator so a mutate-then-raise
                # BaseException cannot strand the process registry frozen.
                freeze_started = True
                framework_registry.freeze()

        try:
            committed = publication.commit(
                publication.stage(before, generation),
                plan,
                validator=validate,
                validator_rollback=rollback_validator if freeze_needed else None,
            )
        except PublicationBusyError as exc:
            if exc.context.get("reason") == "writer_busy":
                time.sleep(0)
                continue
            raise
        except PublicationError as exc:
            if exc.context.get("reason") in {"inventory_changed", "stale_candidate"}:
                continue
            raise
        return _snapshot(committed)
    raise SessionConfigurationError("session transition could not obtain a stable inventory")


def _same_visibility_inventory(left: LocalResourceInventory, right: LocalResourceInventory) -> bool:
    """Compare inherited facts that must stay fixed across visibility setup.

    Available process memory is a volatile, declarative observation rather than
    a visibility control, so it remains part of the retained epoch but not its
    transition fence.
    """

    return (
        left.cpus == right.cpus
        and left.accelerators == right.accelerators
        and left.accelerator_memory == right.accelerator_memory
    )


def _controls(resources: ResourceSpec | None, allocation: SelectedWorldAllocation | None) -> dict[str, str]:
    process = None if allocation is None else allocation.process
    return {
        "memory": "declarative" if (resources and resources.memory is not None) or (process and process.memory is not None) else "undeclared",
        "accelerator_memory": "declarative" if (resources and resources.accelerator_memory) or (process and process.accelerator_memory) else "undeclared",
    }


def _check_current_environment(
    mode: str,
    environment: EnvironmentRequirement,
    requirement_axes: RequirementAxes,
) -> None:
    """Reject an incompatible managed interpreter before process effects begin."""

    if (
        mode != "managed"
        or "environment" not in requirement_axes.enabled
        or not any((environment.requirements, environment.python, environment.excludes, environment.capabilities))
    ):
        return
    report = environment.check(inspect_current(), policy="strict")
    if not report.ok:
        raise SessionConfigurationError("current environment does not satisfy managed session requirements", context={"issues": [issue.code for issue in report.issues]})


def _effect_plan(runtime: RuntimeState, before: SessionGeneration, bootstrap: Any) -> EffectPlan:
    if (
        _same_runtime_process_effects(before.runtime, runtime)
        and before.metadata.get("session_active")
        == (runtime.enforcement is RuntimeEnforcement.STRICT)
    ):
        # Declarative requirements and worker intent may advance independently
        # without invalidating a leased direct invocation.
        return EffectPlan()
    if runtime.mode in {RuntimeMode.INLINE, RuntimeMode.WORKER}:
        visibility = dict(bootstrap.env_updates)
        visibility.update({key: "" for key in _VISIBILITY_KEYS if key not in visibility})
        allocation = runtime.allocation
        if getattr(allocation, "accelerators", {}):
            from dryml.runtime import build_device_visibility_plan

            visibility.update(build_device_visibility_plan(mode=runtime.mode, allocation_view=allocation, policy="assigned").env_updates)
        return EffectPlan(environment=visibility, cpu_affinity=tuple(allocation.cpus))
    if runtime.enforcement is RuntimeEnforcement.STRICT:
        environment = dict(bootstrap.env_updates)
        affinity = None
        visibility_identities = {key.casefold() for key in _VISIBILITY_KEYS}
        for record in publication.effect_journal():
            if record.kind == "environment" and str(record.key).casefold() not in visibility_identities:
                environment[record.key] = record.previous
            elif record.kind == "cpu_affinity":
                affinity = record.previous
        environment.update({key: "" for key in _VISIBILITY_KEYS})
        return EffectPlan(environment=environment, cpu_affinity=affinity)
    restore_environment: dict[str, str | None] = {}
    affinity = None
    for record in publication.effect_journal():
        if record.kind == "environment":
            restore_environment[record.key] = record.previous
        elif record.kind == "cpu_affinity":
            affinity = record.previous
    return EffectPlan(environment=restore_environment, cpu_affinity=affinity)


def _same_runtime_process_effects(left: RuntimeState, right: RuntimeState) -> bool:
    """Compare runtime fields that can change process-global controls."""

    return (
        left.mode is right.mode
        and left.allocation == right.allocation
        and left.spec == right.spec
        and left.enforcement is right.enforcement
    )


def _bootstrap_plan(runtime: RuntimeState):
    visibility = "assigned" if runtime.mode in {RuntimeMode.INLINE, RuntimeMode.WORKER} else ("inherit" if runtime.mode is RuntimeMode.NONE else "none")
    spec = runtime.spec or RuntimeContextSpec(mode=runtime.mode, device_visibility={"policy": visibility})
    frameworks = ("plain", "tensorflow", "torch", "jax") if runtime.enforcement is RuntimeEnforcement.STRICT else ("plain",)
    return build_runtime_bootstrap_plan(spec, runtime.allocation, policy=FrameworkBootstrapPolicy(frameworks))


def _pending_framework_statuses(results: Mapping[str, FrameworkBootstrapResult]) -> dict[str, str]:
    """Publish an honest pending control outcome for each planned root."""

    statuses: dict[str, str] = {}
    registrations = framework_registry.registrations()
    for name in results:
        registration = registrations.get(name)
        if registration is None:
            continue
        for root in registration.roots:
            for control in ("visibility", "threads", "process_memory", "accelerator_memory", "allocator"):
                statuses[f"{name}:{root}:{control}"] = "pending-import"
    return statuses
__all__ = ["allocate_world", "configure", "current", "enforce_requirements", "manage", "mode", "publish_worker_session", "require_env", "reset", "set_mode", "worker_env_request", "worker_world_request"]
