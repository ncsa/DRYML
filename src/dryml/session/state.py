"""Publication-backed persistent state for the public session facade."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

from dryml.environments import EnvironmentRequirement, inspect_current
from dryml.formats import deep_freeze_json
from dryml.runtime import EffectPlan, NoAllocation, RuntimeAllocationView, RuntimeMode, RuntimeState, build_control_plan, build_device_visibility_plan, publication, validate_framework_transition
from dryml.runtime.errors import PublicationError
from dryml.worlds import LocalResourceInventory, ProcessAllocation, ResourceSpec, WorldAllocation, WorldSpec, assign_local_world, local_inventory

from .configuration import _controls, _normalize_environment, _normalize_resources, normalize_configuration, select_world_allocation
from .errors import SessionConfigurationError
from .model import SelectedWorldAllocation, SessionConfiguration, SessionSnapshot, default_requirement_axes, freeze_requirement_axes

_SESSION_CONTROL = "session_configuration"


def _session_operation(name: str):
    """Attach a stable public operation name while preserving failure causes."""

    def decorate(func):
        @wraps(func)
        def call(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (SessionConfigurationError, PublicationError) as exc:
                exc.context["operation"] = name
                raise

        return call

    return decorate


@_session_operation("session.current")
def current() -> SessionSnapshot:
    """Return the current session projection without host observation.

    Returns:
        Immutable snapshot of the current runtime publication generation.

    Side Effects:
        Records an ordinary publication read only; no inventory, environment,
        annotation, framework, or optional-package work occurs.
    """

    return _snapshot(publication.current())


@_session_operation("session.mode")
def mode() -> str:
    """Return the current public mode without host observation.

    Returns:
        Exactly ``python``, ``managed``, or ``orchestrator``.

    Side Effects:
        Records an ordinary publication read only.
    """

    return _configuration(publication.current()).mode


@_session_operation("session.set_mode")
def set_mode(value: str) -> SessionSnapshot:
    """Set a public mode and apply only that mode's default compatibility axes.

    Args:
        value: Target ``python``, ``managed``, or ``orchestrator`` mode.

    Returns:
        Snapshot of the committed generation.

    Raises:
        SessionConfigurationError: If the mode is unsupported.

    Side Effects:
        Entering managed mode can observe inventory and apply reversible runtime
        visibility controls. Leaving managed clears the exact allocation.
    """

    if not isinstance(value, str) or value not in {"python", "managed", "orchestrator"}:
        raise SessionConfigurationError("session mode must be python, managed, or orchestrator")
    before = _configuration(publication.current())
    if value == "managed" and before.mode == "managed" and before.allocation is not None:
        return _publish(value, before.resources, before.allocation, before.environment, default_requirement_axes(value))
    return _publish(value, None, None, before.environment, default_requirement_axes(value), synthesize=value == "managed")


@_session_operation("session.manage")
def manage(*, cpus: int | None = None, memory: str | int | None = None, gpus: int | None = None, accelerator_memory: Any = None) -> SessionSnapshot:
    """Enter managed mode using a new concise current-process allowance.

    Args:
        cpus: Optional positive inherited CPU count.
        memory: Optional positive declarative process-memory allowance.
        gpus: Optional non-negative GPU count.
        accelerator_memory: Scalar or per-GPU positive memory allowance.

    Returns:
        Snapshot of the committed managed generation.

    Raises:
        SessionConfigurationError: If resource data is malformed or cannot fit
            inherited inventory.

    Side Effects:
        Observes lightweight inventory, plans a deterministic exact allocation,
        and may apply reversible device visibility controls.
    """

    supplied = {name: item for name, item in {"cpus": cpus, "memory": memory, "gpus": gpus, "accelerator_memory": accelerator_memory}.items() if item is not None}
    # Validate the caller declaration before observing inventory for defaults.
    _normalize_resources(supplied)
    before = _configuration(publication.current())
    return _publish("managed", None, None, before.environment, before.requirement_axes, synthesize=True, simple_resources=supplied)


@_session_operation("session.allocate_world")
def allocate_world(value: WorldAllocation | Mapping[str, Any], /, *, role: str | None = None, replica: int | None = None) -> SessionSnapshot:
    """Enter managed mode with one selected exact world-allocation process.

    Args:
        value: Exact v1.1 world allocation or typed allocation.
        role: Role selector required for a multi-process allocation.
        replica: Role replica selector paired with ``role``.

    Returns:
        Snapshot of the committed managed generation.

    Raises:
        SessionConfigurationError: If selection is ambiguous, absent, malformed,
            or broader than inherited inventory.

    Side Effects:
        Observes lightweight inventory before publishing reversible controls.
    """

    before = _configuration(publication.current())
    return _publish("managed", None, (value, role, replica), before.environment, before.requirement_axes, exact=True)


@_session_operation("session.require_env")
def require_env(*requirements: str, python: str | None = None, excludes: Any = (), capabilities: Any = ()) -> SessionSnapshot:
    """Atomically merge current-process environment requirements.

    Args:
        requirements: PEP 508 distribution requirements.
        python: Optional Python version specifier.
        excludes: Iterable of excluded distribution names.
        capabilities: Iterable of required capability names.

    Returns:
        Snapshot of the committed generation.

    Raises:
        SessionConfigurationError: If input is malformed, constraints conflict,
            or an enabled managed environment check fails.

    Side Effects:
        A non-empty requirement in managed mode with the environment axis enabled
        lightweight-inspects the current interpreter before publication.
    """

    if not requirements and python is None and not excludes and not capabilities:
        raise SessionConfigurationError("require_env requires at least one constraint")
    before = _configuration(publication.current())
    try:
        addition = _normalize_environment({"requirements": requirements, "python": python, "excludes": excludes, "capabilities": capabilities})
        environment = before.environment.merge(addition, sources=("session",))
    except Exception as exc:
        raise SessionConfigurationError(str(exc), context=getattr(exc, "context", {})) from exc
    return _publish(before.mode, before.resources, before.allocation, environment, before.requirement_axes)


@_session_operation("session.enforce_requirements")
def enforce_requirements(*, environment: bool, world: bool, runtime: bool) -> SessionSnapshot:
    """Replace all identity-bearing compatibility-axis values atomically.

    Args:
        environment: Enable current-environment compatibility checks.
        world: Retain world compatibility policy state.
        runtime: Retain runtime compatibility policy state.

    Returns:
        Snapshot of the committed generation.

    Raises:
        SessionConfigurationError: If any value is not an exact boolean.

    Side Effects:
        May inspect the current interpreter only when enabling a non-empty
        managed environment requirement.
    """

    axes = {"environment": environment, "world": world, "runtime": runtime}
    if any(type(item) is not bool for item in axes.values()):
        raise SessionConfigurationError("session requirement axes must be exact booleans")
    before = _configuration(publication.current())
    return _publish(before.mode, before.resources, before.allocation, before.environment, axes)


@_session_operation("session.configure")
def configure(
    *,
    mode: str,
    resources: Mapping[str, Any] | None = None,
    allocation: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    requirement_axes: Mapping[str, bool] | None = None,
    restage_retries: int = 2,
    **extra: Any,
) -> SessionSnapshot:
    """Replace every public session category from one closed declaration.

    Args:
        mode: Required target public mode.
        resources: Optional concise managed resource declaration.
        allocation: Optional exact allocation plus paired selectors.
        environment: Optional complete replacement software requirement.
        requirement_axes: Optional complete replacement axis mapping.
        restage_retries: Operation-local publication retry count from 0 through
            16; it is not persisted or part of configuration identity.
        extra: Rejected unknown and deprecated source-v1 fields.

    Returns:
        Snapshot of the complete committed replacement.

    Raises:
        SessionConfigurationError: If categories are malformed or inconsistent.
        PublicationError: If runtime publication cannot safely complete.

    Side Effects:
        Validates all categories before observing inventory, then may apply
        reversible visibility controls as one generation publication.
    """

    if type(restage_retries) is not int or not 0 <= restage_retries <= 16:
        raise SessionConfigurationError("restage_retries must be an operation-local integer from 0 through 16")
    try:
        candidate = normalize_configuration(mode=mode, resources=resources, allocation=allocation, environment=environment, requirement_axes=requirement_axes, **extra)
    except SessionConfigurationError:
        raise
    allocation_value = None if allocation is None else (allocation["value"], allocation.get("role"), allocation.get("replica"))
    return _publish(candidate.mode, candidate.resources, allocation_value, candidate.environment, candidate.requirement_axes, synthesize=candidate.mode == "managed" and allocation_value is None, exact=allocation_value is not None, restage_retries=restage_retries)


@_session_operation("session.reset")
def reset() -> SessionSnapshot:
    """Restore the ordinary Python baseline and clear every facade category.

    Returns:
        Snapshot of the reset generation.

    Raises:
        PublicationError: If owned process effects cannot safely be restored.

    Side Effects:
        Restores only publication-owned reversible process effects.
    """

    candidate = SessionConfiguration("python", environment=EnvironmentRequirement(), requirement_axes=default_requirement_axes("python"), controls=_controls(None, None))
    before = publication.current()
    if candidate == _configuration(before) and not publication.effect_journal():
        return _snapshot(before)
    runtime = _runtime_for(candidate)
    return _snapshot(publication.reset(runtime))


def _publish(
    target_mode: str,
    resources: ResourceSpec | None,
    allocation: SelectedWorldAllocation | tuple[Any, str | None, int | None] | None,
    environment: EnvironmentRequirement,
    requirement_axes: Mapping[str, bool],
    *,
    synthesize: bool = False,
    exact: bool = False,
    simple_resources: Mapping[str, Any] | None = None,
    restage_retries: int = 2,
) -> SessionSnapshot:
    """Observe, validate, plan, and publish one complete session generation."""

    if type(restage_retries) is not int or not 0 <= restage_retries <= 16:
        raise SessionConfigurationError("restage_retries must be an operation-local integer from 0 through 16")
    for attempt in range(restage_retries + 1):
        observed = local_inventory() if target_mode in {"managed", "orchestrator"} else None
        selected = allocation
        candidate_resources = resources
        if target_mode == "managed":
            assert observed is not None
            if exact:
                assert isinstance(selected, tuple)
                selected = select_world_allocation(selected[0], role=selected[1], replica=selected[2], inventory=observed)
            elif synthesize:
                candidate_resources = _managed_resources(simple_resources or {}, observed) if simple_resources is not None else (resources or _managed_resources({}, observed))
                selected = _synthesize(candidate_resources, observed)
            if not isinstance(selected, SelectedWorldAllocation):
                raise SessionConfigurationError("managed session requires one exact selected allocation")
        else:
            candidate_resources = None
            selected = None
        candidate = SessionConfiguration(target_mode, candidate_resources, selected, environment, requirement_axes, _controls(candidate_resources, selected))
        _check_current_environment(candidate)
        runtime = _runtime_for(candidate)
        before = publication.current()
        if target_mode == "python":
            if candidate == _configuration(before) and not publication.effect_journal():
                return _snapshot(before)
            return _snapshot(publication.reset(runtime))
        effects = _effects_for(runtime, before.runtime)
        if effects.changes_process:
            validate_framework_transition(runtime)

        def observe() -> LocalResourceInventory:
            """Fence resource-dependent plans against inventory replacement."""

            fresh = local_inventory()
            if target_mode == "managed" and fresh.visibility_identity != observed.visibility_identity:
                raise PublicationError("session inventory changed while staging", context={"reason": "inventory_changed"})
            return fresh

        statuses = _control_statuses(runtime, effects)
        if not effects.changes_process:
            statuses = {**before.statuses, **statuses}
        try:
            committed = publication.publish(runtime, inventory_observer=observe, effects=effects, statuses=statuses, restage_retries=restage_retries)
        except PublicationError as exc:
            if exc.context.get("reason") == "inventory_changed" and attempt < restage_retries:
                continue
            raise
        return _snapshot(committed)
    raise SessionConfigurationError("session transition could not obtain stable inventory")


def _configuration(generation: Any) -> SessionConfiguration:
    """Rebuild the durable facade configuration held by a runtime generation."""

    data = getattr(generation.runtime, "controls", {}).get(_SESSION_CONTROL)
    if not isinstance(data, Mapping):
        return SessionConfiguration("python", environment=EnvironmentRequirement(), requirement_axes=default_requirement_axes("python"), controls=_controls(None, None))
    try:
        return SessionConfiguration.from_payload(data)
    except Exception as exc:
        raise SessionConfigurationError("published session metadata is invalid", context={"error": type(exc).__name__}) from exc


def _snapshot(generation: Any) -> SessionSnapshot:
    """Project the exact current generation, including finalized statuses."""

    configuration = _configuration(generation)
    statuses = dict(generation.statuses)
    statuses.setdefault("memory", configuration.controls.get("memory", "undeclared"))
    statuses.setdefault("accelerator_memory", configuration.controls.get("accelerator_memory", "undeclared"))
    return SessionSnapshot(configuration.mode, configuration.resources, configuration.allocation, configuration.environment, configuration.requirement_axes, configuration.controls, statuses, generation.runtime, generation.number, generation.health, generation.inventory)


def _managed_resources(supplied: Mapping[str, Any], inventory: LocalResourceInventory) -> ResourceSpec:
    """Complete omitted managed resource defaults from one observed inventory."""

    data = dict(supplied)
    data.setdefault("cpus", len(inventory.cpus))
    data.setdefault("gpus", 0)
    if "memory" not in data and inventory.memory is not None:
        data["memory"] = inventory.memory
    return _normalize_resources(data)


def _synthesize(resources: ResourceSpec, inventory: LocalResourceInventory) -> SelectedWorldAllocation:
    """Bind one concise process request into an exact local allocation."""

    world = WorldSpec.from_payload({"roles": {"main": {"replicas": 1, "process": {"resources": resources.to_data()}}}})
    allocation = assign_local_world(world, inventory=inventory)
    return select_world_allocation(allocation, inventory=inventory)


def _runtime_for(configuration: SessionConfiguration) -> RuntimeState:
    """Project a public candidate into the target's closed low-level modes."""

    controls = deep_freeze_json({_SESSION_CONTROL: configuration.to_payload()})
    if configuration.mode == "managed":
        selected = configuration.allocation
        assert selected is not None
        process = selected.process
        allocation = RuntimeAllocationView(selected.role, process.replica, process.rank, process.local_rank, process.cpus, process.memory, process.accelerators, process.accelerator_memory, process.env, metadata=process.metadata)
        return RuntimeState(RuntimeMode.INLINE, allocation, controls=controls)
    return RuntimeState(RuntimeMode.NONE if configuration.mode == "python" else RuntimeMode.ORCHESTRATOR, NoAllocation, controls=controls)


def _effects_for(runtime: RuntimeState, previous: RuntimeState) -> EffectPlan:
    """Plan owned reversible effects for one session runtime transition.

    Managed sessions own their exact allocation environment, device visibility,
    CPU affinity when the platform seam supports it, and process-memory limits
    only when a supported setter exists. Non-managed transitions release all
    session-owned allocation controls except orchestrator visibility hiding.
    """

    if runtime.mode == previous.mode and runtime.allocation == previous.allocation:
        return EffectPlan()
    previous_allocation = previous.allocation if isinstance(previous.allocation, RuntimeAllocationView) else None
    allocation = runtime.allocation if isinstance(runtime.allocation, RuntimeAllocationView) else None
    visibility = dict(build_device_visibility_plan(mode=runtime.mode, allocation=runtime.allocation).env_updates)
    environment = {} if allocation is None else dict(allocation.env)
    environment.update(visibility)
    visibility_keys = set(visibility)
    previous_environment = {} if previous_allocation is None else dict(previous_allocation.env)
    releases = tuple(sorted(name for name in previous_environment if name not in environment and name not in visibility_keys))
    affinity = None
    memory = None
    if allocation is not None and allocation.cpus and publication.supports_cpu_affinity:
        affinity = allocation.cpus
    if allocation is not None and allocation.memory is not None and publication.supports_process_memory:
        memory = allocation.memory
    return EffectPlan(
        environment=environment,
        cpu_affinity=affinity,
        process_memory=memory,
        release_environment=releases,
        release_cpu_affinity=previous_allocation is not None and affinity is None,
        release_process_memory=previous_allocation is not None and memory is None,
    )


def _control_statuses(runtime: RuntimeState, effects: EffectPlan) -> dict[str, str]:
    """Return truthful pre-import statuses for session-owned control effects."""

    allocation = runtime.allocation if isinstance(runtime.allocation, RuntimeAllocationView) else None
    affinity_requested = allocation is not None and bool(allocation.cpus)
    memory_requested = allocation is not None and allocation.memory is not None
    accelerator_memory = allocation is not None and bool(allocation.accelerator_memory)
    statuses = {name: getattr(value, "value", value) for name, value in build_control_plan(runtime.mode, affinity=affinity_requested, process_memory=memory_requested, accelerator_memory=accelerator_memory).statuses.items()}
    if affinity_requested:
        statuses["affinity"] = "enforced" if effects.cpu_affinity is not None else "unsupported"
    if memory_requested:
        statuses["process_memory"] = "enforced" if effects.process_memory is not None else "unsupported"
    return statuses


def _check_current_environment(configuration: SessionConfiguration) -> None:
    """Validate enabled managed software constraints before process effects."""

    environment = configuration.environment
    if configuration.mode != "managed" or not configuration.requirement_axes["environment"] or not any((environment.requirements, environment.python, environment.excludes, environment.capabilities, environment.tags, environment.dryml_protocol, environment.schema_versions)):
        return
    report = environment.check(inspect_current(), policy="strict")
    if not report.ok:
        raise SessionConfigurationError(
            "current environment does not satisfy managed session requirements",
            context={"category": "incompatible", "issues": [item.code for item in report.issues]},
        )


__all__ = ["allocate_world", "configure", "current", "enforce_requirements", "manage", "mode", "require_env", "reset", "set_mode"]
