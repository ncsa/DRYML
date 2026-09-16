"""Worker-owned activation of truthful backend resource grants.

This module composes immutable runtime intent with verified backend evidence. It
does not reserve resources, open Stores, deserialize workloads, or import
watched frameworks. Those operations must happen only after its scope enters.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dryml.formats import deep_freeze_json
from dryml.worlds import WorldAllocation

from .allocation import RuntimeAllocationView
from .bootstrap import validate_framework_transition
from .context import RuntimeState, publication
from .devices import DeviceVisibilityPolicy, build_device_visibility_plan
from .enforcement import build_control_plan
from .errors import RuntimeTransitionError
from .modes import RuntimeMode
from .publication import EffectPlan, PublicationService, SessionGeneration
from .specs import RuntimeContextSpec


@dataclass(frozen=True, slots=True)
class ExecutionGrant:
    """Verified backend grant available to one disposable execution worker.

    Exact subprocess evidence carries CPU IDs and a ``WorldAllocation`` ID.
    Ray evidence carries only a logical CPU quantity and detached native facts;
    it never gains affinity authority or an allocation identity by conversion.

    Args:
        backend: Backend that supplied the grant, currently ``subprocess`` or
            ``ray``.
        exact_cpus: Verified physical CPU IDs, available only to exact grants.
        logical_cpus: Verified logical CPU capacity, if the backend reported it.
        world_allocation_id: Exact source allocation identity, if available.
        role: Selected workload role, when exact allocation evidence supplies it.
        replica: Selected role replica, when exact allocation evidence supplies it.
        rank: Exact global rank, if supplied.
        local_rank: Exact local rank, if supplied.
        memory: Exact or logical backend-reported memory quantity, if supplied.
        accelerators: Verified accelerator IDs.
        accelerator_memory: Verified per-accelerator memory limits.
        env: Backend-provided process environment controls.
        native_evidence: Detached backend-native diagnostic evidence.

    Raises:
        RuntimeTransitionError: If exact and logical evidence are mixed or
            malformed.
    """

    backend: str
    exact_cpus: tuple[int, ...] = ()
    logical_cpus: int | None = None
    world_allocation_id: str | None = None
    role: str | None = None
    replica: int | None = None
    rank: int | None = None
    local_rank: int | None = None
    memory: int | None = None
    accelerators: Mapping[str, tuple[str | int, ...]] = field(default_factory=dict)
    accelerator_memory: Mapping[str, Mapping[str | int, int]] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    native_evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and detach the non-authoritative native evidence."""
        if self.backend not in {"subprocess", "ray"}:
            raise RuntimeTransitionError("execution grant backend must be subprocess or ray")
        cpus = tuple(sorted(self.exact_cpus))
        if len(cpus) != len(set(cpus)) or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in cpus):
            raise RuntimeTransitionError("execution grant CPU IDs must be unique non-negative integers")
        if self.logical_cpus is not None and (isinstance(self.logical_cpus, bool) or not isinstance(self.logical_cpus, int) or self.logical_cpus < 0):
            raise RuntimeTransitionError("execution grant logical CPU capacity must be a non-negative integer")
        if self.memory is not None and (isinstance(self.memory, bool) or not isinstance(self.memory, int) or self.memory < 0):
            raise RuntimeTransitionError("execution grant memory must be a non-negative integer")
        exact = self.world_allocation_id is not None
        if self.backend == "ray" and (cpus or exact):
            raise RuntimeTransitionError("Ray grants cannot claim exact CPU IDs or a world allocation identity")
        if exact and (not isinstance(self.role, str) or not self.role or self.replica is None):
            raise RuntimeTransitionError("exact execution grants require role and replica evidence")
        if not isinstance(self.accelerators, Mapping) or not isinstance(self.accelerator_memory, Mapping) or not isinstance(self.env, Mapping):
            raise RuntimeTransitionError("execution grant resource fields must be mappings")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.env.items()):
            raise RuntimeTransitionError("execution grant environment must be a string mapping")
        accelerators = {}
        for kind, values in self.accelerators.items():
            if not isinstance(kind, str) or not kind or isinstance(values, (str, bytes)):
                raise RuntimeTransitionError("execution grant accelerator IDs are invalid")
            try:
                identifiers = tuple(values)
            except TypeError as exc:
                raise RuntimeTransitionError("execution grant accelerator IDs are invalid") from exc
            if len(identifiers) != len(set(identifiers)) or any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in identifiers):
                raise RuntimeTransitionError("execution grant accelerator IDs are invalid")
            accelerators[kind] = identifiers
        accelerator_memory = {}
        for kind, limits in self.accelerator_memory.items():
            if kind not in accelerators or not isinstance(limits, Mapping):
                raise RuntimeTransitionError("execution grant accelerator memory is invalid")
            values = dict(limits)
            if set(values) - set(accelerators[kind]) or any(
                isinstance(device, bool) or not isinstance(device, (str, int))
                or isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for device, value in values.items()
            ):
                raise RuntimeTransitionError("execution grant accelerator memory is invalid")
            accelerator_memory[kind] = values
        object.__setattr__(self, "exact_cpus", cpus)
        object.__setattr__(self, "accelerators", MappingProxyType(accelerators))
        object.__setattr__(self, "accelerator_memory", MappingProxyType({key: MappingProxyType(value) for key, value in accelerator_memory.items()}))
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))
        object.__setattr__(self, "native_evidence", deep_freeze_json(self.native_evidence))

    @property
    def is_exact(self) -> bool:
        """Return whether this grant carries a source world-allocation identity."""
        return self.world_allocation_id is not None

    @property
    def thread_capacity(self) -> int | None:
        """Return an exact or logical CPU count suitable for thread planning only."""
        return len(self.exact_cpus) if self.is_exact else self.logical_cpus

    @classmethod
    def from_world_allocation(cls, allocation: WorldAllocation, *, role: str, replica: int = 0, native_evidence: Mapping[str, Any] | None = None) -> "ExecutionGrant":
        """Select one exact subprocess process grant from a world allocation.

        Args:
            allocation: Verified exact backend allocation.
            role: Required selected role name.
            replica: Required selected role replica.
            native_evidence: Optional detached backend evidence.

        Returns:
            A grant retaining only the selected process's exact authority.

        Raises:
            RuntimeTransitionError: If selection is absent or ambiguous.
        """
        selected = tuple(item for item in allocation.roles.get(role, ()) if item.replica == replica)
        if len(selected) != 1:
            raise RuntimeTransitionError("execution grant must select exactly one allocated process")
        process = selected[0]
        return cls("subprocess", process.cpus, len(process.cpus), allocation.semantic_id, role, process.replica, process.rank, process.local_rank, process.memory, process.accelerators, process.accelerator_memory, process.env, native_evidence or {})

    @classmethod
    def ray(cls, *, logical_cpus: int | None, memory: int | None = None, accelerators: Mapping[str, tuple[str | int, ...]] | None = None, accelerator_memory: Mapping[str, Mapping[str | int, int]] | None = None, native_evidence: Mapping[str, Any] | None = None) -> "ExecutionGrant":
        """Create a logical Ray grant without fabricating physical allocation facts.

        Args:
            logical_cpus: Verified integer Ray CPU capacity, if reported.
            memory: Verified logical Ray memory quantity, if reported.
            accelerators: Verified Ray accelerator IDs, if reported.
            accelerator_memory: Verified per-accelerator memory quantities.
            native_evidence: Optional detached native Ray observation.

        Returns:
            A logical-only Ray grant.
        """
        return cls("ray", logical_cpus=logical_cpus, memory=memory, accelerators=accelerators or {}, accelerator_memory=accelerator_memory or {}, native_evidence=native_evidence or {})

    @classmethod
    def from_worker_setup(cls, context: Any, *, role: str = "main", replica: int = 0) -> "ExecutionGrant":
        """Build one grant from the generic worker's verified setup context.

        Args:
            context: Generic ``WorkerSetupContext``-shaped evidence value.
            role: Exact allocation role selected by core setup.
            replica: Exact allocation replica selected by core setup.

        Returns:
            A subprocess exact or baseline grant, or a Ray logical grant.

        Raises:
            RuntimeTransitionError: If backend evidence is unsupported or has
                malformed logical resource evidence.
        """
        backend = getattr(context, "backend", None)
        native = getattr(context, "native_grant", {})
        if backend == "subprocess":
            allocation = getattr(context, "allocation", None)
            if allocation is None:
                return cls("subprocess", native_evidence=native if isinstance(native, Mapping) else {})
            if not isinstance(allocation, WorldAllocation):
                raise RuntimeTransitionError("subprocess setup allocation evidence is invalid")
            return cls.from_world_allocation(allocation, role=role, replica=replica, native_evidence=native)
        if backend == "ray":
            resources = native.get("resources", {}) if isinstance(native, Mapping) else {}
            value = resources.get("CPU") if isinstance(resources, Mapping) else None
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise RuntimeTransitionError("Ray setup requires an integer logical CPU quantity")
            memory = native.get("memory_bytes") if isinstance(native, Mapping) else None
            accelerators = native.get("accelerator_ids", {}) if isinstance(native, Mapping) else {}
            accelerator_memory = native.get("accelerator_memory", {}) if isinstance(native, Mapping) else {}
            if memory is not None and (isinstance(memory, bool) or not isinstance(memory, int) or memory < 0):
                raise RuntimeTransitionError("Ray setup memory evidence must be a non-negative integer")
            if not isinstance(accelerators, Mapping) or not isinstance(accelerator_memory, Mapping):
                raise RuntimeTransitionError("Ray setup accelerator evidence must be mappings")
            return cls.ray(logical_cpus=value, memory=memory, accelerators=accelerators, accelerator_memory=accelerator_memory, native_evidence=native if isinstance(native, Mapping) else {})
        raise RuntimeTransitionError("worker setup backend has no supported runtime grant")


@dataclass(frozen=True, slots=True)
class RuntimeActivation:
    """Published runtime state held by one active execution activation scope.

    Args:
        grant: Truthful backend evidence used for the active worker scope.
        state: Published ``INLINE`` runtime state projected from that grant.
        generation: Publication generation that installed the state and effects.

    Side Effects:
        This immutable value does not own the scope. The surrounding
        :func:`activation_scope` restores reversible effects on exit.
    """

    grant: ExecutionGrant
    state: RuntimeState
    generation: SessionGeneration


def _requires_exact_intent(spec: RuntimeContextSpec) -> bool:
    """Return whether declared controls require physical allocation authority."""
    return spec.world_allocation_id is not None or bool(spec.limits.get("affinity") or spec.limits.get("cpu_affinity"))


def _validate_framework_intent(spec: RuntimeContextSpec, grant: ExecutionGrant) -> None:
    """Reject unowned or over-capacity framework thread controls before import."""
    if "threads" in spec.limits or "thread_count" in spec.limits:
        raise RuntimeTransitionError("generic runtime thread limits are unsupported; declare a framework-owned thread control")
    from .frameworks import framework_registry

    registrations = framework_registry.registrations()
    for name, controls in spec.framework.items():
        if name not in registrations:
            raise RuntimeTransitionError("runtime framework controls require a registered framework")
        for key in ("threads", "num_threads", "interop_threads", "num_interop_threads"):
            value = controls.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeTransitionError("framework thread controls must be positive integers")
            if grant.thread_capacity is not None and value > grant.thread_capacity:
                raise RuntimeTransitionError("framework thread control exceeds verified backend CPU capacity")


def _state_for(spec: RuntimeContextSpec, grant: ExecutionGrant) -> RuntimeState:
    """Project runtime intent and grant evidence without mutating the process."""
    if spec.mode is not RuntimeMode.INLINE:
        raise RuntimeTransitionError("execution activation requires INLINE runtime intent")
    if _requires_exact_intent(spec) and not grant.is_exact:
        raise RuntimeTransitionError("runtime intent requires exact allocation evidence")
    if spec.world_allocation_id is not None and spec.world_allocation_id != grant.world_allocation_id:
        raise RuntimeTransitionError("runtime intent world allocation does not match the backend grant")
    if grant.is_exact:
        allocation = RuntimeAllocationView(grant.role, grant.replica, grant.rank, grant.local_rank, grant.exact_cpus, grant.memory, grant.accelerators, grant.accelerator_memory, grant.env, grant.world_allocation_id, logical_cpus=grant.thread_capacity, grant_provenance="exact")
    else:
        allocation = RuntimeAllocationView("main", 0, cpus=(), memory=grant.memory, accelerators=grant.accelerators, accelerator_memory=grant.accelerator_memory, logical_cpus=grant.logical_cpus, grant_provenance="logical" if grant.backend == "ray" else "baseline")
    return RuntimeState(RuntimeMode.INLINE, allocation, spec)


def _effects_for(state: RuntimeState, grant: ExecutionGrant, service: PublicationService) -> tuple[EffectPlan, dict[str, str]]:
    """Plan only controls backed by the active grant and publication seams."""
    spec = state.spec
    assert spec is not None
    declared_visibility = spec.visibility if isinstance(spec.visibility, Mapping) else {}
    policy = declared_visibility.get("policy")
    explicit = declared_visibility.get("devices")
    visibility = build_device_visibility_plan(mode=state.mode, allocation=state.allocation, policy=DeviceVisibilityPolicy(policy) if policy else None, explicit_devices=explicit)
    environment = {**grant.env, **spec.env, **visibility.env_updates}
    affinity = grant.exact_cpus if grant.is_exact and grant.exact_cpus and service.supports_cpu_affinity else None
    memory = grant.memory if grant.is_exact and grant.memory is not None and service.supports_process_memory else None
    statuses = {name: value.value for name, value in build_control_plan(state.mode, affinity=bool(grant.exact_cpus), process_memory=grant.memory is not None, framework_controls=bool(spec.framework), accelerator_memory=bool(grant.accelerator_memory)).statuses.items()}
    if grant.exact_cpus:
        statuses["affinity"] = "enforced" if affinity is not None else "unsupported"
    if grant.memory is not None:
        statuses["process_memory"] = "enforced" if memory is not None else "unsupported"
    return EffectPlan(environment=environment, cpu_affinity=affinity, process_memory=memory), statuses


def _watched_framework_loaded() -> bool:
    """Return whether a registered watched framework executed during activation."""
    from .frameworks import framework_registry

    return any(root in sys.modules for registration in framework_registry.registrations().values() for root in registration.roots)


@contextmanager
def activation_scope(spec: RuntimeContextSpec, grant: ExecutionGrant, *, service: PublicationService | None = None) -> Iterator[RuntimeActivation]:
    """Activate one worker runtime before Store access or workload deserialization.

    Args:
        spec: Immutable runtime intent owned by :class:`RuntimeContextSpec`.
        grant: Verified backend grant for this worker only.
        service: Optional injectable publication authority; the process-global
            authority is used in production.

    Yields:
        The published runtime state and generation for the active worker scope.

    Raises:
        RuntimeTransitionError: If the worker does not begin from a healthy,
            effect-free ``NONE`` baseline or intent exceeds the grant.
        PublicationError: If publication or effect restoration fails.

    Side Effects:
        Publishes pre-import controls and restores only journal-owned reversible
        effects at scope exit. A watched framework observed during the scope marks
        the worker terminal after restoration so a one-shot worker is retired.
    """
    if not isinstance(spec, RuntimeContextSpec) or not isinstance(grant, ExecutionGrant):
        raise TypeError("execution activation requires RuntimeContextSpec and ExecutionGrant")
    owner = publication if service is None else service
    baseline = owner.current()
    if baseline.health != "healthy" or getattr(baseline.runtime, "mode", None) is not RuntimeMode.NONE or owner.effect_journal():
        raise RuntimeTransitionError("execution activation requires a healthy NONE baseline with no owned effects")
    state = _state_for(spec, grant)
    validate_framework_transition(state)
    _validate_framework_intent(spec, grant)
    effects, statuses = _effects_for(state, grant, owner)
    generation = owner.commit(owner.stage(baseline, state, statuses=statuses), effects)
    try:
        yield RuntimeActivation(grant, state, generation)
    finally:
        watched = _watched_framework_loaded()
        owner.reset(baseline.runtime)
        if watched:
            owner.fail_status_finalization(None, RuntimeError("watched framework imported during execution activation"))


__all__ = ["ExecutionGrant", "RuntimeActivation", "activation_scope"]
