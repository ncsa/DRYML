"""High-level dispatch planning API for the local subprocess backend."""

from __future__ import annotations

import os
import shutil
import sys
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Callable

from dryml import worlds
from dryml.environments import CurrentEnvironmentSpec, PythonExecutableSpec
from dryml import runtime
from dryml.runtime import RuntimeAllocationView

from .errors import DispatchPlanningError
from .normalize import is_definition_or_cdef, normalize_user_operation
from .operations import PickledCallable
from .provenance import (
    project_environment_config,
    project_runtime_config,
    project_world_allocation_spec,
    project_world_spec,
)
from .protocol import DispatchResult, ExecutionEnvelope
from .requirements import DispatchExplanation, DispatchPlanningResolution, RequirementPolicy, _validate_sprint8_policies, effective_requirement_policy, explanation_for, parse_analysis_policy, resolve_dispatch_plan
from .recipes import attach_recipe_id, make_execution_recipe
from .specs import attach_dispatch_id, make_dispatch_spec
from .stores import require_supported_plan, same_host_dir_store, select_marshal_plan

_UNSET = object()


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    """Resolved one-operation local subprocess dispatch plan.

    Attributes:
        dispatch_spec: Canonical dispatch specification and planning metadata.
        execution_recipe: Canonical backend execution recipe.
        envelope: Launch-only worker envelope with assigned resources.
        store: Store used for marshalled operation and execution records.
        resolution: Requirement and resolver decisions used to build the plan.
    """

    dispatch_spec: Mapping[str, Any]
    execution_recipe: Mapping[str, Any]
    envelope: ExecutionEnvelope
    store: Any
    resolution: DispatchPlanningResolution | None = None
    extension: Any | None = None


class Dispatcher:
    """Plan and run one operation through a dispatch backend.

    Args:
        backend: Optional dispatch backend; local subprocess is the default.
        store: Optional default store used by plan and execution methods.
        environment_candidates: Finite sequence of resolver candidates retained
            across calls. Per-call candidates override this value.
        environment_registry: Optional explicit resolver registry.
        inventory: Optional injected local capacity facts.
        inventory_policy: ``"lightweight"`` or opt-in ``"external"`` discovery.
        resolver_policy: Optional resolver policy, currently
            ``"first_compatible"`` only.

    Retained candidates must be a concrete sequence so repeated notebook calls
    cannot consume a stateful iterator hidden behind an iterable wrapper.
    Per-call values retain ordinary iterable support and override defaults
    without mutating them.
    """

    def __init__(self, *, backend: Any | None = None, store: Any | None = None, environment_candidates: Any | None = None, environment_registry: Any | None = None, inventory: Any | None = None, inventory_policy: str = "lightweight", resolver_policy: str | None = None):
        from .backends import LocalSubprocessBackend

        self.backend = backend if backend is not None else LocalSubprocessBackend()
        self.store = store
        if environment_candidates is not None:
            if isinstance(environment_candidates, (str, bytes)) or not isinstance(environment_candidates, Sequence):
                raise TypeError(
                    "Dispatcher environment_candidates must be a finite re-iterable sequence; "
                    "pass a list or tuple when retaining candidates across calls"
                )
        self.environment_candidates = environment_candidates
        self.environment_registry = environment_registry
        self.inventory = inventory
        self.inventory_policy = inventory_policy
        self.resolver_policy = resolver_policy

    def plan(
        self,
        operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable,
        method_name: str | None = None,
        *,
        store: Any | None = None,
        environment: Any | Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
        world: Mapping[str, Any] | None = None,
        requirement_policy: str | None = None,
        analysis_policy: Any | None = None,
        environment_candidates: Any = _UNSET,
        environment_registry: Any = _UNSET,
        inventory: Any = _UNSET,
        inventory_policy: str | None = None,
        resolver_policy: str | None = None,
        record_policy: str = "descriptive",
        allow_pickle: bool = False,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        callbacks: Any = (),
        rerun: bool = False,
    ) -> DispatchPlan:
        """Build a requirement-checked dispatch plan and launch-only envelope.

        Args:
            operation: Explicit operation spec, callable, or pickled callable.
            method_name: Optional method for a DRYML definition/object target.
            store: Per-call store overriding the dispatcher default.
            environment: Explicit environment candidate.
            runtime: Explicit worker runtime specification.
            world: Explicit requested world payload, object, or canonical envelope.
            requirement_policy: ``"strict"``, ``"warn"``, or ``"ignore"``.
            analysis_policy: A `CodeAnalysisContext` compatibility value or a
                closed mapping. Only mapping `dynamic_trace=True` or an already
                validated `DynamicTracePolicy` requests one trusted,
                current-process trace; it is otherwise default-off.
            environment_candidates: Per-call ordered resolver candidates.
            environment_registry: Per-call explicit resolver registry.
            inventory: Per-call local inventory reused for synthesis/allocation.
            inventory_policy: ``"lightweight"`` or opt-in ``"external"``.
            resolver_policy: Resolver policy, currently ``"first_compatible"``.
            record_policy: Persistence policy for execution provenance.
            allow_pickle: Permit a non-importable callable transport.
            args: Positional arguments for Python-shaped calls.
            kwargs: Keyword arguments for Python-shaped calls.
            callbacks: Invocation-local callbacks for a bound managed method.
            rerun: Explicit managed rerun selection. It is excluded from the
                operation identity.

        Returns:
            A launchable local-subprocess plan with actual assigned resources.

        Explicit candidates select a target but never override hard requirements.
        Resolver candidates and registries run only after explicit/default/current
        environment slots are absent. Inventory is reused for synthesis and
        allocation.
        """

        _report("dryml.dispatch.plan.start", "Building dispatch plan")
        target_store = store or self.store
        if target_store is None:
            if is_definition_or_cdef(operation):
                normalize_user_operation(operation, method_name, args=args, kwargs=kwargs)
            raise DispatchPlanningError("Dispatcher.plan requires a store for local subprocess marshalling")
        extension = _make_dispatch_extension(
            operation,
            args=args,
            kwargs=kwargs,
            callbacks=callbacks,
            rerun=rerun,
        )
        if extension is not None and not same_host_dir_store(target_store):
            raise DispatchPlanningError(
                "writable managed dispatch requires a same-host DirStore"
            )
        # Managed operations have always been checked workloads.  The facade's
        # ordinary Python-mode bypass applies to direct calls, not this existing
        # managed-operation dispatch boundary.
        effective_requirement_policy = "strict" if extension is not None and requirement_policy is None else requirement_policy
        effective_inventory_policy = self.inventory_policy if inventory_policy is None else inventory_policy
        effective_resolver_policy = self.resolver_policy if resolver_policy is None else resolver_policy
        _validate_sprint8_policies(effective_inventory_policy, effective_resolver_policy)
        effective_candidates = self.environment_candidates if environment_candidates is _UNSET else environment_candidates
        effective_registry = self.environment_registry if environment_registry is _UNSET else environment_registry
        effective_inventory = self.inventory if inventory is _UNSET else inventory
        analysis_request = parse_analysis_policy(analysis_policy)
        _report("dryml.dispatch.requirements.gather", "Gathering environment/world/runtime requirements")
        normalized = normalize_user_operation(operation, method_name, store=target_store, allow_pickle=allow_pickle, args=args, kwargs=kwargs, trace_enabled=analysis_request.requested)
        op_spec = dict(normalized.operation_spec)
        launch = dict(normalized.launch)
        _report("dryml.dispatch.requirements.merge", "Merging requirements and defaults", operation_id=op_spec.get("id"))
        try:
            resolution = resolve_dispatch_plan(
                normalized,
                environment=environment,
                world=world,
                runtime_spec=runtime,
                requirement_policy=effective_requirement_policy,
                analysis_policy=analysis_policy,
                _analysis_request=analysis_request,
                environment_candidates=effective_candidates,
                environment_registry=effective_registry,
                inventory=effective_inventory,
                inventory_policy=effective_inventory_policy,
                resolver_policy=effective_resolver_policy,
                emit_warnings=True,
                single_worker_only=True,
            )
        except BaseException:
            _cleanup_launch(launch)
            raise
        if not resolution.launchable:
            _cleanup_launch(launch)
            if resolution.dynamic_trace is not None and resolution.dynamic_trace.data["status"] != "complete":
                raise DispatchPlanningError(
                    "requested dynamic trace did not produce complete planning evidence",
                    context={"dynamic_trace": resolution.dynamic_trace.to_data()},
                )
            if (
                resolution.dynamic_trace is not None
                and any(item.code == "dryml.dispatch.pickle_environment_restriction" for item in resolution.diagnostics)
            ):
                raise DispatchPlanningError(
                    "PickledCallable dispatch is restricted to the same Python executable",
                    context={"dynamic_trace": resolution.dynamic_trace.to_data()},
                )
            if any(item.code == "dryml.dispatch.pickle_environment_restriction" for item in resolution.diagnostics):
                raise DispatchPlanningError(
                    "PickledCallable dispatch is restricted to the same Python executable",
                    context={"environment": project_environment_config(resolution.environment_selection.candidate)},
                )
            if any(item.code == "dryml.dispatch.single_subprocess_world_unsupported" for item in resolution.diagnostics):
                raise DispatchPlanningError("selected world requires multiple workers; use plan_world() or run_world()")
            allocation_failure = next((item for item in resolution.diagnostics if item.code in {"dryml.dispatch.local_allocation_failed", "dryml.dispatch.local_allocation_requirement_failed"}), None)
            if allocation_failure is not None:
                detail = allocation_failure.data.get("message") if isinstance(allocation_failure.data, Mapping) else None
                raise DispatchPlanningError(
                    f"dispatch plan is not launchable: {detail or allocation_failure.message}",
                    context=dict(allocation_failure.data),
                )
            raise DispatchPlanningError(
                "dispatch plan is not launchable; call dispatch.explain(...) for requirement diagnostics",
                context={"planning": resolution.metadata()},
            )
        env_data = dict(resolution.environment_selection.candidate)
        if launch.get("call_transport") == "pickle_small" and not _same_python_environment(env_data):
            _cleanup_launch(launch)
            raise DispatchPlanningError(
                "PickledCallable dispatch is restricted to the same Python executable",
                context={"environment": project_environment_config(env_data)},
            )
        runtime_data = dict(resolution.runtime_selection.candidate)
        world_data = dict(resolution.world_selection.candidate)
        try:
            from .local_world import allocate_local_world

            selected_inventory = effective_inventory or resolution.local_inventory
            allocation_world = _subprocess_allocation_world(world_data)
            launch_world_spec = worlds.attach_world_id(worlds.make_world_spec(worlds.WorldSpec.from_data(world_data)))
            provenance_world_spec = project_world_spec(launch_world_spec)
            allocation_plan = allocate_local_world(
                allocation_world,
                inventory=selected_inventory,
                allocation_backend_kind="local_subprocess",
                requested_world_id=provenance_world_spec["id"],
            )
            _require_allocation_satisfies_requirement(allocation_plan.world_allocation, resolution.requirements.world_requirement, requirement_policy)
            launch_allocation_spec = allocation_plan.world_allocation_spec
            provenance_allocation_spec = project_world_allocation_spec(
                launch_allocation_spec,
                world_id=provenance_world_spec["id"],
            )
            key = allocation_plan.worker_keys[0]
            allocation = allocation_plan.world_allocation.runtime_view(
                key.role,
                key.replica,
                world_allocation_id=provenance_allocation_spec["id"],
            )
            allocation_data = _allocation_to_json(allocation, world_id=provenance_world_spec.get("id"))
            allocation_data["metadata"] = {
                **allocation_data["metadata"],
                "backend": "local_subprocess",
                # The canonical world spec is persisted separately. Do not copy
                # process.env values into execution metadata.
                "requested_world_id": provenance_world_spec["id"],
                "requested_world_backend": world_data.get("backend", {}).get("kind"),
            }
            resolution = replace(
                resolution,
                canonical_world_spec=launch_world_spec,
                world_allocation_summary={
                    "backend": "local_subprocess",
                    "allocation_policy": "disjoint_local",
                    "workers": [{"role": key.role, "replica": key.replica, "cpus": list(allocation.cpus), "memory": allocation.memory, "accelerators": {name: list(values) for name, values in allocation.accelerators.items()}}],
                },
            )
        except BaseException:
            _cleanup_launch(launch)
            raise
        try:
            marshal = select_marshal_plan(target_store, query_index="none")
            require_supported_plan(marshal)
        except BaseException:
            _cleanup_launch(launch)
            raise
        _report("dryml.dispatch.store.prepare", "Preparing shared DirStore marshalling", operation_id=op_spec.get("id"), data={"strategy": marshal.strategy})
        try:
            planning_metadata = resolution.metadata()
            dispatch = attach_dispatch_id(
                make_dispatch_spec(
                    operation_id=op_spec["id"],
                    operation=op_spec,
                    environment={"policy": resolution.environment_selection.source, "spec": project_environment_config(env_data)},
                    world={"policy": resolution.world_selection.source, "spec": provenance_world_spec["payload"]},
                    runtime={"policy": "worker", "spec": project_runtime_config(runtime_data)},
                    records={"record_policy": record_policy, "provenance": record_policy != "none"},
                    execution={"backend": "local_subprocess"},
                    metadata=planning_metadata,
                )
            )
            _report("dryml.dispatch.recipe.build", "Building execution recipe", operation_id=op_spec.get("id"), data={"dispatch_id": dispatch["id"]})
            recipe = attach_recipe_id(
                make_execution_recipe(
                    dispatch_id=dispatch["id"],
                    operation_id=op_spec["id"],
                    backend={"name": "dryml.local_subprocess", "kind": "local_subprocess", "version": "1"},
                    input_plan={"materialize_cdefs": [], "ref_cdefs": []},
                    output_plan={"record_policy": record_policy, "provenance": record_policy != "none"},
                    store_plan={"strategy": marshal.strategy, "roles": [ref.role for ref in marshal.store_refs]},
                    log_plan={"stdout": "capture", "stderr": "capture"},
                    constraints={"portable": launch.get("call_transport") != "pickle_small"},
                    annotation_report=planning_metadata,
                )
            )
            envelope = ExecutionEnvelope(
                dispatch_spec=dispatch,
                execution_recipe=recipe,
                operation_spec=op_spec,
                environment_spec=env_data,
                runtime_spec=runtime_data,
                allocation_view=allocation_data,
                store_refs=marshal.store_refs,
                transfer={"strategy": marshal.strategy},
                record_policy=record_policy,
                reporting={"planning": planning_metadata},
                launch={
                    **launch,
                    "world_id": provenance_world_spec["id"],
                    "world_allocation_id": provenance_allocation_spec["id"],
                    "world_spec": launch_world_spec,
                    "world_allocation_spec": launch_allocation_spec,
                    "provenance_world_spec": provenance_world_spec,
                    "provenance_world_allocation_spec": provenance_allocation_spec,
                    "parent_persisted_specs": record_policy != "none",
                },
            )
        except BaseException:
            _cleanup_launch(launch)
            raise
        if record_policy != "none":
            try:
                target_store.records.write_spec(op_spec, family="operation")
                target_store.records.write_spec(dispatch, family="dispatch")
                target_store.records.write_spec(recipe, family="execution_recipe")
                target_store.records.write_spec(provenance_world_spec, family="world")
                target_store.records.write_spec(provenance_allocation_spec, family="world_allocation")
            except BaseException:
                _cleanup_launch(launch)
                raise
        result = DispatchPlan(
            dispatch,
            recipe,
            envelope,
            target_store,
            resolution,
            extension,
        )
        return extension.decorate_plan(result) if extension is not None else result

    def submit(
        self,
        operation: DispatchPlan | Mapping[str, Any] | Callable[..., Any] | PickledCallable,
        method_name: str | None = None,
        **kwargs: Any,
    ):
        """Submit a prebuilt plan or plan and submit a user operation.

        Passing a :class:`DispatchPlan` preserves the original advanced API.
        Other targets are normalized through :meth:`plan` first.

        Args:
            operation: Existing plan or operation accepted by :meth:`plan`.
            method_name: Optional DRYML object method name.
            **kwargs: Planning arguments forwarded when ``operation`` is not a plan.

        Returns:
            Backend future for the submitted single-subprocess dispatch.
        """

        plan = operation if isinstance(operation, DispatchPlan) else self.plan(operation, method_name, **kwargs)
        if plan.extension is not None:
            return plan.extension.submit(plan, self.backend)
        return self.backend.submit(plan)

    def plan_world(
        self,
        operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable,
        method_name: str | None = None,
        *,
        world: Mapping[str, Any] | Any | None = None,
        store: Any | None = None,
        environment: Any | Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
        requirement_policy: str | None = None,
        analysis_policy: Any | None = None,
        record_policy: str = "descriptive",
        allow_pickle: bool = False,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        inventory: Any = _UNSET,
        inventory_policy: str | None = None,
        environment_candidates: Any = _UNSET,
        environment_registry: Any = _UNSET,
        resolver_policy: str | None = None,
        oversubscribe: bool = False,
    ):
        """Build a coordinated local-world dispatch plan.

        The first local-world operation policy intentionally runs the same
        operation spec in every allocated role/replica worker. User code can
        branch on ``dryml.runtime.require_workload_allocation(...)`` or the
        ``DRYML_WORLD_*`` environment variables.

        Args:
            operation: Explicit operation spec, callable, or pickled callable.
            method_name: Optional method for a DRYML definition/object target.
            world: Explicit world or ``None`` to synthesize a hard requirement.
            store: Per-call store overriding the dispatcher default.
            environment: Explicit environment candidate.
            runtime: Explicit worker runtime specification.
            requirement_policy: ``"strict"``, ``"warn"``, or ``"ignore"``.
            analysis_policy: Optional code-analysis policy.
            record_policy: Persistence policy for execution provenance.
            allow_pickle: Permit a non-importable callable transport.
            args: Positional arguments for Python-shaped calls.
            kwargs: Keyword arguments for Python-shaped calls.
            inventory: Per-call inventory reused for synthesis/allocation.
            inventory_policy: ``"lightweight"`` or opt-in ``"external"``.
            environment_candidates: Per-call ordered resolver candidates.
            environment_registry: Per-call explicit resolver registry.
            resolver_policy: Resolver policy, currently ``"first_compatible"``.
            oversubscribe: Permit the explicit local-world allocator to share IDs.

        Returns:
            A local-world plan containing one launch plan per assigned worker.

        An omitted ``world`` is synthesized only when no higher-precedence world
        exists and a hard world requirement is present. ``inventory`` and
        ``inventory_policy`` use the same semantics as :meth:`plan` and are
        retained for synthesis plus actual local allocation.
        """

        from .local_world import LOCAL_WORLD_BACKEND_IDENTITY, LocalWorldPlan, WorkerLaunchPlan, allocate_local_world

        _report("dryml.dispatch.world.plan.start", "Building local world dispatch plan")
        if _dispatch_extension_enabled(operation):
            raise DispatchPlanningError(
                "managed operations support one single local subprocess only; local-world execution is unsupported"
            )
        target_store = store or self.store
        if target_store is None:
            if is_definition_or_cdef(operation):
                normalize_user_operation(operation, method_name, args=args, kwargs=kwargs)
            raise DispatchPlanningError("Dispatcher.plan_world requires a store for shared DirStore marshalling")
        effective_inventory_policy = self.inventory_policy if inventory_policy is None else inventory_policy
        effective_resolver_policy = self.resolver_policy if resolver_policy is None else resolver_policy
        _validate_sprint8_policies(effective_inventory_policy, effective_resolver_policy)
        effective_candidates = self.environment_candidates if environment_candidates is _UNSET else environment_candidates
        effective_registry = self.environment_registry if environment_registry is _UNSET else environment_registry
        effective_inventory = self.inventory if inventory is _UNSET else inventory
        analysis_request = parse_analysis_policy(analysis_policy)
        normalized = normalize_user_operation(operation, method_name, store=target_store, allow_pickle=allow_pickle, args=args, kwargs=kwargs, trace_enabled=analysis_request.requested)
        op_spec = dict(normalized.operation_spec)
        launch = dict(normalized.launch)
        try:
            resolution = resolve_dispatch_plan(
                normalized,
                environment=environment,
                world=world,
                runtime_spec=runtime,
                requirement_policy=requirement_policy,
                analysis_policy=analysis_policy,
                _analysis_request=analysis_request,
                environment_candidates=effective_candidates,
                environment_registry=effective_registry,
                inventory=effective_inventory,
                inventory_policy=effective_inventory_policy,
                resolver_policy=effective_resolver_policy,
                emit_warnings=True,
            )
        except BaseException:
            _cleanup_launch(launch)
            raise
        if not resolution.launchable:
            _cleanup_launch(launch)
            if resolution.dynamic_trace is not None and resolution.dynamic_trace.data["status"] != "complete":
                raise DispatchPlanningError(
                    "requested dynamic trace did not produce complete planning evidence",
                    context={"dynamic_trace": resolution.dynamic_trace.to_data()},
                )
            if (
                resolution.dynamic_trace is not None
                and any(item.code == "dryml.dispatch.pickle_environment_restriction" for item in resolution.diagnostics)
            ):
                raise DispatchPlanningError(
                    "PickledCallable dispatch is restricted to the same Python executable",
                    context={"dynamic_trace": resolution.dynamic_trace.to_data()},
                )
            if any(item.code == "dryml.dispatch.pickle_environment_restriction" for item in resolution.diagnostics):
                raise DispatchPlanningError(
                    "PickledCallable dispatch is restricted to the same Python executable",
                    context={"environment": project_environment_config(resolution.environment_selection.candidate)},
                )
            raise DispatchPlanningError(
                "dispatch world plan is not launchable; call dispatch.explain(...) for requirement diagnostics",
                context={"planning": resolution.metadata()},
            )
        env_data = dict(resolution.environment_selection.candidate)
        if launch.get("call_transport") == "pickle_small" and not _same_python_environment(env_data):
            _cleanup_launch(launch)
            raise DispatchPlanningError(
                "PickledCallable dispatch is restricted to the same Python executable",
                context={"environment": project_environment_config(env_data)},
            )
        runtime_data = dict(resolution.runtime_selection.candidate)
        try:
            selected_inventory = effective_inventory or resolution.local_inventory
            if selected_inventory is None:
                selected_inventory = worlds.local_inventory(policy=effective_inventory_policy)
                resolution = replace(
                    resolution,
                    # Allocation still needs the discovered inventory, but a
                    # requirement-free fallback must not make host observations
                    # part of dispatch intent identity.
                    inventory_summary=(
                        selected_inventory.summary()
                        if resolution.world_synthesis is not None or resolution.requirements.world_requirement is not None
                        else None
                    ),
                    local_inventory=selected_inventory,
                )
            allocation_plan = allocate_local_world(resolution.world_selection.candidate, inventory=selected_inventory, oversubscribe=oversubscribe)
            _require_allocation_satisfies_requirement(allocation_plan.world_allocation, resolution.requirements.world_requirement, requirement_policy)
        except BaseException:
            _cleanup_launch(launch)
            raise
        launch_world_spec = allocation_plan.world_spec
        launch_allocation_spec = allocation_plan.world_allocation_spec
        world_spec = project_world_spec(launch_world_spec)
        allocation_spec = project_world_allocation_spec(
            launch_allocation_spec,
            world_id=world_spec["id"],
        )
        allocation_workers = []
        for key in allocation_plan.worker_keys:
            allocation = allocation_plan.world_allocation.runtime_view(
                key.role,
                key.replica,
                world_allocation_id=allocation_spec["id"],
            )
            allocation_workers.append({
                "role": key.role,
                "replica": key.replica,
                "cpus": list(allocation.cpus),
                "memory": allocation.memory,
                "accelerators": {name: list(values) for name, values in allocation.accelerators.items()},
            })
        resolution = replace(
            resolution,
            canonical_world_spec=launch_world_spec,
                world_allocation_summary={
                    "backend": "local_world",
                    "allocation_policy": "oversubscribed_local" if oversubscribe else "disjoint_local",
                    "workers": allocation_workers,
                },
        )
        try:
            marshal = select_marshal_plan(target_store, query_index="none")
            require_supported_plan(marshal)
        except BaseException:
            _cleanup_launch(launch)
            raise
        try:
            planning_metadata = resolution.metadata()
            dispatch = attach_dispatch_id(
                make_dispatch_spec(
                operation_id=op_spec["id"],
                operation=op_spec,
                environment={"policy": resolution.environment_selection.source, "spec": project_environment_config(env_data)},
                world={"policy": resolution.world_selection.source, "spec": world_spec},
                runtime={"policy": "worker", "spec": project_runtime_config(runtime_data)},
                records={"record_policy": record_policy, "provenance": record_policy != "none"},
                execution={"backend": "local_world"},
                metadata=planning_metadata,
                )
            )
            recipe = attach_recipe_id(
                make_execution_recipe(
                dispatch_id=dispatch["id"],
                operation_id=op_spec["id"],
                backend=LOCAL_WORLD_BACKEND_IDENTITY,
                input_plan={"materialize_cdefs": [], "ref_cdefs": []},
                output_plan={"record_policy": record_policy, "provenance": record_policy != "none"},
                store_plan={"strategy": marshal.strategy, "roles": [ref.role for ref in marshal.store_refs]},
                log_plan={"stdout": "capture_per_worker", "stderr": "capture_per_worker"},
                constraints={"portable": launch.get("call_transport") != "pickle_small", "local_only": True},
                annotation_report=planning_metadata,
                )
            )
        except BaseException:
            _cleanup_launch(launch)
            raise
        try:
            worker_plans = []
            for key in allocation_plan.worker_keys:
                allocation = allocation_plan.world_allocation.runtime_view(key.role, key.replica, world_allocation_id=allocation_spec["id"])
                launch_data = dict(launch)
                launch_data.update({
                    "world_id": world_spec.get("id"),
                    "world_allocation_id": allocation_spec.get("id"),
                    "world_spec": launch_world_spec,
                    "world_allocation_spec": launch_allocation_spec,
                    "provenance_world_spec": world_spec,
                    "provenance_world_allocation_spec": allocation_spec,
                    "parent_persisted_specs": record_policy != "none",
                })
                envelope = ExecutionEnvelope(
                    dispatch_spec=dispatch,
                    execution_recipe=recipe,
                    operation_spec=op_spec,
                    environment_spec=env_data,
                    runtime_spec=runtime_data,
                    allocation_view=_allocation_to_json(allocation, world_id=world_spec.get("id")),
                    store_refs=marshal.store_refs,
                    transfer={"strategy": marshal.strategy},
                    record_policy=record_policy,
                    reporting={"planning": planning_metadata},
                    launch=launch_data,
                )
                worker_plans.append(WorkerLaunchPlan(key, dispatch, recipe, envelope, target_store))
        except BaseException:
            _cleanup_launch(launch)
            raise
        if record_policy != "none":
            try:
                target_store.records.write_spec(op_spec, family="operation")
                target_store.records.write_spec(dispatch, family="dispatch")
                target_store.records.write_spec(recipe, family="execution_recipe")
                _report("dryml.dispatch.world.allocation.write", "Writing world allocation spec", operation_id=op_spec.get("id"), data={"world_id": world_spec.get("id"), "world_allocation_id": allocation_spec.get("id")})
                target_store.records.write_spec(world_spec, family="world")
                target_store.records.write_spec(allocation_spec, family="world_allocation")
            except BaseException:
                _cleanup_launch(launch)
                raise
        return LocalWorldPlan(dispatch, recipe, op_spec, world_spec, allocation_spec, tuple(worker_plans), target_store)

    def submit_world(self, plan: Any):
        """Submit a local-world plan to the local-world coordinator.

        Args:
            plan: Plan returned by :meth:`plan_world`.

        Returns:
            A local-world future that owns worker and artifact cleanup.
        """

        from .local_world import LocalWorldBackend

        backend = self.backend if isinstance(self.backend, LocalWorldBackend) else LocalWorldBackend()
        return backend.submit(plan)

    def run_world(self, operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, **kwargs: Any):
        """Plan, launch, and wait for an explicit local-world dispatch.

        Args:
            operation: Python-shaped callable or explicit operation specification.
            method_name: Optional DRYML object method name.
            **kwargs: Arguments accepted by :meth:`plan_world`, plus optional
                worker-result ``timeout`` in seconds.

        Returns:
            The completed local-world result collection.
        """

        plan = self.plan_world(operation, method_name, **{key: value for key, value in kwargs.items() if key not in {"timeout"}})
        future = self.submit_world(plan)
        return future.result(timeout=kwargs.get("timeout"))

    def run(self, operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, **kwargs: Any) -> DispatchResult:
        """Plan, submit, wait, and return a public ``DispatchResult``.

        Args:
            operation: Python-shaped callable or explicit operation specification.
            method_name: Optional DRYML object method name.
            **kwargs: Arguments accepted by :meth:`plan`, plus optional worker
                result ``timeout`` in seconds.

        Returns:
            The completed dispatch result.
        """

        plan = self.plan(operation, method_name, **{key: value for key, value in kwargs.items() if key not in {"timeout"}})
        future = self.submit(plan)
        response = future.result(timeout=kwargs.get("timeout"))
        _report("dryml.dispatch.complete", "Dispatch complete", operation_id=response.operation_id, data={"status": response.status})
        return DispatchResult.from_worker_response(response)

    def explain(
        self,
        operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable,
        method_name: str | None = None,
        *,
        store: Any | None = None,
        environment: Any | Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
        world: Mapping[str, Any] | None = None,
        requirement_policy: str | None = None,
        analysis_policy: Any | None = None,
        environment_candidates: Any = _UNSET,
        environment_registry: Any = _UNSET,
        inventory: Any = _UNSET,
        inventory_policy: str | None = None,
        resolver_policy: str | None = None,
        allow_pickle: bool = False,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> DispatchExplanation:
        """Explain a dispatch request without launching work or creating records.

        Explanation follows the same one-time normalization, requirement policy,
        candidate precedence, and checks as :meth:`plan`. It may run a bounded
        code/environment probe when static discovery is incomplete. Resolver,
        registry, inventory, and policy arguments accept the same values as
        :meth:`plan`, but explanation does not create allocation records or
        activate a workload allocation. With explicit mapping
        `analysis_policy.dynamic_trace`, it executes the same one eligible,
        trusted current-process trace as planning; this is not sandboxed and has
        no hard timeout. A requested trace is structural evidence, so rejected,
        incomplete, or over-limit evidence makes the explanation non-launchable.

        Args:
            operation: Python-shaped callable or explicit operation specification.
            method_name: Optional DRYML object method name.
            store: Optional store used only for operation normalization.
            environment: Explicit environment candidate.
            runtime: Explicit runtime specification.
            world: Explicit requested world.
            requirement_policy: Optional strict, warn, or ignore policy.
            analysis_policy: Optional code-analysis policy.
            environment_candidates: Per-call ordered resolver candidates.
            environment_registry: Per-call explicit environment registry.
            inventory: Per-call local inventory used by synthesis/allocation checks.
            inventory_policy: Local inventory discovery policy.
            resolver_policy: Environment resolver selection policy.
            allow_pickle: Permit pickled callable normalization where supported.
            args: Positional arguments for Python-shaped operation normalization.
            kwargs: Keyword arguments for Python-shaped operation normalization.

        Returns:
            A non-launching explanation with the equivalent plan's resolution.
        """

        target_store = store or self.store
        effective_inventory_policy = self.inventory_policy if inventory_policy is None else inventory_policy
        effective_resolver_policy = self.resolver_policy if resolver_policy is None else resolver_policy
        _validate_sprint8_policies(effective_inventory_policy, effective_resolver_policy)
        effective_candidates = self.environment_candidates if environment_candidates is _UNSET else environment_candidates
        effective_registry = self.environment_registry if environment_registry is _UNSET else environment_registry
        effective_inventory = self.inventory if inventory is _UNSET else inventory
        analysis_request = parse_analysis_policy(analysis_policy)
        normalized = normalize_user_operation(
            operation,
            method_name,
            store=target_store,
            allow_pickle=allow_pickle,
            args=args,
            kwargs=kwargs,
            persist_object=False,
            trace_enabled=analysis_request.requested,
        )
        effective_requirement_policy = (
            "strict"
            if _dispatch_extension_enabled(operation) and requirement_policy is None
            else requirement_policy
        )
        try:
            return explanation_for(
                normalized,
                environment=environment,
                world=world,
                runtime_spec=runtime,
                requirement_policy=effective_requirement_policy,
                analysis_policy=analysis_policy,
                _analysis_request=analysis_request,
                environment_candidates=effective_candidates,
                environment_registry=effective_registry,
                inventory=effective_inventory,
                inventory_policy=effective_inventory_policy,
                resolver_policy=effective_resolver_policy,
                single_worker_only=True,
            )
        finally:
            _cleanup_launch(normalized.launch)


def plan(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, *, backend: Any | str | None = None, store: Any | None = None, **kwargs: Any) -> DispatchPlan:
    """Build a local-subprocess plan.

    Args:
        operation: Python-shaped callable or explicit operation specification.
        method_name: Optional DRYML object method name.
        backend: Optional backend object or ``"local_subprocess"``.
        store: Store used for operation marshalling.
        **kwargs: Arguments accepted by :meth:`Dispatcher.plan`.

    Returns:
        A validated dispatch plan.
    """

    if backend in (None, "local_subprocess"):
        backend_obj = None
    else:
        backend_obj = backend
    return Dispatcher(backend=backend_obj, store=store).plan(operation, method_name, **kwargs)


def run(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, *, backend: Any | str | None = None, store: Any | None = None, **kwargs: Any) -> DispatchResult:
    """Plan, submit, and wait for an operation.

    Args:
        operation: Python-shaped callable or explicit operation specification.
        method_name: Optional DRYML object method name.
        backend: Optional backend object or ``"local_subprocess"``.
        store: Store used for marshalling and optional records.
        **kwargs: Arguments accepted by :meth:`Dispatcher.run`.

    Returns:
        The completed public dispatch result.
    """

    if backend in (None, "local_subprocess"):
        backend_obj = None
    else:
        backend_obj = backend
    return Dispatcher(backend=backend_obj, store=store).run(operation, method_name, **kwargs)


def explain(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, *, backend: Any | str | None = None, store: Any | None = None, **kwargs: Any) -> DispatchExplanation:
    """Explain a Python-shaped or explicit operation without launching it.

    Args:
        operation: Python-shaped callable or explicit operation specification.
        method_name: Optional DRYML object method name.
        backend: ``None`` or ``"local_subprocess"``; other backends are not
            supported by this facade.
        store: Optional store used for operation normalization.
        **kwargs: Arguments accepted by :meth:`Dispatcher.explain`.

    Returns:
        A non-persisting, non-launching ``DispatchExplanation``. Bounded local
        discovery and code/environment probes may still run.

    Raises:
        DispatchPlanningError: If the backend or request cannot be planned.

    When ``analysis_policy`` explicitly requests dynamic tracing, explanation
    executes the eligible trusted target once in the current process, just like
    planning. That execution is not sandboxed and has no hard timeout. A failed
    or incomplete requested trace makes the explanation non-launchable.
    """

    if backend not in (None, "local_subprocess"):
        raise DispatchPlanningError("dispatch.explain currently supports the local_subprocess planner")
    return Dispatcher(store=store).explain(operation, method_name, **kwargs)


def run_world(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, *, store: Any | None = None, **kwargs: Any):
    """Plan, launch, and wait for a local-world operation.

    Args:
        operation: Python-shaped callable or explicit operation specification.
        method_name: Optional DRYML object method name.
        store: Store used for shared marshalling and records.
        **kwargs: Arguments accepted by :meth:`Dispatcher.run_world`.

    Returns:
        The completed local-world result collection.
    """

    return Dispatcher(store=store).run_world(operation, method_name, **kwargs)


def plan_world(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, *, store: Any | None = None, **kwargs: Any):
    """Build a local-world plan, synthesizing an omitted required world.

    Args:
        operation: Python-shaped callable or explicit operation specification.
        method_name: Optional DRYML object method name.
        store: Store used for shared marshalling and records.
        **kwargs: Arguments accepted by :meth:`Dispatcher.plan_world`.

    Returns:
        A validated local-world execution plan.
    """

    return Dispatcher(store=store).plan_world(operation, method_name, **kwargs)


def submit(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, *, backend: Any | str | None = None, store: Any | None = None, **kwargs: Any):
    """Plan and submit an operation without waiting for completion.

    Args:
        operation: Python-shaped callable or explicit operation specification.
        method_name: Optional DRYML object method name.
        backend: Optional backend object or ``"local_subprocess"``.
        store: Store used for operation marshalling.
        **kwargs: Arguments accepted by :meth:`Dispatcher.submit`.

    Returns:
        A backend future for the launched operation.
    """

    dispatcher = Dispatcher(backend=None if backend in (None, "local_subprocess") else backend, store=store)
    return dispatcher.submit(operation, method_name, **kwargs)


def _environment_data(environment: Any | Mapping[str, Any] | None) -> dict[str, Any]:
    if environment is None:
        return CurrentEnvironmentSpec().to_data()
    if isinstance(environment, Mapping):
        return dict(environment)
    if hasattr(environment, "to_data"):
        return environment.to_data()
    if isinstance(environment, str):
        return PythonExecutableSpec(environment).to_data()
    raise DispatchPlanningError("unsupported environment spec", context={"type": type(environment).__name__})


def _same_python_environment(env_data: Mapping[str, Any]) -> bool:
    kind = env_data.get("kind")
    if kind == "current":
        return True
    if kind == "python":
        return os.path.abspath(os.fspath(env_data.get("executable", ""))) == os.path.abspath(sys.executable)
    return False


def _subprocess_allocation_world(world: Mapping[str, Any]) -> dict[str, Any]:
    """Map a local_subprocess request to the local assignment representation."""

    requested = worlds.WorldSpec.from_data(world)
    if requested.backend.get("kind") != "local_subprocess":
        return dict(world)
    data = requested.to_data()
    data["backend"] = {"kind": "local", "parameters": {}}
    return data


def _require_allocation_satisfies_requirement(allocation: worlds.WorldAllocation, requirement: worlds.WorldRequirement | None, policy: RequirementPolicy | str | None) -> None:
    """Validate actual allocation compatibility and enforce the active policy."""

    if requirement is None:
        return
    report = worlds.check_allocation_satisfies_requirement(allocation, requirement)
    if not report.ok:
        effective_policy = effective_requirement_policy(policy, runtime.enforcement())
        if effective_policy is RequirementPolicy.WARN:
            import warnings

            warnings.warn("actual local allocation does not satisfy the hard world requirement", RuntimeWarning, stacklevel=3)
            return report
        if effective_policy is RequirementPolicy.IGNORE:
            return report
        raise DispatchPlanningError(
            "actual local allocation does not satisfy the hard world requirement",
            context={
                "issues": [
                    {
                        "path": issue.path,
                        "message": issue.message,
                        "expected": issue.expected,
                        "actual": issue.actual,
                    }
                    for issue in report.issues
                ]
            },
        )
    return report


def _cleanup_launch(launch: Mapping[str, Any]) -> None:
    """Remove launch-only temporary files when planning never returns a plan."""

    for path in launch.get("cleanup_paths") or ():
        shutil.rmtree(path, ignore_errors=True)


def _dispatch_extension_enabled(operation: Any) -> bool:
    return inspect.getattr_static(
        type(operation), "__dryml_dispatch_extension__", False
    ) is True


def _make_dispatch_extension(
    operation: Any,
    *,
    args: Any,
    kwargs: Any,
    callbacks: Any,
    rerun: Any,
):
    if not _dispatch_extension_enabled(operation):
        if callbacks not in ((), None) or rerun is not False:
            raise DispatchPlanningError(
                "callbacks and rerun are valid only for bound managed methods"
            )
        return None
    factory = object.__getattribute__(operation, "__dryml_make_dispatch_extension__")
    return factory(
        args=() if args is None else args,
        kwargs={} if kwargs is None else kwargs,
        callbacks=callbacks,
        rerun=rerun,
    )


def allocation_from_json(data: Mapping[str, Any] | None) -> RuntimeAllocationView:
    """Build a CPU-only real worker allocation from envelope JSON."""

    data = dict(data or {})
    return RuntimeAllocationView(
        world_allocation_id=data.get("world_allocation_id"),
        role=data.get("role", "worker"),
        replica=data.get("replica", 0),
        rank=data.get("rank", 0),
        local_rank=data.get("local_rank", 0),
        cpus=tuple(data.get("cpus") or ()),
        memory=data.get("memory"),
        accelerators=data.get("accelerators") or {},
        env=data.get("env") or {},
        metadata=data.get("metadata") or {},
    )


def _allocation_to_json(allocation: RuntimeAllocationView, *, world_id: str | None = None) -> dict[str, Any]:
    env = dict(allocation.env)
    if world_id is not None:
        env["DRYML_WORLD_ID"] = world_id
    if allocation.world_allocation_id is not None:
        env["DRYML_WORLD_ALLOCATION_ID"] = allocation.world_allocation_id
    return {
        "world_allocation_id": allocation.world_allocation_id,
        "role": allocation.role,
        "replica": allocation.replica,
        "rank": allocation.rank,
        "local_rank": allocation.local_rank,
        "cpus": list(allocation.cpus),
        "memory": allocation.memory,
        "accelerators": {key: list(value) for key, value in sorted(allocation.accelerators.items())},
        "env": env,
        "metadata": dict(allocation.metadata),
    }


def _local_cpu_ids() -> list[int]:
    try:
        affinity = os.sched_getaffinity(0)
    except Exception:
        count = os.cpu_count() or 1
        return list(range(max(1, count)))
    return sorted(int(cpu) for cpu in affinity) or [0]


def _report(name: str, message: str, *, operation_id: str | None = None, data: Mapping[str, Any] | None = None) -> None:
    try:
        from dryml import reporting

        reporting.step(name, message, operation_id=operation_id, data=data or {})
    except Exception:
        pass


__all__ = ["DispatchPlan", "Dispatcher", "allocation_from_json", "explain", "plan", "plan_world", "run", "run_world", "submit"]
