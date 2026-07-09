"""High-level dispatch planning API for the local subprocess backend."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from dryml.environments import CurrentEnvironmentSpec, PythonExecutableSpec
from dryml.runtime import RuntimeAllocationView, RuntimeMode
from dryml.runtime.specs import RuntimeContextSpec

from .errors import DispatchPlanningError
from .normalize import normalize_user_operation
from .operations import PickledCallable
from .protocol import DispatchResult, ExecutionEnvelope
from .recipes import attach_recipe_id, make_execution_recipe
from .specs import attach_dispatch_id, make_dispatch_spec
from .stores import require_supported_plan, select_marshal_plan


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    """Resolved one-operation local subprocess dispatch plan."""

    dispatch_spec: Mapping[str, Any]
    execution_recipe: Mapping[str, Any]
    envelope: ExecutionEnvelope
    store: Any


class Dispatcher:
    """Plan and run one operation through a dispatch backend."""

    def __init__(self, *, backend: Any | None = None, store: Any | None = None):
        from .backends import LocalSubprocessBackend

        self.backend = backend if backend is not None else LocalSubprocessBackend()
        self.store = store

    def plan(
        self,
        operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable,
        method_name: str | None = None,
        *,
        store: Any | None = None,
        environment: Any | Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
        world: Mapping[str, Any] | None = None,
        record_policy: str = "descriptive",
        allow_pickle: bool = False,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> DispatchPlan:
        """Build DispatchSpec, ExecutionRecipe, and launch-only envelope.

        ``operation`` may be an explicit OperationSpec, an importable callable,
        a non-importable callable with ``allow_pickle=True``, or a DRYML
        Definition/CDef/Object paired with ``method_name``.
        """

        _report("dryml.dispatch.plan.start", "Building dispatch plan")
        target_store = store or self.store
        if target_store is None:
            raise DispatchPlanningError("Dispatcher.plan requires a store for local subprocess marshalling")
        _report("dryml.dispatch.requirements.gather", "Gathering environment/world/runtime requirements")
        normalized = normalize_user_operation(operation, method_name, store=target_store, allow_pickle=allow_pickle, args=args, kwargs=kwargs)
        op_spec = dict(normalized.operation_spec)
        launch = dict(normalized.launch)
        _report("dryml.dispatch.requirements.merge", "Merging requirements and defaults", operation_id=op_spec.get("id"))
        env_data = _environment_data(environment)
        if launch.get("call_transport") == "pickle_small" and not _same_python_environment(env_data):
            raise DispatchPlanningError("PickledCallable dispatch is restricted to the same Python executable", context={"environment": env_data})
        runtime_data = runtime or RuntimeContextSpec(mode=RuntimeMode.WORKER, device_visibility={"policy": "assigned"}, metadata={"source": "dryml.dispatch.local_subprocess"}).to_data()
        allocation_data = {"role": "worker", "replica": 0, "rank": 0, "local_rank": 0, "cpus": _local_cpu_ids(), "accelerators": {}, "env": {}, "metadata": {"backend": "local_subprocess"}}
        marshal = select_marshal_plan(target_store, query_index="none")
        require_supported_plan(marshal)
        _report("dryml.dispatch.store.prepare", "Preparing shared DirStore marshalling", operation_id=op_spec.get("id"), data={"strategy": marshal.strategy})
        dispatch = attach_dispatch_id(
            make_dispatch_spec(
                operation_id=op_spec["id"],
                operation=op_spec,
                environment={"policy": "current", "spec": env_data},
                world=world or {"policy": "single_worker"},
                runtime={"policy": "worker", "spec": runtime_data},
                records={"record_policy": record_policy, "provenance": record_policy != "none"},
                execution={"backend": "local_subprocess"},
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
            launch=launch,
        )
        return DispatchPlan(dispatch, recipe, envelope, target_store)

    def submit(
        self,
        operation: DispatchPlan | Mapping[str, Any] | Callable[..., Any] | PickledCallable,
        method_name: str | None = None,
        **kwargs: Any,
    ):
        """Submit a prebuilt plan or plan and submit a user operation.

        Passing a :class:`DispatchPlan` preserves the original advanced API.
        Other targets are normalized through :meth:`plan` first.
        """

        plan = operation if isinstance(operation, DispatchPlan) else self.plan(operation, method_name, **kwargs)
        return self.backend.submit(plan)

    def plan_world(
        self,
        operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable,
        method_name: str | None = None,
        *,
        world: Mapping[str, Any] | Any | None,
        store: Any | None = None,
        environment: Any | Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
        record_policy: str = "descriptive",
        allow_pickle: bool = False,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        inventory: Any | None = None,
        oversubscribe: bool = False,
    ):
        """Build a coordinated local-world dispatch plan.

        The first local-world operation policy intentionally runs the same
        operation spec in every allocated role/replica worker. User code can
        branch on ``dryml.runtime.require_workload_allocation(...)`` or the
        ``DRYML_WORLD_*`` environment variables.
        """

        from .local_world import LOCAL_WORLD_BACKEND_IDENTITY, LocalWorldPlan, WorkerLaunchPlan, allocate_local_world

        _report("dryml.dispatch.world.plan.start", "Building local world dispatch plan")
        target_store = store or self.store
        if target_store is None:
            raise DispatchPlanningError("Dispatcher.plan_world requires a store for shared DirStore marshalling")
        normalized = normalize_user_operation(operation, method_name, store=target_store, allow_pickle=allow_pickle, args=args, kwargs=kwargs)
        op_spec = dict(normalized.operation_spec)
        launch = dict(normalized.launch)
        env_data = _environment_data(environment)
        if launch.get("call_transport") == "pickle_small" and not _same_python_environment(env_data):
            raise DispatchPlanningError("PickledCallable dispatch is restricted to the same Python executable", context={"environment": env_data})
        runtime_data = runtime or RuntimeContextSpec(mode=RuntimeMode.WORKER, device_visibility={"policy": "assigned"}, metadata={"source": "dryml.dispatch.local_world"}).to_data()
        allocation_plan = allocate_local_world(world, inventory=inventory, oversubscribe=oversubscribe)
        world_spec = allocation_plan.world_spec
        allocation_spec = allocation_plan.world_allocation_spec
        _report("dryml.dispatch.world.allocation.write", "Writing world allocation spec", operation_id=op_spec.get("id"), data={"world_id": world_spec.get("id"), "world_allocation_id": allocation_spec.get("id")})
        if record_policy != "none":
            target_store.records.write_spec(world_spec, family="world")
            target_store.records.write_spec(allocation_spec, family="world_allocation")
        marshal = select_marshal_plan(target_store, query_index="none")
        require_supported_plan(marshal)
        dispatch = attach_dispatch_id(
            make_dispatch_spec(
                operation_id=op_spec["id"],
                operation=op_spec,
                environment={"policy": "current", "spec": env_data},
                world={"policy": "local_world", "spec": world_spec},
                runtime={"policy": "worker", "spec": runtime_data},
                records={"record_policy": record_policy, "provenance": record_policy != "none"},
                execution={"backend": "local_world"},
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
            )
        )
        if record_policy != "none":
            target_store.records.write_spec(op_spec, family="operation")
            target_store.records.write_spec(dispatch, family="dispatch")
            target_store.records.write_spec(recipe, family="execution_recipe")
        worker_plans = []
        for key in allocation_plan.worker_keys:
            allocation = allocation_plan.world_allocation.runtime_view(key.role, key.replica, world_allocation_id=allocation_spec["id"])
            launch_data = dict(launch)
            launch_data.update({"world_id": world_spec.get("id"), "world_allocation_id": allocation_spec.get("id"), "world_spec": world_spec, "world_allocation_spec": allocation_spec, "parent_persisted_specs": record_policy != "none"})
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
                launch=launch_data,
            )
            worker_plans.append(WorkerLaunchPlan(key, dispatch, recipe, envelope, target_store))
        return LocalWorldPlan(dispatch, recipe, op_spec, world_spec, allocation_spec, tuple(worker_plans), target_store)

    def submit_world(self, plan: Any):
        """Submit a local-world plan to the local-world coordinator."""

        from .local_world import LocalWorldBackend

        backend = self.backend if isinstance(self.backend, LocalWorldBackend) else LocalWorldBackend()
        return backend.submit(plan)

    def run_world(self, operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, **kwargs: Any):
        """Plan, launch, and wait for an explicit local-world dispatch."""

        plan = self.plan_world(operation, method_name, **{key: value for key, value in kwargs.items() if key not in {"timeout"}})
        future = self.submit_world(plan)
        return future.result(timeout=kwargs.get("timeout"))

    def run(self, operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, **kwargs: Any) -> DispatchResult:
        """Plan, submit, wait, and return a public ``DispatchResult``."""

        plan = self.plan(operation, method_name, **{key: value for key, value in kwargs.items() if key not in {"timeout"}})
        future = self.submit(plan)
        response = future.result(timeout=kwargs.get("timeout"))
        _report("dryml.dispatch.complete", "Dispatch complete", operation_id=response.operation_id, data={"status": response.status})
        return DispatchResult.from_worker_response(response)


def plan(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, *, backend: Any | str | None = None, store: Any | None = None, **kwargs: Any) -> DispatchPlan:
    """Build a dispatch plan for a Python-shaped or explicit OperationSpec target."""

    if backend in (None, "local_subprocess"):
        backend_obj = None
    else:
        backend_obj = backend
    return Dispatcher(backend=backend_obj, store=store).plan(operation, method_name, **kwargs)


def run(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, *, backend: Any | str | None = None, store: Any | None = None, **kwargs: Any) -> DispatchResult:
    """Plan, submit, and wait for a Python-shaped or explicit operation."""

    if backend in (None, "local_subprocess"):
        backend_obj = None
    else:
        backend_obj = backend
    return Dispatcher(backend=backend_obj, store=store).run(operation, method_name, **kwargs)


def run_world(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, *, store: Any | None = None, **kwargs: Any):
    """Convenience wrapper for ``Dispatcher(...).run_world(...)``."""

    return Dispatcher(store=store).run_world(operation, method_name, **kwargs)


def submit(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, method_name: str | None = None, *, backend: Any | str | None = None, store: Any | None = None, **kwargs: Any):
    """Plan and submit a Python-shaped or explicit operation."""

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


__all__ = ["DispatchPlan", "Dispatcher", "allocation_from_json", "plan", "run", "run_world", "submit"]
