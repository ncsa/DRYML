"""High-level dispatch planning API for the local subprocess backend."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from dryml.environments import CurrentEnvironmentSpec, PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec, validate_operation_spec
from dryml.runtime import RuntimeAllocationView, RuntimeMode
from dryml.runtime.specs import RuntimeContextSpec

from .errors import DispatchPlanningError
from .operations import PickledCallable, write_pickled_callable
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
        """Build DispatchSpec, ExecutionRecipe, and launch-only envelope."""

        _report("dryml.dispatch.plan.start", "Building dispatch plan")
        target_store = store or self.store
        if target_store is None:
            raise DispatchPlanningError("Dispatcher.plan requires a store for local subprocess marshalling")
        _report("dryml.dispatch.requirements.gather", "Gathering environment/world/runtime requirements")
        op_spec, launch = self._normalize_operation(operation, allow_pickle=allow_pickle, args=args, kwargs=kwargs or {})
        _report("dryml.dispatch.requirements.merge", "Merging requirements and defaults", operation_id=op_spec.get("id"))
        env_data = _environment_data(environment)
        if launch.get("call_transport") == "pickle_small" and not _same_python_environment(env_data):
            raise DispatchPlanningError("PickledCallable dispatch is restricted to the same Python executable", context={"environment": env_data})
        runtime_data = runtime or RuntimeContextSpec(mode=RuntimeMode.WORKER, device_visibility={"policy": "assigned"}, metadata={"source": "dryml.dispatch.local_subprocess"}).to_data()
        allocation_data = {"role": "worker", "replica": 0, "rank": 0, "local_rank": 0, "cpus": [], "accelerators": {}, "env": {}, "metadata": {"backend": "local_subprocess"}}
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

    def submit(self, plan: DispatchPlan):
        """Submit a prebuilt plan to the configured backend."""

        return self.backend.submit(plan)

    def run(self, operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, **kwargs: Any) -> DispatchResult:
        """Plan, submit, wait, and return a public ``DispatchResult``."""

        plan = self.plan(operation, **{key: value for key, value in kwargs.items() if key not in {"timeout"}})
        future = self.submit(plan)
        response = future.result(timeout=kwargs.get("timeout"))
        _report("dryml.dispatch.complete", "Dispatch complete", operation_id=response.operation_id, data={"status": response.status})
        return DispatchResult.from_worker_response(response)

    def _normalize_operation(self, operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, *, allow_pickle: bool, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(operation, Mapping):
            return attach_operation_id(validate_operation_spec(operation)), {}
        if isinstance(operation, PickledCallable):
            allow_pickle = True
            func = operation.callable
        else:
            func = operation
        if not allow_pickle:
            raise DispatchPlanningError("callable dispatch requires allow_pickle=True or an OperationSpec import path")
        work_dir = tempfile.mkdtemp(prefix="dryml-dispatch-pickle-")
        pickle_path = os.path.join(work_dir, "callable.pkl")
        write_pickled_callable(func, pickle_path)
        op = attach_operation_id(make_function_call_spec("dryml.dispatch.operations:import_function", args=list(args), kwargs=dict(kwargs)))
        return op, {"call_transport": "pickle_small", "pickle_path": pickle_path, "portable": False, "same_environment_only": True}


def run(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, *, backend: Any | str | None = None, store: Any | None = None, **kwargs: Any) -> DispatchResult:
    """Convenience wrapper for ``Dispatcher(...).run(...)``."""

    if backend in (None, "local_subprocess"):
        backend_obj = None
    else:
        backend_obj = backend
    return Dispatcher(backend=backend_obj, store=store).run(operation, **kwargs)


def submit(operation: Mapping[str, Any] | Callable[..., Any] | PickledCallable, *, backend: Any | str | None = None, store: Any | None = None, **kwargs: Any):
    """Plan and submit an operation, returning a backend future."""

    dispatcher = Dispatcher(backend=None if backend in (None, "local_subprocess") else backend, store=store)
    return dispatcher.submit(dispatcher.plan(operation, **kwargs))


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


def _report(name: str, message: str, *, operation_id: str | None = None, data: Mapping[str, Any] | None = None) -> None:
    try:
        from dryml import reporting

        reporting.step(name, message, operation_id=operation_id, data=data or {})
    except Exception:
        pass


__all__ = ["DispatchPlan", "Dispatcher", "allocation_from_json", "run", "submit"]
