"""Worker entrypoint for ``python -m dryml.dispatch.worker``."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from typing import Any, Mapping

from dryml.core.repo import Repo
from dryml.operations import validate_operation_spec
from dryml.operations.errors import OperationSpecError
from dryml.records import ExecutionErrorInfo, ExecutionLogRef, ExecutionRecord, StorageRef, write_execution_record
from dryml.records.execution import persistence_safe_execution_error, transient_execution_error
from dryml.runtime import RequirementAxes, RuntimeEnforcement, RuntimeMode, activate
from dryml.runtime.specs import RuntimeContextSpec

from .backends import BACKEND_IDENTITY
from .errors import DispatchSpecError, WorkerProtocolError
from .operations import canonicalize_result, execute_operation
from .planner import _allocation_to_json, allocation_from_json
from .protocol import ACCELERATOR_MEMORY_FEATURE, DISPATCH_WORKER_PROTOCOL_SCHEMA, DISPATCH_WORKER_PROTOCOL_VERSION, WORKER_SESSION_FEATURE, ExecutionEnvelope, WorkerHandshakeRequest, WorkerHandshakeResponse, WorkerResponse, load_envelope, write_json_file
from .recipes import validate_execution_recipe
from .specs import validate_dispatch_spec
from .stores import open_worker_store, validate_worker_store_access


FEATURES = (
    "operation.function_call",
    "operation.method_call",
    "call.import_ref",
    "call.pickle_small",
    "store.dir",
    "runtime.worker",
    WORKER_SESSION_FEATURE,
    "records.execution",
    "managed.operation.v1",
)


def main(argv: list[str] | None = None) -> int:
    """Run the dispatch worker protocol."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--handshake", required=True)
    parser.add_argument("--response", required=True)
    ns = parser.parse_args(argv)
    try:
        envelope = load_envelope(ns.request)
        _validate_envelope_ids(envelope)
        allocation = allocation_from_json(envelope.allocation_view)
        _publish_selected_worker_session(envelope, allocation)
        allocation_features = _allocation_features(envelope, allocation)
        stores, store_status, supported, diagnostics = _open_and_validate_stores(
            envelope,
            features=allocation_features,
        )
        if not supported:
            handshake = _handshake(envelope, status="unsupported", store_status=store_status, diagnostics=diagnostics, features=allocation_features)
            write_json_file(ns.handshake, handshake.to_json())
            response = WorkerResponse(
                status="unsupported",
                operation_id=envelope.operation_id,
                dispatch_id=envelope.dispatch_spec.get("id"),
                recipe_id=envelope.execution_recipe.get("id"),
                error={"type": "WorkerHandshakeError", "message": "worker handshake unsupported"},
                diagnostics=diagnostics,
            )
            write_json_file(ns.response, response.to_json())
            return 1
        repo = Repo(stores=stores)
        handshake = _handshake(envelope, status="ok", store_status=store_status, features=allocation_features)
        write_json_file(ns.handshake, handshake.to_json())
        barrier_response = _wait_for_start_barrier(envelope, stores[0] if stores else None)
        if barrier_response is not None:
            write_json_file(ns.response, barrier_response.to_json())
            return 1
        managed_ticket = None
        if envelope.launch.get("managed") is not None:
            from dryml.managed.dispatch import wait_for_managed_ticket

            managed_ticket = wait_for_managed_ticket(envelope.launch)
        response = _execute(
            envelope,
            repo,
            stores[0] if stores else None,
            managed_ticket=managed_ticket,
            allocation=allocation,
        )
        write_json_file(ns.response, response.to_json())
        return 0 if response.status == "ok" else 1
    except Exception as exc:
        try:
            envelope = locals().get("envelope")
            if isinstance(envelope, ExecutionEnvelope):
                error = persistence_safe_execution_error(exc)
                handshake = _handshake(envelope, status="failed", diagnostics=({"message": "worker initialization failed", "error_type": error["type"], "code": error["metadata"]["code"]},))
                write_json_file(ns.handshake, handshake.to_json())
                response = _failure_response(envelope, exc, store=(locals().get("stores") or [None])[0])
            else:
                response = WorkerResponse(status="failed", error=transient_execution_error(exc), diagnostics=({"message": "worker failed before envelope validation"},))
            write_json_file(ns.response, response.to_json())
        except Exception as response_exc:
            print(
                f"worker failure response could not be written: {type(response_exc).__name__}",
                file=sys.stderr,
            )
        return 1


def _open_and_validate_stores(envelope: ExecutionEnvelope, *, features: tuple[str, ...] = FEATURES):
    statuses: dict[str, Any] = {}
    stores = []
    diagnostics = []
    for ref in envelope.store_refs:
        try:
            statuses[ref.label] = validate_worker_store_access(ref)
            stores.append(open_worker_store(ref))
        except Exception as exc:
            error = persistence_safe_execution_error(exc)
            diagnostics.append({"message": "worker Store validation failed", "error_type": error["type"], "code": error["metadata"]["code"]})
    request = WorkerHandshakeRequest.from_json(envelope.handshake)
    missing = sorted(set(request.required_features) - set(features))
    if request.min_protocol > DISPATCH_WORKER_PROTOCOL_VERSION:
        diagnostics.append({"message": "unsupported worker protocol version", "min_protocol": request.min_protocol, "worker_protocol": DISPATCH_WORKER_PROTOCOL_VERSION})
    if missing:
        diagnostics.append({"message": "missing required worker features", "features": missing})
    return stores, statuses, not diagnostics, tuple(diagnostics)


def _allocation_features(envelope: ExecutionEnvelope, allocation: Any) -> tuple[str, ...]:
    """Advertise accelerator-memory support only after exact parser retention."""

    if "accelerator_memory" not in envelope.allocation_view:
        return FEATURES
    retained = _allocation_to_json(allocation).get("accelerator_memory")
    if retained != envelope.allocation_view["accelerator_memory"]:
        raise WorkerProtocolError(
            "worker allocation parser did not retain accelerator-memory limits"
        )
    return (*FEATURES, ACCELERATOR_MEMORY_FEATURE)


def _validate_envelope_ids(envelope: ExecutionEnvelope) -> None:
    try:
        validate_operation_spec(envelope.operation_spec)
        validate_dispatch_spec(envelope.dispatch_spec)
        validate_execution_recipe(envelope.execution_recipe)
    except (OperationSpecError, DispatchSpecError) as exc:
        raise WorkerProtocolError(
            "execution envelope contains an invalid canonical spec",
            context={"error_type": type(exc).__name__, "error": str(exc)},
        ) from exc

    operation_id = envelope.operation_spec.get("id")
    dispatch_id = envelope.dispatch_spec.get("id")
    recipe_id = envelope.execution_recipe.get("id")
    dispatch_operation_id = envelope.dispatch_spec.get("payload", {}).get("operation_id")
    recipe_payload = envelope.execution_recipe.get("payload", {})
    recipe_operation_id = recipe_payload.get("operation_id")
    recipe_dispatch_id = recipe_payload.get("dispatch_id")
    mismatches = {}
    if operation_id is None:
        mismatches["operation_spec.id"] = operation_id
    if dispatch_id is None:
        mismatches["dispatch_spec.id"] = dispatch_id
    if recipe_id is None:
        mismatches["execution_recipe.id"] = recipe_id
    if dispatch_operation_id != operation_id:
        mismatches["dispatch_spec.payload.operation_id"] = dispatch_operation_id
    if recipe_operation_id != operation_id:
        mismatches["execution_recipe.payload.operation_id"] = recipe_operation_id
    if recipe_dispatch_id != dispatch_id:
        mismatches["execution_recipe.payload.dispatch_id"] = recipe_dispatch_id
    if mismatches:
        raise WorkerProtocolError("execution envelope operation/dispatch/recipe IDs are inconsistent", context={"operation_id": operation_id, "dispatch_id": dispatch_id, "mismatches": mismatches})


def _publish_selected_worker_session(envelope: ExecutionEnvelope, allocation: Any) -> None:
    """Validate and publish v2 worker state before any handshake or Store work."""

    from dryml.environments import PythonExecutableSpec, spec_from_data
    from dryml.session.state import publish_worker_session
    from dryml.worlds import WorldSpec

    environment = spec_from_data(envelope.environment_spec)
    if isinstance(environment, PythonExecutableSpec) and os.path.abspath(environment.executable) != os.path.abspath(sys.executable):
        raise WorkerProtocolError("selected Python environment does not match worker interpreter")
    world = WorldSpec.from_data(envelope.world_spec)
    runtime_spec = RuntimeContextSpec.from_data(envelope.runtime_spec)
    try:
        axes = RequirementAxes(tuple(envelope.requirement_axes))
        publish_worker_session(
            environment=environment,
            world=world,
            runtime_spec=runtime_spec,
            allocation=allocation,
            requirement_policy=envelope.requirement_policy,
            requirement_axes=axes,
        )
    except Exception as exc:
        if isinstance(exc, WorkerProtocolError):
            raise
        raise WorkerProtocolError("worker selected-session validation failed", context={"error": type(exc).__name__}) from exc


def _handshake(envelope: ExecutionEnvelope, *, status: str, store_status: Mapping[str, Any] | None = None, diagnostics: tuple[Mapping[str, Any], ...] = (), features: tuple[str, ...] = FEATURES) -> WorkerHandshakeResponse:
    return WorkerHandshakeResponse(
        status=status,
        protocol_schema=DISPATCH_WORKER_PROTOCOL_SCHEMA,
        protocol_version=DISPATCH_WORKER_PROTOCOL_VERSION,
        dryml_version=_dryml_version(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        pid=os.getpid(),
        features=features,
        operation_kinds=("function_call", "method_call"),
        call_transports=("import_ref", "pickle_small"),
        store_ref_kinds=("dir_store",),
        record_schemas={"record": 1, "dispatch": 1, "execution_recipe": 1},
        runtime_modes=("worker",),
        environment_kind=envelope.environment_spec.get("kind"),
        process_group=(os.name == "posix"),
        store_status=store_status or {},
        world_id=envelope.launch.get("world_id") or (envelope.allocation_view.get("env") or {}).get("DRYML_WORLD_ID"),
        world_allocation_id=envelope.allocation_view.get("world_allocation_id") or envelope.launch.get("world_allocation_id"),
        worker_key={key: envelope.allocation_view.get(key) for key in ("role", "replica", "rank", "local_rank")},
        diagnostics=diagnostics,
    )


def _execute(
    envelope: ExecutionEnvelope,
    repo: Repo,
    store: Any,
    *,
    managed_ticket: Mapping[str, Any] | None = None,
    allocation: Any | None = None,
) -> WorkerResponse:
    allocation = allocation_from_json(envelope.allocation_view) if allocation is None else allocation
    runtime_spec = RuntimeContextSpec.from_data(envelope.runtime_spec)
    try:
        with activate(mode=RuntimeMode.WORKER, allocation=allocation, spec=runtime_spec, env=allocation.env, restore_environ=False, enforcement=RuntimeEnforcement.STRICT):
            _report("dryml.dispatch.worker.execute", "Running operation in worker", operation_id=envelope.operation_id)
            if managed_ticket is not None:
                from dryml.managed.dispatch import execute_managed_operation

                execution = execute_managed_operation(
                    dict(envelope.operation_spec),
                    repo=repo,
                    store=store,
                    ticket=managed_ticket,
                )
                if execution.status == "ok":
                    return WorkerResponse(
                        status="ok",
                        operation_id=envelope.operation_id,
                        dispatch_id=envelope.dispatch_spec.get("id"),
                        recipe_id=envelope.execution_recipe.get("id"),
                        result_canonical={
                            "early_completed": bool(
                                execution.managed_result.get("early_completed", False)
                            )
                        },
                        managed_result=execution.managed_result,
                    )
                error = None
                if execution.error is not None:
                    error = transient_execution_error(execution.error)
                return WorkerResponse(
                    status=execution.status,
                    operation_id=envelope.operation_id,
                    dispatch_id=envelope.dispatch_spec.get("id"),
                    recipe_id=envelope.execution_recipe.get("id"),
                    error=error if execution.status != "cancelled" else None,
                    cancellation=execution.cancellation,
                    diagnostics=({"message": "managed worker execution did not complete"},),
                    managed_result=execution.managed_result,
                )
            result, consumed_cdefs = execute_operation(dict(envelope.operation_spec), repo=repo, envelope_launch=dict(envelope.launch))
            canonical, produced_cdefs = canonicalize_result(result, repo=repo, store=store, record_policy=envelope.record_policy)
        record_id = _write_worker_record(envelope, store, "ok", consumed_cdef_ids=consumed_cdefs, produced_cdef_ids=produced_cdefs)
        return WorkerResponse(
            status="ok",
            operation_id=envelope.operation_id,
            dispatch_id=envelope.dispatch_spec.get("id"),
            recipe_id=envelope.execution_recipe.get("id"),
            result_canonical=canonical,
            result_cdef_ids=produced_cdefs,
            produced_record_ids=(),
            execution_record_id=record_id,
            stdout_ref=StorageRef.self_product(path="stdout.txt", role="stdout").to_json() if record_id else None,
            stderr_ref=StorageRef.self_product(path="stderr.txt", role="stderr").to_json() if record_id else None,
        )
    except BaseException as exc:
        return _failure_response(envelope, exc, store=store)


def _failure_response(envelope: ExecutionEnvelope, exc: BaseException, *, store: Any) -> WorkerResponse:
    error = transient_execution_error(exc)
    record_id = _write_worker_record(envelope, store, "failed", error=error, diagnostics=({"message": "worker execution failed"},))
    return WorkerResponse(
        status="failed",
        operation_id=envelope.operation_id,
        dispatch_id=envelope.dispatch_spec.get("id"),
        recipe_id=envelope.execution_recipe.get("id"),
        execution_record_id=record_id,
        error=error,
        diagnostics=({"message": "worker execution failed"},),
    )


def _write_worker_record(envelope: ExecutionEnvelope, store: Any, status: str, *, error: Mapping[str, Any] | None = None, cancellation: Mapping[str, Any] | None = None, diagnostics: tuple[Mapping[str, Any], ...] = (), consumed_cdef_ids: tuple[str, ...] = (), produced_cdef_ids: tuple[str, ...] = ()) -> str | None:
    if envelope.record_policy == "none" or store is None:
        return None
    _persist_provenance_specs(store, envelope)
    _report("dryml.dispatch.result.save", "Saving dispatch outputs", operation_id=envelope.operation_id, data={"status": status})
    logs = (
        ExecutionLogRef("stdout", StorageRef.self_product(path="stdout.txt", role="stdout"), "text/plain"),
        ExecutionLogRef("stderr", StorageRef.self_product(path="stderr.txt", role="stderr"), "text/plain"),
    )
    execution = ExecutionRecord(
        execution_kind="python",
        operation_id=envelope.operation_id,
        backend=_backend_identity(envelope),
        status=status,
        dispatch_id=envelope.dispatch_spec.get("id"),
        recipe_id=envelope.execution_recipe.get("id"),
        world_id=envelope.launch.get("world_id"),
        world_allocation_id=envelope.allocation_view.get("world_allocation_id") or envelope.launch.get("world_allocation_id"),
        consumed_cdef_ids=consumed_cdef_ids,
        produced_cdef_ids=produced_cdef_ids,
        logs=logs,
        error=ExecutionErrorInfo.from_json(
            persistence_safe_execution_error(error)
        ) if error else None,
        cancellation=cancellation,
        diagnostics=diagnostics,
        metadata=_execution_metadata(envelope),
        extra=_execution_extra(envelope),
    )
    _report("dryml.dispatch.execution_record.write", "Writing execution record", operation_id=envelope.operation_id, data={"status": status})
    return write_execution_record(store.records, execution).record_id


def _persist_provenance_specs(store: Any, envelope: ExecutionEnvelope) -> None:
    if envelope.launch.get("parent_persisted_specs") is True:
        return
    record_io = store.records
    record_io.write_spec(envelope.operation_spec, family="operation")
    record_io.write_spec(envelope.dispatch_spec, family="dispatch")
    record_io.write_spec(envelope.execution_recipe, family="execution_recipe")
    if isinstance(envelope.launch.get("provenance_world_spec"), Mapping):
        record_io.write_spec(envelope.launch["provenance_world_spec"], family="world")
    if isinstance(envelope.launch.get("provenance_world_allocation_spec"), Mapping):
        record_io.write_spec(envelope.launch["provenance_world_allocation_spec"], family="world_allocation")


def _wait_for_start_barrier(envelope: ExecutionEnvelope, store: Any) -> WorkerResponse | None:
    coordination = envelope.launch.get("coordination") or {}
    start_path = coordination.get("start_path")
    if not start_path:
        return None
    cancel_path = coordination.get("cancel_path")
    timeout = float(coordination.get("start_timeout", 60.0))
    deadline = time.monotonic() + timeout
    while True:
        if os.path.exists(start_path):
            return None
        if cancel_path and os.path.exists(cancel_path):
            cancellation = {"requested": True, "reason": "start_barrier_cancelled"}
            record_id = _write_worker_record(envelope, store, "cancelled", cancellation=cancellation, diagnostics=({"message": "worker cancelled before start barrier release"},))
            return WorkerResponse(status="cancelled", operation_id=envelope.operation_id, dispatch_id=envelope.dispatch_spec.get("id"), recipe_id=envelope.execution_recipe.get("id"), execution_record_id=record_id, cancellation=cancellation, diagnostics=({"message": "worker cancelled before start barrier release"},))
        if time.monotonic() >= deadline:
            error = {"type": "TimeoutError", "message": "worker timed out waiting for start barrier"}
            record_id = _write_worker_record(envelope, store, "timeout", error=error, diagnostics=({"message": "worker start barrier timeout"},))
            return WorkerResponse(status="timeout", operation_id=envelope.operation_id, dispatch_id=envelope.dispatch_spec.get("id"), recipe_id=envelope.execution_recipe.get("id"), execution_record_id=record_id, error=error, diagnostics=({"message": "worker start barrier timeout"},))
        time.sleep(0.01)


def _backend_identity(envelope: ExecutionEnvelope) -> Mapping[str, Any]:
    if envelope.execution_recipe.get("payload", {}).get("backend", {}).get("kind") == "local_world":
        return {"name": "dryml.local_world", "kind": "local_world", "version": "1"}
    return BACKEND_IDENTITY


def _execution_metadata(envelope: ExecutionEnvelope) -> dict[str, Any]:
    alloc = envelope.allocation_view
    metadata = dict(alloc.get("metadata") or {})
    for field_name in ("role", "replica", "rank", "local_rank"):
        metadata[field_name] = alloc.get(field_name)
    env = alloc.get("env") or {}
    for key, name in (("DRYML_WORLD_SIZE", "world_size"), ("DRYML_WORLD_ROLE_SIZE", "role_size")):
        if key in env:
            try:
                metadata[name] = int(env[key])
            except Exception:
                metadata[name] = env[key]
    coordination = envelope.launch.get("coordination") or {}
    if coordination.get("group_id"):
        metadata["coordination_group_id"] = coordination.get("group_id")
    return metadata


def _execution_extra(envelope: ExecutionEnvelope) -> dict[str, Any]:
    alloc = envelope.allocation_view
    return {
        "worker_key": {
            "role": alloc.get("role"),
            "replica": alloc.get("replica"),
            "rank": alloc.get("rank"),
            "local_rank": alloc.get("local_rank"),
        }
    }


def _dryml_version() -> str | None:
    try:
        import dryml

        return getattr(dryml, "__version__", None)
    except Exception:
        return None


def _report(name: str, message: str, *, operation_id: str | None = None, data: Mapping[str, Any] | None = None) -> None:
    try:
        from dryml import reporting

        reporting.step(name, message, operation_id=operation_id, data=data or {})
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
