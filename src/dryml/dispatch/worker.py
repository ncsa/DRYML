"""Worker entrypoint for ``python -m dryml.dispatch.worker``."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import traceback
from typing import Any, Mapping

from dryml.core2.repo import Repo
from dryml.records import ExecutionErrorInfo, ExecutionLogRef, ExecutionRecord, StorageRef, write_execution_record
from dryml.runtime import RuntimeMode, activate
from dryml.runtime.specs import RuntimeContextSpec

from .backends import BACKEND_IDENTITY
from .operations import canonicalize_result, execute_operation
from .planner import allocation_from_json
from .protocol import DISPATCH_WORKER_PROTOCOL_SCHEMA, DISPATCH_WORKER_PROTOCOL_VERSION, ExecutionEnvelope, WorkerHandshakeRequest, WorkerHandshakeResponse, WorkerResponse, load_envelope, write_json_file
from .stores import open_worker_store, validate_worker_store_access


FEATURES = (
    "operation.function_call",
    "operation.method_call",
    "call.import_ref",
    "call.pickle_small",
    "store.dir",
    "runtime.worker",
    "records.execution",
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
        stores, store_status = _open_and_validate_stores(envelope)
        repo = Repo(stores=stores)
        handshake = _handshake(envelope, status="ok", store_status=store_status)
        write_json_file(ns.handshake, handshake.to_json())
        response = _execute(envelope, repo, stores[0] if stores else None)
        write_json_file(ns.response, response.to_json())
        return 0 if response.status == "ok" else 1
    except Exception as exc:
        try:
            envelope = locals().get("envelope")
            if isinstance(envelope, ExecutionEnvelope):
                handshake = _handshake(envelope, status="failed", diagnostics=({"message": str(exc), "type": type(exc).__name__},))
                write_json_file(ns.handshake, handshake.to_json())
                response = _failure_response(envelope, exc, store=(locals().get("stores") or [None])[0])
            else:
                response = WorkerResponse(status="failed", error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, diagnostics=({"message": "worker failed before envelope validation"},))
            write_json_file(ns.response, response.to_json())
        except Exception:
            traceback.print_exc()
        return 1


def _open_and_validate_stores(envelope: ExecutionEnvelope):
    statuses: dict[str, Any] = {}
    stores = []
    diagnostics = []
    for ref in envelope.store_refs:
        try:
            statuses[ref.label] = validate_worker_store_access(ref)
            stores.append(open_worker_store(ref))
        except Exception as exc:
            diagnostics.append({"message": str(exc), "type": type(exc).__name__, "context": getattr(exc, "context", {})})
    request = WorkerHandshakeRequest.from_json(envelope.handshake)
    missing = sorted(set(request.required_features) - set(FEATURES))
    if diagnostics or request.min_protocol > DISPATCH_WORKER_PROTOCOL_VERSION or missing:
        details = tuple(diagnostics + ([{"message": "missing required worker features", "features": missing}] if missing else []))
        handshake = _handshake(envelope, status="unsupported", store_status=statuses, diagnostics=details)
        raise RuntimeError("worker handshake unsupported: " + repr(handshake.to_json()))
    return stores, statuses


def _handshake(envelope: ExecutionEnvelope, *, status: str, store_status: Mapping[str, Any] | None = None, diagnostics: tuple[Mapping[str, Any], ...] = ()) -> WorkerHandshakeResponse:
    return WorkerHandshakeResponse(
        status=status,
        protocol_schema=DISPATCH_WORKER_PROTOCOL_SCHEMA,
        protocol_version=DISPATCH_WORKER_PROTOCOL_VERSION,
        dryml_version=_dryml_version(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        pid=os.getpid(),
        features=FEATURES,
        operation_kinds=("function_call", "method_call"),
        call_transports=("import_ref", "pickle_small"),
        store_ref_kinds=("dir_store",),
        record_schemas={"record": 1, "dispatch": 1, "execution_recipe": 1},
        runtime_modes=("worker",),
        environment_kind=envelope.environment_spec.get("kind"),
        process_group=(os.name == "posix"),
        store_status=store_status or {},
        diagnostics=diagnostics,
    )


def _execute(envelope: ExecutionEnvelope, repo: Repo, store: Any) -> WorkerResponse:
    allocation = allocation_from_json(envelope.allocation_view)
    runtime_spec = RuntimeContextSpec.from_data(envelope.runtime_spec or {"mode": "worker", "device_visibility": {"policy": "assigned"}})
    try:
        with activate(mode=RuntimeMode.WORKER, allocation=allocation, spec=runtime_spec, restore_environ=False):
            _report("dryml.dispatch.worker.execute", "Running operation in worker", operation_id=envelope.operation_id)
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
            produced_record_ids=(record_id,) if record_id else (),
            execution_record_id=record_id,
            stdout_ref=StorageRef.self_product(path="stdout.txt", role="stdout").to_json() if record_id else None,
            stderr_ref=StorageRef.self_product(path="stderr.txt", role="stderr").to_json() if record_id else None,
        )
    except BaseException as exc:
        return _failure_response(envelope, exc, store=store)


def _failure_response(envelope: ExecutionEnvelope, exc: BaseException, *, store: Any) -> WorkerResponse:
    error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
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


def _write_worker_record(envelope: ExecutionEnvelope, store: Any, status: str, *, error: Mapping[str, Any] | None = None, diagnostics: tuple[Mapping[str, Any], ...] = (), consumed_cdef_ids: tuple[str, ...] = (), produced_cdef_ids: tuple[str, ...] = ()) -> str | None:
    if envelope.record_policy == "none" or store is None:
        return None
    _report("dryml.dispatch.result.save", "Saving dispatch outputs", operation_id=envelope.operation_id, data={"status": status})
    logs = (
        ExecutionLogRef("stdout", StorageRef.self_product(path="stdout.txt", role="stdout"), "text/plain"),
        ExecutionLogRef("stderr", StorageRef.self_product(path="stderr.txt", role="stderr"), "text/plain"),
    )
    execution = ExecutionRecord(
        execution_kind="python",
        operation_id=envelope.operation_id,
        backend=BACKEND_IDENTITY,
        status=status,
        dispatch_id=envelope.dispatch_spec.get("id"),
        recipe_id=envelope.execution_recipe.get("id"),
        consumed_cdef_ids=consumed_cdef_ids,
        produced_cdef_ids=produced_cdef_ids,
        logs=logs,
        error=ExecutionErrorInfo.from_json(error) if error else None,
        diagnostics=diagnostics,
    )
    _report("dryml.dispatch.execution_record.write", "Writing execution record", operation_id=envelope.operation_id, data={"status": status})
    return write_execution_record(store.records, execution).record_id


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
