"""Managed method bridge for the existing local subprocess dispatch path."""

from __future__ import annotations

import inspect
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from dryml.formats.refs import format_cdef_id
from dryml.operations import resolve_call_arguments
from dryml.records import (
    ExecutionCancellationInfo,
    ExecutionErrorInfo,
    ExecutionLogRef,
    ExecutionRecord,
    ExecutionRecordLink,
    RealizationRecord,
    ResolvedRecord,
    RepresentationSpec,
    StorageRef,
    attach_spec_id,
    validate_attempt_id,
    validate_realization_id,
    write_execution_record,
)
from dryml.records.execution import persistence_safe_execution_error

from .callbacks import CallbackCoordinator, ControlRequest, preflight_callbacks
from .context import OperationContext, OperationResult, OutputEffect, _OperationInterrupted
from .declarations import resolve_definition_path
from .descriptor import BoundManagedMethod, ManagedMethod
from .errors import (
    CallbackFailure,
    ManagedOutputError,
    ManagedRerunRequiredError,
)
from .events import OperationEvent, ProgressSnapshot
from .runtime import (
    ManagedInvocationResult,
    _effective_capabilities,
    _invocation_from_state,
    _managed_inputs,
    _managed_record_inputs,
    _operation_preflight,
    _validate_bound_inputs,
    _validate_managed_inputs,
)
from .resolution import resolve_inputs
from .state import OperationKey, declaration_fingerprint
from .store import ManagedOperationStore


MANAGED_LAUNCH_SCHEMA = "dryml.managed.operation_launch.v1"
MANAGED_TICKET_SCHEMA = "dryml.managed.write_ticket.v1"
MANAGED_RESULT_SCHEMA = "dryml.managed.operation_result.v1"
MANAGED_MAILBOX_SCHEMA = "dryml.managed.mailbox.v1"


class ManagedDispatchRequest:
    """Invocation-local coordinator policy excluded from dispatch identities."""

    def __init__(
        self,
        bound: BoundManagedMethod,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        callbacks: Any = (),
        rerun: bool = False,
    ):
        if not isinstance(bound, BoundManagedMethod):
            raise TypeError("managed dispatch requires a bound managed method")
        if not isinstance(rerun, bool):
            raise TypeError("rerun must be a bool")
        self.bound = bound
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self.rerun = rerun
        descriptor = bound._descriptor
        dynamic = _operation_preflight(
            bound.__self__, descriptor.method_name, self.args, self.kwargs
        )
        self.capabilities = _effective_capabilities(descriptor.declaration, dynamic)
        self.callbacks = preflight_callbacks(
            callbacks,
            resumable=self.capabilities.resumable,
            checkpoint_schema=self.capabilities.checkpoint_schema,
            early_completion=self.capabilities.early_completion,
        )
        self.outputs = descriptor.output_declarations(bound.__self__)
        self.input_refs = _managed_inputs(
            bound.__self__, descriptor.method_name, self.args, self.kwargs
        )

    def decorate_plan(self, plan: Any) -> Any:
        """Add only versioned launch transport and handshake requirements."""

        launch_dir = tempfile.mkdtemp(prefix="dryml-managed-launch-")
        launch = dict(plan.envelope.launch)
        cleanup = list(launch.get("cleanup_paths") or ())
        cleanup.append(launch_dir)
        launch["cleanup_paths"] = cleanup
        launch["managed"] = {
            "schema": MANAGED_LAUNCH_SCHEMA,
            "schema_version": 1,
            "ticket_path": os.path.join(launch_dir, "ticket.json"),
            "coordinator_pid": os.getpid(),
            "callback_policy": {
                "count": len(self.callbacks),
                "strict": any(not item.fail_soft for item in self.callbacks),
                "controls": sorted(
                    {
                        control.name.lower()
                        for item in self.callbacks
                        for control in item.controls
                    }
                ),
            },
        }
        handshake = dict(plan.envelope.handshake)
        required = list(handshake.get("required_features") or ())
        if "managed.operation.v1" not in required:
            required.append("managed.operation.v1")
        handshake["required_features"] = required
        envelope = replace(plan.envelope, launch=launch, handshake=handshake)
        return replace(plan, envelope=envelope, extension=self)

    def submit(self, plan: Any, backend: Any):
        """Handshake first, then acquire the coordinator lease and release work."""

        future = backend.submit(plan)
        handshake = future.wait_for_handshake(timeout=future.handshake_timeout)
        if handshake is None or handshake.status != "ok":
            return future
        try:
            prepared = self._prepare(plan)
        except BaseException:
            future.cancel(reason="managed_prepare_failed", record=False)
            raise
        if isinstance(prepared, ManagedInvocationResult):
            future.cancel(reason="managed_reuse", record=False)
            return _CompletedManagedFuture(_reuse_response(plan, prepared))
        ticket = prepared.ticket(plan)
        from dryml.dispatch.protocol import write_json_file

        write_json_file(plan.envelope.launch["managed"]["ticket_path"], ticket)
        return ManagedDispatchFuture(future, prepared)

    def _prepare(self, plan: Any):
        selected = plan.store
        producer = self.bound.__self__
        descriptor = self.bound._descriptor
        key = OperationKey.from_producer(producer, descriptor.method_name)
        fingerprint = declaration_fingerprint(
            descriptor.method_name,
            descriptor.declaration,
            producer=producer,
        )
        managed_store = ManagedOperationStore(selected)
        namespace = managed_store._read_namespace(key, missing_ok=True)
        advance = (
            namespace is not None
            and namespace.current_declaration_fingerprint != fingerprint
        )
        operation = managed_store.operation(key, fingerprint)
        validator = getattr(producer, "__dryml_managed_validate_invocation__", None)
        if validator is not None:
            control = operation._read_control(missing_ok=True)
            validator(
                descriptor.method_name,
                self.args,
                dict(self.kwargs),
                store=selected,
                operation=operation,
                has_active=operation.active(missing_ok=True) is not None,
                has_pending=(
                    control is not None
                    and control.pending_realization_id is not None
                ),
                rerun=self.rerun,
            )
        if not advance:
            control = operation._read_control(missing_ok=True)
            pending_id = None if control is None else control.pending_realization_id
            if pending_id is not None and not self.rerun and not self.capabilities.resumable:
                raise ManagedRerunRequiredError(
                    "pending realization cannot resume under the current whole-pipeline capabilities; explicit rerun is required"
                )
        else:
            pending_id = None
        if pending_id is not None and not self.rerun:
            pending = operation._read_realization(pending_id)
            _validate_bound_inputs(
                self.input_refs,
                pending.consumed_records,
                pending.consumed_record_links,
                selected,
            )
            preflight_inputs = pending.consumed_records
            preflight_record_inputs = pending.consumed_record_links
        else:
            preflight_inputs = (
                resolve_inputs(self.input_refs, store=selected)
                if self.input_refs
                else ()
            )
            preflight_record_inputs = ()
        _validate_managed_inputs(
            producer,
            descriptor.method_name,
            self.args,
            self.kwargs,
            store=selected,
            consumed=preflight_inputs,
            record_inputs=preflight_record_inputs,
        )
        lease = operation.acquire(advance_declaration=advance)
        try:
            current_control = operation._read_control(missing_ok=True)
            current_pending_id = None if current_control is None else current_control.pending_realization_id
            if current_pending_id is not None and not self.rerun:
                current_pending = operation._read_realization(current_pending_id)
                consumed = current_pending.consumed_records
                record_inputs = current_pending.consumed_record_links
                _validate_bound_inputs(self.input_refs, consumed, record_inputs, selected)
            else:
                consumed = resolve_inputs(self.input_refs, store=selected) if self.input_refs else ()
                record_inputs = _managed_record_inputs(
                    producer,
                    descriptor.method_name,
                    self.args,
                    self.kwargs,
                    store=selected,
                )
            _validate_managed_inputs(
                producer,
                descriptor.method_name,
                self.args,
                self.kwargs,
                store=selected,
                consumed=consumed,
                record_inputs=record_inputs,
            )

            def active_inputs_valid(active):
                if active.realization_record_id is None:
                    return False
                realization = RealizationRecord.from_envelope(
                    selected.records.read_record(active.realization_record_id)
                )
                execution = ExecutionRecord.from_envelope(
                    selected.records.read_record(realization.execution_record_id)
                )
                direct = tuple(
                    link for link in execution.consumed_records
                    if link.producer_cdef_id is None
                )
                return realization.consumed_records == consumed and direct == record_inputs

            decision = lease.prepare(
                resumable=self.capabilities.resumable,
                rerun=self.rerun,
                active_inputs_valid=active_inputs_valid,
                consumed_records=tuple(consumed),
                consumed_record_links=tuple(record_inputs),
            )
            if decision.action == "reuse":
                result = _invocation_from_state(selected, decision.action, decision.realization)
                lease.release()
                return result
            coordinator = CallbackCoordinator(self.callbacks)
            context = OperationContext(
                producer=producer,
                method=descriptor.method_name,
                outputs=self.outputs,
                lease=lease,
                realization_id=decision.realization.realization_id,
                coordinator=coordinator,
                checkpoint_schema=self.capabilities.checkpoint_schema,
                early_completion=self.capabilities.early_completion,
                is_resume=decision.action == "resume",
                consumed_records=tuple(consumed),
                consumed_record_links=tuple(record_inputs),
            )
            return _ManagedCoordinatorSession(
                plan=plan,
                request=self,
                lease=lease,
                decision=decision,
                context=context,
                consumed=tuple(consumed),
                record_inputs=tuple(record_inputs),
            )
        except BaseException:
            lease.release()
            raise


@dataclass(slots=True)
class _ManagedCoordinatorSession:
    plan: Any
    request: ManagedDispatchRequest
    lease: Any
    decision: Any
    context: OperationContext
    consumed: tuple[Any, ...]
    record_inputs: tuple[Any, ...]
    last_sequence: int = 0
    finished: bool = False
    ticket_data: dict[str, Any] | None = None

    def ticket(self, plan: Any) -> dict[str, Any]:
        if self.ticket_data is not None:
            return self.ticket_data
        state = self.lease.operation._read_realization(self.decision.realization.realization_id)
        workspace = self.context.writer.workspace
        event_path = workspace / "worker-event.json"
        control_path = workspace / "coordinator-control.json"
        checkpoint_path = self.context.checkpoint_path
        self.ticket_data = {
            "schema": MANAGED_TICKET_SCHEMA,
            "schema_version": 1,
            "operation_id": plan.envelope.operation_id,
            "store_path": os.path.abspath(os.fspath(plan.store.base_dir)),
            "producer_cdef_id": self.lease.operation.key.producer_cdef_id,
            "method": self.lease.operation.key.method,
            "declaration_fingerprint": self.lease.operation.declaration_fingerprint,
            "realization_id": state.realization_id,
            "attempt_id": state.current_attempt_id,
            "fence_epoch": self.lease.epoch,
            "workspace": str(workspace.resolve()),
            "event_path": str(event_path.resolve()),
            "control_path": str(control_path.resolve()),
            "checkpoint_schema": self.request.capabilities.checkpoint_schema,
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path.resolve()),
            "early_completion": self.request.capabilities.early_completion,
            "is_resume": self.decision.action == "resume",
            "consumed_records": [item.to_json() for item in self.consumed],
            "consumed_record_links": [item.to_json() for item in self.record_inputs],
            "coordinator_pid": os.getpid(),
            "control_timeout": 300.0,
        }
        return self.ticket_data

    def poll(self) -> bool:
        """Apply at most one mailbox intent and acknowledge its control generation."""

        ticket = self.ticket(self.plan)
        event_path = Path(ticket["event_path"])
        if not event_path.exists():
            return False
        from dryml.dispatch.protocol import read_json_file, write_json_file

        wire = read_json_file(str(event_path))
        if (
            not isinstance(wire, Mapping)
            or wire.get("schema") != MANAGED_MAILBOX_SCHEMA
            or wire.get("schema_version") != 1
        ):
            raise ManagedOutputError("managed worker event mailbox is malformed")
        sequence = wire.get("sequence")
        if type(sequence) is not int or sequence != self.last_sequence + 1:
            if sequence == self.last_sequence:
                return False
            raise ManagedOutputError("managed worker event sequence is stale or discontinuous")
        kind = wire.get("kind")
        checkpoint_id = None
        if kind == "event":
            if set(wire) != {"schema", "schema_version", "sequence", "kind", "event"}:
                raise ManagedOutputError("managed worker event fields are malformed")
            self.context.apply_worker_event(_event_from_json(wire.get("event")))
        elif kind == "checkpoint_intent":
            if set(wire) != {
                "schema", "schema_version", "sequence", "kind",
                "checkpoint_schema", "metadata",
            }:
                raise ManagedOutputError("managed checkpoint intent fields are malformed")
            if wire.get("checkpoint_schema") != self.request.capabilities.checkpoint_schema:
                raise ManagedOutputError("worker checkpoint schema does not match launch capability")
            metadata = wire.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                raise ManagedOutputError("worker checkpoint metadata must be a mapping")
            checkpoint_id = self.context.commit_worker_checkpoint(
                wire["checkpoint_schema"], metadata=metadata
            )
            if self.context.coordinator.poll() is ControlRequest.CHECKPOINT:
                self.context.coordinator.consume_checkpoint()
        else:
            raise ManagedOutputError("managed worker mailbox intent kind is unsupported")
        self.last_sequence = sequence
        control = self.context.coordinator.poll()
        write_json_file(
            ticket["control_path"],
            {
                "schema": MANAGED_MAILBOX_SCHEMA,
                "schema_version": 1,
                "sequence": sequence,
                "control": control.name.lower(),
                "checkpoint_id": checkpoint_id,
            },
        )
        return True

    def finish(self, response: Any) -> Any:
        if self.finished:
            return response
        while self.poll():
            pass
        managed = response.managed_result
        if response.status == "ok":
            if not isinstance(managed, Mapping) or managed.get("status") != "ok":
                return self._finish_incomplete(
                    replace(
                        response,
                        status="failed",
                        error={"type": "WorkerProtocolError", "message": "managed worker omitted a successful structured result"},
                    ),
                    status="failed",
                )
            try:
                self.lease.assert_current()
                for spec in managed.get("representations") or ():
                    self.plan.store.records.write_spec(spec, family="representation")
                effects = managed.get("effects")
                if not isinstance(effects, Mapping):
                    raise ManagedOutputError("managed worker effects must be a mapping")
                for slot, value in effects.items():
                    self.context.register_output_effect(_effect_from_json(slot, value))
                result = self.context.validate_result(
                    OperationResult(bool(managed.get("early_completed", False)))
                )
                execution = ExecutionRecord(
                    execution_kind="python",
                    operation_id=self.plan.envelope.operation_id,
                    backend={"name": "dryml.local_subprocess", "kind": "local_subprocess", "version": "1"},
                    status="ok",
                    dispatch_id=self.plan.dispatch_spec.get("id"),
                    recipe_id=self.plan.execution_recipe.get("id"),
                    realization_id=self.decision.realization.realization_id,
                    consumed_records=tuple(
                        ExecutionRecordLink.from_resolved(item) for item in self.consumed
                    ) + tuple(self.record_inputs),
                    logs=_execution_logs(),
                )
                publication = self.context.writer.finalize(
                    self.context.output_records(),
                    execution,
                    primary_output_slot=self.request.outputs.primary.slot,
                    required_output_slots=self.request.outputs.slots,
                    activate=True,
                )
                final = {
                    "schema": MANAGED_RESULT_SCHEMA,
                    "schema_version": 1,
                    "status": "ok",
                    "action": self.decision.action,
                    "realization_id": self.decision.realization.realization_id,
                    "realization_record_id": publication.realization_record.record_id,
                    "outputs": {
                        slot: ref.to_json()
                        for slot, ref in publication.output_records.items()
                    },
                    "consumed_records": [item.to_json() for item in self.consumed],
                    "diagnostics": list(self.context.diagnostics),
                    "early_completed": result.early_completed,
                }
                self.finished = True
                self.lease.release()
                return replace(
                    response,
                    execution_record_id=publication.execution_record.record_id,
                    produced_record_ids=tuple(
                        publication.output_records[slot].record_id
                        for slot in self.request.outputs.slots
                    ),
                    managed_result=final,
                )
            except BaseException as exc:
                failed = replace(
                    response,
                    status="failed",
                    error=persistence_safe_execution_error(exc),
                    diagnostics=(*response.diagnostics, {"message": "managed coordinator finalization failed"}),
                )
                return self._finish_incomplete(failed, status="failed")
        status = "interrupted" if response.status == "cancelled" or (managed or {}).get("status") == "interrupted" else "failed"
        return self._finish_incomplete(response, status=status)

    def _finish_incomplete(self, response: Any, *, status: str) -> Any:
        if self.finished:
            return response
        checkpoint = self.context.checkpoint_head
        diagnostic = None
        if response.error:
            failure = persistence_safe_execution_error(response.error)
            diagnostic = f"{failure['type']}: {failure['metadata']['code']}"
        elif response.cancellation:
            diagnostic = f"cancelled: {str(response.cancellation.get('reason', 'requested'))[:384]}"
        try:
            state = self.lease.operation._read_realization(
                self.decision.realization.realization_id
            )
            if state.status == "completed":
                # Publication may have completed before pointer-last activation
                # failed. Preserve that verified inactive realization rather
                # than trying to rewrite it as incomplete.
                pass
            elif status == "interrupted":
                self.lease.interrupt(
                    self.decision.realization.realization_id,
                    checkpoint_head=checkpoint,
                    diagnostic=diagnostic or "managed dispatch interrupted",
                    resumable=checkpoint is not None,
                )
            else:
                self.lease.fail(
                    self.decision.realization.realization_id,
                    checkpoint_head=checkpoint,
                    diagnostic=diagnostic or "managed dispatch failed",
                    resumable=checkpoint is not None,
                )
        finally:
            self.finished = True
            self.lease.release()
        worker_managed = dict(response.managed_result or {})
        worker_managed.update(
            {
                "schema": MANAGED_RESULT_SCHEMA,
                "schema_version": 1,
                "status": response.status,
                "action": self.decision.action,
                "realization_id": self.decision.realization.realization_id,
                "checkpoint_head": checkpoint,
                "diagnostics": list(self.context.diagnostics),
            }
        )
        record_id = response.execution_record_id or _write_incomplete_execution(
            self.plan, response, self.decision.realization.realization_id
        )
        return replace(
            response,
            execution_record_id=record_id,
            managed_result=worker_managed,
        )


class ManagedDispatchFuture:
    """Future that services worker intents while retaining coordinator ownership."""

    def __init__(self, inner: Any, session: _ManagedCoordinatorSession):
        self.inner = inner
        self.session = session
        self.worker_response = None
        self._monitor_error: BaseException | None = None
        self._stop_monitor = threading.Event()
        self._completed = threading.Event()
        self._monitor = threading.Thread(
            target=self._monitor_worker,
            name="dryml-managed-dispatch-coordinator",
            daemon=True,
        )
        self._monitor.start()

    def done(self) -> bool:
        return self._completed.is_set()

    def wait_for_handshake(self, *, timeout=None):
        return self.inner.wait_for_handshake(timeout=timeout)

    def result(self, timeout: float | None = None):
        from dryml.dispatch.errors import DispatchTimeout

        if self._completed.wait(timeout):
            return self.worker_response
        if self.inner.done():
            # ``timeout`` governs worker execution. Pointer-last coordinator
            # finalization has already started and must finish classification.
            self._completed.wait()
            return self.worker_response
        self._stop_monitor.set()
        try:
            self.inner.result(timeout=0)
        except DispatchTimeout:
            self._stop_and_join_monitor()
            response = self.inner._response
            self.worker_response = self.session._finish_incomplete(
                response, status="failed"
            )
            self.inner._response = self.worker_response
            self._completed.set()
            raise
        self._completed.wait()
        return self.worker_response

    def cancel(self, *, grace=None, reason="user", record=True) -> bool:
        if self._completed.is_set():
            return False
        self._stop_monitor.set()
        cancelled = self.inner.cancel(grace=grace, reason=reason, record=record)
        self._stop_and_join_monitor()
        response = self.inner._response
        if response is not None and not self.session.finished:
            self.worker_response = self.session._finish_incomplete(
                response, status="interrupted"
            )
            self.inner._response = self.worker_response
        self._completed.set()
        return cancelled

    def _monitor_worker(self) -> None:
        try:
            while not self._stop_monitor.is_set() and not self.inner.done():
                self.session.poll()
                self._stop_monitor.wait(0.01)
            if self._stop_monitor.is_set():
                return
            response = self.inner.result(timeout=0)
            self.worker_response = self.session.finish(response)
            self.inner._response = self.worker_response
            self.inner._persist_logs(self.worker_response.execution_record_id)
        except BaseException as exc:
            self._monitor_error = exc
            if not self.inner.done():
                self.inner.cancel(reason="managed_coordinator_error", record=True)
            response = self.inner._response
            if response is None:
                from dryml.dispatch.protocol import WorkerResponse

                response = WorkerResponse(
                    status="failed",
                    operation_id=self.session.plan.envelope.operation_id,
                    dispatch_id=self.session.plan.dispatch_spec.get("id"),
                    recipe_id=self.session.plan.execution_recipe.get("id"),
                    error=persistence_safe_execution_error(exc),
                    diagnostics=({"message": "managed coordinator event loop failed"},),
                )
            elif response.status == "cancelled":
                response = replace(
                    response,
                    status="failed",
                    error=persistence_safe_execution_error(exc),
                    cancellation=None,
                )
            self.worker_response = self.session._finish_incomplete(
                response, status="failed"
            )
            self.inner._response = self.worker_response
        finally:
            if self.worker_response is not None:
                self._completed.set()

    def _stop_and_join_monitor(self) -> None:
        self._stop_monitor.set()
        if threading.current_thread() is not self._monitor:
            self._monitor.join(timeout=5)

    def exception(self, timeout=None):
        try:
            self.result(timeout=timeout)
        except BaseException as exc:
            return exc
        return None


class _CompletedManagedFuture:
    def __init__(self, response):
        self.worker_response = response

    def done(self):
        return True

    def wait_for_handshake(self, *, timeout=None):
        del timeout
        return None

    def result(self, timeout=None):
        del timeout
        return self.worker_response

    def cancel(self, **_kwargs):
        return False


@dataclass(frozen=True, slots=True)
class ManagedWorkerExecution:
    status: str
    managed_result: Mapping[str, Any]
    error: BaseException | None = None
    cancellation: Mapping[str, Any] | None = None


def wait_for_managed_ticket(launch: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Wait after handshake for the coordinator's fenced write ticket."""

    managed = launch.get("managed")
    if managed is None:
        return None
    if not isinstance(managed, Mapping) or set(managed) != {
        "schema", "schema_version", "ticket_path", "coordinator_pid", "callback_policy"
    }:
        raise ManagedOutputError("managed launch context is malformed")
    if managed.get("schema") != MANAGED_LAUNCH_SCHEMA or managed.get("schema_version") != 1:
        raise ManagedOutputError("managed launch context schema is unsupported")
    ticket_path = managed.get("ticket_path")
    if not isinstance(ticket_path, str) or not os.path.isabs(ticket_path):
        raise ManagedOutputError("managed ticket path must be absolute")
    deadline = time.monotonic() + 60.0
    from dryml.dispatch.protocol import read_json_file

    while time.monotonic() < deadline:
        if os.path.exists(ticket_path):
            return validate_managed_ticket(read_json_file(ticket_path))
        if not _process_alive(managed.get("coordinator_pid")):
            raise ManagedOutputError("managed coordinator exited before issuing a write ticket")
        time.sleep(0.01)
    raise ManagedOutputError("managed coordinator did not issue a write ticket")


def validate_managed_ticket(value: Any) -> dict[str, Any]:
    """Validate the narrow launch-only worker write ticket."""

    fields = {
        "schema", "schema_version", "operation_id", "store_path",
        "producer_cdef_id", "method", "declaration_fingerprint",
        "realization_id", "attempt_id", "fence_epoch", "workspace",
        "event_path", "control_path", "checkpoint_schema", "checkpoint_path",
        "early_completion", "is_resume", "coordinator_pid", "control_timeout",
        "consumed_records", "consumed_record_links",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ManagedOutputError("managed write ticket fields are malformed")
    data = dict(value)
    if data["schema"] != MANAGED_TICKET_SCHEMA or data["schema_version"] != 1:
        raise ManagedOutputError("managed write ticket schema is unsupported")
    for name in (
        "operation_id", "store_path", "producer_cdef_id", "method",
        "declaration_fingerprint", "realization_id", "attempt_id", "workspace",
        "event_path", "control_path",
    ):
        if not isinstance(data[name], str) or not data[name]:
            raise ManagedOutputError(f"managed write ticket {name} is malformed")
    for name in ("store_path", "workspace", "event_path", "control_path"):
        if not os.path.isabs(data[name]):
            raise ManagedOutputError(f"managed write ticket {name} must be absolute")
    if data["checkpoint_path"] is not None and (
        not isinstance(data["checkpoint_path"], str)
        or not os.path.isabs(data["checkpoint_path"])
    ):
        raise ManagedOutputError("managed checkpoint path must be absolute or null")
    if type(data["fence_epoch"]) is not int or data["fence_epoch"] < 1:
        raise ManagedOutputError("managed write ticket fence is malformed")
    validate_realization_id(data["realization_id"])
    validate_attempt_id(data["attempt_id"])
    if type(data["coordinator_pid"]) is not int or data["coordinator_pid"] < 1:
        raise ManagedOutputError("managed write ticket coordinator PID is malformed")
    if type(data["early_completion"]) is not bool or type(data["is_resume"]) is not bool:
        raise ManagedOutputError("managed write ticket capabilities are malformed")
    try:
        data["consumed_records"] = tuple(
            item if isinstance(item, ResolvedRecord) else ResolvedRecord.from_json(item)
            for item in data["consumed_records"]
        )
        data["consumed_record_links"] = tuple(
            item if isinstance(item, ExecutionRecordLink)
            else ExecutionRecordLink.from_json(item, default_required=True)
            for item in data["consumed_record_links"]
        )
    except Exception as exc:
        raise ManagedOutputError(
            f"managed write ticket consumed inputs are malformed: {exc}"
        ) from exc
    if any(item.producer_cdef_id is not None for item in data["consumed_record_links"]):
        raise ManagedOutputError("managed direct record inputs cannot claim logical resolution")
    if data["checkpoint_schema"] is not None and (
        not isinstance(data["checkpoint_schema"], str)
        or not data["checkpoint_schema"]
    ):
        raise ManagedOutputError("managed write ticket checkpoint schema is malformed")
    if (
        isinstance(data["control_timeout"], bool)
        or not isinstance(data["control_timeout"], (int, float))
        or data["control_timeout"] <= 0
    ):
        raise ManagedOutputError("managed write ticket control timeout is malformed")
    workspace = Path(data["workspace"])
    if Path(data["event_path"]).parent != workspace or Path(data["control_path"]).parent != workspace:
        raise ManagedOutputError("managed event/control channels must belong to the attempt workspace")
    return data


def execute_managed_operation(operation_spec, *, repo, store, ticket) -> ManagedWorkerExecution:
    """Run only the underlying managed implementation under a worker context."""

    ticket = validate_managed_ticket(ticket)
    if operation_spec.get("id") != ticket["operation_id"]:
        raise ManagedOutputError("managed ticket operation identity mismatch")
    if os.path.abspath(os.fspath(store.base_dir)) != ticket["store_path"]:
        raise ManagedOutputError("managed ticket Store locator mismatch")
    managed_root = Path(store.managed_control_root()).resolve()
    workspace = Path(ticket["workspace"])
    try:
        workspace.relative_to(managed_root)
    except ValueError as exc:
        raise ManagedOutputError(
            "managed attempt workspace is outside the selected Store"
        ) from exc
    if workspace.name != f"{ticket['fence_epoch']:020d}-{ticket['attempt_id']}":
        raise ManagedOutputError("managed attempt workspace does not match its fence")
    call = resolve_call_arguments(
        operation_spec,
        materialize_cdef=lambda cdef_id: _materialize(repo, cdef_id),
        make_cdef_ref=lambda cdef_id: cdef_id,
    )
    if call.kind != "method_call" or call.method != ticket["method"]:
        raise ManagedOutputError("managed ticket requires its exact method call")
    if operation_spec.get("payload", {}).get("subject") != ticket["producer_cdef_id"]:
        raise ManagedOutputError("managed ticket producer identity mismatch")
    subject = call.subject
    descriptor = inspect.getattr_static(type(subject), call.method)
    if not isinstance(descriptor, ManagedMethod):
        raise ManagedOutputError("managed dispatch target is not a managed descriptor")
    fingerprint = declaration_fingerprint(
        call.method, descriptor.declaration, producer=subject
    )
    if fingerprint != ticket["declaration_fingerprint"]:
        raise ManagedOutputError("managed worker declaration differs from coordinator declaration")
    outputs = descriptor.output_declarations(subject)
    context = _WorkerOperationContext(
        producer=subject,
        method=call.method,
        outputs=outputs,
        store=store,
        ticket=ticket,
    )
    token = OperationContext.activate(context)
    try:
        context.publish_event("started")
        value = descriptor.__func__(subject, *call.args, **call.kwargs)
        context.completion_point()
        result = context.validate_result(value)
        return ManagedWorkerExecution(
            "ok", context.result("ok", early_completed=result.early_completed)
        )
    except _OperationInterrupted as exc:
        return ManagedWorkerExecution(
            "cancelled",
            context.result("interrupted"),
            error=exc,
            cancellation={"requested": True, "reason": "managed_interrupt"},
        )
    except BaseException as exc:
        return ManagedWorkerExecution("failed", context.result("failed"), error=exc)
    finally:
        OperationContext.deactivate(token)


class _WorkerOperationContext:
    """Worker-side attempt writer with no Store control/publication authority."""

    def __init__(self, *, producer, method, outputs, store, ticket):
        self.producer = producer
        self.method = method
        self.outputs = outputs
        self.store = store
        self.ticket = ticket
        self.realization_id = ticket["realization_id"]
        self.checkpoint_schema = ticket["checkpoint_schema"]
        self.early_completion = ticket["early_completion"]
        self.is_resume = ticket["is_resume"]
        self.consumed_records = ticket["consumed_records"]
        self.consumed_record_links = ticket["consumed_record_links"]
        self.workspace = Path(ticket["workspace"])
        if not self.workspace.is_dir():
            raise ManagedOutputError("managed attempt workspace is missing")
        self._effects: dict[str, OutputEffect] = {}
        self._representations: dict[str, Mapping[str, Any]] = {}
        self._sequence = 0
        self._event_sequence = 0
        self._pending_control = ControlRequest.NONE
        self._checkpoint_head = None
        self._checkpoint_provider = None
        self._graceful_stop_requested = False

    @property
    def checkpoint_head(self):
        return self._checkpoint_head

    @property
    def checkpoint_path(self):
        value = self.ticket["checkpoint_path"]
        return None if value is None else Path(value)

    @property
    def output_effects(self):
        return MappingProxyType(dict(self._effects))

    @property
    def diagnostics(self):
        return ()

    def progress(self, current, *, total=None, message=None, metrics=None):
        progress = ProgressSnapshot(current, total, message, metrics or {})
        self._send_event(
            OperationEvent(
                self._next_sequence(), "progress", progress_snapshot=progress
            )
        )
        return progress

    def safe_point(self, *, checkpoint=None):
        if checkpoint is not None:
            self._checkpoint_provider = checkpoint
        self._send_event(OperationEvent(self._next_sequence(), "safe_point"))
        return self._service_control(checkpoint)

    def completion_point(self):
        self._send_event(OperationEvent(self._next_sequence(), "completed"))
        self._service_control(self._checkpoint_provider)

    def _service_control(self, checkpoint):
        control = self._pending_control
        needs_checkpoint = control in {
            ControlRequest.CHECKPOINT,
            ControlRequest.INTERRUPT,
            ControlRequest.FAIL,
        } and self.checkpoint_schema is not None
        before = self._checkpoint_head
        if needs_checkpoint and checkpoint is not None:
            checkpoint()
            if self._checkpoint_head == before:
                self.commit_checkpoint()
        if control is ControlRequest.FAIL:
            raise CallbackFailure("coordinator callback failed")
        if control is ControlRequest.INTERRUPT:
            raise _OperationInterrupted()
        if control is ControlRequest.GRACEFUL_STOP:
            self._graceful_stop_requested = True
        if control is ControlRequest.CHECKPOINT:
            self._pending_control = ControlRequest.NONE
        return control

    def write_checkpoint(self, path, chunks):
        if self.checkpoint_schema is None:
            raise ManagedOutputError("operation did not declare checkpoint capability")
        self._write_stream(self.workspace / "checkpoint-staging", path, chunks)

    def commit_checkpoint(self, *, metadata=None):
        if self.checkpoint_schema is None:
            raise ManagedOutputError("operation did not declare checkpoint capability")
        ack = self._mailbox(
            "checkpoint_intent",
            checkpoint_schema=self.checkpoint_schema,
            metadata=dict(metadata or {}),
        )
        checkpoint_id = ack.get("checkpoint_id")
        if not isinstance(checkpoint_id, str):
            raise ManagedOutputError("coordinator did not acknowledge checkpoint publication")
        self._checkpoint_head = checkpoint_id
        self._send_event(
            OperationEvent(
                self._next_sequence(),
                "checkpoint",
                payload={"checkpoint_id": checkpoint_id},
            )
        )
        return checkpoint_id

    def write_output(
        self,
        slot,
        path,
        chunks,
        *,
        representation,
        record_kind=None,
        subject_cdef_id=None,
    ):
        declaration = self.outputs.get(slot)
        if declaration is None:
            raise ManagedOutputError(f"operation wrote undeclared output slot {slot!r}")
        representation_id = self._representation(representation)
        kind = record_kind or declaration.kind
        if kind == "object":
            kind = "stored_state"
        if kind not in {"data", "stored_state"}:
            raise ManagedOutputError(f"managed output kind {kind!r} is unsupported")
        if subject_cdef_id is None and declaration.subject_path is not None:
            subject = resolve_definition_path(
                self.producer.definition, declaration.subject_path
            )
            subject_cdef_id = format_cdef_id(subject.stable_hash())
        if kind == "stored_state" and subject_cdef_id is None:
            subject_cdef_id = format_cdef_id(self.producer.definition.stable_hash())
        effect = OutputEffect(slot, representation_id, kind, subject_cdef_id)
        prior = self._effects.get(slot)
        if prior is not None and prior != effect:
            raise ManagedOutputError(
                f"output slot {slot!r} changed representation or ownership"
            )
        self._write_stream(self.workspace / "outputs" / slot, path, chunks)
        self._effects[slot] = effect

    def validate_result(self, value):
        result = OperationResult() if value is None else value
        if not isinstance(result, OperationResult):
            raise ManagedOutputError(
                "managed implementation must return OperationResult or None"
            )
        if result.early_completed and not self.early_completion:
            raise ManagedOutputError("operation returned unsupported early completion")
        if self._graceful_stop_requested and not result.early_completed:
            raise ManagedOutputError(
                "graceful stop requires an explicit early-completed result"
            )
        return result

    def publish_event(self, kind):
        self._send_event(OperationEvent(self._next_sequence(), kind))

    def result(self, status, *, early_completed=False):
        return {
            "schema": MANAGED_RESULT_SCHEMA,
            "schema_version": 1,
            "status": status,
            "effects": {
                slot: _effect_to_json(effect)
                for slot, effect in sorted(self._effects.items())
            },
            "representations": list(self._representations.values()),
            "checkpoint_head": self._checkpoint_head,
            "early_completed": early_completed,
        }

    def _representation(self, value):
        if isinstance(value, RepresentationSpec):
            envelope = value.to_envelope()
        elif isinstance(value, Mapping):
            envelope = attach_spec_id(value, family="representation")
        elif isinstance(value, str):
            self.store.records.read_spec(value, family="representation")
            return value
        else:
            raise TypeError("representation must be a representation spec or ID")
        self._representations[envelope["id"]] = envelope
        return envelope["id"]

    def _send_event(self, event):
        ack = self._mailbox("event", event=_event_to_json(event))
        control = ControlRequest[ack["control"].upper()]
        if control > self._pending_control:
            self._pending_control = control

    def _mailbox(self, kind, **payload):
        self._sequence += 1
        sequence = self._sequence
        wire = {
            "schema": MANAGED_MAILBOX_SCHEMA,
            "schema_version": 1,
            "sequence": sequence,
            "kind": kind,
            **payload,
        }
        from dryml.dispatch.protocol import read_json_file, write_json_file

        write_json_file(self.ticket["event_path"], wire)
        deadline = time.monotonic() + float(self.ticket["control_timeout"])
        while time.monotonic() < deadline:
            path = self.ticket["control_path"]
            if os.path.exists(path):
                ack = read_json_file(path)
                if isinstance(ack, Mapping) and ack.get("sequence") == sequence:
                    return _validate_control_ack(ack)
            if not _process_alive(self.ticket["coordinator_pid"]):
                raise ManagedOutputError("managed coordinator exited while work was running")
            time.sleep(0.01)
        raise ManagedOutputError("managed coordinator did not service a worker safe point")

    def _write_stream(self, root, path, chunks):
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ManagedOutputError("managed product path must stay within the attempt workspace")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = self.workspace / "partials" / uuid.uuid4().hex
        partial.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("xb") as handle:
            for chunk in chunks:
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise ManagedOutputError("managed product chunks must be bytes-like")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            if target.read_bytes() != partial.read_bytes():
                raise ManagedOutputError("immutable attempt file already exists with different bytes")
            partial.unlink()
        else:
            os.replace(partial, target)

    def _next_sequence(self):
        self._event_sequence += 1
        return self._event_sequence


def _event_to_json(event):
    return {
        "sequence": event.sequence,
        "kind": event.kind,
        "progress": None if event.progress_snapshot is None else event.progress_snapshot.to_json(),
        "payload": dict(event.payload),
    }


def _event_from_json(value):
    if not isinstance(value, Mapping) or set(value) != {"sequence", "kind", "progress", "payload"}:
        raise ManagedOutputError("managed operation event is malformed")
    return OperationEvent(
        value["sequence"],
        value["kind"],
        progress_snapshot=ProgressSnapshot.from_json(value["progress"]),
        payload=value["payload"],
    )


def _effect_to_json(effect):
    return {
        "slot": effect.slot,
        "representation_id": effect.representation_id,
        "record_kind": effect.record_kind,
        "subject_cdef_id": effect.subject_cdef_id,
    }


def _effect_from_json(slot, value):
    fields = {"slot", "representation_id", "record_kind", "subject_cdef_id"}
    if not isinstance(value, Mapping) or set(value) != fields or value.get("slot") != slot:
        raise ManagedOutputError("managed output effect is malformed")
    return OutputEffect(**dict(value))


def _validate_control_ack(value):
    fields = {
        "schema", "schema_version", "sequence", "control", "checkpoint_id"
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ManagedOutputError("managed coordinator acknowledgement is malformed")
    if value.get("schema") != MANAGED_MAILBOX_SCHEMA or value.get("schema_version") != 1:
        raise ManagedOutputError("managed coordinator acknowledgement schema is unsupported")
    control = value.get("control")
    if not isinstance(control, str) or control.upper() not in ControlRequest.__members__:
        raise ManagedOutputError("managed coordinator control is unsupported")
    checkpoint_id = value.get("checkpoint_id")
    if checkpoint_id is not None and not isinstance(checkpoint_id, str):
        raise ManagedOutputError("managed coordinator checkpoint acknowledgement is malformed")
    return value


def _materialize(repo, cdef_id):
    from dryml.dispatch.operations import _materialize_cdef

    return _materialize_cdef(repo, cdef_id)


def _process_alive(pid):
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _execution_logs():
    return (
        ExecutionLogRef("stdout", StorageRef.self_product(path="stdout.txt", role="stdout"), "text/plain"),
        ExecutionLogRef("stderr", StorageRef.self_product(path="stderr.txt", role="stderr"), "text/plain"),
    )


def _write_incomplete_execution(plan, response, realization_id):
    if plan.envelope.record_policy == "none":
        return None
    execution = ExecutionRecord(
        execution_kind="python",
        operation_id=plan.envelope.operation_id,
        backend={"name": "dryml.local_subprocess", "kind": "local_subprocess", "version": "1"},
        status=response.status,
        dispatch_id=plan.dispatch_spec.get("id"),
        recipe_id=plan.execution_recipe.get("id"),
        realization_id=realization_id,
        logs=_execution_logs(),
        error=ExecutionErrorInfo.from_json(
            persistence_safe_execution_error(response.error)
        ) if response.error else None,
        cancellation=ExecutionCancellationInfo.from_json(response.cancellation) if response.cancellation else None,
        diagnostics=response.diagnostics,
    )
    return write_execution_record(plan.store.records, execution).record_id


def _reuse_response(plan, invocation):
    from dryml.dispatch.protocol import WorkerResponse

    managed = {
        "schema": MANAGED_RESULT_SCHEMA,
        "schema_version": 1,
        "status": "ok",
        "action": invocation.action,
        "realization_id": invocation.realization_id,
        "realization_record_id": invocation.realization_record_id,
        "outputs": {slot: ref.to_json() for slot, ref in invocation.outputs.items()},
        "consumed_records": [item.to_json() for item in invocation.consumed_records],
        "diagnostics": list(invocation.diagnostics),
        "early_completed": invocation.early_completed,
    }
    return WorkerResponse(
        status="ok",
        operation_id=plan.envelope.operation_id,
        dispatch_id=plan.dispatch_spec.get("id"),
        recipe_id=plan.execution_recipe.get("id"),
        produced_record_ids=tuple(ref.record_id for ref in invocation.outputs.values()),
        managed_result=managed,
    )


__all__ = [
    "MANAGED_LAUNCH_SCHEMA",
    "MANAGED_RESULT_SCHEMA",
    "ManagedDispatchFuture",
    "ManagedDispatchRequest",
    "ManagedWorkerExecution",
    "execute_managed_operation",
    "validate_managed_ticket",
    "wait_for_managed_ticket",
]
