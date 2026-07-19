"""Local coordinator for managed method execution, resume, reuse, and results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from dryml.core2.repo import get_default_repo
from dryml.operations import attach_operation_id, make_method_call_spec
from dryml.records import (
    ExecutionRecord,
    ExecutionRecordLink,
    LocatedRecordRef,
    RealizationRecord,
    require_product_integrity,
)
from dryml.records.execution import persistence_safe_execution_error

from .callbacks import CallbackCoordinator, preflight_callbacks
from .context import (
    OperationContext,
    OperationPreflight,
    _OperationInterrupted,
)
from .declarations import ManagedMethodDeclaration
from .descriptor import BoundManagedMethod
from .errors import (
    ManagedCapabilityError,
    ManagedInterruptedError,
    ManagedOutputError,
    ManagedRerunRequiredError,
)
from .events import ProgressSnapshot
from .refs import ManagedOutputRef
from .resolution import resolve_inputs
from .state import OperationKey, RealizationState, declaration_fingerprint
from .store import ManagedOperationStore, resolve_managed_store


@dataclass(frozen=True, slots=True)
class ManagedStatus:
    """Bounded Store-backed lifecycle status for one current declaration."""

    status: str
    active_realization_id: str | None = None
    pending_realization_id: str | None = None
    checkpoint_head: str | None = None
    progress: ProgressSnapshot | None = None


@dataclass(frozen=True, slots=True)
class ManagedInvocationResult:
    """Exact effects returned by local completion or completed-result reuse."""

    action: str
    realization_id: str
    realization_record_id: str
    outputs: Mapping[str, LocatedRecordRef]
    consumed_records: tuple[Any, ...]
    diagnostics: tuple[str, ...] = ()
    early_completed: bool = False


@dataclass(frozen=True, slots=True)
class _EffectiveCapabilities:
    resumable: bool
    checkpoint_schema: str | None
    early_completion: bool


def should_use_managed_runtime(bound: BoundManagedMethod, kwargs: Mapping[str, Any]) -> bool:
    """Return whether a call selected lifecycle context instead of U1 direct mode."""

    if any(name in kwargs for name in ("repo", "store", "callbacks", "rerun")):
        return True
    default_repo = get_default_repo()
    if default_repo is None or not default_repo.stores:
        return False
    resolve_managed_store(default_repo, target=bound.__self__)
    return True


def invoke_managed(
    bound: BoundManagedMethod,
    *args: Any,
    repo: Any | None = None,
    store: Any | None = None,
    callbacks: Any = (),
    rerun: bool = False,
    **kwargs: Any,
) -> ManagedInvocationResult:
    """Execute, resume, rerun, or reuse one managed method in one Store."""

    if not isinstance(bound, BoundManagedMethod):
        raise TypeError("invoke_managed requires a bound managed method")
    if not isinstance(rerun, bool):
        raise TypeError("rerun must be a bool")
    descriptor = bound._descriptor
    producer = bound.__self__
    declaration = descriptor.declaration
    selected = resolve_managed_store(repo, store=store, target=producer)
    outputs = descriptor.output_declarations(producer)
    dynamic = _operation_preflight(producer, descriptor.method_name, args, kwargs)
    capabilities = _effective_capabilities(declaration, dynamic)
    normalized_callbacks = preflight_callbacks(
        callbacks,
        resumable=capabilities.resumable,
        checkpoint_schema=capabilities.checkpoint_schema,
        early_completion=capabilities.early_completion,
    )
    input_refs = _managed_inputs(producer, descriptor.method_name, args, kwargs)
    operation_spec = attach_operation_id(
        make_method_call_spec(
            OperationKey.from_producer(producer, descriptor.method_name).producer_cdef_id,
            descriptor.method_name,
        )
    )

    key = OperationKey.from_producer(producer, descriptor.method_name)
    fingerprint = declaration_fingerprint(
        descriptor.method_name,
        declaration,
        producer=producer,
    )
    managed_store = ManagedOperationStore(selected)
    namespace = managed_store._read_namespace(key, missing_ok=True)
    advance = namespace is not None and namespace.current_declaration_fingerprint != fingerprint
    operation = managed_store.operation(key, fingerprint)
    validator = getattr(producer, "__dryml_managed_validate_invocation__", None)
    if validator is not None:
        control = operation._read_control(missing_ok=True)
        validator(
            descriptor.method_name,
            args,
            dict(kwargs),
            store=selected,
            operation=operation,
            has_active=operation.active(missing_ok=True) is not None,
            has_pending=control is not None and control.pending_realization_id is not None,
            rerun=rerun,
        )
    control = None if advance else operation._read_control(missing_ok=True)
    pending_id = None if control is None else control.pending_realization_id
    if not advance:
        pending_cannot_resume = pending_id is not None and not rerun
        pending_cannot_resume = pending_cannot_resume and not capabilities.resumable
        if pending_cannot_resume:
            raise ManagedRerunRequiredError(
                "pending realization cannot resume under the current whole-pipeline capabilities; explicit rerun is required"
            )
    pending_state = (
        operation._read_realization(pending_id)
        if pending_id is not None and not rerun
        else None
    )
    if pending_state is not None:
        _validate_bound_inputs(
            input_refs,
            pending_state.consumed_records,
            pending_state.consumed_record_links,
            selected,
        )
        preflight_inputs = pending_state.consumed_records
        preflight_record_inputs = pending_state.consumed_record_links
    else:
        preflight_inputs = resolve_inputs(input_refs, store=selected) if input_refs else ()
        preflight_record_inputs = ()
    _validate_managed_inputs(
        producer,
        descriptor.method_name,
        args,
        kwargs,
        store=selected,
        consumed=preflight_inputs,
        record_inputs=preflight_record_inputs,
    )

    with operation.acquire(advance_declaration=advance) as lease:
        current_control = operation._read_control(missing_ok=True)
        current_pending_id = None if current_control is None else current_control.pending_realization_id
        if current_pending_id is not None and not rerun:
            current_pending = operation._read_realization(current_pending_id)
            consumed = current_pending.consumed_records
            record_inputs = current_pending.consumed_record_links
            _validate_bound_inputs(input_refs, consumed, record_inputs, selected)
        else:
            consumed = resolve_inputs(input_refs, store=selected) if input_refs else ()
            record_inputs = _managed_record_inputs(
                producer, descriptor.method_name, args, kwargs, store=selected
            )
        _validate_managed_inputs(
            producer,
            descriptor.method_name,
            args,
            kwargs,
            store=selected,
            consumed=consumed,
            record_inputs=record_inputs,
        )
        if preflight_inputs and consumed != preflight_inputs:
            # Both vectors were individually stable. The under-lease vector is
            # the execution-time authority and is recorded exactly.
            preflight_inputs = consumed

        def active_inputs_valid(active: RealizationState) -> bool:
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
            resumable=capabilities.resumable,
            rerun=rerun,
            active_inputs_valid=active_inputs_valid,
            consumed_records=tuple(consumed),
            consumed_record_links=tuple(record_inputs),
        )
        if decision.action == "reuse":
            return _invocation_from_state(selected, decision.action, decision.realization)

        selected.records.write_spec(operation_spec, family="operation")
        coordinator = CallbackCoordinator(normalized_callbacks)
        context = OperationContext(
            producer=producer,
            method=descriptor.method_name,
            outputs=outputs,
            lease=lease,
            realization_id=decision.realization.realization_id,
            coordinator=coordinator,
            checkpoint_schema=capabilities.checkpoint_schema,
            early_completion=capabilities.early_completion,
            is_resume=decision.action == "resume",
            consumed_records=tuple(consumed),
            consumed_record_links=tuple(record_inputs),
        )
        context.publish_terminal("started")
        token = context.activate()
        try:
            implementation_result = bound.__func__(producer, *args, **kwargs)
            context.completion_point()
            result = context.validate_result(implementation_result)
            output_records = context.output_records()
            execution = ExecutionRecord(
                execution_kind="python",
                operation_id=operation_spec["id"],
                backend={"name": "dryml.managed.local", "kind": "local"},
                status="ok",
                realization_id=decision.realization.realization_id,
                consumed_records=tuple(
                    ExecutionRecordLink.from_resolved(item) for item in consumed
                ) + tuple(record_inputs),
            )
            publication = context.writer.finalize(
                output_records,
                execution,
                primary_output_slot=outputs.primary.slot,
                required_output_slots=outputs.slots,
                activate=True,
            )
            return ManagedInvocationResult(
                action=decision.action,
                realization_id=decision.realization.realization_id,
                realization_record_id=publication.realization_record.record_id,
                outputs=MappingProxyType(dict(publication.output_records)),
                consumed_records=tuple(consumed),
                diagnostics=context.diagnostics,
                early_completed=result.early_completed,
            )
        except _OperationInterrupted as exc:
            checkpoint = context.checkpoint_head
            lease.interrupt(
                decision.realization.realization_id,
                checkpoint_head=checkpoint,
                diagnostic="operation interrupted at a safe point",
                resumable=checkpoint is not None,
            )
            context.publish_terminal("interrupted")
            raise ManagedInterruptedError(
                "managed operation was interrupted at a safe point"
            ) from exc
        except Exception as exc:
            checkpoint = context.checkpoint_head
            failure = persistence_safe_execution_error(exc)
            diagnostic = f"{failure['type']}: {failure['metadata']['code']}"
            lease.fail(
                decision.realization.realization_id,
                checkpoint_head=checkpoint,
                diagnostic=diagnostic,
                resumable=checkpoint is not None,
            )
            context.publish_terminal("failed")
            raise
        finally:
            OperationContext.deactivate(token)


def managed_status(
    bound: BoundManagedMethod,
    *,
    repo: Any | None = None,
    store: Any | None = None,
) -> ManagedStatus:
    """Read current status without creating operation state."""

    selected, operation = _bound_operation(
        bound, repo=repo, store=store, writable=False
    )
    del selected
    control = operation._read_control(missing_ok=True)
    if control is None:
        return ManagedStatus("not_started")
    active = operation.active(missing_ok=True)
    pending = (
        operation._read_realization(control.pending_realization_id)
        if control.pending_realization_id is not None
        else None
    )
    status = pending.status if pending is not None else ("completed" if active is not None else "not_started")
    return ManagedStatus(
        status=status,
        active_realization_id=None if active is None else active.realization_id,
        pending_realization_id=None if pending is None else pending.realization_id,
        checkpoint_head=control.checkpoint_head,
        progress=control.progress,
    )


def managed_history(bound: BoundManagedMethod, *, repo=None, store=None) -> tuple[RealizationState, ...]:
    """Return retained current-generation realization history."""

    _selected, operation = _bound_operation(
        bound, repo=repo, store=store, writable=False
    )
    return operation.history() if operation.control_path.exists() else ()


def managed_results(bound: BoundManagedMethod, *, repo=None, store=None) -> Mapping[str, LocatedRecordRef]:
    """Return exact active output record refs without computing dependencies."""

    selected, operation = _bound_operation(
        bound, repo=repo, store=store, writable=False
    )
    active = operation.active(missing_ok=True)
    if active is None or active.realization_record_id is None:
        return MappingProxyType({})
    return _output_refs(selected, active.realization_record_id)


def managed_activate(
    bound: BoundManagedMethod,
    realization_id: str,
    *,
    repo=None,
    store=None,
) -> RealizationState:
    """Explicitly activate one completed compatible retained realization."""

    _selected, operation = _bound_operation(bound, repo=repo, store=store)
    with operation.acquire() as lease:
        return lease.activate(realization_id)


def _bound_operation(
    bound: BoundManagedMethod, *, repo=None, store=None, writable: bool = True
):
    producer = bound.__self__
    selected = resolve_managed_store(
        repo, store=store, target=producer, writable=writable
    )
    key = OperationKey.from_producer(producer, bound._descriptor.method_name)
    fingerprint = declaration_fingerprint(
        bound._descriptor.method_name,
        bound._descriptor.declaration,
        producer=producer,
    )
    managed_store = ManagedOperationStore(selected, writable=writable)
    namespace = managed_store._read_namespace(key, missing_ok=True)
    operation = managed_store.operation(key, fingerprint)
    if namespace is None or namespace.current_declaration_fingerprint != fingerprint:
        return selected, operation
    return selected, operation


def _operation_preflight(producer, method, args, kwargs) -> OperationPreflight:
    provider = getattr(producer, "__dryml_managed_preflight__", None)
    if provider is None:
        return OperationPreflight()
    value = provider(method, args, dict(kwargs))
    if not isinstance(value, OperationPreflight):
        raise TypeError("__dryml_managed_preflight__ must return OperationPreflight")
    return value


def _effective_capabilities(
    declaration: ManagedMethodDeclaration,
    dynamic: OperationPreflight,
) -> _EffectiveCapabilities:
    resumable = declaration.resumable and dynamic.resumable is not False
    early = declaration.early_completion and dynamic.early_completion is not False
    if dynamic.resumable is True and not declaration.resumable:
        raise ManagedCapabilityError("dynamic pipeline cannot add undeclared resume capability")
    if dynamic.early_completion is True and not declaration.early_completion:
        raise ManagedCapabilityError("dynamic pipeline cannot add undeclared early completion capability")
    checkpoint = declaration.checkpoint_schema if resumable else None
    if dynamic.checkpoint_schema is not None:
        if not resumable or dynamic.checkpoint_schema != declaration.checkpoint_schema:
            raise ManagedCapabilityError("dynamic checkpoint schema is incompatible with the declaration")
        checkpoint = dynamic.checkpoint_schema
    if resumable and checkpoint is None:
        raise ManagedCapabilityError("resumable operation requires a compatible checkpoint schema")
    return _EffectiveCapabilities(resumable, checkpoint, early)


def _managed_inputs(producer, method, args, kwargs) -> tuple[ManagedOutputRef, ...]:
    provider = getattr(producer, "__dryml_managed_inputs__", None)
    if provider is not None:
        refs = tuple(provider(method, args, dict(kwargs)))
    else:
        refs = tuple(_walk_refs((args, kwargs)))
    if any(not isinstance(ref, ManagedOutputRef) for ref in refs):
        raise TypeError("managed input provider must return only ManagedOutputRef values")
    return refs


def _managed_record_inputs(producer, method, args, kwargs, *, store) -> tuple[ExecutionRecordLink, ...]:
    provider = getattr(producer, "__dryml_managed_record_inputs__", None)
    if provider is None:
        return ()
    values = tuple(provider(method, args, dict(kwargs), store=store))
    links = []
    for value in values:
        if not isinstance(value, ExecutionRecordLink):
            raise TypeError("managed record input provider must return ExecutionRecordLink values")
        if value.producer_cdef_id is not None:
            raise TypeError("managed direct record inputs cannot claim logical-output resolution")
        envelope = store.records.read_record(value.record_id)
        require_product_integrity(store.records, envelope)
        links.append(value)
    return tuple(links)


def _validate_bound_inputs(input_refs, consumed, record_inputs, store) -> None:
    if len(input_refs) != len(consumed):
        raise ManagedCapabilityError("pending realization input contract no longer matches")
    for ref, resolved in zip(input_refs, consumed):
        key = OperationKey.from_producer(ref.producer, ref.method)
        if (
            resolved.producer_cdef_id != key.producer_cdef_id
            or resolved.method != ref.method
            or resolved.output_slot != ref.slot
        ):
            raise ManagedCapabilityError("pending realization input binding is incompatible")
        require_product_integrity(store.records, store.records.read_record(resolved.record_id))
    for link in record_inputs:
        require_product_integrity(store.records, store.records.read_record(link.record_id))


def _validate_managed_inputs(
    producer,
    method,
    args,
    kwargs,
    *,
    store,
    consumed,
    record_inputs,
) -> None:
    """Run an optional producer check over exact resolved input records."""

    validator = getattr(producer, "__dryml_managed_validate_inputs__", None)
    if validator is not None:
        validator(
            method,
            args,
            dict(kwargs),
            store=store,
            consumed_records=tuple(consumed),
            consumed_record_links=tuple(record_inputs),
        )


def _walk_refs(value, *, _seen=None):
    seen = set() if _seen is None else _seen
    if isinstance(value, ManagedOutputRef):
        yield value
        return
    marker = id(value)
    if marker in seen:
        return
    if isinstance(value, (tuple, list)):
        seen.add(marker)
        for item in value:
            yield from _walk_refs(item, _seen=seen)
    elif isinstance(value, Mapping):
        seen.add(marker)
        for item in value.values():
            yield from _walk_refs(item, _seen=seen)


def _invocation_from_state(selected, action, state):
    if state.realization_record_id is None:
        raise ManagedOutputError("completed realization has no immutable realization record")
    realization = RealizationRecord.from_envelope(
        selected.records.read_record(state.realization_record_id)
    )
    return ManagedInvocationResult(
        action=action,
        realization_id=state.realization_id,
        realization_record_id=state.realization_record_id,
        outputs=_output_refs(selected, state.realization_record_id),
        consumed_records=realization.consumed_records,
    )


def _output_refs(selected, realization_record_id):
    realization = RealizationRecord.from_envelope(
        selected.records.read_record(realization_record_id)
    )
    store_ref = selected.catalog_key()
    return MappingProxyType({
        output.slot: LocatedRecordRef(store_ref, output.record_id)
        for output in realization.outputs
    })


__all__ = [
    "ManagedInvocationResult",
    "ManagedStatus",
    "invoke_managed",
    "managed_activate",
    "managed_history",
    "managed_results",
    "managed_status",
    "should_use_managed_runtime",
]
