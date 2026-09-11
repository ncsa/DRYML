"""Synchronous start, resume, rerun, status, and request lifecycle execution."""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID, uuid4

from dryml.core import Object
from dryml.core.reference_values import StateRef

from .context import _create_context
from .control import ControlSnapshot, ManagedControlStore
from .errors import (
    ManagedConfigError,
    ManagedConflictError,
    ManagedControlError,
    ManagedInterrupted,
    ManagedPublicationError,
    ManagedRecoveryError,
    ManagedRerunRequiredError,
)
from .identity import argument_digest, operation_digest
from .model import InterruptRequestResult, ManagedStatus
from .storage import (
    _acquire_state_ownership,
    publish_managed_state,
    resolve_stores,
    state_ref_for_digest,
    validate_state_ref,
)


def invoke(descriptor, instance: object, args: tuple[object, ...], managed, kwargs: dict[str, object]) -> object:
    """Execute one bound managed method synchronously through selected authority.

    Args:
        descriptor: Checked managed declaration retained by the bound view.
        instance: Exact live method receiver.
        args: Caller positional ordinary arguments.
        managed: ``ManagedConfig`` or ``None`` caller policy.
        kwargs: Caller ordinary keyword arguments.

    Returns:
        The wrapped method's ordinary result, only after final state and completed
        control authority are both published.

    Raises:
        ManagedError: For lifecycle, storage, compatibility, or cleanup failures.
        BaseException: The original method exception after best-effort failure
        publication, except ``KeyboardInterrupt`` which becomes
        ``ManagedInterrupted`` chained from it.
    """

    from .config import ManagedConfig

    _require_receiver(instance)
    if managed is not None and type(managed) is not ManagedConfig:
        raise ManagedConfigError(message="managed must be a ManagedConfig or None")
    config = ManagedConfig() if managed is None else managed
    options = config.snapshot()
    supplied = dict(kwargs)
    supplied["managed"] = managed
    arguments = argument_digest(descriptor, instance, args, supplied)
    operation_id = operation_digest(instance.object_ref, descriptor.member)
    stores = resolve_stores(instance, state_repo=options.state_repo, control_store=options.control_store)
    # Managed invocation mutates an already materialized Object graph, so it uses
    # the runtime's existing admission boundary rather than bypassing strict mode.
    from dryml.runtime import materialization_admission

    try:
        with materialization_admission(operation="managed invocation"):
            return _invoke_selected(
                descriptor, instance, args, kwargs, arguments, operation_id, stores, options,
            )
    finally:
        stores.close()


def _invoke_selected(descriptor, instance, args, kwargs, arguments, operation_id, stores, options):
    """Execute one already-admitted managed lifecycle against selected Stores."""

    state_repo = stores.state_repo
    context = None
    try:
        control = ManagedControlStore(stores.control_store, state_repo)
        current = control.reconcile(operation_id)
        expected_ownership = None if current is None else current.ownership
        with _acquire_state_ownership(state_repo, instance, expected_ownership=expected_ownership) as ownership:
            # State topology, graph reservation, and every lifetime lock are now
            # held, so control bootstrap cannot precede failed ownership admission.
            control.initialize()
            current = control.reconcile(operation_id)
            if current is not None and {key: list(value) if isinstance(value, tuple) else value for key, value in current.ownership.items()} != ownership.ownership:
                raise ManagedConflictError("ownership_changed", "managed ownership changed during admission")
            if current is None:
                running = _new_running(
                    operation_id, instance.object_ref.digest(), arguments,
                    descriptor.member, ownership.ownership, generation=1,
                )
                running = control.create_initial(running)
                is_resuming = False
                checkpoint = None
            else:
                running, is_resuming, checkpoint = _enter_existing(
                    control, current, descriptor, arguments, options.rerun,
                )
                if is_resuming:
                    try:
                        state_repo.restore_state_ref_into(
                            instance, checkpoint, reservation=ownership.reservation,
                        )
                    except BaseException as error:
                        _record_failure(control, running, "restore_error", error)
                        raise
            context = _create_context(
                control_store=stores.control_store,
                operation_id=operation_id, attempt_id=running.attempt_id,
                owner_id=running.owner_id, is_resuming=is_resuming,
                checkpoint_state_ref=checkpoint, obj=instance, state_repo=state_repo,
                ownership=ownership, control=control, callbacks=options.callbacks,
            )
            try:
                result = descriptor._target(instance, *args, managed=context, **kwargs)
                context._raise_if_interrupted()
            except ManagedInterrupted as error:
                if context._terminal_interrupted:
                    raise
                _record_failure(control, running, context._failure_code or "method_error", error)
                raise
            except KeyboardInterrupt as error:
                if context._safe_point_error is error:
                    # Save-boundary control flow has its own immutable core
                    # report; it must not be recast as a workload interruption.
                    _record_base_exception_failure(control, running, error)
                    raise
                _record_interruption(control, running, error)
                raise ManagedInterrupted("interrupted", "managed method received KeyboardInterrupt") from error
            except BaseException as error:
                if isinstance(error, ManagedPublicationError) and error.outcome == "indeterminate":
                    # A visible pending intent is authoritative uncertainty; a
                    # generic failure transition must not replace it.
                    raise
                if isinstance(error, (SystemExit, GeneratorExit)):
                    _record_base_exception_failure(control, running, error)
                else:
                    _record_failure(control, running, context._failure_code or "method_error", error)
                raise
            report = None
            try:
                final_state, report = publish_managed_state(
                    state_repo, instance, reservation=ownership.reservation,
                )
                validate_state_ref(state_repo, final_state)
                _completion_boundary("final_state_published")
                control.transition_running_owner(
                    operation_id, attempt_id=running.attempt_id, owner_id=running.owner_id,
                    build=lambda current: replace(
                        current, owner_id=None, generation=current.generation + 1,
                        state="completed", interrupt_request=None,
                        final_digest=final_state.digest(), failure_code=None,
                    ),
                )
                _completion_boundary("final_associated")
            except BaseException as error:
                if report is not None and getattr(error, "report", None) is None:
                    error.report = report
                if not isinstance(error, ManagedPublicationError) or error.outcome == "not_committed":
                    _record_failure(control, running, "publication_error", error)
                raise
            return result
    finally:
        if context is not None:
            context._deactivate()
        pass


def status(descriptor, instance: object, *, state_repo=None, control_store=None) -> ManagedStatus:
    """Project exactly one caller-selected authority without activating workload.

    Args:
        descriptor: Checked managed declaration for the bound member.
        instance: Exact live receiver used only for identity and ownership evidence.
        state_repo: Optional explicit selected state Repo or Store.
        control_store: Optional explicit selected control Store.

    Returns:
        An immutable selected-authority status.  A definitely ownerless running
        snapshot is projected as ``failed``/``owner_lost`` without rewriting it.

    Raises:
        ManagedControlError: If current authority requires reconciliation.
        ManagedRecoveryError: If a retained state association is incomplete.
    """

    _require_receiver(instance)
    operation_id = operation_digest(instance.object_ref, descriptor.member)
    stores = resolve_stores(instance, state_repo=state_repo, control_store=control_store, require_writable=False)
    try:
        control = ManagedControlStore(stores.control_store, stores.state_repo)
        initial = control.inspect(operation_id)
        if initial is None or initial.state != "running":
            return _status_from_snapshot(initial, operation_id, stores.state_repo)
        for _ in range(3):
            observed, ownerless, stable = control.observe_running_owner(operation_id)
            if observed is None or observed.state != "running":
                return _status_from_snapshot(observed, operation_id, stores.state_repo)
            if stable:
                return _status_from_snapshot(observed, operation_id, stores.state_repo, owner_lost=ownerless)
        return _status_from_snapshot(control.inspect(operation_id), operation_id, stores.state_repo)
    finally:
        stores.close()


def request_interrupt(descriptor, instance: object, *, state_repo=None,
                      control_store=None, expected_attempt_id: str | None = None) -> InterruptRequestResult:
    """Request interruption of a selected running invocation without creating one.

    A full nonblocking lock probe and unchanged generation prove owner loss.  The
    request then returns ``not_running`` without a control write; contention and
    adapter failures remain inconclusive/actionable respectively.
    """

    if expected_attempt_id is not None:
        _validate_attempt_id(expected_attempt_id)
    _require_receiver(instance)
    operation_id = operation_digest(instance.object_ref, descriptor.member)
    stores = resolve_stores(instance, state_repo=state_repo, control_store=control_store)
    try:
        control = ManagedControlStore(stores.control_store, stores.state_repo)
        initial = control.inspect(operation_id)
        if initial is None or initial.state != "running":
            return _request_result("not_running", operation_id, initial)
        for _ in range(3):
            outcome, observed = control.request_interrupt_if_running(
                operation_id, expected_attempt_id=expected_attempt_id,
            )
            if outcome != "changed":
                return _request_result(outcome, operation_id, observed)
        raise ManagedConflictError("generation_conflict", "selected running authority changed during interruption request")
    finally:
        stores.close()


def _enter_existing(control, current, descriptor, arguments: str, rerun: bool):
    """Choose and commit resume or fresh rerun for one owned selected snapshot."""

    if current.state == "completed" and not rerun:
        raise ManagedRerunRequiredError("already_completed", "selected operation is already completed")
    if rerun:
        return _transition_to_running(control, current, new_attempt=True, arguments=arguments), False, None
    if current.state not in {"running", "interrupted", "failed"}:
        raise ManagedRecoveryError("invalid_current", "selected authority cannot be resumed")
    if not descriptor.resumable or current.argument_digest != arguments or current.checkpoint_digest is None:
        raise ManagedRerunRequiredError("rerun_required", "selected operation needs explicit rerun")
    checkpoint = state_ref_for_digest(control.state_repo, current.checkpoint_digest)
    validate_state_ref(control.state_repo, checkpoint)
    return _transition_to_running(control, current, new_attempt=False, arguments=arguments), True, checkpoint


def _transition_to_running(control, current, *, new_attempt: bool, arguments: str) -> ControlSnapshot:
    """Commit a replacement owner and clear any old request before workload entry."""

    proposed = ControlSnapshot(
        current.operation_id, current.object_ref_digest,
        arguments if new_attempt else current.argument_digest,
        current.member, uuid4().hex if new_attempt else current.attempt_id, uuid4().hex,
        current.generation + 1, "running", None,
        None if new_attempt else current.checkpoint_digest, None, None, current.ownership,
    )
    return control.transition(
        current.operation_id, proposed, expected_generation=current.generation,
        expected_attempt_id=current.attempt_id,
        expected_owner_id=current.owner_id if current.state == "running" else None,
    )


def _new_running(operation_id: str, object_ref_digest: str, arguments: str,
                 member: str, ownership: dict[str, object], *, generation: int) -> ControlSnapshot:
    """Build the first running authority for an absent operation lineage."""

    return ControlSnapshot(
        operation_id, object_ref_digest, arguments, member, uuid4().hex, uuid4().hex,
        generation, "running", None, None, None, None, ownership,
    )


def _record_failure(control, running: ControlSnapshot, code: str, original: BaseException) -> None:
    """Best-effort terminal failure publication that never hides its original error."""

    try:
        control.transition_running_owner(
            running.operation_id, attempt_id=running.attempt_id, owner_id=running.owner_id,
            build=lambda current: replace(
                current, owner_id=None, generation=current.generation + 1, state="failed",
                interrupt_request=None, final_digest=None, failure_code=code,
            ),
        )
    except ManagedConflictError:
        # A replacement owner or terminal state is authoritative; stale cleanup
        # must not overwrite it or replace the original method failure.
        return
    except ManagedPublicationError as error:
        if error.outcome == "indeterminate":
            raise ManagedPublicationError(
                "indeterminate", "managed failure publication requires reconciliation",
            ) from original
        raise ManagedControlError("failure_recording_failed", "could not publish managed failure") from original
    except BaseException:
        raise ManagedControlError("failure_recording_failed", "could not publish managed failure") from original


def _record_base_exception_failure(control, running: ControlSnapshot, original: BaseException) -> None:
    """Attempt terminal cleanup for non-interruption BaseExceptions without masking them."""

    try:
        _record_failure(control, running, "method_error", original)
    except BaseException:
        pass


def _record_interruption(control, running: ControlSnapshot, original: KeyboardInterrupt) -> None:
    """Commit an honest terminal interruption without saving mutated live state.

    Args:
        control: Selected current-authority adapter for the active invocation.
        running: Invocation's owner-fenced running snapshot.
        original: Escaping caller interruption whose cause must be preserved.

    Raises:
        ManagedControlError: If interrupted authority cannot be committed safely,
            chained from ``original`` rather than a storage-side cleanup error.

    Side Effects:
        Transitions the active authority to ``interrupted`` while retaining its
        last associated checkpoint, if any.
    """

    try:
        control.transition_running_owner(
            running.operation_id, attempt_id=running.attempt_id, owner_id=running.owner_id,
            build=lambda current: replace(
                current, owner_id=None, generation=current.generation + 1,
                state="interrupted", interrupt_request=None, final_digest=None,
                failure_code=None,
            ),
        )
    except ManagedConflictError:
        raise ManagedControlError("interruption_recording_failed", "could not publish managed interruption") from original
    except ManagedPublicationError as error:
        raise ManagedPublicationError(
            error.outcome, "could not publish managed interruption",
        ) from original
    except BaseException:
        raise ManagedControlError("interruption_recording_failed", "could not publish managed interruption") from original


def _require_receiver(instance: object) -> None:
    """Require a materialized DRYML Object before managed lifecycle access."""

    if not isinstance(instance, Object):
        raise ManagedConfigError("invalid_receiver", "managed operations require a materialized Object receiver")


def _status_from_snapshot(snapshot, operation_id: str, state_repo, *, owner_lost: bool = False) -> ManagedStatus:
    """Convert validated selected current authority into its immutable public view."""

    if snapshot is None:
        return ManagedStatus("not_started", operation_id, None, 0, None, None, False, None)
    checkpoint = None if snapshot.checkpoint_digest is None else state_ref_for_digest(state_repo, snapshot.checkpoint_digest)
    final = None if snapshot.final_digest is None else state_ref_for_digest(state_repo, snapshot.final_digest)
    return ManagedStatus(
        "failed" if owner_lost else snapshot.state, snapshot.operation_id,
        snapshot.attempt_id, snapshot.generation, checkpoint, final,
        snapshot.interrupt_request is not None, "owner_lost" if owner_lost else snapshot.failure_code,
    )


def _request_result(outcome: str, operation_id: str, snapshot) -> InterruptRequestResult:
    """Build one immutable request result from the selected observed authority."""

    return InterruptRequestResult(
        outcome, operation_id, None if snapshot is None else snapshot.attempt_id,
        0 if snapshot is None else snapshot.generation,
    )


def _validate_attempt_id(attempt_id: str) -> None:
    """Validate the optional stale-attempt precondition before authority reads."""

    if type(attempt_id) is not str:
        raise ManagedConfigError(message="expected_attempt_id must be UUID hex")
    try:
        parsed = UUID(attempt_id)
    except ValueError as error:
        raise ManagedConfigError(message="expected_attempt_id must be UUID hex") from error
    if parsed.hex != attempt_id:
        raise ManagedConfigError(message="expected_attempt_id must be lowercase UUID hex")


def _completion_boundary(stage: str) -> None:
    """Provide a no-op in-process seam for deterministic final-save crash tests."""


__all__ = ["invoke", "request_interrupt", "status"]
