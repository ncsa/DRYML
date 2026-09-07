"""Invocation-scoped safe-point authority exposed to managed method bodies."""

from __future__ import annotations

import os
from threading import get_ident

from dataclasses import replace

from dryml.core.reference_values import StateRef

from .errors import ManagedContextError, ManagedControlError, ManagedInterrupted


class ManagedContext:
    """Read-only authority for one active managed invocation.

    Instances are created only by the managed runtime after state ownership and
    the running control snapshot are established.  They become inactive when the
    method exits and cannot transfer across a thread or process.
    """

    def __init__(self, token, *, state_store, control_store, operation_id: str,
                  attempt_id: str, owner_id: str, is_resuming: bool,
                  checkpoint_state_ref: StateRef | None, obj, state_repo,
                  ownership, control, callbacks) -> None:
        """Initialize private runtime authority; callers receive no construction API."""

        if token is not _CONTEXT_TOKEN:
            raise ManagedContextError("context_construction", "managed contexts are runtime-created")
        self._state_store = state_store
        self._control_store = control_store
        self._operation_id = operation_id
        self._attempt_id = attempt_id
        self._owner_id = owner_id
        self._is_resuming = is_resuming
        self._checkpoint_state_ref = checkpoint_state_ref
        self._obj = obj
        self._state_repo = state_repo
        self._ownership = ownership
        self._control = control
        self._callbacks = callbacks
        self._pid = os.getpid()
        self._thread_id = get_ident()
        self._active = True
        self._checkpointing = False
        self._interruption_cause: BaseException | None = None
        self._terminal_interrupted = False
        self._failure_code: str | None = None
        self._safe_point_error: BaseException | None = None

    @property
    def active(self) -> bool:
        """Return whether this context remains usable by its creating caller."""

        return (
            self._active and not self._terminal_interrupted
            and self._pid == os.getpid() and self._thread_id == get_ident()
        )

    @property
    def state_store(self):
        """Return the caller-selected state ``DirStore`` without changing it."""

        return self._state_store

    @property
    def control_store(self):
        """Return the caller-selected control ``DirStore`` without changing it."""

        return self._control_store

    @property
    def operation_id(self) -> str:
        """Return the stable identity of this managed member invocation."""

        return self._operation_id

    @property
    def attempt_id(self) -> str:
        """Return the current durable attempt identifier."""

        return self._attempt_id

    @property
    def is_resuming(self) -> bool:
        """Return whether this method entry followed exact checkpoint restoration."""

        return self._is_resuming

    @property
    def checkpoint_state_ref(self) -> StateRef | None:
        """Return the retained associated checkpoint, if this attempt has one."""

        return self._checkpoint_state_ref

    def checkpoint(self) -> StateRef:
        """Publish a complete checkpoint and service an already-pending request.

        Returns:
            The exact StateRef associated with this invocation's current attempt.

        Raises:
            ManagedContextError: If this context is inactive or checkpoint entry is
                recursive.
            ManagedInterrupted: If a request was durable before the post-callback
                safe-point decision.

        Side Effects:
            Saves the managed graph, commits its control association, invokes
            callbacks in configured order, and may commit terminal interruption.
        """

        return self._checkpoint(cause=None, force_interrupt=False)

    def interrupt(self, *, cause: BaseException | None = None) -> None:
        """Publish a checkpoint, notify callbacks, then terminate this invocation.

        Args:
            cause: Optional exact exception retained as the outer interruption's
                exception cause.

        Raises:
            ManagedContextError: If the context is inactive, recursive, or
                ``cause`` is not a BaseException.
            ManagedInterrupted: Always after the interruption transition commits.
        """

        self._require_active()
        if cause is not None and not isinstance(cause, BaseException):
            raise ManagedContextError("invalid_cause", "interruption cause must be a BaseException or None")
        self._checkpoint(cause=cause, force_interrupt=True)
        raise AssertionError("managed interruption safe point must not return")

    def _checkpoint(self, *, cause: BaseException | None, force_interrupt: bool) -> StateRef:
        """Perform one nonrecursive safe-point publication while ownership is live."""

        self._require_active()
        if self._checkpointing:
            raise ManagedContextError("recursive_checkpoint", "managed checkpoint entry is not recursive")
        if self._safe_point_error is not None:
            raise self._safe_point_error
        self._checkpointing = True
        try:
            try:
                self._ownership.require_owner()
                state_ref = self._state_repo.save_object(
                    self._obj, store=self._state_store, main=False, alias=None,
                    deep_capture=True, federated=False, reservation=self._ownership.reservation,
                )
                validate_state_ref = _validate_state_ref()
                validate_state_ref(self._state_store, state_ref)
                _checkpoint_boundary("state_published")
                self._control.transition_running_owner(
                    self._operation_id, attempt_id=self._attempt_id, owner_id=self._owner_id,
                    build=lambda current: replace(
                        current, generation=current.generation + 1,
                        checkpoint_digest=state_ref.digest(), final_digest=None, failure_code=None,
                    ),
                )
                self._checkpoint_state_ref = state_ref
                _checkpoint_boundary("checkpoint_associated")
            except BaseException as error:
                self._latch_safe_point_error("publication_error", error)
                raise
            try:
                _checkpoint_boundary("callbacks_started")
                for callback in self._callbacks:
                    callback(self._obj, self)
                _checkpoint_boundary("callbacks_finished")
            except BaseException as error:
                self._latch_safe_point_error("callback_error", error)
                raise
            try:
                interrupted = self._control.transition_running_owner(
                    self._operation_id, attempt_id=self._attempt_id, owner_id=self._owner_id,
                    build=lambda current: _interrupted_snapshot(current) if (
                        force_interrupt or current.interrupt_request is not None
                    ) else None,
                )
            except BaseException as error:
                if force_interrupt:
                    self._latch_safe_point_error("publication_error", error)
                    raise ManagedControlError(
                        "interruption_recording_failed",
                        "could not commit managed interruption",
                    ) from error
                self._latch_safe_point_error("publication_error", error)
                raise
            if interrupted is not None:
                self._interruption_cause = cause
                self._terminal_interrupted = True
                _checkpoint_boundary("interrupted_transition")
                error = ManagedInterrupted("interrupted", "managed operation stopped at a checkpoint")
                if cause is not None:
                    raise error from cause
                raise error
            _checkpoint_boundary("interruption_decided")
            return state_ref
        finally:
            self._checkpointing = False

    def _raise_if_interrupted(self) -> None:
        """Prevent method code from converting terminal safe-point outcomes into success."""

        if self._safe_point_error is not None:
            raise self._safe_point_error
        if not self._terminal_interrupted:
            return
        error = ManagedInterrupted("interrupted", "managed operation stopped at a checkpoint")
        if self._interruption_cause is not None:
            raise error from self._interruption_cause
        raise error

    def _latch_safe_point_error(self, code: str, error: BaseException) -> None:
        """Retain the first genuine safe-point failure for the runtime terminal guard."""

        if self._safe_point_error is None:
            self._safe_point_error = error
            self._failure_code = code

    def _require_active(self) -> None:
        """Reject stale, cross-thread, or inherited context use."""

        if not self.active:
            raise ManagedContextError("inactive_context", "managed context is inactive or belongs to another caller")

    def _deactivate(self) -> None:
        """Retire this private authority after runtime cleanup completes."""

        self._active = False


_CONTEXT_TOKEN = object()


def _create_context(**kwargs) -> ManagedContext:
    """Create the runtime-only context after a running snapshot is committed."""

    return ManagedContext(_CONTEXT_TOKEN, **kwargs)


def _validate_state_ref():
    """Load the storage validator lazily to keep context metadata import-light."""

    from .storage import validate_state_ref

    return validate_state_ref


def _interrupted_snapshot(current):
    """Build one terminal interruption replacement from freshly locked authority."""

    return replace(
        current, owner_id=None, generation=current.generation + 1,
        state="interrupted", interrupt_request=None, final_digest=None, failure_code=None,
    )


def _checkpoint_boundary(stage: str) -> None:
    """Provide a no-op in-process seam for deterministic crash-window tests."""


__all__ = ["ManagedContext"]
