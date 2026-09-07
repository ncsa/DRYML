"""Invocation-scoped metadata exposed to managed method bodies.

Checkpoint publication and cooperative interruption are intentionally deferred to
U7.  This U6 context is only the private authority that supplies exact selected
Stores and attempt metadata while a synchronous invocation is active.
"""

from __future__ import annotations

import os
from threading import get_ident

from dryml.core.reference_values import StateRef

from .errors import ManagedContextError


class ManagedContext:
    """Read-only authority for one active managed invocation.

    Instances are created only by the managed runtime after state ownership and
    the running control snapshot are established.  They become inactive when the
    method exits and cannot transfer across a thread or process.
    """

    def __init__(self, token, *, state_store, control_store, operation_id: str,
                 attempt_id: str, is_resuming: bool,
                 checkpoint_state_ref: StateRef | None) -> None:
        """Initialize private runtime authority; callers receive no construction API."""

        if token is not _CONTEXT_TOKEN:
            raise ManagedContextError("context_construction", "managed contexts are runtime-created")
        self._state_store = state_store
        self._control_store = control_store
        self._operation_id = operation_id
        self._attempt_id = attempt_id
        self._is_resuming = is_resuming
        self._checkpoint_state_ref = checkpoint_state_ref
        self._pid = os.getpid()
        self._thread_id = get_ident()
        self._active = True

    @property
    def active(self) -> bool:
        """Return whether this context remains usable by its creating caller."""

        return self._active and self._pid == os.getpid() and self._thread_id == get_ident()

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
        """Reject checkpoint publication until U7 supplies its safe-point protocol.

        Raises:
            ManagedContextError: Always, because U6 deliberately does not expose a
                partial checkpoint implementation outside U7's callback and
                interruption ordering contract.
        """

        self._require_active()
        raise ManagedContextError("checkpoint_unavailable", "checkpoint publication is available with U7")

    def interrupt(self, *, cause: BaseException | None = None) -> None:
        """Reject interruption until U7 supplies its safe-point protocol.

        Args:
            cause: Reserved future explanatory interruption cause.

        Raises:
            ManagedContextError: Always, because U6 must not fabricate an
                interruption transition without an associated checkpoint.
        """

        self._require_active()
        raise ManagedContextError("interrupt_unavailable", "interruption is available with U7")

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


__all__ = ["ManagedContext"]
