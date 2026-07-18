"""Coordinator-owned callbacks and coalesced managed operation controls."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntEnum
from .errors import CallbackFailure, ManagedCapabilityError
from .events import OperationEvent


class ControlRequest(IntEnum):
    """Controls ordered by the precedence used for callback coalescing."""

    NONE = 0
    CHECKPOINT = 1
    GRACEFUL_STOP = 2
    INTERRUPT = 3
    CANCEL = 3
    FAIL = 4


@dataclass(frozen=True, slots=True)
class ManagedCallback:
    """Invocation-scoped callback policy excluded from persistent identity.

    Args:
        callback: Callable receiving :class:`OperationEvent`.
        controls: Controls this callback may return. Undeclared controls are
            callback failures rather than implicit capability escalation.
        fail_soft: Record bounded diagnostics and continue after callback errors.
    """

    callback: Callable[[OperationEvent], ControlRequest | None]
    controls: frozenset[ControlRequest] | set[ControlRequest] = frozenset()
    fail_soft: bool = False

    def __post_init__(self) -> None:
        if not callable(self.callback):
            raise TypeError("managed callback must be callable")
        controls = frozenset(self.controls)
        if any(not isinstance(item, ControlRequest) or item in {ControlRequest.NONE, ControlRequest.FAIL} for item in controls):
            raise ValueError("callback controls contain an unsupported request")
        if not isinstance(self.fail_soft, bool):
            raise TypeError("fail_soft must be a bool")
        object.__setattr__(self, "controls", controls)


def normalize_callbacks(callbacks: Iterable[ManagedCallback | Callable] | None) -> tuple[ManagedCallback, ...]:
    """Normalize raw strict observers and explicit callback policies."""

    if callbacks is None:
        return ()
    if isinstance(callbacks, (ManagedCallback,)) or callable(callbacks):
        callbacks = (callbacks,)
    result = []
    for callback in callbacks:
        result.append(callback if isinstance(callback, ManagedCallback) else ManagedCallback(callback))
    return tuple(result)


def preflight_callbacks(
    callbacks: Iterable[ManagedCallback | Callable] | None,
    *,
    resumable: bool,
    checkpoint_schema: str | None,
    early_completion: bool,
) -> tuple[ManagedCallback, ...]:
    """Reject unsupported callback guarantees before Store mutation."""

    normalized = normalize_callbacks(callbacks)
    checkpoint_capable = resumable and checkpoint_schema is not None
    if any(not callback.fail_soft for callback in normalized) and not checkpoint_capable:
        raise ManagedCapabilityError(
            "strict callback failure requires a resumable operation with a checkpoint schema"
        )
    controls = frozenset(control for callback in normalized for control in callback.controls)
    if ControlRequest.CHECKPOINT in controls and not checkpoint_capable:
        raise ManagedCapabilityError("callback checkpoint control requires compatible checkpoint capability")
    if ControlRequest.GRACEFUL_STOP in controls and not early_completion:
        raise ManagedCapabilityError("callback graceful stop requires declared early completion capability")
    return normalized


class CallbackCoordinator:
    """Execute callbacks in the coordinator and retain only compact control state."""

    def __init__(
        self,
        callbacks: Iterable[ManagedCallback | Callable] = (),
        *,
        max_diagnostics: int = 32,
    ):
        if type(max_diagnostics) is not int or max_diagnostics < 1 or max_diagnostics > 256:
            raise ValueError("max_diagnostics must be between 1 and 256")
        self.callbacks = normalize_callbacks(callbacks)
        self.max_diagnostics = max_diagnostics
        self._control = ControlRequest.NONE
        self._diagnostics: list[str] = []
        self._failure: str | None = None

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return bounded callback diagnostics."""

        return tuple(self._diagnostics)

    def publish(self, event: OperationEvent) -> None:
        """Run each callback once for an event and coalesce returned controls."""

        for callback in self.callbacks:
            if self._failure is not None and not callback.fail_soft:
                continue
            try:
                request = callback.callback(event)
                if request is None:
                    continue
                if not isinstance(request, ControlRequest):
                    raise TypeError("callback returned a non-ControlRequest value")
                if request not in callback.controls:
                    raise ValueError(f"callback returned undeclared control {request.name.lower()}")
                if request > self._control:
                    self._control = request
            except Exception as exc:
                diagnostic = f"callback {type(exc).__name__}: {str(exc)[:384]}"
                self._append_diagnostic(diagnostic)
                if not callback.fail_soft:
                    self._failure = diagnostic
                    self._control = ControlRequest.FAIL

    def poll(self) -> ControlRequest:
        """Return the current highest-precedence compact control request."""

        return self._control

    def consume_checkpoint(self) -> None:
        """Clear a one-shot checkpoint request without clearing sticky stops."""

        if self._control is ControlRequest.CHECKPOINT:
            self._control = ControlRequest.NONE

    def raise_failure(self) -> None:
        """Raise the retained strict callback failure, if any."""

        if self._failure is not None:
            raise CallbackFailure(self._failure)

    def _append_diagnostic(self, diagnostic: str) -> None:
        self._diagnostics.append(diagnostic)
        del self._diagnostics[:-self.max_diagnostics]


__all__ = [
    "CallbackCoordinator",
    "ControlRequest",
    "ManagedCallback",
    "normalize_callbacks",
    "preflight_callbacks",
]
