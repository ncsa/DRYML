"""Immutable public value projections for managed lifecycle APIs."""

from __future__ import annotations

from dataclasses import dataclass

from dryml.core.reference_values import StateRef

from .control import ControlSnapshot


@dataclass(frozen=True, slots=True)
class ManagedStatus:
    """Immutable projection of one selected managed operation authority.

    Args:
        state: One lifecycle state: ``not_started``, ``running``, ``interrupted``,
            ``failed``, or ``completed``.
        operation_id: Stable operation digest.
        attempt_id: Current attempt identifier, or ``None`` before a start.
        generation: Monotonic authority generation.
        checkpoint_state_ref: Last associated checkpoint, if any.
        final_state_ref: Completed state, if any.
        interruption_requested: Whether current authority records a request.
        failure_code: Static failure code, if current authority is failed.

    Side Effects:
        None. U6 reads selected lifecycle authority into this value.
    """

    state: str
    operation_id: str
    attempt_id: str | None
    generation: int
    checkpoint_state_ref: StateRef | None
    final_state_ref: StateRef | None
    interruption_requested: bool
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class InterruptRequestResult:
    """Immutable result of a cooperative interruption request.

    Args:
        outcome: One of ``requested``, ``already_requested``, ``not_running``, or
            ``stale_attempt``.
        operation_id: Stable operation digest addressed by the request.
        attempt_id: Attempt observed while processing the request, if any.
        generation: Committed authority generation observed by the request.

    Side Effects:
        None. U6 publishes and reports the selected request outcome.
    """

    outcome: str
    operation_id: str
    attempt_id: str | None
    generation: int


__all__ = ["ControlSnapshot", "InterruptRequestResult", "ManagedStatus"]
