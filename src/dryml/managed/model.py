"""Immutable metadata-only managed status and interruption result projections."""

from dataclasses import dataclass
from dryml.core.node import Node
from .control import ControlSnapshot


@dataclass(frozen=True, slots=True)
class ManagedStatus:
    """Selected authority projection, with no private-data availability promise.

    state/operation_id identify lifecycle and stable operation; run_id and
    invocation_id are distinct UUID hex or None before admission. generation is
    the authority fence. checkpoint/final are detached SavedNodes or None;
    interruption_requested is bool, failure_code optional static text, and
    producer_invocation_id identifies historical completion rather than a later
    replay caller. Construction returns an immutable value, performs no IO or
    retention, and imposes no additional validation beyond its owning reader.
    """
    state: str
    operation_id: str
    run_id: str | None
    invocation_id: str | None
    generation: int
    checkpoint: Node | None
    final: Node | None
    interruption_requested: bool
    failure_code: str | None
    producer_invocation_id: str | None


@dataclass(frozen=True, slots=True)
class InterruptRequestResult:
    """Immutable outcome of selected run/invocation/generation request admission.

    outcome is requested/already_requested/not_running/stale_run/stale_invocation/
    stale_generation/changed; operation_id is stable. run_id/invocation_id are the
    observed UUIDs or None, generation is the observed fence or zero. Construction
    performs no IO or validation; the control reader owns field admission. This
    value promises request acceptance only, not checkpointing or cancellation.
    """
    outcome: str
    operation_id: str
    run_id: str | None
    invocation_id: str | None
    generation: int


__all__ = ["ControlSnapshot", "InterruptRequestResult", "ManagedStatus"]
