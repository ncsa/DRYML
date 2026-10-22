"""Immutable metadata-only managed status and interruption result projections."""

from dataclasses import dataclass
from dryml.core.node import Node
from .control import ControlSnapshot
from .results import ResultReference, ResultSource


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
    __dryml_execute_adapter__ = "dryml.managed.execute_adapter:ManagedExecuteAdapter"
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


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    """Detached operation/run/current-caller/producer UUID metadata; no IO.

    Strings are admitted by control/admission before construction. Replay preserves
    run/producer and allocates invocation_id for its current caller.
    """
    __dryml_execute_adapter__ = "dryml.managed.execute_adapter:ManagedExecuteAdapter"
    operation_id: str
    run_id: str
    invocation_id: str
    producer_invocation_id: str | None


@dataclass(frozen=True, slots=True)
class InvocationResult:
    """Detached successful metadata with a freshly reopened lazy ordinary value.

    status is ManagedStatus, execution ExecutionIdentity, final_state_ref SavedNode,
    result_reference ResultReference, source ResultSource, cleanup_report tuple.
    Construction/inspection performs no IO and retains no Repo/Store/reader/lease.
    value freshly validates authority and bytes on every access, raising
    ManagedResultError for missing/corrupt/replaced sources without reexecution.
    References retain metadata closure only, never private StateKey data grants.
    cleanup_report entries are (Store domain, full StateKey, StateRemovalOutcome)
    for physical garbage failures only. Retry the outcome's exact incarnation with
    Store.retry_state_removal; grant/association/index failures never enter this
    successful report. Historical final/status/value inspection survives state cleanup.
    """
    __dryml_execute_adapter__ = "dryml.managed.execute_adapter:ManagedExecuteAdapter"

    status: ManagedStatus
    execution: ExecutionIdentity
    final_state_ref: Node
    result_reference: ResultReference
    source: ResultSource
    cleanup_report: tuple = ()

    @property
    def value(self):
        """Load the supported value under its own fresh validated source lifetime."""
        return self.source.read(self.result_reference)


@dataclass(frozen=True, slots=True)
class FailedInvocationResult:
    """Detached failed status/execution and optional recovery; no success value.

    status/execution are owner-produced metadata; checkpoint_state_ref may be None.
    Construction has no IO and makes no availability promise. Attached to raised
    authored/finalization errors as failed_result for inspection after Repo closure.
    """
    __dryml_execute_adapter__ = "dryml.managed.execute_adapter:ManagedExecuteAdapter"
    status: ManagedStatus
    execution: ExecutionIdentity
    checkpoint_state_ref: Node | None


__all__ = ["ControlSnapshot", "InterruptRequestResult", "ManagedStatus", "ExecutionIdentity",
           "InvocationResult", "FailedInvocationResult"]
