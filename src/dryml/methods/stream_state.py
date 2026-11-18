"""Dependency-light execution and explicit state-transfer contracts.

These structural protocols describe local kernel capabilities, not Dataset
policy, a persistence format, or a serialization API. Native executors may
report one signal for an entire native batch. Ordinary iterator pulls need not
construct signals, event wrappers, or exported state at internal edges.
"""

from enum import IntEnum
from collections.abc import Iterator
from typing import Any, Protocol


class PullKind(IntEnum):
    """Distinguish delivery, a pending section handoff, and root completion.

    DELIVERY exposes one authored output (possibly a native batch); SECTION_END
    leaves execution alive until an explicit handoff; ROOT_END is terminal.
    Values are process-local execution facts, not versioned wire-format tags.
    """

    DELIVERY = 0
    SECTION_END = 1
    ROOT_END = 2


class SourceBinding(Protocol):
    """Inert, domain-supplied acquisition over frozen source configuration.

    Domain adapters define binding ownership and availability checks. Retaining a
    binding does not acquire a producer, snapshot payload bytes, or imply recovery.
    """

    def open(self, epoch: int) -> Iterator[Any]:
        """Create an independent owned iterator for a nonnegative logical epoch.

        Acquisition may remain lazy until its first read. Source access errors
        propagate to the owning execution's lifecycle boundary.
        """
        ...


class StreamExecution(Protocol):
    """Structural, out-of-band pull interface for an owned local execution.

    Implementations own their lifecycle and report errors rather than successful
    completion on failure. This protocol does not require state export support.
    """

    @property
    def value(self) -> Any:
        """Return the last delivered value, valid only after DELIVERY.

        The value is borrowed from this execution; ownership and native placement
        are unchanged. Reading before delivery may raise RuntimeError.
        """
        ...

    def pull(self) -> PullKind:
        """Advance once and return a delivery/boundary/root fact without encoding.

        SECTION_END must remain pending until handoff. Acquisition, execution,
        and cleanup failures propagate; they must not become ROOT_END.
        """
        ...

    def close(self) -> None:
        """Release owned resources once; explicit close does not prove root EOF.

        Cleanup failures propagate subject to the executor's primary-error policy.
        """
        ...


class BoundaryHandoff(Protocol):
    """Optional section-transition capability, separate from physical EOF."""

    def handoff(self) -> None:
        """Accept a pending section boundary and permit subsequent execution.

        Raises RuntimeError without a pending boundary or after closure. Policy
        owners decide when to call this operation; it may acquire the next pass
        only after the preceding section's postlude has had its opportunity.
        """
        ...


class StateExport(Protocol):
    """Optional capture-on-request capability; ordinary pulls never require it."""

    def capture_state(self) -> object:
        """Return detached execution state at a caller-established coherent cut.

        Implementations freeze required mutable buffers only on this request and
        propagate unsupported-state or extraction errors. The domain owner defines
        validation, versioning, and persistence; this operation performs no IO.
        """
        ...


class StateRestore(Protocol):
    """Optional explicit restoration capability for validated local state."""

    def restore_state(self, state: object) -> None:
        """Install domain-validated state before advancing execution.

        Args:
            state: Detached state in this executor's supported domain schema.

        Raises:
            TypeError, ValueError: For unsupported or incompatible state.
            RuntimeError: If the executor cannot restore at its current lifecycle.

        No producer acquisition is implied; domain adapters specify any lazy
        source restoration and must propagate restoration failure explicitly.
        """
        ...


__all__ = ["BoundaryHandoff", "PullKind", "SourceBinding", "StateExport", "StateRestore", "StreamExecution"]
