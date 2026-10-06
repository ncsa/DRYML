"""Bounded error types for the additive Execute contract."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .future import ExecutionFuture
    from .models import AdmissionReport


def _format_raw_traceback(error: BaseException, limit_bytes: int) -> str | None:
    """Bound unredacted traceback text without replacing the original failure."""

    if limit_bytes <= 0:
        return None
    try:
        parts = bytearray()
        for fragment in traceback.TracebackException.from_exception(
                error, limit=64, capture_locals=False).format():
            chunk = fragment.encode("utf-8", errors="replace")
            remaining = limit_bytes - len(parts)
            parts.extend(chunk[:remaining])
            if len(chunk) >= remaining:
                break
        return parts.decode("utf-8", errors="ignore") or None
    except Exception:
        return None


class ExecutionError(RuntimeError):
    """Report a bounded Execute failure with optional admission evidence.

    Args:
        message: A safe, bounded explanation of the failure.
        report: Optional environment/world admission evidence retained for callers.

    Side Effects:
        None. The report is retained by reference because its domain owner defines
        its immutable value semantics.
    """

    def __init__(self, message: str, *, report: "AdmissionReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


class RemoteExecutionError(ExecutionError):
    """Report a worker failure without reconstructing its exception class.

    Args:
        message: Safe summary of the worker failure.
        remote_type: Bounded worker exception type name.
        remote_traceback: Optional bounded raw worker traceback text, enabled by
            default in backend configuration. It is not redacted.

    Side Effects:
        Retains diagnostic strings only; it does not import worker exception code.
    """

    def __init__(self, message: str, *, remote_type: str, remote_traceback: str | None = None) -> None:
        super().__init__(message)
        self.remote_type = remote_type
        self.remote_traceback = remote_traceback

    def __str__(self) -> str:
        """Display the summary, plus raw worker text if enabled and available.

        Returns:
            The worker type summary and any available unredacted
            traceback. Raw text may contain paths, exception values, or secrets.
        """

        summary = super().__str__()
        if self.remote_traceback is None:
            return summary
        return f"{summary}\nRemote traceback (raw, unredacted):\n{self.remote_traceback}"


class BackendUnavailableError(ExecutionError):
    """Report that a selected backend cannot be initialized or used."""


class AdmissionError(ExecutionError):
    """Report a rejected environment, world, or backend admission request."""


class ExecutionDeadlineExceeded(ExecutionError):
    """Report confirmed termination after an execution deadline expires."""


class ExecutionUncertainError(ExecutionError):
    """Report an outcome that cannot safely be classified as successful."""


class CleanupError(ExecutionError):
    """Report incomplete cleanup while retaining a live recovery reference.

    Args:
        message: Safe explanation of incomplete cleanup.
        report: Optional admission evidence for the affected execution.
        execution: Optional live coordinator Future that can reconcile cleanup.
    Side Effects:
        Retains an optional Future recovery handle without serializing or
        rendering it. Internal cleanup ownership remains with the coordinator.
    """

    def __init__(self, message: str, *, report: "AdmissionReport | None" = None, execution: "ExecutionFuture[object] | None" = None) -> None:
        super().__init__(message, report=report)
        self.execution = execution


__all__ = [
    "AdmissionError",
    "BackendUnavailableError",
    "CleanupError",
    "ExecutionDeadlineExceeded",
    "ExecutionError",
    "ExecutionUncertainError",
    "RemoteExecutionError",
]
