"""Explicit, effect-free admission barrier for domain compatibility reports."""

from __future__ import annotations

from typing import Protocol

from .errors import RequirementBarrierError, RequirementError


class AdmissionReport(Protocol):
    """Structural domain report that exposes a fail-closed admission decision.

    Properties:
        admission_ok: An exact ``bool`` independent of any domain-specific,
            policy-dependent reporting property such as ``ok``.
    """

    @property
    def admission_ok(self) -> bool:
        """Return whether explicit hard-requirement admission is allowed."""


def require_admission(report: AdmissionReport, *, operation: str | None = None) -> None:
    """Require an exact policy-independent domain admission decision.

    Args:
        report: A structural domain report exposing exact-boolean
            ``admission_ok``.
        operation: Optional bounded label describing the requested operation.

    Raises:
        RequirementError: If the report or operation shape is malformed.
        RequirementBarrierError: If admission is explicitly false. The raised
            error retains the identical report object.

    Side Effects:
        None. This function does not invoke protected work, bind targets, or
        mutate runtime, session, or global state.
    """

    if operation is not None and (
        type(operation) is not str
        or len(operation) > 512
        or any(ord(char) < 32 or ord(char) == 127 for char in operation)
    ):
        raise RequirementError("invalid admission operation")
    try:
        admitted = report.admission_ok
    except Exception:
        raise RequirementError("invalid admission report") from None
    if type(admitted) is not bool:
        raise RequirementError("invalid admission report")
    if not admitted:
        raise RequirementBarrierError("requirement admission denied", report=report, operation=operation)


__all__ = ["AdmissionReport", "require_admission"]
