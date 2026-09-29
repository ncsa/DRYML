"""Public Dispatch failures that retain one safe preflight report."""

from __future__ import annotations

from .models import DispatchReport


class DispatchError(RuntimeError):
    """Report a trustworthy pre-submission Dispatch failure.

    Args:
        report: Immutable ineligible preflight observation.

    Raises:
        TypeError: If ``report`` is not a Dispatch report.

    Side Effects:
        None. This error does not wrap workload, backend acceptance,
        cancellation, timeout, or cleanup failures after acceptance.
    """

    def __init__(self, report: DispatchReport) -> None:
        """Retain the exact report without formatting caller values."""

        if type(report) is not DispatchReport:
            raise TypeError("dispatch error requires a DispatchReport")
        super().__init__(report.probe_reason)
        self.report = report


__all__ = ["DispatchError"]
