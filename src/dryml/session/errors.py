"""Public errors for invalid persistent session declarations."""

from __future__ import annotations

from typing import Any


class SessionConfigurationError(ValueError):
    """Report a malformed, incompatible, or unpublished session candidate.

    Args:
        message: Human-readable failure explanation.
        context: Optional bounded machine-readable diagnostic details.

    Side Effects:
        None. The exception never changes the published runtime generation.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = dict(context or {})


__all__ = ["SessionConfigurationError"]
