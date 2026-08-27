"""Public errors for invalid persistent session declarations."""

from __future__ import annotations

from functools import wraps
from typing import Any

from dryml.runtime.errors import _redact_diagnostic


class SessionConfigurationError(ValueError):
    """Report a malformed, incompatible, or unpublished session candidate.

    Args:
        message: Human-readable failure explanation.
        context: Optional bounded machine-readable diagnostic details.

    Side Effects:
        None. The exception never changes the published runtime generation.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        message = _redact_diagnostic(str(message))
        super().__init__(message)
        details = dict(context or {})
        details.setdefault("operation", "session_configuration")
        details.setdefault("category", "malformed")
        self.context = _redact_diagnostic(details)


def session_operation(name: str):
    """Decorate a public session operation with stable failure context."""

    def decorate(func):
        @wraps(func)
        def call(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except SessionConfigurationError as exc:
                exc.context["operation"] = name
                raise

        return call

    return decorate


__all__ = ["SessionConfigurationError"]
