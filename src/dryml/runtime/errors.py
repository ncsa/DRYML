"""Typed failures for the PID-bound runtime publication authority."""

from __future__ import annotations

import re
from typing import Any


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(password|passwd|secret|token|api[_-]?key|credential)\s*=\s*[^\s,]+"
)
_LOCAL_PATH = re.compile(r"(?<!\w)(?:/[^\s,]+|[A-Za-z]:[\\/][^\s,]+|\\\\[^\s,]+)")
_SECRET_KEY = re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key|credential)")


def _redact_diagnostic(value: Any, *, key: str | None = None) -> Any:
    """Return a bounded diagnostic projection without recognizable secrets."""

    if key is not None and _SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(name)[:512]: _redact_diagnostic(item, key=str(name))
            for name, item in tuple(value.items())[:64]
        }
    if isinstance(value, (list, tuple)):
        return tuple(_redact_diagnostic(item) for item in value[:64])
    if isinstance(value, str):
        text = _SECRET_ASSIGNMENT.sub(r"\1=<redacted>", value)
        text = _LOCAL_PATH.sub("<local-path>", text)
        return text[:512]
    return value


class RuntimeErrorBase(RuntimeError):
    """Base runtime failure with bounded machine-readable context.

    Args:
        message: Redacted human-readable failure explanation.
        context: Optional diagnostic details. Missing ``operation`` and
            ``category`` fields receive stable class-specific defaults.

    Side Effects:
        None. Construction does not alter runtime publication state.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        message = _redact_diagnostic(str(message))
        super().__init__(message)
        details = dict(context or {})
        defaults = {
            "DeviceVisibilityError": ("device_visibility", "unsupported"),
            "ForkSafetyError": ("runtime_fork_check", "restart-required"),
            "FrameworkImportSafetyError": ("framework_import", "restart-required"),
            "PublicationBusyError": ("runtime_publication", "contention"),
            "PublicationFailedError": ("runtime_publication", "terminal"),
            "PublicationReentryError": ("runtime_publication", "contention"),
            "PublicationError": ("runtime_publication", "publication"),
            "RuntimeSpecError": ("runtime_specification", "malformed"),
            "RuntimeTransitionError": ("runtime_transition", "incompatible"),
        }
        operation, category = defaults.get(type(self).__name__, ("runtime", "runtime"))
        details.setdefault("operation", operation)
        details.setdefault("category", category)
        self.context = _redact_diagnostic(details)


class RuntimeSpecError(RuntimeErrorBase):
    """Raised when a runtime declaration is malformed or unsupported."""


class RuntimeTransitionError(RuntimeErrorBase):
    """Raised when a runtime state violates allocation or mode invariants."""


class DeviceVisibilityError(RuntimeErrorBase):
    """Raised when a mandatory visibility plan cannot be constructed safely."""


class PublicationError(RuntimeTransitionError):
    """Raised when an immutable runtime generation cannot be published."""


class PublicationBusyError(PublicationError):
    """Raised when an active generation lease excludes a transition."""


class PublicationFailedError(PublicationError):
    """Raised after uncertain effects make the process require a restart."""


class PublicationReentryError(PublicationError):
    """Raised when code tries to upgrade a publication reader to a writer."""


class ForkSafetyError(PublicationError):
    """Raised when inherited activated runtime state is used after ``fork()``."""


class FrameworkImportSafetyError(PublicationError):
    """Raised when a watched framework cannot retain mandatory controls."""


__all__ = ["DeviceVisibilityError", "ForkSafetyError", "FrameworkImportSafetyError", "PublicationBusyError", "PublicationError", "PublicationFailedError", "PublicationReentryError", "RuntimeErrorBase", "RuntimeSpecError", "RuntimeTransitionError"]
