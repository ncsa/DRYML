"""Typed failures for the PID-bound runtime publication authority."""

from __future__ import annotations

from typing import Any


class RuntimeErrorBase(RuntimeError):
    """Base runtime failure with bounded machine-readable context."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = dict(context or {})


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


__all__ = ["DeviceVisibilityError", "ForkSafetyError", "PublicationBusyError", "PublicationError", "PublicationFailedError", "PublicationReentryError", "RuntimeErrorBase", "RuntimeSpecError", "RuntimeTransitionError"]
