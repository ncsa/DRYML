"""Runtime mode, allocation, visibility, and bootstrap errors."""

from __future__ import annotations

from typing import Any


class RuntimeErrorBase(RuntimeError):
    """Base runtime error carrying machine-readable context."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = dict(context or {})


class RuntimeSpecError(RuntimeErrorBase):
    """Raised when a runtime context spec is malformed."""


class RuntimeTransitionError(RuntimeErrorBase):
    """Raised for invalid runtime state transitions."""


class NoAllocationError(RuntimeErrorBase):
    """Raised when workload resources are required but no allocation is active."""


class DeviceVisibilityError(RuntimeErrorBase):
    """Raised when device visibility cannot be built or applied safely."""


class FrameworkImportSafetyError(RuntimeErrorBase):
    """Raised when a framework import conflicts with the desired runtime setup."""


class PublicationError(RuntimeTransitionError):
    """Raised when a process-global runtime publication cannot complete safely."""


class PublicationBusyError(PublicationError):
    """Raised when import activity prevents a non-waiting transition."""


class PublicationFailedError(PublicationError):
    """Raised when a prior publication left the process requiring restart."""


class PublicationReentryError(PublicationError):
    """Raised when a transition owner re-enters publication APIs."""


__all__ = [
    "DeviceVisibilityError",
    "FrameworkImportSafetyError",
    "NoAllocationError",
    "PublicationBusyError",
    "PublicationError",
    "PublicationFailedError",
    "PublicationReentryError",
    "RuntimeErrorBase",
    "RuntimeSpecError",
    "RuntimeTransitionError",
]
