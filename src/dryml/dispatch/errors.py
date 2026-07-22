"""Exceptions for dispatch planning, launch, and worker protocol handling."""

from __future__ import annotations


class DispatchSpecError(Exception):
    """Raised when dispatch intent or recipe metadata is malformed."""

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = dict(context or {})


class DispatchPlanningError(Exception):
    """Raised when a dispatch request cannot be planned for a backend."""

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = dict(context or {})


class DispatchLaunchError(Exception):
    """Raised when a worker subprocess cannot be launched."""

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = dict(context or {})


class WorkerProtocolError(Exception):
    """Raised when the worker protocol is missing or malformed."""

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = dict(context or {})


class WorkerHandshakeError(WorkerProtocolError):
    """Raised when the worker handshake is unsupported or failed."""


class DispatchCancelled(Exception):
    """Raised when a local subprocess dispatch is cancelled."""


class DispatchTimeout(TimeoutError):
    """Raised when a local subprocess dispatch exceeds its timeout."""


__all__ = [
    "DispatchCancelled",
    "DispatchLaunchError",
    "DispatchPlanningError",
    "DispatchSpecError",
    "DispatchTimeout",
    "WorkerHandshakeError",
    "WorkerProtocolError",
]
