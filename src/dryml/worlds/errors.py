"""Errors raised while declaring or planning local worlds."""

from typing import Any


class WorldError(ValueError):
    """Base error for malformed world declarations and local plans.

    Args:
        message: Human-readable failure explanation.
        context: Optional bounded machine-readable failure context.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = dict(context or {})


class ResourceValidationError(WorldError):
    """Raised when a resource constraint or concrete resource value is invalid."""


class WorldSpecValidationError(WorldError):
    """Raised when a world requirement, spec, or allocation is malformed."""


class WorldCompatibilityError(WorldError):
    """Raised when a caller requests an unavailable local compatibility action."""


class WorldRequirementError(WorldError):
    """Raised when passive hard world declarations cannot be resolved safely.

    This error reports malformed decorators, attached declaration metadata, and
    bounded diagnostic-combination failures without returning a partial result.
    Its optional ``context`` follows :class:`WorldError`.
    """
