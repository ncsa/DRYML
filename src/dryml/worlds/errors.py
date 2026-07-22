"""Structured errors for world resource, topology, and allocation specs."""

from __future__ import annotations

from typing import Any


class WorldError(ValueError):
    """Base error for invalid world/runtime resource topology data."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = dict(context or {})


class ResourceValidationError(WorldError):
    """Raised when a resource spec or requirement is malformed."""


class WorldSpecValidationError(WorldError):
    """Raised when a world requirement, spec, or allocation is malformed."""


class WorldCompatibilityError(WorldError):
    """Raised when compatibility checking cannot be performed."""


__all__ = [
    "ResourceValidationError",
    "WorldCompatibilityError",
    "WorldError",
    "WorldSpecValidationError",
]
