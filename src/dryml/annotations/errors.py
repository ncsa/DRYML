"""Typed errors for DRYML planning annotations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class AnnotationError(Exception):
    """Base error for sidecar requirement/default annotations."""

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = dict(context or {})


class AnnotationValidationError(AnnotationError):
    """Raised when annotation model data or payloads are malformed."""


class AnnotationMergeError(AnnotationError):
    """Raised when annotation fragments cannot be merged."""


class AnnotationConflictError(AnnotationMergeError):
    """Raised for hard requirement/default override conflicts."""


class AnnotationResolutionError(AnnotationError):
    """Raised when strict resolution cannot produce a valid plan."""


__all__ = [
    "AnnotationConflictError",
    "AnnotationError",
    "AnnotationMergeError",
    "AnnotationResolutionError",
    "AnnotationValidationError",
]
