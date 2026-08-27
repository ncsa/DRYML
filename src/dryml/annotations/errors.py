"""Exceptions raised by the direct, declaration-only annotation API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class AnnotationError(Exception):
    """Base error for annotation declaration, collection, and resolution.

    Args:
        message: Human-readable explanation of the failed annotation operation.
        context: Optional detached machine-readable error context.
    """

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = dict(context or {})


class AnnotationValidationError(AnnotationError):
    """Raised when a closed annotation value or policy is malformed."""


class UnsupportedAnnotationTargetError(AnnotationError):
    """Raised before metadata is attached to a non-extensible target."""


class AnnotationMergeError(AnnotationError):
    """Raised for a direct merge request that cannot form a valid declaration."""


__all__ = ["AnnotationError", "AnnotationMergeError", "AnnotationValidationError", "UnsupportedAnnotationTargetError"]
