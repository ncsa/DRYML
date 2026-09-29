"""Exceptions raised by the passive annotation kernel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class AnnotationError(Exception):
    """Base error for annotation attachment and collection.

    Args:
        message: Human-readable explanation of the failed annotation operation.
        context: Optional detached machine-readable error context.
    """

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = dict(context or {})


class AnnotationValidationError(AnnotationError):
    """Raised when an annotation key, carrier, or attachment tuple is malformed."""


class UnsupportedAnnotationTargetError(AnnotationError):
    """Raised before static attachment or inspection of an unsafe target."""


__all__ = ["AnnotationError", "AnnotationValidationError", "UnsupportedAnnotationTargetError"]
