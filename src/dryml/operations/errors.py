"""Exceptions for dependency-light DRYML operation specifications."""

from __future__ import annotations


class OperationError(Exception):
    """Base class for operation-spec and operation-resolution errors.

    Parameters
    ----------
    message:
        Human-readable error message.
    context:
        Optional structured details useful to callers and tests.
    """

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = dict(context or {})


class OperationSpecError(OperationError):
    """Raised when an operation spec envelope or payload is malformed."""


class OperationResolutionError(OperationError):
    """Raised when operation call arguments cannot be resolved safely."""


__all__ = ["OperationError", "OperationResolutionError", "OperationSpecError"]
