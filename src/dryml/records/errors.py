"""Exceptions for DRYML record and spec sidecars."""

from __future__ import annotations


class RecordError(Exception):
    """Base class for DRYML record sidecar errors.

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


class RecordValidationError(RecordError):
    """Raised when a record envelope or record ID is malformed."""


class SpecValidationError(RecordError):
    """Raised when a spec envelope, family, or spec ID is malformed."""


class StorageRefError(RecordError):
    """Raised when a logical storage reference is malformed or unresolvable."""


class RecordIOError(RecordError):
    """Raised when record/spec sidecar IO cannot complete safely."""


class RecordNotFoundError(RecordIOError):
    """Raised when a requested record sidecar is absent."""


class SpecNotFoundError(RecordIOError):
    """Raised when a requested spec sidecar is absent."""


__all__ = [
    "RecordError",
    "RecordIOError",
    "RecordNotFoundError",
    "RecordValidationError",
    "SpecNotFoundError",
    "SpecValidationError",
    "StorageRefError",
]
