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


class RecordPolicyError(RecordError):
    """Raised when a save/export record policy is invalid or unsafe."""


class RecordClosureError(RecordError):
    """Raised when record/spec closure planning cannot complete."""


class RecordExportError(RecordError):
    """Raised when record/spec/product export cannot complete."""


__all__ = [
    "RecordClosureError",
    "RecordError",
    "RecordExportError",
    "RecordIOError",
    "RecordNotFoundError",
    "RecordPolicyError",
    "RecordValidationError",
    "SpecNotFoundError",
    "SpecValidationError",
    "StorageRefError",
]
