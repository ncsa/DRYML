"""Generic exceptions for DRYML canonical formats."""

from __future__ import annotations


class DrymlFormatError(Exception):
    """Base class for canonical format, ID, envelope, and reference errors.

    Parameters
    ----------
    message:
        Human-readable error message.
    context:
        Optional structured details useful to tests, logs, or user interfaces.
    """

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = dict(context or {})


class CanonicalJSONError(DrymlFormatError):
    """Raised when data cannot be converted to canonical JSON."""


class ContentIDError(DrymlFormatError):
    """Raised when a content ID or content-ID component is invalid."""


class EnvelopeError(DrymlFormatError):
    """Raised when a generic DRYML envelope is malformed."""


class ReferenceParseError(DrymlFormatError):
    """Raised when a reserved reference string or literal escape is invalid."""


__all__ = [
    "CanonicalJSONError",
    "ContentIDError",
    "DrymlFormatError",
    "EnvelopeError",
    "ReferenceParseError",
]
