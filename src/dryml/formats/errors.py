"""Typed failures for the closed v1.1 metadata codec."""


class DrymlFormatError(ValueError):
    """Base failure for malformed canonical metadata.

    Parameters are a human-readable message and bounded machine-readable
    context.  The codec never returns a partial value after this error.
    """

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = dict(context or {})


class CanonicalJSONError(DrymlFormatError):
    """Raised when a value cannot be represented as bounded canonical JSON."""


class EnvelopeError(DrymlFormatError):
    """Raised when a v1.1 envelope is missing, malformed, or mismatched."""


class ContentIDError(DrymlFormatError):
    """Raised when a v1.1 semantic ID is malformed or does not validate."""


__all__ = ["CanonicalJSONError", "ContentIDError", "DrymlFormatError", "EnvelopeError"]
