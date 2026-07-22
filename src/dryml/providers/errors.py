"""Exception types for provider/probe metadata and protocols."""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for structured provider/probe exceptions."""

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = dict(context or {})


class ProviderValidationError(ProviderError):
    """Raised when provider metadata or JSON payloads are malformed."""


class ProviderRegistryError(ProviderError):
    """Raised for duplicate, missing, or unloadable provider registry entries."""


class ProviderProbeError(ProviderError):
    """Raised when a probe subprocess cannot complete successfully."""


class ProviderProtocolError(ProviderError):
    """Raised when worker JSON protocol envelopes are malformed."""


class ProviderReportError(ProviderError):
    """Raised when provider reports or probe-report records are invalid."""


class ProviderCacheError(ProviderError):
    """Raised for invalid probe cache keys or store-backed cache access."""


__all__ = [
    "ProviderCacheError",
    "ProviderError",
    "ProviderProbeError",
    "ProviderProtocolError",
    "ProviderRegistryError",
    "ProviderReportError",
    "ProviderValidationError",
]
