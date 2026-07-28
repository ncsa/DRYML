"""Errors raised while normalizing effect-free public session values."""

from __future__ import annotations

from typing import Any


class SessionConfigurationError(ValueError):
    """Raised when a session configuration candidate is malformed or unsafe."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = dict(context or {})


__all__ = ["SessionConfigurationError"]
