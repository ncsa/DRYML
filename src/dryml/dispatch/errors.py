"""Exceptions for dispatch metadata specs."""

from __future__ import annotations


class DispatchSpecError(Exception):
    """Raised when dispatch intent or recipe metadata is malformed."""

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = dict(context or {})


__all__ = ["DispatchSpecError"]
