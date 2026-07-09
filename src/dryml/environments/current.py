"""Context-local current environment defaults."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_UNSET = object()
_CURRENT_ENVIRONMENT: ContextVar[Any] = ContextVar("dryml_current_environment", default=_UNSET)


def current(default: Any = None) -> Any:
    """Return the context-local current environment default.

    Args:
        default: Value returned when no current environment has been set.

    Returns:
        The environment spec previously set with ``set_current``/``use``, or
        *default* when unset. This function does not probe or validate the
        process environment.
    """

    value = _CURRENT_ENVIRONMENT.get()
    return default if value is _UNSET else value


def set_current(spec: Any) -> Any:
    """Set the context-local current environment default.

    Args:
        spec: Environment spec or user-provided selector for future planning.

    Returns:
        The previous current value, or ``None`` when it was unset.
    """

    previous = current(default=None)
    _CURRENT_ENVIRONMENT.set(spec)
    return previous


def reset_current() -> None:
    """Clear the context-local current environment default."""

    _CURRENT_ENVIRONMENT.set(_UNSET)


@contextmanager
def use(spec: Any) -> Iterator[Any]:
    """Temporarily set the context-local current environment default.

    Args:
        spec: Environment spec or selector scoped to the context body.

    Yields:
        The provided *spec*.
    """

    token = _CURRENT_ENVIRONMENT.set(spec)
    try:
        yield spec
    finally:
        _CURRENT_ENVIRONMENT.reset(token)


__all__ = ["current", "reset_current", "set_current", "use"]
