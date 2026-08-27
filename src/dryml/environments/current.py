"""Context-local opaque environment selector state."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_UNSET = object()
_CURRENT: ContextVar[Any] = ContextVar("dryml_current_environment", default=_UNSET)


def current(default: Any = None) -> Any:
    """Return this context's selector or ``default`` without probing it."""

    value = _CURRENT.get()
    return default if value is _UNSET else value


def set_current(spec: Any) -> Any:
    """Replace this context's opaque selector and return its prior value."""

    previous = current()
    _CURRENT.set(spec)
    return previous


def reset_current() -> None:
    """Clear only this context's selector without runtime side effects."""

    _CURRENT.set(_UNSET)


@contextmanager
def use(spec: Any) -> Iterator[Any]:
    """Scope an opaque selector and restore it after normal or failed exit."""

    token = _CURRENT.set(spec)
    try:
        yield spec
    finally:
        _CURRENT.reset(token)


__all__ = ["current", "reset_current", "set_current", "use"]
