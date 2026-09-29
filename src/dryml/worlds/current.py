"""Context-local requested-world defaults, independent from runtime allocation."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_UNSET = object()
_CURRENT: ContextVar[Any] = ContextVar("dryml_current_world", default=_UNSET)


def current(default: Any = None) -> Any:
    """Return the context-local requested world, or ``default`` while unset."""
    value = _CURRENT.get()
    return default if value is _UNSET else value


def set_current(world: Any) -> Any:
    """Set one context-local requested world and return its prior value."""
    previous = current()
    _CURRENT.set(world)
    return previous


def reset_current() -> None:
    """Clear only the current context's requested-world default."""
    _CURRENT.set(_UNSET)


@contextmanager
def use(world: Any) -> Iterator[Any]:
    """Scope one requested world and restore the prior context on all exits."""
    token = _CURRENT.set(world)
    try:
        yield world
    finally:
        _CURRENT.reset(token)
