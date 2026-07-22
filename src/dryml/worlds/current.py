"""Context-local current requested world defaults."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_UNSET = object()
_CURRENT_WORLD: ContextVar[Any] = ContextVar("dryml_current_world", default=_UNSET)


def current(default: Any = None) -> Any:
    """Return the context-local requested default world.

    Args:
        default: Value returned when no current world has been set.

    Returns:
        The requested world previously set with ``set_current``/``use``, or
        *default* when unset. This is a default for future dispatch/probe
        planning, not the active process allocation.
    """

    value = _CURRENT_WORLD.get()
    return default if value is _UNSET else value


def set_current(world: Any) -> Any:
    """Set the context-local requested default world.

    Args:
        world: Requested world spec or selector for future planning.

    Returns:
        The previous current value, or ``None`` when it was unset.

    Side Effects:
        Updates only a context-local future-planning default. It does not create
        a ``WorldAllocation`` or assign resources to the current process.
    """

    previous = current(default=None)
    _CURRENT_WORLD.set(world)
    return previous


def reset_current() -> None:
    """Clear the context-local requested default world without deallocating work."""

    _CURRENT_WORLD.set(_UNSET)


@contextmanager
def use(world: Any) -> Iterator[Any]:
    """Temporarily set the context-local requested default world.

    Args:
        world: Requested world spec or selector scoped to the context body.

    Yields:
        The provided *world*.

    Side Effects:
        Restores the prior context-local default on normal or exceptional exit;
        no runtime allocation is created or changed.
    """

    token = _CURRENT_WORLD.set(world)
    try:
        yield world
    finally:
        _CURRENT_WORLD.reset(token)


def discover_current(*, default: Any = None) -> Any:
    """Return the context-local requested world or a caller default.

    Explicit context-local state has priority. When unset, this function returns
    *default*. It intentionally does not synthesize worlds, parse ``DRYML_WORLD``
    environment variables, or convert active runtime allocation into a requested
    world. Synthesis and allocation remain explicit operations.

    Args:
        default: Value returned when no explicit current world has been set.

    Returns:
        The explicit current requested world, or *default*.
    """

    return current(default=default)


__all__ = ["current", "discover_current", "reset_current", "set_current", "use"]
