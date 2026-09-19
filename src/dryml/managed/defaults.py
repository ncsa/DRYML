"""Private owner-bound worker defaults for managed invocation resolution."""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import get_ident
from typing import Iterator

from dryml.core.store.dir import DirStore


@dataclass(slots=True)
class _ControlStoreDefaultLease:
    """Retain one worker control Store default for its creating task and thread."""

    control_store: DirStore
    owner: tuple[int, int, int | None]
    active: bool = True


_control_store_default_lease: ContextVar[_ControlStoreDefaultLease | None] = ContextVar(
    "dryml_managed_control_store_default", default=None,
)


def _owner() -> tuple[int, int, int | None]:
    """Return the current process/thread/task identity without retaining a task."""

    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), get_ident(), None if task is None else id(task)


@contextmanager
def _control_store_defaults(control_store: DirStore) -> Iterator[DirStore]:
    """Install one private worker control default for nested managed invocations.

    Args:
        control_store: Exact worker-owned-or-borrowed direct control Store.

    Yields:
        The unchanged selected control Store.

    Raises:
        TypeError: If ``control_store`` is not an exact ``DirStore``.
        RuntimeError: If a different default is already active or a copied context
            attempts to reuse an owner-bound default.

    Side Effects:
        Installs an invocation-local ContextVar lease and marks it inactive on exit.
        It does not open, write, close, or otherwise own the Store.
    """

    if type(control_store) is not DirStore:
        raise TypeError("managed worker control default requires an exact DirStore")
    existing = _control_store_default_lease.get()
    if existing is not None:
        if not existing.active:
            raise RuntimeError("managed worker control default is inactive")
        if existing.owner != _owner():
            raise RuntimeError("managed worker control default belongs to a different thread or task")
        if existing.control_store is not control_store:
            raise RuntimeError("a different managed worker control default is already active")
        yield control_store
        return
    lease = _ControlStoreDefaultLease(control_store, _owner())
    token = _control_store_default_lease.set(lease)
    try:
        yield control_store
    finally:
        lease.active = False
        _control_store_default_lease.reset(token)


def _current_control_store_default() -> DirStore | None:
    """Return the active worker control default after owner validation.

    Returns:
        The exact active direct control Store, or ``None`` outside worker setup.

    Raises:
        RuntimeError: If a copied or inactive context tries to retain a default
            after its owner has exited.

    Side Effects:
        None. This lookup neither opens nor changes Store authority.
    """

    lease = _control_store_default_lease.get()
    if lease is None:
        return None
    if not lease.active:
        raise RuntimeError("managed worker control default is inactive")
    if lease.owner != _owner():
        raise RuntimeError("managed worker control default belongs to a different thread or task")
    return lease.control_store
