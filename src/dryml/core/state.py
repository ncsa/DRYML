"""Process-local ownership for live Object state transitions.

The reservation registry is intentionally independent of Store identity. Store
locks protect durable authority, while this module prevents two cooperating
threads in one process from mutating the same retained live graph at once.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import os
from threading import RLock, get_ident
from typing import Iterable


_REGISTRY_LOCK = RLock()
_OBJECT_OWNERS: dict[object, "StateGraphReservation"] = {}
_LIVE_OWNERS: dict[int, "StateGraphReservation"] = {}
_AUTHORITY = object()


def _clear_inherited_reservations() -> None:
    """Drop parent-process tokens after fork so child state IO cannot deadlock."""

    global _REGISTRY_LOCK, _OBJECT_OWNERS, _LIVE_OWNERS
    # A parent thread can hold the inherited RLock at fork. Replacing the whole
    # process-local registry is the only child-safe reset; acquiring it can hang.
    _REGISTRY_LOCK = RLock()
    _OBJECT_OWNERS = {}
    _LIVE_OWNERS = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_clear_inherited_reservations)


class StateGraphReservation(AbstractContextManager):
    """One active process/thread lease over an exact live state graph.

    Instances are created only by :meth:`Repo.reserve_state_graph`. The lease
    covers the root's immutable ObjectRef, every stateful ObjectId, and the
    exact live nodes observed during preflight. ``release()`` is idempotent for
    the owning process and thread; inherited or foreign callers cannot release
    a token.

    Attributes:
        object_ref: Exact immutable identity of the reserved root graph.
        object_ids: Deterministically ordered owned state identities.
        active: Whether the owning process/thread still holds the lease.
    """

    def __init__(self, authority, object_ref, nodes: Iterable[object], object_ids: Iterable[object]):
        if authority is not _AUTHORITY:
            raise TypeError("StateGraphReservation instances are created by Repo.reserve_state_graph().")
        self.object_ref = object_ref
        self.object_ids = tuple(sorted(set(object_ids), key=lambda value: value.__stable_leaf_bytes__()))
        self._nodes = tuple(nodes)
        self._node_ids = frozenset(id(node) for node in self._nodes)
        self._object_ids = frozenset(self.object_ids)
        self._pid = os.getpid()
        self._thread_id = get_ident()
        self._active = False

    @property
    def active(self) -> bool:
        """Return whether this token remains owned by its creating thread."""

        return self._active and self._pid == os.getpid() and self._thread_id == get_ident()

    @property
    def _held(self) -> bool:
        """Return whether this process still owns the lease, from any thread."""

        return self._active and self._pid == os.getpid()

    def __enter__(self):
        """Return this already-acquired reservation for context-manager use."""

        self._require_owner()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Release this token without suppressing an enclosing exception."""

        self.release()
        return False

    def release(self) -> bool:
        """Release this token when called by its owning process and thread.

        Returns:
            ``True`` only when this invocation removed an active lease.

        Side Effects:
            Removes the exact ObjectId and live-identity ownership entries. A
            repeated owning release is a harmless no-op.
        """

        if not self.active:
            return False
        with _REGISTRY_LOCK:
            if not self.active:
                return False
            for object_id in self.object_ids:
                if _OBJECT_OWNERS.get(object_id) is self:
                    del _OBJECT_OWNERS[object_id]
            for node in self._nodes:
                if _LIVE_OWNERS.get(id(node)) is self:
                    del _LIVE_OWNERS[id(node)]
            self._active = False
            return True

    def _activate(self) -> None:
        with _REGISTRY_LOCK:
            conflicts = [
                object_id for object_id in self.object_ids
                if object_id in _OBJECT_OWNERS
            ]
            conflicts.extend(
                node for node in self._nodes if id(node) in _LIVE_OWNERS
            )
            if conflicts:
                from .repo import RepoSaveError

                raise RepoSaveError("Live state graph is already reserved by another operation.")
            for object_id in self.object_ids:
                _OBJECT_OWNERS[object_id] = self
            for node in self._nodes:
                _LIVE_OWNERS[id(node)] = self
            self._active = True

    def _require_owner(self) -> None:
        if not self.active:
            from .repo import RepoSaveError

            raise RepoSaveError("State graph reservation is inactive or belongs to another process/thread.")

    def _covers(self, nodes: Iterable[object], object_ids: Iterable[object]) -> None:
        self._require_owner()
        if not set(object_ids).issubset(self._object_ids) or not {
                id(node) for node in nodes}.issubset(self._node_ids):
            from .repo import RepoSaveError

            raise RepoSaveError("State graph reservation does not cover this exact live graph.")


def reserve(object_ref, nodes: Iterable[object], object_ids: Iterable[object]) -> StateGraphReservation:
    """Acquire a nonblocking all-or-nothing reservation for preflighted nodes."""

    reservation = StateGraphReservation(_AUTHORITY, object_ref, nodes, object_ids)
    reservation._activate()
    return reservation


def is_reserved(node: object) -> bool:
    """Return whether a current-process graph token owns this live identity.

    The observing thread is intentionally irrelevant: a sibling may not reuse a
    node just because the token's public ``active`` property is owner-relative.
    """

    with _REGISTRY_LOCK:
        owner = _LIVE_OWNERS.get(id(node))
        if owner is None:
            owner = _OBJECT_OWNERS.get(getattr(node, "object_id", None))
        return owner is not None and owner._held


def reserve_node(node: object) -> StateGraphReservation:
    """Temporarily reserve one exact stateful live node during generic reuse."""

    return reserve(node.object_ref, (node,), (node.object_id,))
