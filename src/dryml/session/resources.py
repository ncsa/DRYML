"""Session-owned process-local resource-cache activation and inspection."""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import TYPE_CHECKING, Iterator

from dryml.core import session as _core_session

if TYPE_CHECKING:
    from dryml.core.repo import Repo
    from dryml.core.store.store import Store


class ResourceCache:
    """Inspectable strong-lifetime registry for active local Repo and Store handles.

    Instances are created only by :func:`resource_cache`. They retain borrowed
    selected session resources while active, but inspection never grants callers
    permission to close them. Future participating reconstruction paths use the
    same private admission hooks to register cache-owned handles.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._repos: list[Repo] = []
        self._stores: list[Store] = []
        self._active = True

    def _assert_owner(self) -> None:
        """Reject inspection or admission through a copied active context."""

        if self._active:
            active = _core_session._current_resource_cache()
            if active is not self:
                raise RuntimeError("This Session resource cache is not active in the current owner context.")

    @property
    def repos(self) -> tuple[Repo, ...]:
        """Return a point-in-time tuple of registered live Repo instances.

        Returns:
            Actual local Repo handles retained by this cache, or an empty tuple
            after final teardown.

        Raises:
            RuntimeError: If a copied task/thread attempts inspection while this
                cache remains active for its creating owner.

        Side Effects:
            None. This property performs no reconstruction, Store scan, or cache
            mutation. The returned tuple confers no close ownership.
        """

        self._assert_owner()
        with self._lock:
            return tuple(self._repos)

    @property
    def stores(self) -> tuple[Store, ...]:
        """Return a point-in-time tuple of registered live Store instances.

        Returns:
            Actual local Store handles retained by this cache, or an empty tuple
            after final teardown.

        Raises:
            RuntimeError: If a copied task/thread attempts inspection while this
                cache remains active for its creating owner.

        Side Effects:
            None. This property performs no reconstruction, Store scan, or cache
            mutation. The returned tuple confers no close ownership.
        """

        self._assert_owner()
        with self._lock:
            return tuple(self._stores)

    @staticmethod
    def _contains(resources, candidate: object) -> bool:
        """Test handle identity without using resource equality semantics."""

        return any(existing is candidate for existing in resources)

    def _admit_borrowed_repo(self, repo: Repo) -> None:
        """Retain a selected Repo and its already-connected Store handles strongly.

        This private hook accepts only a live locally selected Repo. It neither
        opens Stores nor treats the handles as cache-owned; its transaction keeps
        membership and raw-close guards aligned if admission fails.
        """

        self._assert_owner()
        stores = tuple(repo.stores)
        with self._lock:
            if not self._active:
                raise RuntimeError("Cannot admit a resource after Session resource cache teardown.")
            additions = []
            if not self._contains(self._repos, repo):
                additions.append((self._repos, repo))
            additions.extend(
                (self._stores, store) for store in stores
                if not self._contains(self._stores, store)
            )
            retained = []
            try:
                for resources, resource in additions:
                    _core_session._lease_resource(self, resource)
                    resources.append(resource)
                    retained.append(resource)
            except BaseException:
                for resource in reversed(retained):
                    _core_session._release_resource(self, resource)
                    for resources in (self._repos, self._stores):
                        index = next(
                            (index for index, existing in enumerate(resources)
                             if existing is resource),
                            None,
                        )
                        if index is not None:
                            resources.pop(index)
                            break
                raise

    def _teardown(self) -> None:
        """Release borrowed memberships and close guards after final activation exit."""

        with self._lock:
            if not self._active:
                return
            self._active = False
            resources = (*self._repos, *self._stores)
            self._repos.clear()
            self._stores.clear()
        for resource in reversed(resources):
            _core_session._release_resource(self, resource)


@contextmanager
def resource_cache() -> Iterator[ResourceCache]:
    """Activate Session-owned local resource caching for the current owner context.

    Yields:
        The active :class:`ResourceCache` for read-only membership inspection.

    Raises:
        RuntimeError: If a copied thread/task uses an inherited activation, an
            incompatible cache is active, or a registered resource cannot be
            retained safely.

    Side Effects:
        Creates one owner-bound cache when disabled, registers the current core
        session Repo and connected Stores as borrowed entries, and clears its
        membership on final exit. Nested use by the same thread/task yields the
        same object. Activation opens no resource and does not change ``current_repo``.

    Lifetime:
        Registered resources are strongly retained and raw ``close()`` calls are
        refused until the outermost activation exits. Borrowed resources survive
        teardown and remain owned by their original caller.
    """

    existing = _core_session._current_resource_cache()
    cache = ResourceCache() if existing is None else existing
    outermost = existing is None
    try:
        with _core_session._resource_cache_scope(cache):
            if outermost:
                repo = _core_session.current_repo()
                if repo is not None:
                    cache._admit_borrowed_repo(repo)
            yield cache
    finally:
        if outermost:
            cache._teardown()


def current_resource_cache() -> ResourceCache | None:
    """Return the active owner-valid Session resource cache without creating one.

    Returns:
        The active :class:`ResourceCache`, or ``None`` when caching is disabled.

    Raises:
        RuntimeError: If a copied thread/task attempts to use an inherited cache
            activation or the activation has become inactive.

    Side Effects:
        None. This function neither opens resources nor changes session selection.
    """

    return _core_session._current_resource_cache()


__all__ = ["ResourceCache", "current_resource_cache", "resource_cache"]
