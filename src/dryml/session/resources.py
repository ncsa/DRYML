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
    selected session resources and cache-owned Store reconstruction handles while
    active, but inspection never grants callers permission to close them.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._repos: list[Repo] = []
        self._stores: list[Store] = []
        self._store_keys: dict[tuple[object, ...], Store] = {}
        self._owned_stores: list[Store] = []
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

            for store in stores:
                self._admit_store_key(store)

    def _admit_store_key(self, store: Store) -> None:
        """Register usable physical/opening evidence without exporting a Store."""

        self._revalidate_store_keys()

    def _revalidate_store_keys(self) -> None:
        """Rebuild Store lookup keys from each handle's current valid evidence.

        A Store retains its lexical lifetime after an external replacement, but
        cannot remain eligible for future lookups. A ZipStore updates its own
        evidence after a successful atomic commit, so it remains the live
        transaction for its newly published archive.
        """

        from dryml.core.repo_definition import _store_cache_key_from_store

        self._store_keys.clear()
        for store in self._stores:
            key = _store_cache_key_from_store(store)
            if key is not None:
                self._store_keys.setdefault(key, store)

    @staticmethod
    def _attach_cleanup_owner(primary: BaseException, store: Store, error: BaseException) -> None:
        """Attach the one retry owner for a Store that could not be closed."""

        from dryml.core.repo_definition import RepoReconstructionError

        cleanup = RepoReconstructionError(
            "Session resource-cache cleanup requires retry.",
            _retained_stores=(store,), cleanup_issues=(type(error).__name__,),
        )
        primary.repo_cleanup_error = cleanup

    def _close_provisional(self, store: Store, primary: BaseException) -> None:
        """Close one unpublished Store without replacing its primary failure."""

        try:
            store.close()
        except BaseException as error:
            self._attach_cleanup_owner(primary, store, error)

    def _acquire_store(self, key: tuple[object, ...], opener):
        """Return a matching Store or open and retain one cache-owned handle.

        Args:
            key: Validated physical identity and opening-settings tuple.
            opener: Zero-argument existing-authority constructor invoked on a miss.

        Returns:
            The matching cached Store or one newly opened cache-owned Store.

        Raises:
            RuntimeError: If the cache is inactive or an opened Store cannot be
                retained safely.
            Exception: Any existing-authority validation or opening error from
                ``opener``. Failed opens never publish a cache entry.

        Side Effects:
            A successful miss retains the returned Store until outer cache exit.
            Cache teardown closes such owned Stores with ``flush=False`` behavior
            where the backend supports buffered state; it never commits buffers.
        """

        self._assert_owner()
        with self._lock:
            if not self._active:
                raise RuntimeError("Cannot acquire a resource after Session resource cache teardown.")
            self._revalidate_store_keys()
            existing = self._store_keys.get(key)
            if existing is not None:
                return existing
            store = opener()
            try:
                from dryml.core.repo_definition import RepoDefinitionError, _store_cache_key_from_store

                if _store_cache_key_from_store(store) != key:
                    raise RepoDefinitionError("Store authority changed while opening.")
                _core_session._lease_resource(self, store)
                self._stores.append(store)
                self._owned_stores.append(store)
                self._revalidate_store_keys()
                if self._store_keys.get(key) is not store:
                    raise RepoDefinitionError("Store authority changed while opening.")
            except BaseException as primary:
                if self._contains(self._stores, store):
                    self._stores.remove(store)
                    self._owned_stores.remove(store)
                    _core_session._release_resource(self, store)
                self._close_provisional(store, primary)
                raise
            return store

    def _retains_store(self, store: Store) -> bool:
        """Return whether this active cache retains the exact Store handle."""

        self._assert_owner()
        with self._lock:
            return self._contains(self._stores, store)

    def _teardown(self) -> None:
        """Release borrowed memberships and discard cache-owned Store handles."""

        with self._lock:
            if not self._active:
                return
            self._active = False
            resources = (*self._repos, *self._stores)
            owned_stores = tuple(self._owned_stores)
            self._repos.clear()
            self._stores.clear()
            self._store_keys.clear()
            self._owned_stores.clear()
        for resource in reversed(resources):
            _core_session._release_resource(self, resource)
        failed_stores = []
        issues = []
        cleanup_control_flow = None
        for store in reversed(owned_stores):
            try:
                store.close()
            except BaseException as error:
                failed_stores.append(store)
                issues.append(type(error).__name__)
                if cleanup_control_flow is None and not isinstance(error, Exception):
                    cleanup_control_flow = error
        if failed_stores:
            from dryml.core.repo_definition import RepoReconstructionError

            cleanup = RepoReconstructionError(
                "Session resource-cache cleanup requires retry.",
                _retained_stores=failed_stores, cleanup_issues=issues,
            )
            if cleanup_control_flow is not None:
                cleanup_control_flow.repo_cleanup_error = cleanup
                raise cleanup_control_flow
            raise cleanup


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
        teardown and remain owned by their original caller. Stores opened through
        participating reconstruction are cache-owned and closed without commit on
        teardown, so dirty ZipStore buffers are discarded rather than published.
    """

    existing = _core_session._current_resource_cache()
    cache = ResourceCache() if existing is None else existing
    outermost = existing is None
    primary = None
    try:
        with _core_session._resource_cache_scope(cache):
            if outermost:
                repo = _core_session.current_repo()
                if repo is not None:
                    cache._admit_borrowed_repo(repo)
            yield cache
    except BaseException as error:
        primary = error
        raise
    finally:
        if outermost:
            try:
                cache._teardown()
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.repo_cleanup_error = getattr(cleanup, "repo_cleanup_error", cleanup)


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
