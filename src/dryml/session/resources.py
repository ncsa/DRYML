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
    selected session resources plus cache-owned reconstructed Repo and Store
    handles while active, but inspection never grants callers permission to
    close them.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._repos: list[Repo] = []
        self._stores: list[Store] = []
        self._repo_records: list[dict[str, object]] = []
        self._repo_keys: dict[tuple[object, ...], Repo] = {}
        self._store_keys: dict[tuple[object, ...], Store] = {}
        self._owned_repos: list[Repo] = []
        self._owned_stores: list[Store] = []
        self._owned_repo_dependencies: dict[int, tuple[Store, ...]] = {}
        self._pending_repo_keys: set[tuple[object, ...]] = set()
        self._pending_store_keys: set[tuple[object, ...]] = set()
        self._staged_stores: list[tuple[tuple[object, ...], Store]] | None = None
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
        from dryml.core.repo_definition import _BOUNDS
        from dryml.formats.canonical import canonical_json_bytes

        # Validate all evidence before leasing or publishing a selected handle.
        # A failed selection transition must leave its previous cache/session state intact.
        try:
            canonical_json_bytes(repo.config, **_BOUNDS)
        except (RecursionError, TypeError, ValueError, OverflowError, UnicodeError):
            raise RuntimeError(
                "Selected Repo is unsupported or closed for Session resource caching."
            ) from None
        if repo._closing or repo._closed or any(
                getattr(store, "_closed_handle", False) for store in stores):
            raise RuntimeError("Selected Repo is unsupported or closed for Session resource caching.")
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

            self._revalidate_store_keys()
            self._register_repo(repo, owned=False)

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

    def _revalidate_repo_keys(self) -> None:
        """Reconcile every live Repo's current semantic snapshot under one lock.

        Repos remain deliberately mutable while cached. A changed record is
        rekeyed before either a hit or miss; a changed record that collides with
        an already canonical entry stays retained but is quarantined from future
        matching rather than replacing that entry.
        """

        from dryml.core.repo_definition import _repo_cache_key_from_repo

        previous_keys = dict(self._repo_keys)
        snapshots = []
        try:
            for record in self._repo_records:
                snapshots.append((record, _repo_cache_key_from_repo(record["repo"])))
        except BaseException:
            # A failed live lookup must not leave a stale semantic hit table.
            self._repo_keys.clear()
            raise

        self._repo_keys.clear()
        candidates: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for record, current in snapshots:
            previous = record["key"]
            if current != previous:
                record["quarantined"] = False
            record["key"] = current
            if current is not None and not record["quarantined"]:
                candidates.setdefault(current, []).append(record)

        for key, records in candidates.items():
            canonical = next(
                (
                    record for record in records
                    if previous_keys.get(record["key"]) is record["repo"]
                ),
                records[0],
            )
            self._repo_keys[key] = canonical["repo"]
            for record in records:
                if record is not canonical:
                    record["quarantined"] = True

    def _register_repo(
            self, repo: Repo, *, owned: bool,
            dependencies: tuple[Store, ...] = ()) -> None:
        """Publish one fully assembled Repo after all dependent Stores are ready."""

        if not self._contains(self._repos, repo):
            self._repos.append(repo)
        if not any(record["repo"] is repo for record in self._repo_records):
            self._repo_records.append({"repo": repo, "key": None, "quarantined": False})
        if owned:
            if not self._contains(self._owned_repos, repo):
                self._owned_repos.append(repo)
            self._owned_repo_dependencies[id(repo)] = dependencies
        self._revalidate_repo_keys()

    @staticmethod
    def _attach_cleanup_owner(primary: BaseException, store: Store, error: BaseException) -> None:
        """Attach the one retry owner for a Store that could not be closed."""

        from dryml.core.repo_definition import RepoReconstructionError

        cleanup = RepoReconstructionError(
            "Session resource-cache cleanup requires retry.",
            _retained_stores=(store,), cleanup_issues=(type(error).__name__,),
        )
        ResourceCache._attach_cleanup(primary, cleanup)

    @staticmethod
    def _attach_cleanup(primary: BaseException, cleanup) -> None:
        """Preserve an existing retry owner while adding cache cleanup work."""

        from dryml.core.repo_definition import RepoReconstructionError

        existing = getattr(primary, "repo_cleanup_error", None)
        if existing is None:
            primary.repo_cleanup_error = cleanup
        elif isinstance(existing, RepoReconstructionError):
            existing._retain_cleanup(
                repos=cleanup._retained_repos,
                stores=cleanup._retained_stores,
                issues=cleanup.cleanup_issues,
            )
        else:
            primary.repo_cleanup_errors = (*getattr(primary, "repo_cleanup_errors", (existing,)), cleanup)

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
            if self._staged_stores is not None:
                staged = next(
                    (store for store_key, store in self._staged_stores if store_key == key),
                    None,
                )
                if staged is not None:
                    return staged
            if key in self._pending_store_keys:
                raise RuntimeError("Recursive reconstruction of the same cached Store is unsupported.")
            self._pending_store_keys.add(key)
            store = None
            try:
                store = opener()
                from dryml.core.repo_definition import RepoDefinitionError, _store_cache_key_from_store

                if _store_cache_key_from_store(store) != key:
                    raise RepoDefinitionError("Store authority changed while opening.")
                if self._staged_stores is not None:
                    self._staged_stores.append((key, store))
                    return store
                _core_session._lease_resource(self, store)
                self._stores.append(store)
                self._owned_stores.append(store)
                self._revalidate_store_keys()
                if self._store_keys.get(key) is not store:
                    raise RepoDefinitionError("Store authority changed while opening.")
            except BaseException as primary:
                if store is not None and self._contains(self._stores, store):
                    self._stores.remove(store)
                    self._owned_stores.remove(store)
                    _core_session._release_resource(self, store)
                if store is not None:
                    self._close_provisional(store, primary)
                raise
            finally:
                self._pending_store_keys.remove(key)
            return store

    def _acquire_repo(self, key: tuple[object, ...], builder):
        """Return a semantic Repo hit or atomically publish one staged miss.

        Args:
            key: Validated current Repo configuration and physical Store evidence.
            builder: Zero-argument constructor which requests dependent Stores
                through this cache and returns a complete borrowed-Store Repo.

        Returns:
            The matching cached Repo or a newly constructed cache-owned Repo.

        Raises:
            RuntimeError: If the cache is inactive or the request recursively
                attempts the same semantic Repo key.
            Exception: Any validation, open, or construction failure. Only Stores
                newly staged for this request are closed on failure.
        """

        self._assert_owner()
        with self._lock:
            if not self._active:
                raise RuntimeError("Cannot acquire a resource after Session resource cache teardown.")
            self._revalidate_store_keys()
            self._revalidate_repo_keys()
            existing = self._repo_keys.get(key)
            if existing is not None:
                return existing
            if key in self._pending_repo_keys:
                raise RuntimeError("Recursive reconstruction of the same cached Repo is unsupported.")
            if self._staged_stores is not None:
                raise RuntimeError("Nested Repo reconstruction is unsupported while another Repo is staging.")
            self._pending_repo_keys.add(key)
            self._staged_stores = []
            repo = None
            try:
                repo = builder()
                from dryml.core.repo_definition import RepoDefinitionError, _repo_cache_key_from_repo, _store_cache_key_from_store

                if _repo_cache_key_from_repo(repo) != key:
                    raise RepoDefinitionError("Repo configuration changed while opening.")
                staged = tuple(self._staged_stores)
                for store_key, store in staged:
                    if _store_cache_key_from_store(store) != store_key:
                        raise RepoDefinitionError("Store authority changed while opening.")
                for _, store in staged:
                    _core_session._lease_resource(self, store)
                    self._stores.append(store)
                    self._owned_stores.append(store)
                _core_session._lease_resource(self, repo)
                self._register_repo(
                    repo,
                    owned=True,
                    dependencies=tuple(
                        store for store in repo.stores
                        if self._contains(self._owned_stores, store)
                    ),
                )
                self._revalidate_store_keys()
                self._revalidate_repo_keys()
                if self._repo_keys.get(key) is not repo:
                    raise RepoDefinitionError("Repo configuration changed while opening.")
                return repo
            except BaseException as primary:
                if repo is not None and self._contains(self._repos, repo):
                    self._repos.remove(repo)
                    self._repo_records[:] = [
                        record for record in self._repo_records if record["repo"] is not repo
                    ]
                    self._owned_repos[:] = [item for item in self._owned_repos if item is not repo]
                    self._owned_repo_dependencies.pop(id(repo), None)
                    if self._repo_keys.get(key) is repo:
                        self._repo_keys.pop(key, None)
                    _core_session._release_resource(self, repo)
                staged = tuple(self._staged_stores or ())
                from dryml.core.repo_definition import RepoReconstructionError

                # Detach staged dependencies before closing their Repo. If that
                # close fails, the retry error becomes their only owner.
                for _, store in staged:
                    if self._contains(self._stores, store):
                        self._stores.remove(store)
                        self._owned_stores.remove(store)
                        _core_session._release_resource(self, store)
                cleanup = primary if isinstance(primary, RepoReconstructionError) else getattr(
                    primary, "repo_cleanup_error", None,
                )
                if isinstance(cleanup, RepoReconstructionError):
                    cleanup._retain_cleanup(stores=tuple(store for _, store in staged))
                elif repo is not None:
                    repo._closing = True
                    try:
                        repo.close(flush=False)
                    except BaseException as cleanup_error:
                        repo._closing = True
                        cleanup = RepoReconstructionError(
                            "Session resource-cache cleanup requires retry.",
                            _retained_repos=(repo,),
                            _retained_stores=tuple(store for _, store in staged),
                            cleanup_issues=(type(cleanup_error).__name__,),
                        )
                        self._attach_cleanup(primary, cleanup)
                    else:
                        cleanup = None
                if not isinstance(cleanup, RepoReconstructionError):
                    for _, store in reversed(staged):
                        self._close_provisional(store, primary)
                self._revalidate_store_keys()
                raise
            finally:
                self._staged_stores = None
                self._pending_repo_keys.remove(key)

    def _teardown(self) -> None:
        """Release memberships, then close owned Repos before owned Stores."""

        with self._lock:
            if not self._active:
                return
            self._active = False
            resources = (*self._repos, *self._stores)
            owned_repos = tuple(self._owned_repos)
            owned_stores = tuple(self._owned_stores)
            self._repos.clear()
            self._stores.clear()
            self._repo_records.clear()
            self._repo_keys.clear()
            self._store_keys.clear()
            self._owned_repos.clear()
            self._owned_stores.clear()
            dependencies = self._owned_repo_dependencies
            self._owned_repo_dependencies = {}
        for resource in reversed(resources):
            _core_session._release_resource(self, resource)
        failed_repos = []
        deferred_stores = []
        failed_stores = []
        issues = []
        cleanup_control_flow = None
        for repo in reversed(owned_repos):
            try:
                repo.close(flush=False)
            except BaseException as error:
                repo._closing = True
                failed_repos.append(repo)
                for store in dependencies.get(id(repo), ()):
                    if not self._contains(deferred_stores, store):
                        deferred_stores.append(store)
                issues.append(type(error).__name__)
                if cleanup_control_flow is None and not isinstance(error, Exception):
                    cleanup_control_flow = error
        for store in reversed(owned_stores):
            if self._contains(deferred_stores, store):
                continue
            try:
                store.close()
            except BaseException as error:
                failed_stores.append(store)
                issues.append(type(error).__name__)
                if cleanup_control_flow is None and not isinstance(error, Exception):
                    cleanup_control_flow = error
        if failed_repos or failed_stores:
            from dryml.core.repo_definition import RepoReconstructionError

            cleanup = RepoReconstructionError(
                "Session resource-cache cleanup requires retry.",
                _retained_repos=failed_repos,
                _retained_stores=(*deferred_stores, *failed_stores),
                cleanup_issues=issues,
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
        teardown and remain owned by their original caller. Repos and Stores
        opened through participating reconstruction are cache-owned; teardown
        closes Repos with ``flush=False`` before Stores, so dirty ZipStore buffers
        are discarded rather than published.
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
                ResourceCache._attach_cleanup(
                    primary, getattr(cleanup, "repo_cleanup_error", cleanup),
                )


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
