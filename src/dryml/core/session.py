from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from os import getpid
from threading import RLock, get_ident
from typing import Any, Literal


ObjectMode = Literal["fresh", "definition", "concrete", "selector", "space", "load_or_build"]
CacheMode = Literal["none", "weak", "strong"]

_UNSET = object()
_OBJECT_MODES = {"fresh", "definition", "concrete", "selector", "space", "load_or_build"}
_CACHE_MODES = {"none", "weak", "strong"}


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Context-local repository, construction-mode, and cache configuration."""

    repo: Any = None
    object_mode: ObjectMode = "fresh"
    cache: CacheMode = "weak"
    repo_owned: bool = False


_DEFAULT_CONFIG = SessionConfig()
_current_config: ContextVar[SessionConfig] = ContextVar(
    "dryml_session_config",
    default=_DEFAULT_CONFIG,
)
_internal_construction: ContextVar[bool] = ContextVar(
    "dryml_internal_construction",
    default=False,
)


@dataclass(slots=True)
class _ResourceCacheLease:
    """Private owner-bound reference to one Session resource-cache protocol."""

    cache: object
    owner: tuple[int, int, int | None]
    depth: int = 1
    active: bool = True


_active_resource_cache: ContextVar[_ResourceCacheLease | None] = ContextVar(
    "dryml_active_resource_cache", default=None,
)
_resource_lock = RLock()
_resource_leases: dict[int, tuple[object, object]] = {}


def _resource_owner() -> tuple[int, int, int | None]:
    """Return the current thread/task identity without retaining a task object."""

    try:
        import asyncio

        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return getpid(), get_ident(), None if task is None else id(task)


def _current_resource_cache() -> object | None:
    """Return the active owner-valid cache protocol without constructing one.

    Raises:
        RuntimeError: If a copied thread/task context attempts to use an active
            cache after the creating owner has exited.
    """

    lease = _active_resource_cache.get()
    if lease is None:
        return None
    if not lease.active:
        raise RuntimeError("The Session resource cache is inactive.")
    if lease.owner != _resource_owner():
        raise RuntimeError(
            "The Session resource cache may only be used by its creating thread and task owner."
        )
    return lease.cache


@contextmanager
def _resource_cache_scope(cache: object):
    """Install one private Session cache protocol for its creating owner.

    Args:
        cache: Opaque public-session cache object implementing ownership policy.

    Yields:
        The supplied cache protocol.

    Raises:
        RuntimeError: If nesting occurs from a copied or inactive context.

    Side Effects:
        Installs and resets a task/thread-owned ContextVar lease. The lower seam
        does not import or construct the public Session facade.
    """

    existing = _active_resource_cache.get()
    if existing is not None:
        active = _current_resource_cache()
        if active is not cache:
            raise RuntimeError("A different Session resource cache is already active.")
        existing.depth += 1
        try:
            yield cache
        finally:
            existing.depth -= 1
        return

    lease = _ResourceCacheLease(cache, _resource_owner())
    token = _active_resource_cache.set(lease)
    try:
        yield cache
    finally:
        lease.active = False
        _active_resource_cache.reset(token)


def _lease_resource(cache: object, resource: object) -> None:
    """Record one strong cache lifetime lease for a raw Repo or Store handle."""

    with _resource_lock:
        existing = _resource_leases.get(id(resource))
        if existing is not None and existing[0] is resource and existing[1] is not cache:
            raise RuntimeError("Resource is already leased by another Session resource cache.")
        _resource_leases[id(resource)] = (resource, cache)


def _release_resource(cache: object, resource: object) -> None:
    """Release one exact cache lifetime lease after its owner has detached it."""

    with _resource_lock:
        existing = _resource_leases.get(id(resource))
        if existing is not None and existing[0] is resource and existing[1] is cache:
            _resource_leases.pop(id(resource), None)


def _assert_resource_close_allowed(resource: object) -> None:
    """Reject raw closure while a Session cache promises a resource lifetime."""

    with _resource_lock:
        existing = _resource_leases.get(id(resource))
        if existing is not None and existing[0] is resource:
            raise RuntimeError("Cannot close a resource while the Session resource cache retains it.")


def _preflight_session_transition(
        old: "SessionConfig", new: "SessionConfig", *, temporary: bool = False) -> None:
    """Reject a selection change before it can close a cache-leased Repo.

    Args:
        old: Currently selected immutable core configuration.
        new: Fully validated replacement configuration.
        temporary: Whether ``new`` must be closed when a scoped configuration exits.

    Raises:
        RuntimeError: If the transition would close a cache-leased Repo or a
            temporary owned Repo could not safely restore its outer selection.
    """

    if temporary and new.repo is not old.repo and new.repo_owned:
        if _current_resource_cache() is not None:
            raise RuntimeError(
                "A temporary owned repository cannot safely restore while the Session resource cache is active."
            )
    if old.repo is not new.repo and old.repo_owned and old.repo is not None:
        _assert_resource_close_allowed(old.repo)


def _preflight_repo_input(
        old: "SessionConfig", repo: object, *, temporary: bool = False) -> None:
    """Reject an unsafe Repo replacement before coercion can open a Store.

    Args:
        old: Current immutable core configuration.
        repo: Raw ``configure``/``config`` Repo input.
        temporary: Whether the input will require scoped owned-Repo cleanup.

    Raises:
        RuntimeError: If replacing an owned leased Repo would close it, or if a
            temporary non-Repo input would require unsafe cleanup. Rejection occurs
            before Store coercion can allocate a new backend handle.
    """

    if repo is _UNSET or repo is old.repo:
        return
    if temporary and repo is not None:
        from .repo import Repo

        if not isinstance(repo, Repo) and _current_resource_cache() is not None:
            raise RuntimeError(
                "A temporary owned repository cannot safely restore while the Session resource cache is active."
            )
    if old.repo_owned and old.repo is not None:
        _assert_resource_close_allowed(old.repo)


def _resource_cache_selected_repo(repo: object) -> None:
    """Offer a changed core selection to the active cache without facade imports."""

    if repo is None:
        return
    cache = _current_resource_cache()
    if cache is not None:
        cache._admit_borrowed_repo(repo)


def get_config() -> SessionConfig:
    """Return the immutable current context configuration without side effects."""

    return _current_config.get()


def current_repo():
    """Return the configured repository, or ``None`` when no repo is selected."""

    return get_config().repo


def current_object_mode() -> ObjectMode:
    """Return the effective object mode after the public orchestrator floor.

    The floor changes only the projected mode.  It deliberately leaves the
    context-local repository, cache, and raw configured mode untouched so they
    restore exactly after orchestration ends.
    """

    from dryml.runtime.context import active_runtime
    from dryml.runtime.modes import RuntimeMode

    cfg = get_config()
    if (
            active_runtime().mode is RuntimeMode.ORCHESTRATOR
            and cfg.object_mode in {"fresh", "load_or_build"}):
        return "definition"
    return cfg.object_mode


def _construction_object_mode() -> ObjectMode:
    """Return the private mode used only by an admitted constructor chain."""

    if _internal_construction.get():
        from dryml.runtime.guards import internal_construction_admitted

        if not internal_construction_admitted():
            return current_object_mode()
        return get_config().object_mode
    return current_object_mode()


def current_cache() -> CacheMode:
    """Return the configured runtime-object cache policy."""

    return get_config().cache


def _validate_object_mode(value: str) -> ObjectMode:
    if value not in _OBJECT_MODES:
        raise ValueError(
            "object_mode must be one of "
            f"{sorted(_OBJECT_MODES)}, got {value!r}."
        )
    return value


def _validated_object_mode(value: str, *, internal_construction: bool = False) -> ObjectMode:
    """Reject public materializing mode selection while orchestrating."""

    mode = _validate_object_mode(value)
    if mode in {"fresh", "load_or_build"}:
        from dryml.runtime.context import active_runtime
        from dryml.runtime.errors import RuntimeTransitionError
        from dryml.runtime.guards import internal_construction_admitted
        from dryml.runtime.modes import RuntimeMode

        if (
                active_runtime().mode is RuntimeMode.ORCHESTRATOR
                and (not internal_construction or not internal_construction_admitted())):
            raise RuntimeTransitionError(
                "orchestration object-mode floor prohibits public fresh/load_or_build selection",
                context={
                    "mode": "orchestrator",
                    "object_mode": mode,
                    "fix": "use definition/concrete/selector/space modes, a fresh managed process, or a future explicit dispatch",
                },
            )
    return mode


def _validate_cache(value: str) -> CacheMode:
    if value not in _CACHE_MODES:
        raise ValueError(f"cache must be one of {sorted(_CACHE_MODES)}, got {value!r}.")
    return value


def _coerce_repo(value):
    if value is None:
        return None, False

    from .repo import Repo

    if isinstance(value, Repo):
        return value, False
    return Repo(stores=value), True


def _close_owned_repo(cfg: SessionConfig) -> None:
    if cfg.repo_owned and cfg.repo is not None:
        cfg.repo.close(flush=True)


def _merged_config(
        base: SessionConfig,
        *,
        repo=_UNSET,
        object_mode=_UNSET,
        cache=_UNSET,
        internal_construction: bool = False) -> SessionConfig:
    updates = {}

    if repo is not _UNSET:
        repo_obj, repo_owned = _coerce_repo(repo)
        updates["repo"] = repo_obj
        updates["repo_owned"] = repo_owned

    if object_mode is not _UNSET:
        updates["object_mode"] = _validated_object_mode(
            object_mode, internal_construction=internal_construction
        )

    if cache is not _UNSET:
        updates["cache"] = _validate_cache(cache)

    return replace(base, **updates)


def configure(*, repo=_UNSET, object_mode=_UNSET, cache=_UNSET) -> SessionConfig:
    """Persist validated core configuration in the current context.

    Args:
        repo: Repo, Store input, ``None``, or omitted current value.
        object_mode: Closed object-mode value or omitted current value.
        cache: Closed cache policy or omitted current value.

    Returns:
        The new immutable configuration.

    Raises:
        ValueError: If a mode or cache value is invalid.
        RuntimeTransitionError: If orchestration prohibits a materializing mode.
        RuntimeError: If replacing an owned Repo would close a handle retained by
            an active Session resource cache.

    Side Effects:
        Replaces context-local state, closes a replaced owned repository, and
        registers a changed selected Repo as a borrowed entry when the optional
        Session resource cache is active.
    """

    old = get_config()
    _preflight_repo_input(old, repo)
    new = _merged_config(old, repo=repo, object_mode=object_mode, cache=cache)
    if repo is not _UNSET and old.repo is not new.repo:
        _preflight_session_transition(old, new)
        _resource_cache_selected_repo(new.repo)
        _close_owned_repo(old)
    _current_config.set(new)
    return new


@contextmanager
def config(*, repo=_UNSET, object_mode=_UNSET, cache=_UNSET):
    """Scope and exactly restore validated core configuration.

    Args and failures match :func:`configure`.

    Raises:
        RuntimeError: If temporary owned Repo cleanup could not safely coexist
            with an active Session resource cache.

    Yields:
        The temporary immutable configuration.

    Side Effects:
        Sets context-local state and closes a temporary owned repository on exit.
        A scoped owned Repo is rejected before entry while an active resource cache
        would prevent its required restoration cleanup.
    """

    old = get_config()
    _preflight_repo_input(old, repo, temporary=True)
    new = _merged_config(old, repo=repo, object_mode=object_mode, cache=cache)
    if repo is not _UNSET and old.repo is not new.repo:
        _preflight_session_transition(old, new, temporary=True)
        _resource_cache_selected_repo(new.repo)
    token = _current_config.set(new)
    try:
        yield new
    finally:
        try:
            if new.repo is not old.repo:
                _preflight_session_transition(new, old)
                _close_owned_repo(new)
        finally:
            _current_config.reset(token)


@contextmanager
def _construction_config(*, object_mode: ObjectMode = "fresh"):
    """Enter private fresh mode only while a materialization admission is active."""

    old = get_config()
    new = _merged_config(
        old, object_mode=object_mode, internal_construction=True
    )
    config_token = _current_config.set(new)
    construction_token = _internal_construction.set(True)
    try:
        yield new
    finally:
        _internal_construction.reset(construction_token)
        _current_config.reset(config_token)


def reset_config() -> SessionConfig:
    """Restore defaults, close an owned Repo, and return the default config.

    Raises:
        RuntimeError: If the owned Repo is retained by an active Session resource
            cache; the existing selection remains installed in that case.
    """

    old = get_config()
    _preflight_session_transition(old, _DEFAULT_CONFIG)
    _close_owned_repo(old)
    _current_config.set(_DEFAULT_CONFIG)
    return _DEFAULT_CONFIG


def close_configured_repo() -> None:
    """Close the current repository only when this configuration owns it."""

    _close_owned_repo(get_config())


def status() -> dict[str, Any]:
    """Return requested and effective core object-mode configuration.

    Returns:
        A detached mapping containing the configured repository/cache, requested
        mode, effective mode, and whether orchestration imposes its definition
        floor. ``object_mode`` remains an alias of ``effective_object_mode``.

    Side Effects:
        Reads the PID-bound public runtime state without changing configuration.
    """

    cfg = get_config()
    effective_mode = current_object_mode()
    from dryml.runtime.context import active_runtime
    from dryml.runtime.modes import RuntimeMode

    orchestrator_floor = active_runtime().mode is RuntimeMode.ORCHESTRATOR
    return {
        "repo": cfg.repo,
        "object_mode": effective_mode,
        "requested_object_mode": cfg.object_mode,
        "effective_object_mode": effective_mode,
        "orchestrator_floor": orchestrator_floor,
        "cache": cfg.cache,
        "repo_owned": cfg.repo_owned,
    }


__all__ = [
    "SessionConfig",
    "configure",
    "config",
    "status",
    "reset_config",
    "get_config",
    "current_repo",
    "current_object_mode",
    "current_cache",
    "close_configured_repo",
]
