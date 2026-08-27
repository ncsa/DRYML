from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Literal


ObjectMode = Literal["fresh", "definition", "concrete", "selector", "space", "load_or_build"]
CacheMode = Literal["none", "weak", "strong"]

_UNSET = object()
_OBJECT_MODES = {"fresh", "definition", "concrete", "selector", "space", "load_or_build"}
_CACHE_MODES = {"none", "weak", "strong"}


@dataclass(frozen=True, slots=True)
class SessionConfig:
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


def get_config() -> SessionConfig:
    return _current_config.get()


def current_repo():
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
    old = get_config()
    new = _merged_config(old, repo=repo, object_mode=object_mode, cache=cache)
    _current_config.set(new)

    if repo is not _UNSET and old.repo is not new.repo:
        _close_owned_repo(old)
    return new


@contextmanager
def config(*, repo=_UNSET, object_mode=_UNSET, cache=_UNSET):
    old = get_config()
    new = _merged_config(old, repo=repo, object_mode=object_mode, cache=cache)
    token = _current_config.set(new)
    try:
        yield new
    finally:
        _current_config.reset(token)
        if new.repo is not old.repo:
            _close_owned_repo(new)


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
    old = get_config()
    _current_config.set(_DEFAULT_CONFIG)
    _close_owned_repo(old)
    return _DEFAULT_CONFIG


def close_configured_repo() -> None:
    _close_owned_repo(get_config())


def status() -> dict[str, Any]:
    """Return the current configuration using the effective object-mode projection."""

    cfg = get_config()
    return {
        "repo": cfg.repo,
        "object_mode": current_object_mode(),
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
