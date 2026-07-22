from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Literal

from dryml.reporting import ReportingConfig, config_from_env


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
    reporting: ReportingConfig = None
    repo_owned: bool = False

    def __post_init__(self) -> None:
        if self.reporting is None:
            object.__setattr__(self, "reporting", config_from_env())


def _default_config() -> SessionConfig:
    return SessionConfig()


_DEFAULT_CONFIG = _default_config()
_current_config: ContextVar[SessionConfig] = ContextVar(
    "dryml_session_config",
    default=_DEFAULT_CONFIG,
)


def get_config() -> SessionConfig:
    return _current_config.get()


def current_repo():
    return get_config().repo


def current_object_mode() -> ObjectMode:
    return get_config().object_mode


def current_cache() -> CacheMode:
    return get_config().cache


def _validate_object_mode(value: str) -> ObjectMode:
    if value not in _OBJECT_MODES:
        raise ValueError(
            "object_mode must be one of "
            f"{sorted(_OBJECT_MODES)}, got {value!r}."
        )
    return value


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
        reporting=_UNSET) -> SessionConfig:
    updates = {}

    if repo is not _UNSET:
        repo_obj, repo_owned = _coerce_repo(repo)
        updates["repo"] = repo_obj
        updates["repo_owned"] = repo_owned

    if object_mode is not _UNSET:
        updates["object_mode"] = _validate_object_mode(object_mode)

    if cache is not _UNSET:
        updates["cache"] = _validate_cache(cache)

    if reporting is not _UNSET:
        updates["reporting"] = ReportingConfig.from_value(reporting, base=base.reporting)

    return replace(base, **updates)


def configure(*, repo=_UNSET, object_mode=_UNSET, cache=_UNSET, reporting=_UNSET) -> SessionConfig:
    old = get_config()
    new = _merged_config(old, repo=repo, object_mode=object_mode, cache=cache, reporting=reporting)
    _current_config.set(new)

    if repo is not _UNSET and old.repo is not new.repo:
        _close_owned_repo(old)
    return new


@contextmanager
def config(*, repo=_UNSET, object_mode=_UNSET, cache=_UNSET, reporting=_UNSET):
    old = get_config()
    new = _merged_config(old, repo=repo, object_mode=object_mode, cache=cache, reporting=reporting)
    token = _current_config.set(new)
    try:
        yield new
    finally:
        _current_config.reset(token)
        if new.repo is not old.repo:
            _close_owned_repo(new)


def reset_config() -> SessionConfig:
    old = get_config()
    default = _default_config()
    _current_config.set(default)
    _close_owned_repo(old)
    return default


def close_configured_repo() -> None:
    _close_owned_repo(get_config())


def status() -> dict[str, Any]:
    cfg = get_config()
    return {
        "repo": cfg.repo,
        "object_mode": cfg.object_mode,
        "cache": cfg.cache,
        "reporting": cfg.reporting.to_data(),
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
