from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Literal

from dryml.reporting import ReportingConfig, config_from_env
from dryml.runtime.errors import RuntimeTransitionError
from dryml.runtime.publication import publication


ObjectMode = Literal["fresh", "definition", "concrete", "selector", "space", "load_or_build"]
CacheMode = Literal["none", "weak", "strong"]

_UNSET = object()
_OBJECT_MODES = {"fresh", "definition", "concrete", "selector", "space", "load_or_build"}
_CACHE_MODES = {"none", "weak", "strong"}


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Context-local core configuration for repository and object behavior.

    ``object_mode_floor`` identifies the public orchestrator control epoch that
    admitted a definition-like override. It is internal metadata: callers only
    observe the effective object mode through :func:`current_object_mode`.
    """

    repo: Any = None
    object_mode: ObjectMode = "fresh"
    cache: CacheMode = "weak"
    reporting: ReportingConfig = None
    repo_owned: bool = False
    object_mode_floor: int | None = None

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
    """Return the raw context-local core configuration.

    Returns:
        The stored configuration. Use :func:`current_object_mode` when the
        effective mode is required during public orchestration.
    """

    return _current_config.get()


def current_repo():
    return get_config().repo


def current_object_mode() -> ObjectMode:
    """Return the effective object mode for the current control epoch.

    An active public orchestrator session projects ``definition`` unless this
    context entered a matching definition-like override. This leaves the raw
    ContextVar baseline intact for restoration when orchestration ends.
    """

    floor = _current_object_mode_floor()
    cfg = get_config()
    if floor is not None and cfg.object_mode_floor != floor:
        return "definition"
    return cfg.object_mode


def current_cache() -> CacheMode:
    return get_config().cache


def _validate_object_mode(value: str) -> ObjectMode:
    if value not in _OBJECT_MODES:
        raise ValueError(
            "object_mode must be one of "
            f"{sorted(_OBJECT_MODES)}, got {value!r}."
        )
    return value


def _current_object_mode_floor() -> int | None:
    """Return the active facade orchestration control epoch, if any."""

    generation = publication.current()
    floor = generation.metadata.get("object_mode_floor")
    return floor if isinstance(floor, int) else None


def _validated_object_mode(
        value: str, *, internal_construction: bool = False) -> tuple[ObjectMode, int | None]:
    """Validate a public mode change against the active orchestration floor."""

    mode = _validate_object_mode(value)
    floor = _current_object_mode_floor()
    if floor is not None and mode in {"fresh", "load_or_build"}:
        from dryml.runtime.guards import internal_construction_admitted

        if not internal_construction or not internal_construction_admitted():
            raise RuntimeTransitionError(
                "orchestration object-mode floor prohibits public fresh/load_or_build selection",
                context={
                    "mode": "orchestrator",
                    "object_mode": mode,
                    "fix": "use definition/concrete/selector/space modes, or materialize through a guarded Definition build",
                },
            )
    return mode, floor


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
        reporting=_UNSET,
        internal_construction: bool = False) -> SessionConfig:
    updates = {}

    if repo is not _UNSET:
        repo_obj, repo_owned = _coerce_repo(repo)
        updates["repo"] = repo_obj
        updates["repo_owned"] = repo_owned

    if object_mode is not _UNSET:
        mode, floor = _validated_object_mode(
            object_mode, internal_construction=internal_construction
        )
        updates["object_mode"] = mode
        updates["object_mode_floor"] = floor

    if cache is not _UNSET:
        updates["cache"] = _validate_cache(cache)

    if reporting is not _UNSET:
        updates["reporting"] = ReportingConfig.from_value(reporting, base=base.reporting)

    return replace(base, **updates)


def configure(*, repo=_UNSET, object_mode=_UNSET, cache=_UNSET, reporting=_UNSET) -> SessionConfig:
    """Persist selected core configuration values in the current context.

    Materializing object modes are rejected before mutation while public
    orchestration is active.
    """
    old = get_config()
    new = _merged_config(old, repo=repo, object_mode=object_mode, cache=cache, reporting=reporting)
    _current_config.set(new)

    if repo is not _UNSET and old.repo is not new.repo:
        _close_owned_repo(old)
    return new


@contextmanager
def config(*, repo=_UNSET, object_mode=_UNSET, cache=_UNSET, reporting=_UNSET):
    """Temporarily override core configuration with exact token restoration.

    The context validates materializing object modes before changing the
    ContextVar. Definition-like modes remain nestable during orchestration.
    """
    old = get_config()
    new = _merged_config(old, repo=repo, object_mode=object_mode, cache=cache, reporting=reporting)
    token = _current_config.set(new)
    try:
        yield new
    finally:
        _current_config.reset(token)
        if new.repo is not old.repo:
            _close_owned_repo(new)


@contextmanager
def _construction_config(*, object_mode: ObjectMode = "fresh"):
    """Enter a private object-construction mode under a live guard admission."""

    old = get_config()
    new = _merged_config(
        old, object_mode=object_mode, internal_construction=True
    )
    token = _current_config.set(new)
    try:
        yield new
    finally:
        _current_config.reset(token)


def reset_config() -> SessionConfig:
    old = get_config()
    default = _default_config()
    _current_config.set(default)
    _close_owned_repo(old)
    return default


def close_configured_repo() -> None:
    _close_owned_repo(get_config())


def status() -> dict[str, Any]:
    """Return current core configuration with its effective object mode."""

    cfg = get_config()
    return {
        "repo": cfg.repo,
        "object_mode": current_object_mode(),
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
