"""Worker-local core setup and execution-context ownership.

The core execution adapter encodes the actual callable separately.  This module
only provides the setup factory consumed by generic Execute before it transfers
that payload, keeping generic Execute free of core imports and resource policy.
"""

from __future__ import annotations

import asyncio
import importlib
import math
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, TypeAlias

from dryml.execute.models import WorkerSetupContext
from dryml.formats import validate_envelope
from dryml.runtime import (
    ExecutionGrant,
    RuntimeContextSpec,
    RuntimeMode,
    activation_scope,
    active_runtime,
)

from .freeze import FrozenDict
from .repo import Repo
from .repo_definition import RepoDefinition
from .session import config, get_config
from .store.dir import DirStore


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Borrowed core handles visible only during one worker invocation.

    Args:
        repo: Reconstructed worker-owned state repository. Workloads borrow it
            and must not close it.
        control_store: Optional selected control Store. It may be a Store already
            held by ``repo`` and is likewise borrowed by workloads.

    Side Effects:
        None. Lifetime is owned by :func:`core_worker_setup`, not this value.
    """

    repo: Repo
    control_store: DirStore | None


@dataclass(slots=True)
class _ContextLease:
    """Track one context's thread/task owner and whether its setup remains live."""

    value: ExecutionContext
    thread_id: int
    task: asyncio.Task[Any] | None
    active: bool = True


_current_context: ContextVar[_ContextLease | None] = ContextVar("dryml_core_execution_context", default=None)
_SETUP_SCHEMA = "dryml.core.execute.v1.1"
_SETUP_KIND = "worker_setup"
_SETUP_PREFIX = "core_setup"
_SETUP_FIELDS = frozenset({"runtime", "repo", "role", "replica", "control_store"})
_SETUP_BOUNDS = {"max_depth": 64, "max_nodes": 65_536, "max_entries": 65_536}
Inherit: TypeAlias = Literal["inherit"]
CacheMode: TypeAlias = Literal["none", "weak", "strong"]
ReturnObjects: TypeAlias = bool | Literal["auto"]
_INHERIT = "inherit"


def _freeze_storage_setup(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Deeply detach one bounded JSON storage-role description.

    Args:
        value: JSON-compatible storage-role mapping supplied by a marshalling
            strategy.

    Returns:
        An immutable recursively frozen mapping detached from ``value``.

    Raises:
        TypeError: If ``value`` is not a string-keyed JSON mapping.
        ValueError: If its nesting, entry count, string size, or numeric values
            exceed the bounded setup grammar.

    Side Effects:
        None. This helper never opens a Store or renders caller values in errors.
    """
    entries = 0

    def freeze(item: Any, depth: int) -> Any:
        nonlocal entries
        entries += 1
        if depth > _SETUP_BOUNDS["max_depth"] or entries > _SETUP_BOUNDS["max_entries"]:
            raise ValueError("core storage setup exceeds bounded structure limits")
        if item is None or type(item) in {bool, int}:
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("core storage setup requires finite JSON numbers")
            return item
        if isinstance(item, str):
            if len(item.encode("utf-8")) > 1_048_576:
                raise ValueError("core storage setup string exceeds the bounded limit")
            return item
        if isinstance(item, Mapping):
            frozen_items = []
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError("core storage setup requires string mapping keys")
                frozen_items.append((key, freeze(child, depth + 1)))
            return FrozenDict(frozen_items)
        if isinstance(item, (list, tuple)):
            return tuple(freeze(child, depth + 1) for child in item)
        raise TypeError("core storage setup requires JSON-compatible values")

    frozen = freeze(value, 0)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class PreparedCoreCall:
    """Immutable strategy output retained between preparation and recovery.

    Args:
        invocation: Strategy-owned bounded invocation bytes. U5 defines their
            callable codec; U4 deliberately does not serialize live call values.
        storage_setup: Detached JSON storage-role description for worker setup.

    Raises:
        TypeError: If invocation is not bytes or storage setup is not a JSON
            mapping.
        ValueError: If storage setup exceeds the closed bounded grammar.

    Side Effects:
        Copies and freezes storage setup. The value owns no Repo, Store, runtime,
        or caller-session resource.
    """

    invocation: bytes
    storage_setup: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate and recursively freeze the strategy-owned handoff data."""
        if not isinstance(self.invocation, bytes):
            raise TypeError("core invocation must be bytes")
        if not isinstance(self.storage_setup, Mapping):
            raise TypeError("core storage setup must be a mapping")
        object.__setattr__(self, "storage_setup", _freeze_storage_setup(self.storage_setup))


class MarshallingStrategy(Protocol):
    """Own storage eligibility and future core call/result transport mechanics."""

    def validate(self, *, repo: Repo | RepoDefinition | None, control_store: DirStore | None) -> None:
        """Validate strategy-specific storage eligibility without mutation."""

    def prepare(
            self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any],
            *, repo: Repo | None, control_store: DirStore | None,
            update_args: bool) -> PreparedCoreCall:
        """Prepare one detached call; implemented by U5's callable codec."""

    def invoke(self, invocation: bytes, *, repo: Repo | None, update_args: bool) -> bytes:
        """Invoke strategy bytes after worker setup; implemented by U5."""

    def recover(
            self, result: bytes, prepared: PreparedCoreCall, *, repo: Repo | None,
            args: tuple[Any, ...], kwargs: Mapping[str, Any],
            return_objects: bool, update_args: bool) -> Any:
        """Recover strategy output; implemented by U5/U6."""


class SharedDirStoreStrategy:
    """Initial marshalling strategy for existing directly shared DirStore authority.

    The strategy accepts only a detached Repo definition containing directly
    reopenable ``DirStore`` descriptors. Call/result codec methods intentionally
    remain unavailable until U5 so this stage cannot pickle live core values.
    """

    def validate(self, *, repo: Repo | RepoDefinition | None, control_store: DirStore | None) -> None:
        """Validate shared-directory eligibility without opening or creating Stores.

        Args:
            repo: Live Repo or detached RepoDefinition selected for this call.
            control_store: Optional separately selected direct Store handle.

        Raises:
            ValueError: If storage is absent or includes a non-directory Store.
            TypeError: If the control binding is not a Store handle.

        Side Effects:
            A live Repo is exported once only when callers invoke this public
            method directly. U4 snapshot preparation supplies a definition so it
            never takes a second export.
        """
        if repo is None:
            raise ValueError("SharedDirStoreStrategy requires configured shared Store authority")
        definition = repo.to_definition() if isinstance(repo, Repo) else repo
        if not isinstance(definition, RepoDefinition):
            raise TypeError("SharedDirStoreStrategy requires a Repo or RepoDefinition")
        data = definition.to_data()
        stores = data["stores"]
        if not stores or any(descriptor.get("kind") != "dir" for descriptor in stores):
            raise ValueError("SharedDirStoreStrategy requires only configured DirStores")
        if control_store is not None and type(control_store) is not DirStore:
            raise ValueError("SharedDirStoreStrategy control_store must be a direct DirStore")

    def prepare(self, *args: Any, **kwargs: Any) -> PreparedCoreCall:
        """Reject premature callable preparation until U5 owns the codec."""
        raise NotImplementedError("SharedDirStoreStrategy callable preparation is implemented by U5")

    def invoke(self, *args: Any, **kwargs: Any) -> bytes:
        """Reject premature invocation until U5 owns the codec."""
        raise NotImplementedError("SharedDirStoreStrategy invocation is implemented by U5")

    def recover(self, *args: Any, **kwargs: Any) -> Any:
        """Reject premature recovery until U5/U6 own result adaptation."""
        raise NotImplementedError("SharedDirStoreStrategy recovery is implemented by U5/U6")


def _validate_strategy_identity(value: object) -> type[SharedDirStoreStrategy]:
    """Accept only the initial importable shared-directory strategy identity."""
    if value is not SharedDirStoreStrategy:
        raise ValueError("only the importable SharedDirStoreStrategy is supported")
    module = importlib.import_module(SharedDirStoreStrategy.__module__)
    resolved: object = module
    for part in SharedDirStoreStrategy.__qualname__.split("."):
        resolved = getattr(resolved, part, None)
    if resolved is not SharedDirStoreStrategy:
        raise ValueError("SharedDirStoreStrategy must retain its importable class identity")
    return SharedDirStoreStrategy


@dataclass(frozen=True, kw_only=True, slots=True)
class CoreOptions:
    """Inert reusable core-execution overrides resolved at submission time.

    Args:
        repo: Live Repo, detached RepoDefinition, explicit ``None``, or
            ``"inherit"``.
        control_store: Direct selected DirStore, explicit ``None``, or
            ``"inherit"``.
        runtime: Worker runtime intent, explicit ``None``, or ``"inherit"``.
        cache: Detached cache-construction policy or ``"inherit"``.
        marshalling: The initial importable SharedDirStoreStrategy class or
            ``"inherit"``.
        return_objects: Live-result policy, ``"auto"``, or ``"inherit"``.
        update_args: Argument-refresh policy or ``"inherit"``.

    Raises:
        TypeError: If a field has an unsupported input type.
        ValueError: If a policy value or strategy identity is unsupported.

    Each field accepts ``"inherit"`` to fall through to executor then session
    defaults. ``None`` is an explicit clearing value for Repo, control Store, or
    runtime selection. Construction validates shape only; it never exports,
    opens, reconstructs, saves, or closes a Store.
    """

    repo: Repo | RepoDefinition | None | Inherit = _INHERIT
    control_store: DirStore | None | Inherit = _INHERIT
    runtime: RuntimeContextSpec | None | Inherit = _INHERIT
    cache: CacheMode | Inherit = _INHERIT
    marshalling: type[MarshallingStrategy] | Inherit = _INHERIT
    return_objects: ReturnObjects | Inherit = _INHERIT
    update_args: bool | Inherit = _INHERIT

    def __post_init__(self) -> None:
        """Validate option shape without inspecting persistent authority."""
        if self.repo != _INHERIT and self.repo is not None and not isinstance(self.repo, (Repo, RepoDefinition)):
            raise TypeError("core repo must be a Repo, RepoDefinition, None, or 'inherit'")
        if self.control_store != _INHERIT and self.control_store is not None and not isinstance(self.control_store, DirStore):
            raise TypeError("control_store must be a DirStore, None, or 'inherit'")
        if self.runtime != _INHERIT and self.runtime is not None and not isinstance(self.runtime, RuntimeContextSpec):
            raise TypeError("runtime must be a RuntimeContextSpec, None, or 'inherit'")
        if self.cache != _INHERIT and (
                not isinstance(self.cache, str) or self.cache not in {"none", "weak", "strong"}):
            raise ValueError("cache must be 'none', 'weak', 'strong', or 'inherit'")
        if self.marshalling != _INHERIT:
            _validate_strategy_identity(self.marshalling)
        if self.return_objects != _INHERIT and not (
                type(self.return_objects) is bool or self.return_objects == "auto"):
            raise ValueError("return_objects must be True, False, 'auto', or 'inherit'")
        if self.update_args != _INHERIT and type(self.update_args) is not bool:
            raise TypeError("update_args must be bool or 'inherit'")


@dataclass(frozen=True, slots=True)
class _EffectiveCoreOptions:
    """Resolved per-submission policy before strategy-specific storage preparation."""

    repo: Repo | RepoDefinition | None
    control_store: DirStore | None
    runtime: RuntimeContextSpec | None
    cache: CacheMode
    marshalling: type[SharedDirStoreStrategy]
    return_objects: ReturnObjects
    update_args: bool


def _resolve_field(name: str, call: CoreOptions | None, executor: CoreOptions | None, terminal: Any) -> Any:
    """Resolve one option field without collapsing explicit ``None`` into inheritance."""
    for options in (call, executor):
        if options is not None:
            value = getattr(options, name)
            if value != _INHERIT:
                return value
    return terminal


def resolve_core_options(
        core: CoreOptions | None = None, *, executor: CoreOptions | None = None) -> _EffectiveCoreOptions:
    """Resolve call, executor, and current-session core settings into one value.

    Args:
        core: Per-call inert overrides, or ``None`` for no per-call override.
        executor: Reusable executor defaults, or ``None`` when absent.

    Returns:
        One immutable effective configuration. Repo/cache session values are read
        once at this boundary; runtime defaults to ``None`` rather than copying
        the caller's active runtime allocation or session state.

    Raises:
        TypeError: If either supplied layer is not CoreOptions or None.
        ValueError: If effective result materialization or updates violate the
            caller's orchestration floor.

    Side Effects:
        Reads the current immutable session and runtime projections only. It does
        not export, open, mutate, or close storage and never changes caller state.
    """
    if core is not None and not isinstance(core, CoreOptions):
        raise TypeError("core must be CoreOptions or None")
    if executor is not None and not isinstance(executor, CoreOptions):
        raise TypeError("executor defaults must be CoreOptions or None")
    session = get_config()
    resolved = _EffectiveCoreOptions(
        repo=_resolve_field("repo", core, executor, session.repo),
        control_store=_resolve_field("control_store", core, executor, None),
        runtime=_resolve_field("runtime", core, executor, None),
        cache=_resolve_field("cache", core, executor, session.cache),
        marshalling=_validate_strategy_identity(_resolve_field(
            "marshalling", core, executor, SharedDirStoreStrategy,
        )),
        return_objects=_resolve_field("return_objects", core, executor, "auto"),
        update_args=_resolve_field("update_args", core, executor, False),
    )
    if active_runtime().mode is RuntimeMode.ORCHESTRATOR and (
            resolved.return_objects is True or resolved.update_args):
        raise ValueError("orchestration mode prohibits live result materialization and argument updates")
    return resolved


def _store_identity(path: str) -> tuple[int, int]:
    """Return one direct Store's physical identity after existing-only validation."""
    try:
        DirStore._validate_existing_root(path)
        evidence = os.stat(path)
    except (OSError, RuntimeError, ValueError):
        raise ValueError("shared Store authority is unavailable") from None
    return evidence.st_dev, evidence.st_ino


def _shared_storage_setup(
        definition: RepoDefinition, control_store: DirStore | None) -> Mapping[str, Any]:
    """Derive every worker storage role only from one detached Repo definition."""
    data = definition.to_data()
    stores = data["stores"]
    if not stores or any(descriptor.get("kind") != "dir" for descriptor in stores):
        raise ValueError("SharedDirStoreStrategy requires only configured DirStores")
    identities = [_store_identity(descriptor["path"]) for descriptor in stores]
    if len(set(identities)) != len(identities):
        raise ValueError("shared Store definition has duplicate physical authority")
    control_descriptor: Mapping[str, Any] | None = None
    if control_store is not None:
        if type(control_store) is not DirStore:
            raise ValueError("SharedDirStoreStrategy control_store must be a direct DirStore")
        if control_store._query_index_config is not None:
            raise ValueError("SharedDirStoreStrategy control_store has an unsupported query policy")
        identity = _store_identity(control_store.base_dir)
        matches = [index for index, candidate in enumerate(identities) if candidate == identity]
        if len(matches) > 1:
            raise ValueError("shared Store control binding is ambiguous")
        if matches:
            control_descriptor = {"repo_store": matches[0]}
        else:
            control_descriptor = {
                "kind": "dir",
                "path": os.path.abspath(control_store.base_dir),
                "query_index": control_store.query_index_policy,
            }
    return {"repo": data, "control_store": control_descriptor}


@dataclass(slots=True)
class _PreparedSharedStorage:
    """Submission-owned recovery Repo plus detached worker storage role data."""

    storage_setup: Mapping[str, Any]
    recovery_repo: Repo
    runtime: RuntimeContextSpec | None
    cache: CacheMode
    marshalling: type[SharedDirStoreStrategy]
    return_objects: ReturnObjects
    update_args: bool
    _closed: bool = False

    def close(self) -> None:
        """Release only this snapshot's reconstructed handles without flushing.

        The original caller Repo and all caller-supplied Store handles remain
        borrowed. Repeated close calls are no-ops after successful release.
        """
        if not self._closed:
            self.recovery_repo.close(flush=False)
            self._closed = True


def prepare_shared_storage(
        core: CoreOptions | None = None, *, executor: CoreOptions | None = None) -> _PreparedSharedStorage:
    """Freeze eligible shared storage and open an owned recovery reconstruction.

    Args:
        core: Per-call core overrides.
        executor: Reusable executor defaults beneath ``core``.

    Returns:
        A submission-owned snapshot with immutable worker storage-role data and a
        fresh recovery Repo. Call ``close()`` after future cleanup to release its
        owned handles with ``flush=False``.

    Raises:
        TypeError: If options or storage bindings have unsupported types.
        ValueError: If storage is absent, includes ZipStore authority, has missing
        or inaccessible directories, duplicate destinations, or violates caller
        materialization floors.

    Side Effects:
        Exports a live Repo exactly once and opens only fresh existing handles for
        the retained recovery Repo. It never saves, creates, installs, or closes
        caller-owned resources.
    """
    effective = resolve_core_options(core, executor=executor)
    strategy = effective.marshalling()
    selected = effective.repo
    if selected is None:
        strategy.validate(repo=None, control_store=effective.control_store)
    if isinstance(selected, Repo):
        definition = selected.to_definition()
    elif isinstance(selected, RepoDefinition):
        definition = selected
    else:
        raise TypeError("core repo must resolve to Repo, RepoDefinition, or None")
    # Validation consumes the detached definition, preserving the one-export cut.
    strategy.validate(repo=definition, control_store=effective.control_store)
    setup = _freeze_storage_setup(_shared_storage_setup(definition, effective.control_store))
    recovery_repo = Repo.from_definition(definition)
    return _PreparedSharedStorage(
        setup,
        recovery_repo,
        effective.runtime,
        effective.cache,
        effective.marshalling,
        effective.return_objects,
        effective.update_args,
    )


def _task() -> asyncio.Task[Any] | None:
    """Return the active asyncio task without requiring an event loop."""
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def current_context() -> ExecutionContext:
    """Return the active worker's borrowed core handles.

    Returns:
        The task- and thread-owned worker execution context.

    Raises:
        RuntimeError: If no setup is active, a copied context outlived its owner,
            or a copied asyncio task attempts to use another task's context.

    Side Effects:
        None. This function never selects an ambient caller repository.
    """
    lease = _current_context.get()
    if lease is None or not lease.active:
        raise RuntimeError("current_context is unavailable outside an active worker setup; copied-task contexts cannot outlive their owner")
    if lease.thread_id != threading.get_ident():
        raise RuntimeError("current_context belongs to a different thread")
    if lease.task is not _task():
        raise RuntimeError("current_context belongs to a different task; copied-task context reuse is not allowed")
    return lease.value


@contextmanager
def worker_context(value: ExecutionContext) -> Iterator[ExecutionContext]:
    """Install one worker context for the owning thread and asyncio task.

    Args:
        value: Borrowed worker Repo/control-Store pair.

    Yields:
        The supplied execution context.

    Raises:
        TypeError: If ``value`` is not an :class:`ExecutionContext`.

    Side Effects:
        Sets a task-local context variable and invalidates copied values on exit.
    """
    if not isinstance(value, ExecutionContext):
        raise TypeError("worker_context requires an ExecutionContext")
    lease = _ContextLease(value, threading.get_ident(), _task())
    token = _current_context.set(lease)
    try:
        yield value
    finally:
        lease.active = False
        _current_context.reset(token)


def _decode_setup(data: Mapping[str, Any]) -> tuple[RuntimeContextSpec, RepoDefinition, str, int, Mapping[str, Any] | None]:
    """Decode one inert, closed core worker-setup envelope before activation.

    The ``dryml.core.execute.v1.1`` ``worker_setup`` payload has exactly
    ``runtime``, ``repo``, ``role``, ``replica``, and ``control_store`` fields.
    Its nested runtime and Repo values retain their owner schemas; this function
    only validates and decodes them and never opens a Store.
    """
    raw_payload = data.get("payload")
    if not isinstance(raw_payload, Mapping):
        raise ValueError("core worker setup requires a v1.1 envelope payload")
    envelope = validate_envelope(
        data, schema=_SETUP_SCHEMA, kind=_SETUP_KIND, prefix=_SETUP_PREFIX,
        identifying_payload=raw_payload, **_SETUP_BOUNDS,
    )
    payload = envelope["payload"]
    if set(payload) != _SETUP_FIELDS:
        raise ValueError("core worker setup payload fields are closed")
    runtime = payload["runtime"]
    repo = payload["repo"]
    role = payload["role"]
    replica = payload["replica"]
    control_store = payload["control_store"]
    if not isinstance(runtime, Mapping) or not isinstance(repo, Mapping):
        raise TypeError("core worker setup runtime and Repo definitions must be envelopes")
    if not isinstance(role, str) or not role or isinstance(replica, bool) or not isinstance(replica, int) or replica < 0:
        raise ValueError("core worker role and replica are invalid")
    if control_store is not None and not isinstance(control_store, Mapping):
        raise TypeError("core worker control Store descriptor must be a mapping or null")
    if isinstance(control_store, Mapping):
        if set(control_store) == {"repo_store"}:
            index = control_store["repo_store"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("core worker control Store index is invalid")
        elif (
            set(control_store) != {"kind", "path", "query_index"}
            or control_store["kind"] != "dir"
            or not isinstance(control_store["path"], str)
            or not isinstance(control_store["query_index"], str)
        ):
            raise ValueError("core worker control Store descriptor is invalid")
    return RuntimeContextSpec.from_data(runtime), RepoDefinition.from_data(repo), role, replica, control_store


def _require_pristine_session() -> None:
    """Reject inherited core session state before a worker opens Store authority."""
    session = get_config()
    if session.repo is not None or session.repo_owned or session.object_mode != "fresh" or session.cache != "weak":
        raise RuntimeError("core worker setup requires a pristine core session")


def _control_store(repo: Repo, descriptor: Any) -> tuple[DirStore | None, bool]:
    """Resolve one explicit optional control Store and whether setup opened it."""
    if descriptor is None:
        return None, False
    if set(descriptor) == {"repo_store"}:
        index = descriptor["repo_store"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(repo.stores):
            raise ValueError("core worker control Store index is invalid")
        store = repo.stores[index]
        if type(store) is not DirStore:
            raise ValueError("core worker control Store must be a DirStore")
        return store, False
    path = Path(descriptor["path"])
    for store in repo.stores:
        if type(store) is DirStore and os.path.samefile(path, store.base_dir):
            return store, False
    return DirStore.open_existing(path, query_index=descriptor["query_index"]), True


@contextmanager
def core_worker_setup(context: WorkerSetupContext, data: Mapping[str, Any]) -> Iterator[None]:
    """Establish worker runtime, Repo/session, and task-owned core context.

    Args:
        context: Generic backend evidence reconstructed after GO.
        data: Detached ``dryml.core.execute.v1.1`` ``worker_setup`` envelope
            containing owner runtime/Repo envelopes, role/replica, and an
            optional explicit control Store role.

    Yields:
        ``None`` after controls are installed and before generic Execute receives
        callable payload bytes.

    Raises:
        TypeError: If setup data has unsupported structural types.
        ValueError: If the closed setup envelope, nested owner envelopes, or
            control Store descriptor is malformed.
        RuntimeError: If inherited session state, activation, Store
            reconstruction, or control Store setup fails. Errors intentionally
            omit the detached setup payload.

    Side Effects:
        Reconstructs only existing worker-owned Stores after runtime activation,
        installs temporary core session/context state, and closes acquired handles
        once in LIFO order with ``flush=False``.
    """
    if not isinstance(context, WorkerSetupContext) or not isinstance(data, Mapping):
        raise TypeError("core worker setup requires WorkerSetupContext and mapping data")
    spec, definition, role, replica, descriptor = _decode_setup(data)
    _require_pristine_session()
    grant = ExecutionGrant.from_worker_setup(context, role=role, replica=replica)
    with ExitStack() as stack:
        stack.enter_context(activation_scope(spec, grant))
        repo = Repo.from_definition(definition)
        stack.callback(repo.close, flush=False)
        control, opened_control = _control_store(repo, descriptor)
        if opened_control:
            assert control is not None
            stack.callback(control.close)
        stack.enter_context(config(repo=repo))
        stack.enter_context(worker_context(ExecutionContext(repo, control)))
        yield None


__all__ = ["ExecutionContext", "core_worker_setup", "current_context", "worker_context"]
