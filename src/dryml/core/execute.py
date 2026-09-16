"""Worker-local core setup and execution-context ownership.

The core execution adapter encodes the actual callable separately.  This module
only provides the setup factory consumed by generic Execute before it transfers
that payload, keeping generic Execute free of core imports and resource policy.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dryml.execute.models import WorkerSetupContext
from dryml.formats import validate_envelope
from dryml.runtime import ExecutionGrant, RuntimeContextSpec, activation_scope

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
