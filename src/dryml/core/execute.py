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
import time
from uuid import uuid4
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, RLock, Thread
from typing import Any, Callable, Literal, Protocol, TypeAlias

from dryml.execute import BackendConfig, ExecutionFuture, ExecutionOutput
from dryml.execute import Executor as _GenericExecutor
from dryml.execute import submit as _generic_submit
from dryml.execute.errors import CleanupError, ExecutionError
from dryml.execute.models import ExecutionIssue, ExecutionSnapshot, WorkerSetupContext
from dryml.environments import EnvironmentRequirement
from dryml.environments.specs import EnvironmentSpec
from dryml.formats import make_envelope, semantic_id, validate_envelope
from dryml.runtime import (
    ExecutionGrant,
    RuntimeContextSpec,
    RuntimeMode,
    activation_scope,
    active_runtime,
    publication,
)
from dryml.worlds import WorldRequirement

from dryml.formats import deep_freeze_json
from .repo import Repo
from .repo_definition import RepoDefinition
from .execute_codec import CoreCallCodecError
from .session import config, get_config
from .store.dir import DirStore
from .reference_values import StateRef


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
_SETUP_FIELDS_WITH_CACHE = _SETUP_FIELDS | {"cache"}
_SETUP_BOUNDS = {"max_depth": 64, "max_nodes": 65_536, "max_entries": 65_536}
Inherit: TypeAlias = Literal["inherit"]
CacheMode: TypeAlias = Literal["none", "weak", "strong"]
ReturnObjects: TypeAlias = bool | Literal["auto"]
_INHERIT = "inherit"
_MISSING = object()


@dataclass(frozen=True, slots=True)
class CorePublicationEvidence:
    """Portable fact for one Store-table-relative publication boundary.

    Args:
        state_ref: Exact StateRef associated with the observed boundary.
        store_index: Index in the frozen Repo Store table, never a Store handle.
        phase: Repo publication phase reported by the owning save path.
        status: Observed completed, failed, uncertain, or unattempted status.
        path: Canonical graph path recorded by the save report.

    The value is detached from the worker's ephemeral StoreReport and is safe for
    Future snapshots may retain it after worker cleanup.
    """

    state_ref: StateRef
    store_index: int
    phase: str
    status: str
    path: str


@dataclass(frozen=True, slots=True)
class CoreRefreshEvidence:
    """One caller-target refresh observation for a future facade.

    Args:
        target: Opaque stable target association, never a live Object.
        state_ref: Exact published snapshot requested for that target.
        status: ``pending``, ``applied``, ``preflight_failed``, ``invalidated``,
            or ``skipped``.

    Coordinator recovery state retains this ledger for one call, so repeated
    future result access cannot retry a partial refresh.
    """

    target: str
    state_ref: StateRef
    status: Literal["pending", "applied", "preflight_failed", "invalidated", "skipped"]


@dataclass(frozen=True, slots=True)
class CoreOutcomeEvidence:
    """Detached exact authority published by a completed worker outcome.

    Args:
        publications: Store-table-relative publication boundaries from Repo save
            reports, including partial publication evidence.
        updates: Exact maximal-root StateRefs selected for optional caller refresh.

    No live Store, Repo, StoreReport, workload argument, or result is retained.
    Publication failure is therefore inspectable without presenting it as rollback.
    """

    publications: tuple[CorePublicationEvidence, ...]
    updates: tuple[StateRef, ...]
    refreshes: tuple[CoreRefreshEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class CoreAdaptationOutcome:
    """Decoded result and evidence retained for a future core facade.

    Args:
        value: Decoded ordinary value or immutable reference graph. It contains no
            live StoreReport or Store handle.
        evidence: Exact detached publication/update evidence.

    A future may cache this immutable outcome for repeated access without
    repeating publication, recovery, or refresh work.
    """

    value: Any
    evidence: CoreOutcomeEvidence


class CoreExecutionError(ExecutionError):
    """A core execution phase failed while retaining detached authority evidence.

    Args:
        message: Bounded phase failure category, never a workload value representation.
        phase: Core adaptation phase which failed.
        execution: Optional future retaining this adaptation.
        evidence: Publication/update/refresh facts completed before the failure.

    This error does not claim rollback or retry. A future retains it directly rather
    than introducing a second outcome-error scheme.
    """

    def __init__(self, message: str, *, phase: str,
                 evidence: CoreOutcomeEvidence | None = None,
                 execution: Any = None) -> None:
        super().__init__(message)
        self.phase = phase
        self.evidence = evidence
        self.execution = execution


class _RefreshLedger:
    """Retain one call's deterministic refresh progress without retaining result data."""

    def __init__(self) -> None:
        self.entries: list[CoreRefreshEvidence] = []
        self.error: CoreExecutionError | None = None

    def begin(self, updates: tuple[tuple[str, StateRef], ...]) -> None:
        if not self.entries:
            self.entries = [CoreRefreshEvidence(token, state, "pending") for token, state in updates]

    def replace(self, index: int, status: Literal["pending", "applied", "preflight_failed", "invalidated", "skipped"]) -> None:
        current = self.entries[index]
        self.entries[index] = CoreRefreshEvidence(current.target, current.state_ref, status)

    def evidence(self, base: CoreOutcomeEvidence) -> CoreOutcomeEvidence:
        return CoreOutcomeEvidence(base.publications, base.updates, tuple(self.entries))


@dataclass(slots=True)
class _CoreRecovery:
    """Coordinator-only caller bindings and refresh progress for one submission.

    A future creates this state immediately after preparation, before caller code can
    mutate arguments. ``PreparedCoreCall`` remains detached transport data.
    """

    targets: Mapping[str, Any]
    ledger: _RefreshLedger = field(default_factory=_RefreshLedger)

    @classmethod
    def bind(cls, prepared: "PreparedCoreCall", args: tuple[Any, ...],
             kwargs: Mapping[str, Any]) -> "_CoreRecovery":
        """Bind prepared opaque tokens to the supplied caller object identities."""
        if not prepared.update_targets:
            return cls({})
        available = {
            (item["path"], item["object_digest"]): item["object"]
            for item in _prepared_update_targets(args, kwargs)
        }
        targets = {}
        for item in prepared.update_targets:
            key = (item["path"], item["object_digest"])
            try:
                targets[item["target"]] = available.pop(key)
            except KeyError as error:
                raise CoreCallCodecError("core execution update targets do not match recovery arguments") from error
        if available:
            raise CoreCallCodecError("core execution recovery arguments have unexpected update targets")
        return cls(targets)


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
    frozen = deep_freeze_json(value, **_SETUP_BOUNDS)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class PreparedCoreCall:
    """Immutable strategy output retained between preparation and recovery.

    Args:
        invocation: Strategy-owned bounded invocation bytes; it never contains
            live core values.
        storage_setup: Detached JSON storage-role description for worker setup.
        update_targets: Detached transport descriptors for optional caller refresh.

    Raises:
        TypeError: If invocation is not bytes or storage setup is not a JSON
            mapping.
        ValueError: If storage setup exceeds the closed bounded grammar.

    Side Effects:
        Copies and freezes storage setup. The value owns no Repo, Store, runtime,
        or caller-session resource. Coordinator recovery state retains original
        caller Objects separately from this transport value.
    """

    invocation: bytes
    storage_setup: Mapping[str, Any]
    update_targets: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Validate and recursively freeze the strategy-owned handoff data."""
        if not isinstance(self.invocation, bytes):
            raise TypeError("core invocation must be bytes")
        if not isinstance(self.storage_setup, Mapping):
            raise TypeError("core storage setup must be a mapping")
        if not isinstance(self.update_targets, tuple):
            raise TypeError("core update targets must be a tuple")
        object.__setattr__(self, "storage_setup", _freeze_storage_setup(self.storage_setup))
        frozen_targets = deep_freeze_json(
            {"targets": list(self.update_targets)}, **_SETUP_BOUNDS,
        )
        object.__setattr__(self, "update_targets", tuple(frozen_targets["targets"]))

    def worker_setup(self, runtime: RuntimeContextSpec | None = None, *, cache: CacheMode = "weak"):
        """Create the generic Execute setup required to invoke this prepared call.

        Args:
            runtime: Optional detached worker runtime specification. ``None`` uses
                the dependency-light inline core runtime.
            cache: Detached core session cache policy installed before invocation.

        Returns:
            A generic ``WorkerSetup`` which opens only the frozen Store table in
            the receiving worker.

        Raises:
            ValueError: If this call did not retain a complete shared Store setup.

        Side Effects:
            None. Factory resolution and Store reconstruction remain worker-local.
        """
        from dryml.execute.models import WorkerSetup

        spec = RuntimeContextSpec(RuntimeMode.INLINE) if runtime is None else runtime
        if not isinstance(spec, RuntimeContextSpec):
            raise TypeError("runtime must be a RuntimeContextSpec or None")
        if cache not in {"none", "weak", "strong"}:
            raise ValueError("cache must be 'none', 'weak', or 'strong'")
        setup = self.storage_setup
        if set(setup) != {"repo", "control_store"} or not isinstance(setup["repo"], Mapping):
            raise ValueError("prepared core call has no complete shared Store setup")
        payload = {
            "runtime": spec.to_data(), "repo": setup["repo"], "role": "main",
            "replica": 0, "control_store": setup["control_store"], "cache": cache,
        }
        envelope = make_envelope(
            schema=_SETUP_SCHEMA, kind=_SETUP_KIND, prefix=_SETUP_PREFIX,
            payload=payload,
            semantic_id=semantic_id(
                _SETUP_PREFIX, _SETUP_SCHEMA, _SETUP_KIND, payload,
                **_SETUP_BOUNDS,
            ),
            **_SETUP_BOUNDS,
        )
        return WorkerSetup(factory="dryml.core.execute:core_worker_setup", data=envelope)

class MarshallingStrategy(Protocol):
    """Own storage eligibility and future core call/result transport mechanics."""

    def validate(self, *, repo: Repo | RepoDefinition | None, control_store: DirStore | None) -> None:
        """Validate strategy-specific storage eligibility without mutation."""

    def prepare(
            self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any],
            *, repo: Repo | None, control_store: DirStore | None,
            update_args: bool) -> PreparedCoreCall:
        """Prepare one detached call through the strategy's bounded codec."""

    def invoke(self, invocation: bytes, *, repo: Repo | None, update_args: bool) -> bytes:
        """Invoke strategy bytes after worker setup."""

    def recover(
            self, result: bytes, prepared: PreparedCoreCall, *, repo: Repo | None,
            args: tuple[Any, ...], kwargs: Mapping[str, Any],
            return_objects: bool, update_args: bool) -> Any:
        """Recover strategy output after worker completion."""


class SharedDirStoreStrategy:
    """Initial marshalling strategy for existing directly shared DirStore authority.

    The strategy freezes directly reopenable ``DirStore`` descriptors once for
    each prepared call. Its result adapter remains unavailable until result
    publication owns recovery of live core values.
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
            method directly. Snapshot preparation supplies a definition so it
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

    def prepare(self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any], *,
                repo: Repo | None, control_store: DirStore | None, update_args: bool,
                selections: Mapping[Any, Any] | None = None,
                _frozen_storage: "_FrozenSharedStorage | None" = None,
                invocation_limit_bytes: int = 67_108_864) -> PreparedCoreCall:
        """Encode one inert whole-call graph without saving or activating annotations.

        ``selections`` is the optional explicit signature selection table. The
        generic backend's resolved invocation budget must be forwarded through
        ``invocation_limit_bytes``; no strategy-local one-megabyte limit applies.
        """
        if repo is None:
            raise ValueError("SharedDirStoreStrategy requires configured shared Store authority")
        from .execute_codec import encode_invocation
        frozen = _frozen_storage or _freeze_shared_storage(repo, control_store)
        if frozen.source_repo is not repo:
            raise ValueError("frozen core storage does not belong to the prepared Repo")
        bindings = _prepared_update_targets(args, kwargs) if update_args else ()
        update_targets = tuple({
            "target": uuid4().hex,
            "path": binding["path"],
            "object_digest": binding["object_digest"],
        } for binding in bindings)
        return PreparedCoreCall(
            encode_invocation(
                fn, args, kwargs, repo=repo, selections=selections,
                store_table=frozen.source_stores, limit_bytes=invocation_limit_bytes,
                update_targets=update_targets,
            ),
            frozen.storage_setup,
            update_targets,
        )

    def bind_recovery(self, prepared: PreparedCoreCall, *, args: tuple[Any, ...],
                      kwargs: Mapping[str, Any]) -> _CoreRecovery:
        """Create coordinator-only refresh state before caller mutation.

        This internal adapter seam returns the sole object retaining caller
        Objects. The future owns it for its lifetime and never serializes it,
        attaches it to ``prepared``, or registers it globally.
        """
        return _CoreRecovery.bind(prepared, args, kwargs)

    def invoke(self, invocation: bytes, *, repo: Repo | None, update_args: bool,
               invocation_limit_bytes: int = 67_108_864,
               result_limit_bytes: int | None = None) -> bytes:
        """Run one reconstructed worker-local call through its single signature boundary."""
        if repo is None:
            raise ValueError("SharedDirStoreStrategy requires worker Repo authority")
        from .execute_codec import invoke_invocation
        from dryml.managed import ManagedConfig
        try:
            control_store = current_context().control_store
        except RuntimeError:
            control_store = None
        return invoke_invocation(
            invocation, repo=repo, invocation_limit_bytes=invocation_limit_bytes,
            result_limit_bytes=result_limit_bytes or invocation_limit_bytes,
            managed_config=ManagedConfig(state_repo=repo, control_store=control_store),
            update_args=update_args,
        )

    def recover(
            self, result: bytes, prepared: PreparedCoreCall, *, repo: Repo | None,
            args: tuple[Any, ...], kwargs: Mapping[str, Any], return_objects: bool,
            update_args: bool, _recovery: _CoreRecovery | None = None,
            _decoded: CoreAdaptationOutcome | None = None,
            result_limit_bytes: int = 67_108_864) -> Any:
        """Recover one tagged outcome and optionally restore original arguments.

        Args:
            result: Bounded worker outcome bytes from :meth:`invoke`.
            prepared: The immutable submission-local handoff retained by the caller.
            repo: Caller-owned recovery Repo.
            args: Original caller positional arguments used only for exact refresh.
            kwargs: Original caller keyword arguments used only for exact refresh.
            return_objects: Whether reference results should be materialized locally.
            update_args: Whether published update StateRefs should be restored into
                the original live argument instances.
            result_limit_bytes: Frozen generic transport budget used to validate
                this delivered outcome.

        Returns:
            The decoded result, or its requested fresh local materialization.

        Raises:
            CoreCallCodecError: If the delivered outcome is malformed or the worker
                reported a failed invocation/publication.
            RepoLoadError: If exact result recovery or requested in-place refresh
                fails. Earlier refreshes remain applied and no refresh is retried.

        Side Effects:
            May materialize result references and restores only caller objects
            explicitly associated with a worker-published update snapshot.
        """
        if repo is None:
            raise ValueError("SharedDirStoreStrategy requires caller Repo authority")
        decoded = _decoded or decode_core_outcome(
            result, repo=repo, result_limit_bytes=result_limit_bytes,
        )
        outcome = decoded.value
        if not outcome["success"]:
            raise CoreExecutionError(
                f"core execution worker outcome failed: {outcome['reason']}",
                phase="invoke", evidence=decoded.evidence,
            )
        updates = tuple((item["target"], StateRef.from_data(item["state"])) for item in outcome["updates"])
        recovery = _recovery or self.bind_recovery(prepared, args=args, kwargs=kwargs)
        targets = recovery.targets
        if updates and not update_args:
            raise CoreCallCodecError("core execution outcome has unexpected update targets")
        if any(target not in targets for target, _ in updates):
            raise CoreCallCodecError("core execution outcome has unassociated update targets")
        recovered = outcome["result"]
        try:
            if return_objects:
                recovered = _recover_result_objects(
                    recovered, repo, outcome["automatic_references"],
                )
        except Exception as error:
            if isinstance(error, CoreExecutionError):
                raise
            raise CoreExecutionError(
                "core execution result recovery failed", phase="recover",
                evidence=decoded.evidence,
            ) from error
        if not update_args:
            return recovered
        ledger = recovery.ledger
        if ledger.error is not None:
            raise ledger.error
        ledger.begin(updates)
        if any(entry.status != "pending" for entry in ledger.entries):
            return recovered
        # Complete result recovery and every non-mutating authority preflight before
        # the first caller object can be restored.
        for index, (_, state) in enumerate(updates):
            try:
                repo.load_state_ref(state, reuse_live="never")
            except Exception as error:
                ledger.replace(index, "preflight_failed")
                for later in range(index + 1, len(ledger.entries)):
                    ledger.replace(later, "skipped")
                ledger.error = CoreExecutionError(
                    "core execution refresh preflight failed", phase="refresh",
                    evidence=ledger.evidence(decoded.evidence),
                )
                raise ledger.error from error
        for index, (token, state) in enumerate(updates):
            try:
                repo.restore_state_ref_into(targets[token], state)
            except Exception as error:
                ledger.replace(index, "invalidated" if getattr(targets[token], "_restore_failed", False) else "preflight_failed")
                for later in range(index + 1, len(ledger.entries)):
                    ledger.replace(later, "skipped")
                ledger.error = CoreExecutionError(
                    "core execution refresh failed", phase="refresh",
                    evidence=ledger.evidence(decoded.evidence),
                )
                raise ledger.error from error
            ledger.replace(index, "applied")
        return recovered


def decode_core_outcome(
        result: bytes, *, repo: Repo, result_limit_bytes: int = 67_108_864,
) -> CoreAdaptationOutcome:
    """Decode a tagged outcome into a value plus detached exact evidence.

        Args:
            result: Bounded bytes returned by :class:`SharedDirStoreStrategy`.
            repo: Caller-owned Repo used only to validate the closed result graph.
            result_limit_bytes: Frozen generic outcome byte budget.

    Returns:
        A future-retainable value/evidence pair. Failed outcomes retain ``value`` as
        their closed transport mapping so callers can inspect its bounded reason.

    Raises:
        CoreCallCodecError: If the transport outcome or evidence grammar is
            malformed. No caller object is materialized or refreshed here.

    Side Effects:
        Reads the supplied Repo only while decoding authority graph nodes; it does
        not save, restore, or close any caller-owned resource.
    """
    from .execute_codec import CoreCallCodecError, decode_outcome

    outcome = decode_outcome(result, repo=repo, limit_bytes=result_limit_bytes)
    publications = []
    publication_statuses = {"completed", "failed", "uncertain", "unattempted"}
    for item in outcome["publications"]:
        if not isinstance(item, Mapping) or set(item) != {"state", "store", "phase", "status", "path"}:
            raise CoreCallCodecError("core execution transport rejected malformed publication evidence")
        if (isinstance(item["store"], bool) or not isinstance(item["store"], int)
                or not 0 <= item["store"] < len(repo.stores)
                or not all(isinstance(item[name], str) for name in ("phase", "status", "path"))
                or item["status"] not in publication_statuses):
            raise CoreCallCodecError("core execution transport rejected malformed publication evidence")
        try:
            state = StateRef.from_data(item["state"])
        except (TypeError, ValueError) as error:
            raise CoreCallCodecError("core execution transport rejected malformed publication evidence") from error
        publications.append(CorePublicationEvidence(state, item["store"], item["phase"], item["status"], item["path"]))
    updates = []
    for item in outcome["updates"]:
        if not isinstance(item, Mapping) or set(item) != {"state", "object", "target"}:
            raise CoreCallCodecError("core execution transport rejected malformed update evidence")
        try:
            state = StateRef.from_data(item["state"])
        except (TypeError, ValueError) as error:
            raise CoreCallCodecError("core execution transport rejected malformed update evidence") from error
        if state.object.to_data() != item["object"] or not isinstance(item["target"], str) or not item["target"]:
            raise CoreCallCodecError("core execution transport rejected mismatched update evidence")
        updates.append(state)
    if len({item["target"] for item in outcome["updates"]}) != len(updates):
        raise CoreCallCodecError("core execution transport rejected duplicate update target")
    return CoreAdaptationOutcome(
        outcome,
        CoreOutcomeEvidence(
            tuple(publications), tuple(updates),
            tuple(CoreRefreshEvidence(item["target"], state, "pending") for item, state in zip(outcome["updates"], updates)),
        ),
    )


def _prepared_update_targets(
        args: tuple[Any, ...], kwargs: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Detach first-appearance live argument associations without saving or selecting.

    These internal records bind recovery arguments to prepared descriptors. The
    worker still selects only delivered materialized values, so a Ref position or
    callable capture cannot create a refresh merely by appearing here.
    """
    from .object import Object

    targets: list[Mapping[str, Any]] = []
    seen: set[int] = set()

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Object):
            if id(value) not in seen:
                seen.add(id(value))
                targets.append({"path": path, "object_digest": value.object_ref.digest(), "object": value})
            return
        if isinstance(value, Mapping):
            if id(value) in seen:
                return
            seen.add(id(value))
            for index, (key, item) in enumerate(value.items()):
                visit(key, f"{path}.key[{index}]")
                visit(item, f"{path}.value[{index}]")
        elif isinstance(value, (tuple, list)):
            if id(value) in seen:
                return
            seen.add(id(value))
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(args, "args")
    visit(kwargs, "kwargs")
    return tuple(targets)


def _recover_result_objects(value: Any, repo: Repo, automatic_references: frozenset[int],
                            memo: dict[int, Any] | None = None) -> Any:
    """Materialize only Execute-created result references in one aggregate boundary.

    Explicit ``Ref[...]`` leaves are intentionally ordinary data. The aggregate
    materialization call gives every auto-converted leaf one fresh graph memo, so
    repeated and diamond result aliases stay shared without reusing caller inputs.
    """
    from .definition import ConcreteDefinition
    from .reference_values import ObjectRef, StateRef

    references: list[Any] = []
    seen_references: set[int] = set()

    def collect(item: Any, seen: set[int]) -> None:
        if isinstance(item, (ConcreteDefinition, ObjectRef, StateRef)):
            if id(item) in automatic_references and id(item) not in seen_references:
                references.append(item)
                seen_references.add(id(item))
            return
        if isinstance(item, (tuple, list, Mapping)):
            if id(item) in seen:
                return
            seen.add(id(item))
            if isinstance(item, Mapping):
                for key, child in item.items():
                    collect(key, seen)
                    collect(child, seen)
            else:
                for child in item:
                    collect(child, seen)

    collect(value, set())
    materialized = dict(zip((id(item) for item in references), repo.materialize_boundary(
        tuple(references), reuse_live="never",
    )))
    memo = {} if memo is None else memo

    def replace(item: Any) -> Any:
        if isinstance(item, (ConcreteDefinition, ObjectRef, StateRef)):
            return materialized.get(id(item), item)
        if isinstance(item, tuple):
            if id(item) not in memo:
                memo[id(item)] = tuple(replace(child) for child in item)
            return memo[id(item)]
        if isinstance(item, list):
            if id(item) not in memo:
                memo[id(item)] = [replace(child) for child in item]
            return memo[id(item)]
        if isinstance(item, Mapping):
            if id(item) not in memo:
                memo[id(item)] = {replace(key): replace(child) for key, child in item.items()}
            return memo[id(item)]
        return item

    return replace(value)


def invoke_prepared_call(invocation: bytes) -> Any:
    """Invoke prepared core bytes inside an already-installed generic worker setup.

    Args:
        invocation: One bounded :class:`PreparedCoreCall` invocation byte string.

    Returns:
        The decoded ordinary worker result.

    Raises:
        RuntimeError: If no active core worker context owns a reconstructed Repo.

    Side Effects:
        Invokes the transported workload once through that worker's core boundary.
    """
    context = current_context()
    from .execute_codec import CoreCallCodecError, decode_outcome

    outcome = decode_outcome(
        SharedDirStoreStrategy().invoke(invocation, repo=context.repo, update_args=False),
        repo=context.repo,
    )
    if not outcome["success"]:
        raise CoreCallCodecError(f"core execution worker outcome failed: {outcome['reason']}")
    return outcome["result"]


def _invoke_prepared_outcome(
        invocation: bytes, invocation_limit_bytes: int, result_limit_bytes: int,
        update_args: bool,
) -> bytes:
    """Return one opaque core outcome from an already configured worker.

    Generic Execute transports only this byte result.  Decoding and recovery stay
    in the submission coordinator, where the frozen caller-side Store snapshot is
    available and no live core value crosses the generic result boundary.

    Args:
        invocation: Prepared strategy invocation bytes.
        invocation_limit_bytes: Captured public backend invocation bound.
        result_limit_bytes: Captured public backend outcome bound.
        update_args: Captured caller-refresh policy.

    Returns:
        The strategy-owned tagged outcome bytes.

    Raises:
        RuntimeError: If called outside the core worker setup scope.
    """
    context = current_context()
    return SharedDirStoreStrategy().invoke(
        invocation, repo=context.repo, update_args=update_args,
        invocation_limit_bytes=invocation_limit_bytes,
        result_limit_bytes=result_limit_bytes,
    )


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

    Side Effects:
        None. This immutable value owns no runtime, Repo, Store, or worker.
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


@dataclass(frozen=True, slots=True)
class _FrozenCoreControls:
    """
    Inert call-entry core controls retained until guarded preparation.

        This package-private value is captured before an external probe can
        run. It
        retains only resolved option values and the entry orchestration floor;
        it does
        not export, open, or otherwise touch selected Store authority.
    """

    effective: _EffectiveCoreOptions
    call_entry_orchestrator: bool


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


def _capture_frozen_core_controls(
        core: CoreOptions | None, *, executor_core: CoreOptions | None,
) -> _FrozenCoreControls:
    """Capture ambient core controls before external preflight work.

    Args:
        core: Inert per-call overrides, if any.
        executor_core: Inert reusable defaults below ``core``, if any.

    Returns:
        A package-private call-entry snapshot suitable for
        :func:`_prepare_frozen_core_submission`.

    Raises:
        TypeError, ValueError: If effective options are invalid at call entry.

    Side Effects:
        Reads immutable session/runtime controls once. It never exports, opens,
        saves, closes, or otherwise accesses a Repo or Store.
    """

    return _FrozenCoreControls(
        resolve_core_options(core, executor=executor_core),
        active_runtime().mode is RuntimeMode.ORCHESTRATOR,
    )


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


@dataclass(frozen=True, slots=True)
class _FrozenSharedStorage:
    """One submission's detached Store table and its originating handle identities."""

    source_repo: Repo
    source_stores: tuple[DirStore, ...]
    definition: RepoDefinition
    storage_setup: Mapping[str, Any]


def _freeze_shared_storage(repo: Repo, control_store: DirStore | None) -> _FrozenSharedStorage:
    """Export one Repo once and retain the exact table used for subsequent pins.

    The handle tuple is intentionally private and exists only while coordinator
    preparation encodes selected Store indexes. Worker setup receives the detached
    definition and never observes the caller's handles.
    """
    with repo._configuration_lock:
        if repo._closing or repo._closed:
            raise ValueError("shared Store authority is unavailable")
        definition = repo.to_definition()
        source_stores = _definition_store_table(definition, tuple(repo.stores))
        setup = _freeze_storage_setup(_shared_storage_setup(definition, control_store))
    return _FrozenSharedStorage(repo, source_stores, definition, setup)


def _definition_store_table(
        definition: RepoDefinition, stores: tuple[object, ...],
) -> tuple[DirStore, ...]:
    """Associate source handles with immutable definition descriptors by identity.

    The caller captures ``stores`` while its Repo configuration is locked.  The
    returned table follows the frozen definition rather than a later mutable Repo
    order, so pinned selection indexes always address the worker's matching Store.
    """
    handles: dict[tuple[int, int], DirStore] = {}
    for store in stores:
        if type(store) is not DirStore:
            raise ValueError("SharedDirStoreStrategy requires only configured DirStores")
        identity = _store_identity(store.base_dir)
        if identity in handles:
            raise ValueError("shared Store definition has duplicate physical authority")
        handles[identity] = store
    try:
        descriptors = definition.to_data()["stores"]
        return tuple(handles[_store_identity(descriptor["path"])] for descriptor in descriptors)
    except (KeyError, TypeError, ValueError):
        raise ValueError("shared Store handles do not match the frozen definition") from None


@dataclass(slots=True)
class _PreparedSharedStorage:
    """Submission-owned recovery Repo plus detached worker storage role data."""

    storage_setup: Mapping[str, Any]
    recovery_repo: Repo
    frozen_storage: _FrozenSharedStorage | None
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


@dataclass(slots=True)
class _FrozenCoreSubmission:
    """
    One owned, validated core payload awaiting generic backend acceptance.

        Args:
            storage: Owned recovery reconstruction and frozen storage
            descriptors.
            prepared: Detached invocation bytes and update descriptors.
            recovery: Coordinator-only caller-object associations.
            effective: Call-entry resolved core controls.
            result_limit_bytes: Frozen generic result budget after setup
            overhead.
            return_objects: Call-entry result policy after the orchestration
            floor.
            call_entry_orchestrator: Whether the caller entered under the
            strict floor.

        This package-internal value is the only handoff between guarded core
        payload
        preparation and backend submission.  Its owner must either transfer it
        to a
        ``CoreExecutionFuture`` or call :meth:`close` after a failed
        acceptance.
    """

    storage: _PreparedSharedStorage
    prepared: PreparedCoreCall
    recovery: _CoreRecovery
    effective: _EffectiveCoreOptions
    result_limit_bytes: int
    return_objects: bool
    call_entry_orchestrator: bool
    _state: str = field(default="pending", init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def close(self) -> None:
        """
        Release only preparation-owned recovery resources after non-acceptance.
        """

        with self._lock:
            if self._state in {"accepted", "closed"}:
                return
            self.storage.close()
            self._state = "closed"

    def _begin_acceptance(self) -> None:
        """Claim this payload for exactly one backend acceptance attempt."""

        with self._lock:
            if self._state != "pending":
                raise RuntimeError(
                    "frozen core submission has already been consumed")
            self._state = "submitting"

    def _accepted(self) -> None:
        """Transfer recovery ownership to the returned core future."""

        with self._lock:
            if self._state != "submitting":
                raise RuntimeError(
                    "frozen core submission acceptance state is invalid")
            self._state = "accepted"


def prepare_shared_storage(
        core: CoreOptions | None = None, *, executor: CoreOptions | None = None,
        _effective: _EffectiveCoreOptions | None = None,
) -> _PreparedSharedStorage:
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
    effective = resolve_core_options(core, executor=executor) if _effective is None else _effective
    strategy = effective.marshalling()
    selected = effective.repo
    if selected is None:
        strategy.validate(repo=None, control_store=effective.control_store)
    if isinstance(selected, Repo):
        frozen = _freeze_shared_storage(selected, effective.control_store)
        definition = frozen.definition
    elif isinstance(selected, RepoDefinition):
        definition = selected
    else:
        raise TypeError("core repo must resolve to Repo, RepoDefinition, or None")
    # Validation consumes the detached definition, preserving the one-export cut.
    strategy.validate(repo=definition, control_store=effective.control_store)
    if not isinstance(selected, Repo):
        # A detached definition is reconstructed once. Its owned handles serve as
        # both preparation pins and recovery authority for this submission.
        setup = _freeze_storage_setup(_shared_storage_setup(definition, effective.control_store))
        recovery_repo = Repo.from_definition(definition)
        try:
            with recovery_repo._configuration_lock:
                source_stores = _definition_store_table(definition, tuple(recovery_repo.stores))
            frozen = _FrozenSharedStorage(recovery_repo, source_stores, definition, setup)
        except BaseException:
            recovery_repo.close(flush=False)
            raise
    else:
        setup = frozen.storage_setup
        recovery_repo = Repo.from_definition(definition)
    return _PreparedSharedStorage(
        setup,
        recovery_repo,
        frozen,
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


def _decode_setup(data: Mapping[str, Any]) -> tuple[RuntimeContextSpec, RepoDefinition, str, int, Mapping[str, Any] | None, CacheMode]:
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
    if set(payload) not in {_SETUP_FIELDS, _SETUP_FIELDS_WITH_CACHE}:
        raise ValueError("core worker setup payload fields are closed")
    runtime = payload["runtime"]
    repo = payload["repo"]
    role = payload["role"]
    replica = payload["replica"]
    control_store = payload["control_store"]
    cache = payload.get("cache", "weak")
    if not isinstance(runtime, Mapping) or not isinstance(repo, Mapping):
        raise TypeError("core worker setup runtime and Repo definitions must be envelopes")
    if not isinstance(role, str) or not role or isinstance(replica, bool) or not isinstance(replica, int) or replica < 0:
        raise ValueError("core worker role and replica are invalid")
    if control_store is not None and not isinstance(control_store, Mapping):
        raise TypeError("core worker control Store descriptor must be a mapping or null")
    if cache not in {"none", "weak", "strong"}:
        raise ValueError("core worker cache policy is invalid")
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
    return RuntimeContextSpec.from_data(runtime), RepoDefinition.from_data(repo), role, replica, control_store, cache


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
    spec, definition, role, replica, descriptor, cache = _decode_setup(data)
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
        stack.enter_context(config(repo=repo, cache=cache))
        stack.enter_context(worker_context(ExecutionContext(repo, control)))
        yield None


@dataclass(frozen=True, slots=True)
class CoreExecutionSnapshot:
    """Immutable combined backend and core-adaptation observation.

    Args:
        backend: The nested generic execution snapshot.
        state: Core lifecycle state, including the adaptation interval.
        phase: Current or terminal core phase.
        cleanup_state: Aggregate generic/recovery-resource cleanup state.
        cleanup_issues: Independent cleanup observations.
        evidence: Detached worker publication and refresh evidence, when delivered.

    The value contains no caller Repo, Store, arguments, or recovered result.
    Cleanup observations may change after a successful adapted result is stored.
    """

    backend: ExecutionSnapshot
    state: str
    phase: str
    cleanup_state: str
    cleanup_issues: tuple[ExecutionIssue, ...]
    evidence: CoreOutcomeEvidence | None


CoreCallback: TypeAlias = Callable[["CoreExecutionFuture"], None]


class CoreExecutionFuture:
    """Adapt one public generic byte Future into a recovered core outcome.

    Only :class:`Executor` and the module one-off helpers construct this facade.
    The generic Future remains the backend owner; this facade registers one public
    completion callback, recovers the byte outcome once, and retains only the
    submission-owned recovery Repo until cleanup.

    Args:
        backend_future: Accepted public generic Future carrying outcome bytes.
        prepared_storage: Owned caller-side recovery snapshot.
        prepared: Detached invocation/storage handoff retained for recovery.
        recovery: Caller-object bindings created before submission returns.
        return_objects: Captured result-materialization decision.
        result_limit_bytes: Captured public generic result limit.
        done_callbacks: Coordinator-local callbacks receiving this facade.
        one_off: Whether generic one-off cleanup should also release recovery state.

    Returns:
        A future facade whose result methods yield a recovered core value rather
        than the generic byte outcome.

    Raises:
        TypeError: If construction receives a non-generic future or invalid
            callback sequence. Public callers receive failures from result,
            recovery, or cleanup methods instead.

    Side Effects:
        Registers one generic completion callback and owns the supplied recovery
        snapshot until cleanup. It borrows ``backend_future`` and never closes a
        caller Repo or Store. Wait timeouts affect only the waiting caller;
        callback and cleanup failures never overwrite an adapted result.
    """

    def __init__(
            self, backend_future: ExecutionFuture[bytes], *,
            prepared_storage: _PreparedSharedStorage, prepared: PreparedCoreCall,
            recovery: _CoreRecovery, return_objects: bool, result_limit_bytes: int,
            done_callbacks: Sequence[CoreCallback] = (), one_off: bool = False,
    ) -> None:
        if not isinstance(backend_future, ExecutionFuture):
            raise TypeError("backend_future must be an ExecutionFuture")
        callbacks = tuple(done_callbacks)
        if not all(callable(callback) for callback in callbacks):
            raise TypeError("done_callbacks entries must be callable")
        self._backend_future = backend_future
        self._storage = prepared_storage
        self._prepared = prepared
        self._recovery = recovery
        self._return_objects = return_objects
        self._result_limit_bytes = result_limit_bytes
        self._condition = Condition(RLock())
        self._state = "pending"
        self._phase = "prepare"
        self._outcome: Any = _MISSING
        self._outcome_kind: str | None = None
        self._evidence: CoreOutcomeEvidence | None = None
        self._callbacks = list(callbacks)
        self._callback_queue: list[CoreCallback] = []
        self._cleanup_state = "pending"
        self._cleanup_issues: list[ExecutionIssue] = []
        self._diagnostic_text_limit_bytes = backend_future._diagnostic_text_limit_bytes
        self._diagnostic_issue_limit = backend_future._diagnostic_issue_limit
        self._cleanup_timeout = backend_future._termination_timeout
        self._storage_cleanup_thread: Thread | None = None
        self._storage_cleanup_done = False
        self._generic_cleanup_complete = False
        self._one_off = one_off
        backend_future.add_done_callback(self._backend_completed)

    @property
    def backend_future(self) -> ExecutionFuture[bytes]:
        """Return borrowed advanced inspection access to the generic byte Future.

        Returns:
            The public generic future carrying internal tagged outcome bytes.

        Side Effects:
            None. The core facade retains generic-future ownership boundaries.
        """
        return self._backend_future

    @property
    def submission_id(self) -> str:
        """Return the immutable backend-assigned submission identifier.

        Returns:
            The opaque identifier assigned by the generic backend.

        Side Effects:
            None. Reading the identifier does not start, cancel, or clean work.
        """
        return self._backend_future.submission_id

    @property
    def output(self) -> ExecutionOutput:
        """Return the generic output owner retained across core adaptation.

        Returns:
            The caller-owned generic output holder for this submission.

        Side Effects:
            None. Output remains available after facade cleanup.
        """
        return self._backend_future.output

    def done(self) -> bool:
        """Return whether core recovery has produced one stable adapted outcome."""
        with self._condition:
            return self._outcome is not _MISSING

    def running(self) -> bool:
        """Return whether generic execution or coordinator adaptation is active."""
        backend = self._backend_future.snapshot()
        with self._condition:
            return self._outcome is _MISSING and (
                backend.state == "running" or self._state == "adapting"
            )

    def cancelled(self) -> bool:
        """Return whether the backend confirmed cancellation before core adaptation."""
        if self._backend_future.cancelled():
            return True
        with self._condition:
            return self._state == "cancelled"

    def snapshot(self) -> CoreExecutionSnapshot:
        """Return nested backend, adaptation, evidence, and cleanup observations."""
        backend = self._backend_future.snapshot()
        with self._condition:
            issues = tuple(backend.cleanup_issues) + tuple(self._cleanup_issues)
            cleanup_state = self._cleanup_state
            if cleanup_state == "pending" and backend.cleanup_state != "pending":
                cleanup_state = backend.cleanup_state
            state = self._state
            if self._outcome is _MISSING:
                if backend.state in {"pending", "admitting", "running"}:
                    state = backend.state
                elif state != "adapting":
                    state = "adapting"
            return CoreExecutionSnapshot(
                backend=backend, state=state, phase=self._phase,
                cleanup_state=cleanup_state, cleanup_issues=issues,
                evidence=self._evidence,
            )

    def result(self, timeout: float | None = None) -> Any:
        """Wait for and return the recovered result or raise its stable error.

        Args:
            timeout: Optional finite nonnegative caller wait in seconds.

        Returns:
            The one recovered core result.

        Raises:
            TimeoutError: If this caller's wait expires.
            BaseException: The stored backend, publication, or recovery failure.

        Side Effects:
            May join or start the single adaptation after generic terminality.
            ``timeout`` never cancels worker or adaptation work.
        """
        outcome = self._wait(timeout)
        with self._condition:
            kind = self._outcome_kind
        if kind in {"error", "cancelled"}:
            assert isinstance(outcome, BaseException)
            raise outcome
        return outcome

    def exception(self, timeout: float | None = None) -> BaseException | None:
        """Wait for and return the core-adapted failure without changing work.

        Args:
            timeout: Optional finite nonnegative caller wait in seconds.

        Returns:
            The stored failure, or ``None`` for a recovered success.

        Raises:
            TimeoutError: If this caller's wait expires.
            BaseException: The stored cancellation outcome.

        Side Effects:
            May join the one adaptation; it never retries execution or refresh.
        """
        outcome = self._wait(timeout)
        with self._condition:
            kind = self._outcome_kind
        if kind == "cancelled":
            assert isinstance(outcome, BaseException)
            raise outcome
        return outcome if kind == "error" else None

    def add_done_callback(self, callback: CoreCallback) -> None:
        """Schedule one callback after core adaptation, outside synchronization.

        Args:
            callback: Callable receiving this core future after adaptation.

        Raises:
            TypeError: If ``callback`` is not callable.

        Side Effects:
            Runs callbacks asynchronously and isolates their failures from the
            stored outcome. A callback does not own execution or cleanup.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._condition:
            if self._outcome is _MISSING:
                self._callbacks.append(callback)
                pending = True
            else:
                pending = False
        if pending:
            # If generic callback delivery was queued after a transient thread
            # launch failure, this consumer starts the same guarded adaptation.
            if self._backend_future.done():
                self._backend_completed(self._backend_future)
            return
        self._dispatch_callbacks((callback,))

    def cancel(self) -> bool:
        """Delegate truthful pre-GO cancellation to the owned generic Future.

        Returns:
            ``True`` only when generic execution confirms cancellation before GO.

        Side Effects:
            Does not claim cancellation of running work or alter stored evidence.
        """
        return self._backend_future.cancel()

    def request_cancel(self) -> bool:
        """Delegate a running-work cancellation request without fabricating success.

        Returns:
            Whether the generic backend accepted a best-effort request.

        Raises:
            ExecutionError: If the backend cannot accept the request.

        Side Effects:
            Requests backend cancellation only; terminality and cleanup remain
            independently observable.
        """
        return self._backend_future.request_cancel()

    def cleanup(self, timeout: float | None = None) -> None:
        """Join generic cleanup and release this facade's owned recovery Repo.

        Raises:
            RuntimeError: If adaptation has not reached terminality.
            CleanupError: If generic or recovery-resource cleanup remains incomplete.

        Side Effects:
            Starts at most one owned recovery-Repo close and bounds this caller's
            join of that synchronous Store operation. A close that outlives the
            caller budget remains observable as incomplete for a later join.
        """
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout must be a finite positive number of seconds")
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be a finite positive number of seconds")
        deadline = time.monotonic() + (self._cleanup_timeout if timeout is None else float(timeout))
        with self._condition:
            if self._outcome is _MISSING:
                raise RuntimeError("cleanup requires a terminal core execution outcome")
            if self._cleanup_state == "complete":
                return
            if self._cleanup_state == "reconciling":
                # The generic Future owns bounded reconciliation. A concurrent
                # caller joins that work rather than closing the recovery Repo twice.
                while self._cleanup_state == "reconciling":
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise CleanupError("core execution cleanup is still running", execution=self)
                    self._condition.wait(remaining)
                if self._cleanup_state == "complete":
                    return
                raise CleanupError("core execution cleanup remains incomplete", execution=self)
            self._cleanup_state = "reconciling"
        failure: BaseException | None = None
        try:
            remaining = self._cleanup_remaining(deadline)
            self._backend_future.cleanup(timeout=remaining)
            self._generic_cleanup_complete = self._backend_future.snapshot().cleanup_state == "complete"
        except BaseException as error:
            failure = error
        storage_failure = self._start_or_join_storage_cleanup(deadline)
        if storage_failure is not None and failure is None:
            failure = storage_failure
        with self._condition:
            if self._storage_cleanup_thread is None:
                self._cleanup_state = "complete" if (
                    failure is None and self._generic_cleanup_complete and self._storage_cleanup_done
                ) else "incomplete"
                self._condition.notify_all()
        if failure is not None:
            if isinstance(failure, CleanupError):
                raise failure
            raise CleanupError("core recovery cleanup failed", execution=self) from failure

    def _cleanup_remaining(self, deadline: float | None) -> float | None:
        """Return this caller's remaining cleanup budget without inventing one."""
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CleanupError("core execution cleanup timed out", execution=self)
        return remaining

    @staticmethod
    def _wait_remaining(deadline: float | None) -> float | None:
        """Return a Future wait budget without converting expiry into cleanup state."""
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    def _start_or_join_storage_cleanup(self, deadline: float | None) -> BaseException | None:
        """Bound joining the synchronous recovery-Repo close without abandoning it.

        Store close remains a synchronous Store operation in its own owner thread;
        this facade only bounds callers joining that operation.  A later cleanup
        joins the same close rather than opening a second close attempt.
        """
        with self._condition:
            thread = self._storage_cleanup_thread
            if not self._storage_cleanup_done and thread is None:
                thread = Thread(
                    target=self._close_storage, name="dryml-core-execute-cleanup", daemon=True,
                )
                self._storage_cleanup_thread = thread
                try:
                    thread.start()
                except BaseException:
                    self._storage_cleanup_thread = None
                    self._append_cleanup_issue_locked()
                    return CleanupError("core recovery cleanup could not start", execution=self)
        while thread is not None and thread.is_alive():
            remaining = self._cleanup_remaining(deadline)
            thread.join(remaining)
            if thread.is_alive() and deadline is not None and time.monotonic() >= deadline:
                return CleanupError("core recovery cleanup is still running", execution=self)
        with self._condition:
            return None if self._storage_cleanup_done else CleanupError(
                "core recovery cleanup remains incomplete", execution=self,
            )

    def _close_storage(self) -> None:
        """Release the facade-owned reconstruction handles without blocking callers."""
        try:
            self._storage.close()
        except BaseException:
            with self._condition:
                self._storage_cleanup_thread = None
                self._append_cleanup_issue_locked()
                self._cleanup_state = "incomplete"
                self._condition.notify_all()
        else:
            with self._condition:
                self._storage_cleanup_thread = None
                self._storage_cleanup_done = True
                self._cleanup_state = "complete" if self._generic_cleanup_complete else "incomplete"
                self._condition.notify_all()

    def _append_cleanup_issue_locked(self) -> None:
        """Record one configured-bounded, non-sensitive facade cleanup failure."""
        backend_issues = len(self._backend_future.snapshot().cleanup_issues)
        if backend_issues + len(self._cleanup_issues) >= self._diagnostic_issue_limit:
            return
        message = "cleanup failure".encode("utf-8")[:self._diagnostic_text_limit_bytes].decode("utf-8", errors="ignore")
        self._cleanup_issues.append(ExecutionIssue("core_recovery_cleanup_failed", message))

    def __await__(self) -> Generator[Any, None, Any]:
        """Await the same single adapted outcome without cancelling backend work."""
        return self._await_result().__await__()

    async def _await_result(self) -> Any:
        """Bridge the coordinator callback into the current event loop."""
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[Any] = loop.create_future()

        def completed(future: CoreExecutionFuture) -> None:
            def deliver() -> None:
                if waiter.done():
                    return
                try:
                    waiter.set_result(future.result())
                except BaseException as error:
                    waiter.set_exception(error)
            try:
                loop.call_soon_threadsafe(deliver)
            except RuntimeError:
                pass

        self.add_done_callback(completed)
        try:
            return await asyncio.shield(waiter)
        finally:
            self._remove_callback(completed)
            if not waiter.done():
                waiter.cancel()

    def _backend_completed(self, _: ExecutionFuture[bytes]) -> None:
        """Run exactly one coordinator adaptation after generic terminality."""
        with self._condition:
            if self._outcome is not _MISSING or self._state == "adapting":
                return
            self._state = "adapting"
            self._phase = "recover"
        phase = "backend"
        try:
            data = self._backend_future.result()
            if not isinstance(data, bytes):
                raise TypeError("generic core transport result must be bytes")
            phase = "recover"
            decoded = decode_core_outcome(
                data, repo=self._storage.recovery_repo,
                result_limit_bytes=self._result_limit_bytes,
            )
            # Recovery runs in generic callback threads, which do not inherit the
            # caller's ContextVars.  Install only this frozen call's owned Repo and
            # cache policy; the scope restores the callback thread's original state.
            with config(repo=self._storage.recovery_repo, cache=self._storage.cache):
                value = SharedDirStoreStrategy().recover(
                    data, self._prepared, repo=self._storage.recovery_repo,
                    args=(), kwargs={}, return_objects=self._return_objects,
                    update_args=self._storage.update_args, _recovery=self._recovery,
                    _decoded=decoded, result_limit_bytes=self._result_limit_bytes,
                )
            with self._condition:
                self._evidence = self._recovery.ledger.evidence(decoded.evidence)
        except BaseException as error:
            if isinstance(error, CoreExecutionError):
                error.execution = self
            evidence = getattr(error, "evidence", None)
            if isinstance(evidence, CoreOutcomeEvidence):
                with self._condition:
                    self._evidence = evidence
            backend_state = self._backend_future.snapshot().state
            state = "cancelled" if self._backend_future.cancelled() else (
                "uncertain" if backend_state == "uncertain" else "failed"
            )
            terminal_phase = error.phase if isinstance(error, CoreExecutionError) else phase
            self._finish(error, state, "cancelled" if state == "cancelled" else "error", phase=terminal_phase)
        else:
            self._finish(value, "succeeded", "result", phase="refresh" if self._storage.update_args else "recover")
        if self._one_off:
            try:
                self.cleanup()
            except CleanupError:
                pass

    def _finish(self, outcome: Any, state: str, kind: str, *, phase: str) -> None:
        """Store one adapted terminal outcome and dispatch detached callbacks."""
        with self._condition:
            if self._outcome is not _MISSING:
                return
            self._outcome = outcome
            self._outcome_kind = kind
            self._state = state
            self._phase = phase
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
            self._condition.notify_all()
        self._dispatch_callbacks(callbacks)

    def _wait(self, timeout: float | None) -> Any:
        if timeout is not None:
            if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                    or not math.isfinite(timeout) or timeout < 0):
                raise ValueError("timeout must be a finite nonnegative number of seconds")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        # A generic callback can be temporarily queued when callback-thread start
        # fails.  Any core consumer therefore joins or starts the one adaptation
        # after the backend reaches terminality instead of waiting forever for a
        # later generic callback registration.
        try:
            self._backend_future.result(timeout=self._wait_remaining(deadline))
        except TimeoutError:
            raise TimeoutError("core execution did not finish before timeout") from None
        except BaseException:
            pass
        self._backend_completed(self._backend_future)
        with self._condition:
            if self._outcome is _MISSING and timeout is not None:
                while self._outcome is _MISSING:
                    remaining = self._wait_remaining(deadline)
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("core execution did not finish before timeout")
                    self._condition.wait(remaining)
            while self._outcome is _MISSING:
                self._condition.wait()
            return self._outcome

    def _dispatch_callbacks(self, callbacks: Sequence[CoreCallback]) -> None:
        """Isolate callbacks and retain launch failures for a later safe retry."""
        with self._condition:
            pending = tuple(self._callback_queue) + tuple(callbacks)
            self._callback_queue.clear()
        failed: list[CoreCallback] = []
        for callback in pending:
            for _ in range(2):
                try:
                    Thread(target=self._run_callback, args=(callback,), name="dryml-core-execute-callback", daemon=True).start()
                    break
                except BaseException:
                    pass
            else:
                failed.append(callback)
        with self._condition:
            self._callback_queue.extend(failed)

    def _remove_callback(self, callback: CoreCallback) -> None:
        """Release an undelivered await bridge after its local waiter is cancelled."""
        with self._condition:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                try:
                    self._callback_queue.remove(callback)
                except ValueError:
                    pass

    def _run_callback(self, callback: CoreCallback) -> None:
        try:
            callback(self)
        except BaseException:
            pass


def _validate_core_validators(
    validators: Sequence[Callable[[],
                                  None]]) -> tuple[Callable[[], None], ...]:
    """
    Validate and freeze coordinator preflight callbacks without invoking them.
    """

    frozen = tuple(validators)
    if not all(callable(validator) for validator in frozen):
        raise TypeError("core preparation validators must be callable")
    return frozen


def _run_core_validators(validators: Sequence[Callable[[], None]]) -> None:
    """
    Run one frozen validator sequence at a caller-defined preparation fence.
    """

    for validator in validators:
        validator()


def _check_orchestration_floor(
        effective: _EffectiveCoreOptions, *, call_entry_orchestrator: bool,
) -> None:
    """
    Preserve call-entry restrictions and reject later stricter core controls.
    """

    current_orchestrator = active_runtime().mode is RuntimeMode.ORCHESTRATOR
    if current_orchestrator and (effective.return_objects is True
                                 or effective.update_args):
        raise ValueError(
            "orchestration mode prohibits live result materialization and "
            "argument updates"
        )
    if call_entry_orchestrator and effective.return_objects is True:
        raise ValueError(
            "call entered under the orchestration result-materialization floor"
        )


def _prepare_frozen_core_submission(
    config: BackendConfig,
    executor_core: CoreOptions | None,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    kwargs: Mapping[str, Any] | None,
    core: CoreOptions | None,
    validators: Sequence[Callable[[], None]] = (),
    frozen_controls: _FrozenCoreControls | None = None,
) -> _FrozenCoreSubmission:
    """
    Freeze core controls and prepare a discardable payload before backend
    acceptance.

        Args:
            config: Generic backend configuration whose transport budgets are
            frozen.
            executor_core: Reusable core defaults beneath ``core``.
            fn: Exact callable to encode.
            args: Positional workload values.
            kwargs: Optional string-keyed workload mapping.
            core: Per-call inert core overrides.
            validators: Coordinator-owned drift guards run before preparation,
            after
                serialization, and by the paired acceptance helper.
            frozen_controls: Optional call-entry control capture. When
            supplied,
                preparation never rereads ambient session or runtime controls.

        Returns:
            One owned frozen submission.  Callers must pass it to
            :func:`_submit_frozen_core_submission` or close it on abandonment.

        Raises:
            TypeError, ValueError, CoreCallCodecError: For malformed controls,
                unavailable borrowed storage, guard failure, or transport
                preparation.

        Side Effects:
            Exports only the selected call-entry Repo and opens a fresh owned
            recovery
            Repo.  A failed guard or serialization releases that owned
            reconstruction;
            caller-owned Repo and Store handles are never closed.
    """

    if kwargs is not None and not isinstance(kwargs, Mapping):
        raise TypeError("kwargs must be a mapping or None")
    call_kwargs = {} if kwargs is None else dict(kwargs)
    if not all(isinstance(key, str) for key in call_kwargs):
        raise TypeError("kwargs keys must be strings")
    frozen_validators = _validate_core_validators(validators)
    if frozen_controls is None:
        frozen_controls = _capture_frozen_core_controls(
            core, executor_core=executor_core)
    elif type(frozen_controls) is not _FrozenCoreControls:
        raise TypeError(
            "frozen_controls must be a _FrozenCoreControls or None")
    effective = frozen_controls.effective
    call_entry_orchestrator = frozen_controls.call_entry_orchestrator
    _run_core_validators(frozen_validators)
    with publication.lease():
        _check_orchestration_floor(
            effective, call_entry_orchestrator=call_entry_orchestrator)
    storage = prepare_shared_storage(_effective=effective)
    try:
        from dryml.execute._worker import setup_result_bytes_limit

        core_result_limit = setup_result_bytes_limit(config.result_limit_bytes)
        if core_result_limit <= 0:
            raise ValueError("configured result limit cannot transport a core outcome with worker setup")
        strategy = effective.marshalling()
        preparation_repo = storage.recovery_repo if isinstance(effective.repo, RepoDefinition) else effective.repo
        prepared = strategy.prepare(
            fn, args, call_kwargs, repo=preparation_repo,
            control_store=effective.control_store, update_args=effective.update_args,
            _frozen_storage=storage.frozen_storage,
            invocation_limit_bytes=config.invocation_limit_bytes,
        )
        recovery = strategy.bind_recovery(prepared, args=args, kwargs=call_kwargs)
        _run_core_validators(frozen_validators)
        with publication.lease():
            _check_orchestration_floor(
                effective, call_entry_orchestrator=call_entry_orchestrator)
        # ``auto`` remains constrained by the call-entry floor even after a
        # later relaxation, so delayed preparation cannot widen caller
        # authority.
        return_objects = effective.return_objects is True or (
            effective.return_objects == "auto" and not call_entry_orchestrator
        )
        return _FrozenCoreSubmission(
            storage, prepared, recovery, effective, core_result_limit,
            return_objects, call_entry_orchestrator,
        )
    except BaseException as error:
        cleanup_failure = None
        try:
            storage.close()
        except BaseException:
            cleanup_failure = CleanupError(
                "core execution preparation cleanup failed")
        if cleanup_failure is not None:
            raise error from cleanup_failure
        raise


def _submit_frozen_core_submission(
        submitter: Callable[..., ExecutionFuture[bytes]],
        config: BackendConfig,
        frozen: _FrozenCoreSubmission,
        *,
        environment: EnvironmentRequirement | None,
        environment_spec: EnvironmentSpec | None,
        world: WorldRequirement | None,
        execution_timeout: Any,
        stream_output: bool | None,
        done_callbacks: Sequence[CoreCallback],
        output: ExecutionOutput | None,
        one_off: bool,
        validators: Sequence[Callable[[], None]] = (),
) -> CoreExecutionFuture:
    """
    Validate and accept one frozen core payload through an existing backend
    owner.

        Args:
            submitter: Generic backend submission owner.
            config: Frozen backend configuration used during preparation.
            frozen: Submission returned by
            :func:`_prepare_frozen_core_submission`.
            environment: Optional typed worker software requirement.
            environment_spec: Optional exact existing worker selector.
            world: Optional typed worker world requirement.
            execution_timeout: Generic workload deadline control.
            stream_output: Generic live-output control.
            done_callbacks: Core future callbacks.
            output: Generic output holder.
            one_off: Whether accepted future cleanup also owns generic one-off
            cleanup.
            validators: Same coordinator-owned drift guards used for
            preparation.

        Returns:
            The existing CoreExecutionFuture after generic backend acceptance.

        Raises:
            Exception: Propagates guard or generic acceptance failures after
            releasing
                only the frozen submission's owned recovery resources.

        Side Effects:
            Rechecks the current orchestration floor under a short publication
            lease.
            On rejection before acceptance it closes only ``frozen.storage``.
    """

    if not isinstance(frozen, _FrozenCoreSubmission):
        raise TypeError("frozen must be a _FrozenCoreSubmission")
    backend_future: ExecutionFuture[bytes] | None = None
    try:
        callbacks = tuple(done_callbacks)
        if not all(callable(callback) for callback in callbacks):
            raise TypeError("done_callbacks entries must be callable")
        frozen_validators = _validate_core_validators(validators)
        frozen._begin_acceptance()
        _run_core_validators(frozen_validators)
        with publication.lease():
            _check_orchestration_floor(
                frozen.effective,
                call_entry_orchestrator=frozen.call_entry_orchestrator,
            )
            _check_borrowed_core_storage(frozen)
            backend_future = submitter(
                _invoke_prepared_outcome,
                frozen.prepared.invocation,
                config.invocation_limit_bytes,
                frozen.result_limit_bytes,
                frozen.effective.update_args,
                kwargs=None,
                environment=environment,
                environment_spec=environment_spec,
                world=world,
                execution_timeout=execution_timeout,
                stream_output=stream_output,
                output=output,
                worker_setup=frozen.prepared.worker_setup(
                    frozen.effective.runtime, cache=frozen.effective.cache),
            )
            future = CoreExecutionFuture(
                backend_future,
                prepared_storage=frozen.storage,
                prepared=frozen.prepared,
                recovery=frozen.recovery,
                return_objects=frozen.return_objects,
                result_limit_bytes=frozen.result_limit_bytes,
                done_callbacks=callbacks,
                one_off=one_off,
            )
            frozen._accepted()
            return future
    except BaseException as error:
        if backend_future is not None:
            # Generic acceptance already occurred. Do not discard its retryable
            # lifecycle handle if facade construction cannot retain it.
            try:
                frozen.close()
            except BaseException as cleanup_error:
                raise CleanupError(
                    "core execution acceptance cleanup failed",
                    execution=backend_future,
                ) from cleanup_error
            raise CleanupError(
                "core execution acceptance could not retain the backend "
                "future",
                execution=backend_future,
            ) from error
        try:
            frozen.close()
        except BaseException as cleanup_error:
            raise CleanupError(
                "core execution acceptance cleanup failed", execution=frozen,
            ) from cleanup_error
        raise


def _check_borrowed_core_storage(frozen: _FrozenCoreSubmission) -> None:
    """
    Reject a borrowed Repo closed after preparation but before acceptance.
    """

    repo = frozen.effective.repo
    if isinstance(repo, Repo):
        with repo._configuration_lock:
            if repo._closing or repo._closed:
                raise ValueError("shared Store authority is unavailable")


def _submit_core_call(
    submitter: Callable[..., ExecutionFuture[bytes]],
    config: BackendConfig,
    executor_core: CoreOptions | None,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    kwargs: Mapping[str, Any] | None,
    core: CoreOptions | None,
    environment: EnvironmentRequirement | None,
    environment_spec: EnvironmentSpec | None = None,
    world: WorldRequirement | None,
    execution_timeout: Any,
    stream_output: bool | None,
    done_callbacks: Sequence[CoreCallback],
    output: ExecutionOutput | None,
    one_off: bool,
) -> CoreExecutionFuture:
    """
    Prepare then submit one core call through the guarded frozen-owner seam.
    """

    if not isinstance(done_callbacks, Sequence):
        raise TypeError("done_callbacks must be a finite sequence")
    if not all(callable(callback) for callback in done_callbacks):
        raise TypeError("done_callbacks entries must be callable")
    controls = _capture_frozen_core_controls(core, executor_core=executor_core)
    frozen = _prepare_frozen_core_submission(
        config, executor_core, fn, args, kwargs=kwargs, core=core,
        frozen_controls=controls,
    )
    return _submit_frozen_core_submission(
        submitter, config, frozen, environment=environment,
        environment_spec=environment_spec, world=world,
        execution_timeout=execution_timeout, stream_output=stream_output,
        done_callbacks=done_callbacks, output=output, one_off=one_off,
    )


class Executor:
    """Own one generic backend lifetime while adapting core byte outcomes.

    Args:
        config: Explicit generic backend configuration and its operational limits.
        core: Optional inert defaults resolved beneath per-call core options.

    Construction does not open Stores or start a worker. Each submission freezes
    caller session/core controls once, then delegates execution to exactly one
    generic executor and retains its own reconstruction resources until cleanup.

    Raises:
        TypeError: If configuration or reusable options have the wrong type.

    Side Effects:
        Owns the generic executor and each accepted facade's recovery resources;
        it borrows caller Repo and Store handles.
    """

    def __init__(self, config: BackendConfig, *, core: CoreOptions | None = None) -> None:
        if not isinstance(config, BackendConfig):
            raise TypeError("config must be a BackendConfig")
        if core is not None and not isinstance(core, CoreOptions):
            raise TypeError("core must be CoreOptions or None")
        self._config = config
        self._core = core
        self._generic = _GenericExecutor(config)
        self._condition = Condition(RLock())
        self._futures: set[CoreExecutionFuture] = set()

    def start(self) -> "Executor":
        """Start the owned generic backend and return this core facade.

        Returns:
            This executor after generic backend initialization.

        Raises:
            ExecutionError: If the configured backend cannot start.

        Side Effects:
            Starts only the owned generic backend; it does not export or open core
            storage until a submission.
        """
        self._generic.start()
        return self

    def __enter__(self) -> "Executor":
        """Start this executor for a context-managed core execution lifetime."""
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close owned work without suppressing a context-body exception."""
        self.close()

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        kwargs: Mapping[str, Any] | None = None,
        core: CoreOptions | None = None,
        environment: EnvironmentRequirement | None = None,
        environment_spec: EnvironmentSpec | None = None,
        world: WorldRequirement | None = None,
        execution_timeout: float | None | Literal["inherit"] = "inherit",
        stream_output: bool | None = None,
        done_callbacks: Sequence[CoreCallback] = (),
        output: ExecutionOutput | None = None,
    ) -> CoreExecutionFuture:
        """
        Prepare and submit one core-aware call through the owned generic
        executor.

                Args:
                    fn: Trusted synchronous core-aware callable.
                    args: Positional workload values.
                    kwargs: Optional string-keyed workload mapping.
                    core: Per-call inert core overrides.
                    environment: Optional typed software requirement,
                    independent of a pin.
                    environment_spec: Optional existing exact selector resolved
                    once when
                        this submission begins; failure never falls back to a
                        candidate.
                    world: Optional typed world requirement.
                    execution_timeout: Inherited, disabled, or positive
                    workload deadline.
                    stream_output: Optional generic live-output override.
                    done_callbacks: Core-future callbacks.
                    output: Optional caller-owned generic output holder.

                Returns:
                    A CoreExecutionFuture with the one recovered
                    result/evidence outcome.

                Raises:
                    TypeError: If controls or workload mapping are invalid.
                    ValueError: If frozen storage, materialization floor, or
                    call graph is
                        unsupported before generic acceptance.

                Side Effects:
                    Exports eligible live Repo authority once, opens an owned
                    recovery Repo,
                    and submits exactly one generic worker call. Accepted
                    asynchronous
                    failures remain on the returned future; caller handles are
                    borrowed.
        """
        future = _submit_core_call(
            self._generic.submit,
            self._config,
            self._core,
            fn,
            args,
            kwargs=kwargs,
            core=core,
            environment=environment,
            environment_spec=environment_spec,
            world=world,
            execution_timeout=execution_timeout,
            stream_output=stream_output,
            done_callbacks=done_callbacks,
            output=output,
            one_off=False,
        )
        with self._condition:
            self._futures.add(future)
        return future

    def run(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        kwargs: Mapping[str, Any] | None = None,
        core: CoreOptions | None = None,
        environment: EnvironmentRequirement | None = None,
        environment_spec: EnvironmentSpec | None = None,
        world: WorldRequirement | None = None,
        execution_timeout: float | None | Literal["inherit"] = "inherit",
        stream_output: bool | None = None,
        done_callbacks: Sequence[CoreCallback] = (),
        output: ExecutionOutput | None = None,
    ) -> Any:
        """
        Submit one call and return its recovered result without closing this
        executor.

                Args:
                    fn: Trusted synchronous core-aware callable.
                    args: Positional workload values.
                    kwargs: Optional string-keyed workload mapping.
                    core: Per-call inert core overrides.
                    environment: Optional typed software requirement,
                    independent of a pin.
                    environment_spec: Optional existing exact selector resolved
                    once when
                        this call begins; failure never falls back to a
                        candidate.
                    world: Optional typed world requirement.
                    execution_timeout: Inherited, disabled, or positive
                    workload deadline.
                    stream_output: Optional generic live-output override.
                    done_callbacks: Core-future callbacks.
                    output: Optional caller-owned generic output holder.

                Returns:
                    The recovered core result.

                Raises:
                    BaseException: Validation, backend, publication, or
                    recovery failure.

                Side Effects:
                    Has submit's ownership effects but does not close this
                    reusable executor.
        """
        return self.submit(
            fn,
            *args,
            kwargs=kwargs,
            core=core,
            environment=environment,
            environment_spec=environment_spec,
            world=world,
            execution_timeout=execution_timeout,
            stream_output=stream_output,
            done_callbacks=done_callbacks,
            output=output,
        ).result()

    def with_options(
        self,
        *,
        core: CoreOptions | None = None,
        environment: EnvironmentRequirement | None = None,
        environment_spec: EnvironmentSpec | None = None,
        world: WorldRequirement | None = None,
        execution_timeout: float | None | Literal["inherit"] = "inherit",
        stream_output: bool | None = None,
        done_callbacks: Sequence[CoreCallback] = (),
        output: ExecutionOutput | None = None,
    ) -> "ExecutorView":
        """
        Return a non-owning view whose call keywords remain workload data.

                Args:
                    core: Reusable core overrides for calls through the view.
                    environment: Optional typed software requirement bound to
                    later calls.
                    environment_spec: Optional exact selector retained inertly
                    and resolved
                        separately when each view call begins.
                    world: Optional typed world requirement bound to later
                    calls.
                    execution_timeout: Inherited, disabled, or positive
                    workload deadline.
                    stream_output: Optional generic live-output override.
                    done_callbacks: Core-future callbacks.
                    output: Optional caller-owned generic output holder.

                Returns:
                    A lightweight view retaining this executor.

                Raises:
                    TypeError: If ``core`` is not CoreOptions or None.

                Side Effects:
                    None. The view owns no backend, future set, or storage
                    resource.
        """
        if core is not None and not isinstance(core, CoreOptions):
            raise TypeError("core must be CoreOptions or None")
        return ExecutorView(
            executor=self,
            core=core,
            environment=environment,
            world=world,
            execution_timeout=execution_timeout,
            stream_output=stream_output,
            done_callbacks=tuple(done_callbacks),
            output=output,
            environment_spec=environment_spec,
        )

    def discover(
        self,
        *,
        environment: EnvironmentRequirement | None = None,
        environment_spec: EnvironmentSpec | None = None,
        world: WorldRequirement | None = None,
        timeout: float | None = None,
    ) -> Any:
        """
        Delegate a bounded non-reserving generic query with an exact pin
        option.

                Args:
                    environment: Optional typed software requirement.
                    environment_spec: Optional exact selector resolved for this
                    discovery
                        only; unsupported or mismatched targets are not
                        reported viable.
                    world: Optional typed world requirement.
                    timeout: Optional positive query timeout.

                Returns:
                    The generic non-reserving discovery snapshot.

                Raises:
                    TypeError, ValueError, TimeoutError, ExecutionError:
                    Forwarded generic
                        validation, bounded-observation, or backend failures.

                Side Effects:
                    May initialize and query the selected generic backend but
                    neither
                    submits workload code nor creates an environment.
        """
        return self._generic.discover(environment=environment,
                                      environment_spec=environment_spec,
                                      world=world,
                                      timeout=timeout)

    def resources(self, *, timeout: float | None = None) -> Any:
        """Delegate bounded resource inspection to the owned generic backend."""
        return self._generic.resources(timeout=timeout)

    def close(self, *, cancel: bool = False, timeout: float | None = None) -> None:
        """Join core adaptation/cleanup before releasing the generic backend owner.

        Args:
            cancel: Whether to request cancellation for unfinished calls first.
            timeout: Optional finite nonnegative total caller close budget.

        Returns:
            ``None`` after owned cleanup completes.

        Raises:
            TypeError: If controls have invalid types.
            ValueError: If ``timeout`` is invalid.
            CleanupError: If an owned future or backend cannot reconcile cleanup.

        Side Effects:
            Stops future acceptance through the generic owner, joins accepted core
            adaptation, and releases only executor/facade-owned resources.
        """
        if not isinstance(cancel, bool):
            raise TypeError("cancel must be bool")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout must be a finite nonnegative number of seconds")
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError("timeout must be a finite nonnegative number of seconds")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            futures = tuple(self._futures)
        if cancel:
            for future in futures:
                if not future.done():
                    if not future.cancel():
                        try:
                            future.request_cancel()
                        except ExecutionError:
                            pass
        for future in futures:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                future.result(timeout=remaining)
            except BaseException:
                pass
            future.cleanup(timeout=remaining)
        self._generic.close(cancel=cancel, timeout=timeout)


@dataclass(frozen=True, slots=True)
class ExecutorView:
    """
    Bind core/generic controls to one parent without owning another backend.

        Args:
            executor: Parent core executor that owns backend and future
            lifetime.
            core: Frozen reusable core overrides.
            environment: Frozen generic environment requirement or selection.
            world: Frozen generic world requirement or selection.
            execution_timeout: Frozen generic workload deadline control.
            stream_output: Frozen generic live-output control.
            done_callbacks: Frozen core-future callback sequence.
            output: Optional caller-owned generic output holder.
            environment_spec: Optional exact selector forwarded by every call
            and
                resolved independently for each submission.

        Every keyword received by :meth:`submit` or :meth:`run` is workload
        data;
        controls are fixed when this value is created. The view borrows its
        parent and
        cannot start, close, or allocate a second backend.

        Side Effects:
            Construction retains the parent reference only and does not start
            work or
            acquire storage.
    """

    executor: Executor
    core: CoreOptions | None
    environment: EnvironmentRequirement | None
    world: WorldRequirement | None
    execution_timeout: float | None | Literal["inherit"]
    stream_output: bool | None
    done_callbacks: tuple[CoreCallback, ...]
    output: ExecutionOutput | None
    environment_spec: EnvironmentSpec | None = None

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> CoreExecutionFuture:
        """Submit workload data with bound controls; no keyword is interpreted as a control.

        Args:
            fn: Trusted synchronous core-aware callable.
            args: Positional workload values.
            kwargs: Workload keyword values, including control-named values.

        Returns:
            The parent-owned CoreExecutionFuture.

        Raises:
            BaseException: The parent submission's synchronous validation failure.

        Side Effects:
            Delegates to the parent; this view owns no submission resource.
        """
        return self.executor.submit(
            fn,
            *args,
            kwargs=kwargs,
            core=self.core,
            environment=self.environment,
            environment_spec=self.environment_spec,
            world=self.world,
            execution_timeout=self.execution_timeout,
            stream_output=self.stream_output,
            done_callbacks=self.done_callbacks,
            output=self.output,
        )

    def run(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Submit view-bound workload data and return the adapted result.

        Args:
            fn: Trusted synchronous core-aware callable.
            args: Positional workload values.
            kwargs: Workload keyword values, including control-named values.

        Returns:
            The parent future's recovered core result.

        Raises:
            BaseException: The parent submission or recovered execution failure.

        Side Effects:
            Delegates to the parent without closing it or owning a backend.
        """
        return self.submit(fn, *args, **kwargs).result()


def submit(
    fn: Callable[..., Any],
    /,
    *args: Any,
    backend: BackendConfig,
    kwargs: Mapping[str, Any] | None = None,
    core: CoreOptions | None = None,
    environment: EnvironmentRequirement | None = None,
    environment_spec: EnvironmentSpec | None = None,
    world: WorldRequirement | None = None,
    execution_timeout: float | None | Literal["inherit"] = "inherit",
    stream_output: bool | None = None,
    done_callbacks: Sequence[CoreCallback] = (),
    output: ExecutionOutput | None = None,
) -> CoreExecutionFuture:
    """
    Submit one explicit-backend core call with generic one-off ownership.

        Args:
            fn: Trusted synchronous core-aware callable.
            args: Positional workload values.
            backend: Required explicit generic backend configuration.
            kwargs: Optional string-keyed workload mapping.
            core: Per-call inert core overrides.
            environment: Optional typed software requirement.
            environment_spec: Optional exact selector resolved once at
            submission entry;
                an unavailable or incompatible pin never falls back to
                candidates.
            world: Optional typed world requirement.
            execution_timeout: Inherited, disabled, or positive workload
            deadline.
            stream_output: Optional generic live-output override.
            done_callbacks: Core-future callbacks.
            output: Optional caller-owned generic output holder.

        Returns:
            A CoreExecutionFuture retaining the one-off owner's recovered
            outcome.

        Raises:
            TypeError: If ``backend`` or controls are invalid.
            ValueError: If core preparation rejects the call before acceptance.

        Side Effects:
            Creates a hidden generic one-off owner following its bounded
            cleanup policy.
            The returned facade owns only its recovery Repo and never closes
            caller-borrowed Repo or Store handles.
    """
    if not isinstance(backend, BackendConfig):
        raise TypeError("backend must be a BackendConfig")
    return _submit_core_call(
        lambda *call_args, **controls: _generic_submit(
            *call_args, backend=backend, **controls),
        backend,
        None,
        fn,
        args,
        kwargs=kwargs,
        core=core,
        environment=environment,
        environment_spec=environment_spec,
        world=world,
        execution_timeout=execution_timeout,
        stream_output=stream_output,
        done_callbacks=done_callbacks,
        output=output,
        one_off=True,
    )


def run(
    fn: Callable[..., Any],
    /,
    *args: Any,
    backend: BackendConfig,
    kwargs: Mapping[str, Any] | None = None,
    core: CoreOptions | None = None,
    environment: EnvironmentRequirement | None = None,
    environment_spec: EnvironmentSpec | None = None,
    world: WorldRequirement | None = None,
    execution_timeout: float | None | Literal["inherit"] = "inherit",
    stream_output: bool | None = None,
    done_callbacks: Sequence[CoreCallback] = (),
    output: ExecutionOutput | None = None,
) -> Any:
    """
    Run one explicit-backend core call and reconcile its hidden owner.

        Args:
            fn: Trusted synchronous core-aware callable.
            args: Positional workload values.
            backend: Required explicit generic backend configuration.
            kwargs: Optional string-keyed workload mapping.
            core: Per-call inert core overrides.
            environment: Optional typed software requirement.
            environment_spec: Optional exact selector resolved once at
            submission entry;
                an unavailable or incompatible pin never falls back to
                candidates.
            world: Optional typed world requirement.
            execution_timeout: Inherited, disabled, or positive workload
            deadline.
            stream_output: Optional generic live-output override.
            done_callbacks: Core-future callbacks.
            output: Optional caller-owned generic output holder.

        Returns:
            The recovered core result after one-off cleanup completes.

        Raises:
            BaseException: The adapted workload failure, with cleanup chained
            when
                both fail, or CleanupError after a successful result cannot
                clean up.

        Side Effects:
            Creates and reconciles a hidden one-off owner. It never closes
            caller
            Repo or Store handles.
    """
    future = submit(
        fn,
        *args,
        backend=backend,
        kwargs=kwargs,
        core=core,
        environment=environment,
        environment_spec=environment_spec,
        world=world,
        execution_timeout=execution_timeout,
        stream_output=stream_output,
        done_callbacks=done_callbacks,
        output=output,
    )
    try:
        result = future.result()
    except BaseException as error:
        try:
            future.cleanup()
        except CleanupError as cleanup_error:
            raise error from cleanup_error
        raise
    future.cleanup()
    return result


__all__ = [
    "CoreAdaptationOutcome", "CoreExecutionError", "CoreExecutionFuture",
    "CoreExecutionSnapshot", "CoreOptions", "CoreOutcomeEvidence",
    "CorePublicationEvidence", "CoreRefreshEvidence", "ExecutionContext", "Executor",
    "ExecutorView", "PreparedCoreCall", "SharedDirStoreStrategy", "core_worker_setup",
    "current_context", "decode_core_outcome", "prepare_shared_storage", "run", "submit",
    "worker_context",
]
