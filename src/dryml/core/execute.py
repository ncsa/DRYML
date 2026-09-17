"""Worker-local core setup and execution-context ownership.

The core execution adapter encodes the actual callable separately.  This module
only provides the setup factory consumed by generic Execute before it transfers
that payload, keeping generic Execute free of core imports and resource policy.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import threading
from uuid import uuid4
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, TypeAlias

from dryml.execute.models import WorkerSetupContext
from dryml.execute.errors import ExecutionError
from dryml.formats import make_envelope, semantic_id, validate_envelope
from dryml.runtime import (
    ExecutionGrant,
    RuntimeContextSpec,
    RuntimeMode,
    activation_scope,
    active_runtime,
)

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
_SETUP_BOUNDS = {"max_depth": 64, "max_nodes": 65_536, "max_entries": 65_536}
Inherit: TypeAlias = Literal["inherit"]
CacheMode: TypeAlias = Literal["none", "weak", "strong"]
ReturnObjects: TypeAlias = bool | Literal["auto"]
_INHERIT = "inherit"


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
    frozen = deep_freeze_json(value, max_depth=64, max_nodes=65_536, max_entries=65_536)
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
        frozen_targets = deep_freeze_json({"targets": list(self.update_targets)}, max_depth=64, max_nodes=65_536, max_entries=65_536)
        object.__setattr__(self, "update_targets", tuple(frozen_targets["targets"]))

    def worker_setup(self, runtime: RuntimeContextSpec | None = None):
        """Create the generic Execute setup required to invoke this prepared call.

        Args:
            runtime: Optional detached worker runtime specification. ``None`` uses
                the dependency-light inline core runtime.

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
        setup = self.storage_setup
        if set(setup) != {"repo", "control_store"} or not isinstance(setup["repo"], Mapping):
            raise ValueError("prepared core call has no complete shared Store setup")
        payload = {
            "runtime": spec.to_data(), "repo": setup["repo"], "role": "main",
            "replica": 0, "control_store": setup["control_store"],
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
            update_args: bool, _recovery: _CoreRecovery | None = None) -> Any:
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
        decoded = decode_core_outcome(result, repo=repo)
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


def decode_core_outcome(result: bytes, *, repo: Repo) -> CoreAdaptationOutcome:
    """Decode a tagged outcome into a value plus detached exact evidence.

    Args:
        result: Bounded bytes returned by :class:`SharedDirStoreStrategy`.
        repo: Caller-owned Repo used only to validate the closed result graph.

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

    outcome = decode_outcome(result, repo=repo)
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
    definition = repo.to_definition()
    setup = _freeze_storage_setup(_shared_storage_setup(definition, control_store))
    source_stores = tuple(repo.stores)
    if not all(type(store) is DirStore for store in source_stores):
        raise ValueError("SharedDirStoreStrategy requires only configured DirStores")
    return _FrozenSharedStorage(repo, source_stores, definition, setup)


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
        frozen = _freeze_shared_storage(selected, effective.control_store)
        definition = frozen.definition
    elif isinstance(selected, RepoDefinition):
        definition = selected
    else:
        raise TypeError("core repo must resolve to Repo, RepoDefinition, or None")
    # Validation consumes the detached definition, preserving the one-export cut.
    strategy.validate(repo=definition, control_store=effective.control_store)
    if not isinstance(selected, Repo):
        # A detached definition has no live handles to pin. Build only its worker
        # table; public callable preparation still requires a live source Repo.
        setup = _freeze_storage_setup(_shared_storage_setup(definition, effective.control_store))
        frozen = None
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


__all__ = [
    "CoreAdaptationOutcome", "CoreExecutionError", "CoreOutcomeEvidence",
    "CorePublicationEvidence", "CoreRefreshEvidence", "ExecutionContext",
    "SharedDirStoreStrategy", "core_worker_setup", "current_context",
    "decode_core_outcome", "worker_context",
]
