"""Authoritative DirStore-local managed operation control and activation."""

from __future__ import annotations

import os
import tempfile
import threading
import uuid
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping

from dryml.core.repo import Repo, get_default_repo, make_store
from dryml.core.store.store import Store
from dryml.formats.canonical import canonical_json_bytes, canonical_json_load_bytes
from dryml.formats.errors import CanonicalJSONError

from .errors import (
    AmbiguousManagedStoreError,
    ManagedActivationIndeterminateError,
    ManagedInputValidationRequiredError,
    ManagedLeaseConflictError,
    ManagedRerunRequiredError,
    ManagedStateError,
    ManagedStoreUnsupportedError,
    ManagedTakeoverRequiredError,
    StaleManagedLeaseError,
    StaleManagedResultError,
)
from .locking import PlatformFileLock, process_is_alive
from .events import ProgressSnapshot
from .state import (
    ActivationEvent,
    GenerationControl,
    MAX_DIAGNOSTICS,
    NamespaceState,
    OperationDecision,
    OperationKey,
    RealizationState,
    validate_declaration_fingerprint,
)


MANAGED_SNAPSHOT_CAPABILITY = "managed-snapshot-v1"
MANAGED_CAPABILITIES = frozenset({
    MANAGED_SNAPSHOT_CAPABILITY,
    "managed-control-v1",
    "managed-locking-v1",
    "managed-activation-v1",
    "managed-durable-products-v1",
})
_ACTIVATION_READ_ATTEMPTS = 3


class _TransientManagedStateReadError(ManagedStateError):
    """A managed-state read failed without proving persisted bytes invalid."""


def resolve_managed_store(
    repo: Repo | Store | None = None,
    *,
    store: Store | str | os.PathLike[str] | None = None,
    target: Any | None = None,
    writable: bool = True,
) -> Store:
    """Resolve one managed Store without relying on Store order.

    Explicit Store selection wins. Otherwise an explicit or active default Repo
    may provide an object binding or exactly one suitable Store. ``writable``
    requires the complete live lifecycle contract; read-only selection accepts
    verified managed snapshot Stores. Multiple candidates fail closed so state
    cannot split across authorities.
    """

    if not isinstance(writable, bool):
        raise TypeError("writable must be a bool")

    if isinstance(repo, Store):
        if store is not None and make_store(store).catalog_key() != repo.catalog_key():
            raise AmbiguousManagedStoreError("repo Store and explicit Store select different authorities")
        store = repo
        repo = None
    if store is not None:
        selected = make_store(store)
        if writable:
            _require_managed_capabilities(selected)
        else:
            _require_managed_snapshot_capability(selected)
        return selected
    selected_repo = repo if repo is not None else get_default_repo()
    if not isinstance(selected_repo, Repo):
        raise TypeError("repo must be a Repo, Store, or None")
    capability = "managed-control-v1" if writable else MANAGED_SNAPSHOT_CAPABILITY
    candidates = selected_repo.store_candidates(target, capability=capability)
    if writable:
        candidates = tuple(candidate for candidate in candidates if _has_managed_capabilities(candidate))
    if not candidates:
        if writable:
            raise AmbiguousManagedStoreError("no managed-capable Store is available")
        raise AmbiguousManagedStoreError(
            "no managed snapshot-capable Store is available"
        )
    if len(candidates) != 1:
        raise AmbiguousManagedStoreError(
            "multiple managed-capable Stores are available; select one explicitly"
        )
    return candidates[0]


class ManagedOperationStore:
    """Facade over one Store's versioned managed operation namespace."""

    def __init__(self, store: Store, *, writable: bool = True):
        if not isinstance(store, Store):
            store = make_store(store)
        if not isinstance(writable, bool):
            raise TypeError("writable must be a bool")
        if writable:
            _require_managed_capabilities(store)
        else:
            _require_managed_snapshot_capability(store)
        self.store = store
        self.writable = writable
        self._snapshot_only = not _has_managed_capabilities(store)
        root = store.managed_control_root() if writable else store.managed_snapshot_root()
        self.root = Path(root)

    @classmethod
    def resolve(
        cls,
        repo: Repo | Store | None = None,
        *,
        store: Store | str | os.PathLike[str] | None = None,
        target: Any | None = None,
        writable: bool = True,
    ) -> "ManagedOperationStore":
        """Resolve a context and return its managed control facade."""

        return cls(
            resolve_managed_store(
                repo, store=store, target=target, writable=writable
            ),
            writable=writable,
        )

    def _require_writable(self) -> None:
        """Reject lifecycle mutation through a read-only snapshot facade."""

        if not self.writable:
            raise ManagedStoreUnsupportedError(
                "managed snapshot Stores do not support live lifecycle mutation"
            )

    def operation(self, key: OperationKey, declaration_fingerprint: str) -> "OperationControl":
        """Return direct control for one operation declaration generation."""

        if not isinstance(key, OperationKey):
            raise TypeError("key must be an OperationKey")
        validate_declaration_fingerprint(declaration_fingerprint)
        return OperationControl(self, key, declaration_fingerprint)

    def generations(self, key: OperationKey) -> tuple[str, ...]:
        """Return declaration generations in durable introduction order."""

        namespace = self._read_namespace(key, missing_ok=True)
        return () if namespace is None else namespace.generations

    def history(self, key: OperationKey) -> tuple[RealizationState, ...]:
        """Return retained realizations across every declaration generation."""

        result = []
        for fingerprint in self.generations(key):
            result.extend(self.operation(key, fingerprint).history())
        return tuple(result)

    def _operation_dir(self, key: OperationKey) -> Path:
        digest = key.producer_cdef_id.rsplit("-", 1)[1]
        return self.root / "operations" / digest[:2] / digest / key.method

    def _namespace_path(self, key: OperationKey) -> Path:
        return self._operation_dir(key) / "namespace.json"

    def _read_namespace(self, key: OperationKey, *, missing_ok: bool = False) -> NamespaceState | None:
        path = self._namespace_path(key)
        if missing_ok and not path.exists():
            return None
        state = NamespaceState.from_json(_read_json(path, "operation namespace"))
        if state.key != key:
            raise ManagedStateError("operation namespace identity does not match its direct path")
        return state


class OperationControl:
    """Direct lookup and lifecycle control for one declaration generation."""

    def __init__(self, managed_store: ManagedOperationStore, key: OperationKey, fingerprint: str):
        self.managed_store = managed_store
        self.key = key
        self.declaration_fingerprint = fingerprint

    @property
    def operation_dir(self) -> Path:
        return self.managed_store._operation_dir(self.key)

    @property
    def generation_dir(self) -> Path:
        return self.operation_dir / "generations" / self.declaration_fingerprint

    @property
    def control_path(self) -> Path:
        return self.generation_dir / "control.json"

    @property
    def active_pointer_path(self) -> Path:
        return self.generation_dir / "active.json"

    @property
    def _lock_path(self) -> Path:
        return self.operation_dir / "owner.lock"

    @property
    def _owner_path(self) -> Path:
        return self.operation_dir / "owner.json"

    @property
    def _realizations_dir(self) -> Path:
        return self.generation_dir / "realizations"

    @property
    def _activations_dir(self) -> Path:
        return self.generation_dir / "activations"

    @property
    def attempts_dir(self) -> Path:
        """Return the retained fence-isolated attempt workspace root."""

        return self.generation_dir / "attempts"

    def acquire(self, *, advance_declaration: bool = False, takeover: bool = False) -> "OperationLease":
        """Acquire lifetime ownership and advance the monotonic fence.

        ``takeover`` never breaks a live OS lock. It only authorizes recovery
        after an operator has made the lock available while its prior PID still
        appears live, for example after cooperative lock handoff.
        """

        self.managed_store._require_writable()
        lock = PlatformFileLock(self._lock_path)
        try:
            lock.acquire()
        except ManagedLeaseConflictError as exc:
            if takeover:
                raise ManagedLeaseConflictError(
                    "managed operation is already owned; stop the current owner before explicit takeover"
                ) from exc
            raise
        try:
            namespace = self.managed_store._read_namespace(self.key, missing_ok=True)
            if namespace is None:
                namespace = NamespaceState(
                    key=self.key,
                    current_declaration_fingerprint=self.declaration_fingerprint,
                    generations=(self.declaration_fingerprint,),
                    fence_epoch=0,
                )
            elif namespace.current_declaration_fingerprint != self.declaration_fingerprint:
                if not advance_declaration:
                    raise ManagedStateError(
                        "declaration fingerprint is not the current declaration generation"
                    )
                generations = namespace.generations
                if self.declaration_fingerprint in generations:
                    raise ManagedStateError(
                        "an older declaration generation cannot become current again"
                    )
                generations = (*generations, self.declaration_fingerprint)
                namespace = replace(
                    namespace,
                    current_declaration_fingerprint=self.declaration_fingerprint,
                    generations=generations,
                )

            owner = self._read_owner(missing_ok=True)
            if owner is not None and owner["status"] == "owned" and process_is_alive(owner["pid"]):
                if not takeover:
                    raise ManagedTakeoverRequiredError(
                        "released ownership still names a live process; explicit operator takeover is required"
                    )

            epoch = namespace.fence_epoch + 1
            namespace = replace(namespace, fence_epoch=epoch)
            _write_json(self.managed_store._namespace_path(self.key), namespace.to_json())
            control = self._read_control(missing_ok=True)
            if control is None:
                control = GenerationControl(self.declaration_fingerprint, epoch)
            control = self._reconcile_control(replace(control, fence_epoch=epoch))
            _write_json(self.control_path, control.to_json())
            _write_json(
                self._owner_path,
                {
                    "schema_version": 1,
                    "status": "owned",
                    "epoch": epoch,
                    "pid": os.getpid(),
                    "heartbeat": 0,
                },
            )
            return OperationLease(self, lock, epoch)
        except Exception:
            lock.release()
            raise

    def status(self) -> dict[str, Any]:
        """Return current bounded control plus direct active selection."""

        control = self._read_control()
        active = self.active(missing_ok=True)
        return {"control": control, "active": active}

    def history(self) -> tuple[RealizationState, ...]:
        """Return all retained realizations for this declaration generation."""

        if not self._realizations_dir.exists():
            return ()
        states = []
        for path in self._realizations_dir.iterdir():
            if path.suffix != ".json":
                continue
            state = RealizationState.from_json(_read_json(path, "realization state"))
            if state.declaration_fingerprint != self.declaration_fingerprint:
                raise ManagedStateError("realization declaration does not match its generation")
            if path.stem != state.realization_id:
                raise ManagedStateError("realization ID does not match its direct path")
            if self.managed_store._snapshot_only and (
                state.status != "completed" or state.realization_record_id is None
            ):
                raise ManagedStateError(
                    "managed snapshot contains an incomplete realization"
                )
            states.append(state)
        return tuple(sorted(states, key=lambda item: item.sequence))

    def active(self, *, missing_ok: bool = False) -> RealizationState | None:
        """Resolve the direct active pointer and validate its immutable event."""

        if not self.active_pointer_path.exists():
            if missing_ok:
                return None
            raise ManagedStateError("operation generation has no active realization")
        event = ActivationEvent.from_json(_read_json(self.active_pointer_path, "active pointer"))
        event_path = self._activation_path(event)
        if not event_path.exists():
            raise ManagedStateError("active pointer references a missing activation event")
        authoritative = ActivationEvent.from_json(_read_json(event_path, "activation event"))
        if authoritative != event:
            raise ManagedStateError("active pointer does not match its activation event")
        state = self._read_realization(event.realization_id)
        if state.status != "completed":
            raise ManagedStateError("active realization is not completed")
        return state

    def active_event(self, *, missing_ok: bool = False) -> ActivationEvent | None:
        """Return the validated immutable event behind the active pointer."""

        if not self.active_pointer_path.exists():
            if missing_ok:
                return None
            raise ManagedStateError("operation generation has no active realization")
        event = ActivationEvent.from_json(_read_json(self.active_pointer_path, "active pointer"))
        event_path = self._activation_path(event)
        if not event_path.exists():
            raise ManagedStateError("active pointer references a missing activation event")
        authoritative = ActivationEvent.from_json(_read_json(event_path, "activation event"))
        if authoritative != event:
            raise ManagedStateError("active pointer does not match its activation event")
        return event

    def rebuild_active_pointer(self, *, takeover: bool = False) -> RealizationState:
        """Rebuild the derived pointer while holding a fresh fenced lease."""

        with self.acquire(takeover=takeover) as lease:
            return lease.rebuild_active_pointer()

    def _rebuild_active_pointer(self) -> RealizationState:
        """Rebuild the derived direct pointer from immutable activation events."""

        events = self._activation_events()
        if not events:
            raise ManagedStateError("operation generation has no activation events")
        event = max(events, key=lambda item: item.sequence)
        state = self._read_realization(event.realization_id)
        if state.status != "completed":
            raise ManagedStateError("latest activation event selects a non-completed realization")
        _write_json(self.active_pointer_path, event.to_json())
        return state

    def _read_control(self, *, missing_ok: bool = False) -> GenerationControl | None:
        if missing_ok and not self.control_path.exists():
            return None
        control = GenerationControl.from_json(_read_json(self.control_path, "generation control"))
        if control.declaration_fingerprint != self.declaration_fingerprint:
            raise ManagedStateError("generation control declaration fingerprint mismatch")
        if self.managed_store._snapshot_only and (
            control.pending_realization_id is not None
            or control.current_attempt_id is not None
            or control.reserved_realization_id is not None
        ):
            raise ManagedStateError(
                "managed snapshot contains live or incomplete control state"
            )
        return control

    def _read_realization(self, realization_id: str) -> RealizationState:
        path = self._realization_path(realization_id)
        state = RealizationState.from_json(_read_json(path, "realization state"))
        if state.realization_id != realization_id:
            raise ManagedStateError("realization ID does not match requested identity")
        if state.declaration_fingerprint != self.declaration_fingerprint:
            raise ManagedStateError("realization belongs to a different declaration generation")
        if self.managed_store._snapshot_only and (
            state.status != "completed" or state.realization_record_id is None
        ):
            raise ManagedStateError(
                "managed snapshot contains an incomplete realization"
            )
        return state

    def _write_realization(self, state: RealizationState) -> None:
        _write_json(self._realization_path(state.realization_id), state.to_json())

    def _realization_path(self, realization_id: str) -> Path:
        return self._realizations_dir / f"{realization_id}.json"

    def _reconcile_control(self, control: GenerationControl) -> GenerationControl:
        """Recover bounded current state and migrate legacy sequence metadata."""

        pending = None
        if control.pending_realization_id is not None:
            pending = self._read_realization(control.pending_realization_id)
            if pending.status in {"completed", "abandoned"}:
                pending = None
        next_sequence = control.next_realization_sequence
        latest_realization_id = control.latest_realization_id
        reserved_realization_id = control.reserved_realization_id
        if next_sequence is None:
            history = self.history()
            incomplete = [
                item
                for item in history
                if item.status in {"running", "interrupted", "failed"} and (
                    pending is None
                    or item.realization_id != pending.realization_id
                )
            ]
            if incomplete:
                if pending is not None or len(incomplete) != 1:
                    raise ManagedStateError(
                        "operation generation has ambiguous retained pending realizations"
                    )
                pending = incomplete[0]
            latest = max(
                history,
                key=lambda item: (item.sequence, item.realization_id),
                default=None,
            )
            next_sequence = 1 if latest is None else latest.sequence + 1
            latest_realization_id = (
                None if latest is None else latest.realization_id
            )
        else:
            if latest_realization_id is not None:
                latest = self._read_realization(latest_realization_id)
                if latest.sequence >= next_sequence:
                    raise ManagedStateError(
                        "latest realization is inconsistent with the next sequence"
                    )
            if reserved_realization_id is not None:
                if pending is not None:
                    raise ManagedStateError(
                        "generation control has both pending and reserved realizations"
                    )
                reservation_path = self._realization_path(reserved_realization_id)
                if reservation_path.exists():
                    reserved = self._read_realization(reserved_realization_id)
                    if reserved.sequence != next_sequence - 1:
                        raise ManagedStateError(
                            "reserved realization sequence does not match its allocation"
                        )
                    if reserved.status not in {"running", "interrupted", "failed"}:
                        raise ManagedStateError(
                            "reserved realization has an invalid recovery status"
                        )
                    pending = reserved
                    latest_realization_id = reserved.realization_id
                reserved_realization_id = None
        return replace(
            control,
            pending_realization_id=None if pending is None else pending.realization_id,
            current_attempt_id=None if pending is None else pending.current_attempt_id,
            checkpoint_head=None if pending is None else pending.checkpoint_head,
            diagnostics=() if pending is None else pending.diagnostics,
            next_realization_sequence=next_sequence,
            latest_realization_id=latest_realization_id,
            reserved_realization_id=reserved_realization_id,
        )

    def _activation_path(self, event: ActivationEvent) -> Path:
        return self._activations_dir / f"{event.sequence:020d}-{event.activation_id}.json"

    def _activation_events(self) -> tuple[ActivationEvent, ...]:
        if not self._activations_dir.exists():
            return ()
        events = []
        seen_sequences = set()
        for path in self._activations_dir.iterdir():
            if path.suffix != ".json":
                continue
            event = ActivationEvent.from_json(_read_json(path, "activation event"))
            if event.declaration_fingerprint != self.declaration_fingerprint:
                raise ManagedStateError("activation event belongs to a different generation")
            if path != self._activation_path(event):
                raise ManagedStateError("activation event identity does not match its path")
            if event.sequence in seen_sequences:
                raise ManagedStateError("activation history contains duplicate sequence numbers")
            seen_sequences.add(event.sequence)
            events.append(event)
        return tuple(sorted(events, key=lambda item: item.sequence))

    def _require_exact_activation_event(self, expected: ActivationEvent) -> None:
        """Fail unless ``expected`` is the latest immutable selection event."""

        path = self._activation_path(expected)
        authoritative = ActivationEvent.from_json(
            _read_json(path, "activation event")
        )
        if authoritative != expected:
            raise ManagedStateError(
                "published activation event does not match the proposed activation"
            )
        events = self._activation_events()
        if not events or events[-1] != expected:
            raise ManagedStateError(
                "published activation event is not the latest authoritative selection"
            )

    def _retry_activation_read(self, reader):
        """Retry transient activation reads without swallowing exact mismatches."""

        last_error = None
        for _attempt in range(_ACTIVATION_READ_ATTEMPTS):
            try:
                return True, reader()
            except (OSError, _TransientManagedStateReadError) as exc:
                last_error = exc
        return False, last_error

    def _read_owner(self, *, missing_ok: bool = False) -> dict[str, Any] | None:
        if missing_ok and not self._owner_path.exists():
            return None
        data = _read_json(self._owner_path, "owner diagnostics")
        fields = {"schema_version", "status", "epoch", "pid", "heartbeat"}
        if set(data) != fields or data.get("schema_version") != 1:
            raise ManagedStateError("owner diagnostics schema is malformed")
        if data.get("status") not in {"owned", "released"}:
            raise ManagedStateError("owner diagnostics status is malformed")
        if type(data.get("epoch")) is not int or data["epoch"] < 1:
            raise ManagedStateError("owner diagnostics epoch is malformed")
        if type(data.get("pid")) is not int or data["pid"] < 1:
            raise ManagedStateError("owner diagnostics pid is malformed")
        if type(data.get("heartbeat")) is not int or data["heartbeat"] < 0:
            raise ManagedStateError("owner diagnostics heartbeat is malformed")
        return data


def _serialized_mutation(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._mutation_lock:
            self._assert_no_unreconciled_reservation()
            result = method(self, *args, **kwargs)
            self._heartbeat()
            return result

    return guarded


class OperationLease:
    """Sole Store control writer while its platform lock remains held."""

    def __init__(self, operation: OperationControl, lock: PlatformFileLock, epoch: int):
        self.operation = operation
        self._lock = lock
        self.epoch = epoch
        self._released = False
        self._owner_pid = os.getpid()
        self._heartbeat_count = 0
        self._mutation_lock = threading.RLock()

    def __enter__(self) -> "OperationLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        """Publish released diagnostics and relinquish OS ownership."""

        with self._mutation_lock:
            if self._released:
                return
            try:
                self._assert_current()
                _write_json(
                    self.operation._owner_path,
                    {
                        "schema_version": 1,
                        "status": "released",
                        "epoch": self.epoch,
                        "pid": self._owner_pid,
                        "heartbeat": self._heartbeat_count,
                    },
                )
            finally:
                self._released = True
                self._lock.release()

    @_serialized_mutation
    def prepare(
        self,
        *,
        resumable: bool,
        rerun: bool = False,
        active_inputs_valid: bool | Callable[[RealizationState], bool] | None = None,
        realization_id: str | None = None,
        consumed_records: tuple[Any, ...] = (),
        consumed_record_links: tuple[Any, ...] = (),
    ) -> OperationDecision:
        """Model normal resume/reuse precedence or start an explicit rerun.

        ``active_inputs_valid`` is the U2 hook for U4's concurrency-stable input
        resolution. Reuse fails closed unless that later layer supplies ``True``.
        """

        self._assert_current()
        if not isinstance(resumable, bool) or not isinstance(rerun, bool):
            raise TypeError("resumable and rerun must be bool values")
        control = self.operation._read_control()
        assert control is not None
        pending = None
        if control.pending_realization_id is not None:
            pending = self.operation._read_realization(control.pending_realization_id)
        if pending is not None and not rerun and realization_id is not None:
            if realization_id != pending.realization_id:
                raise ManagedStateError(
                    "normal resume cannot select a different realization ID"
                )
        if (pending is None or rerun) and realization_id is not None:
            if self.operation._realization_path(realization_id).exists():
                raise ManagedStateError("realization ID already exists in retained history")
        if pending is not None and not rerun:
            if not pending.resumable:
                raise ManagedRerunRequiredError(
                    "pending realization is not resumable; explicit rerun is required"
                )
            attempt_id = _new_id("attempt")
            pending = replace(
                pending,
                status="running",
                attempt_ids=(*pending.attempt_ids, attempt_id),
                current_attempt_id=attempt_id,
            )
            self.operation._write_realization(pending)
            self._write_control_for(pending)
            return OperationDecision("resume", pending)
        if pending is not None:
            abandoned = replace(pending, status="abandoned", current_attempt_id=None)
            self.operation._write_realization(abandoned)

        active = self.operation.active(missing_ok=True)
        if pending is None and active is not None and not rerun:
            validation = active_inputs_valid
            if callable(validation):
                validation = validation(active)
            if validation is None:
                raise ManagedInputValidationRequiredError(
                    "completed reuse requires a concurrency-stable input validation result"
                )
            if validation is not True:
                raise StaleManagedResultError(
                    "active result inputs are stale; explicit rerun is required"
                )
            return OperationDecision("reuse", active)

        realization_id = realization_id or _new_id("realization")
        attempt_id = _new_id("attempt")
        if control.next_realization_sequence is None:
            raise ManagedStateError(
                "generation control sequence metadata was not reconciled"
            )
        sequence = control.next_realization_sequence
        reservation = replace(
            control,
            pending_realization_id=None,
            current_attempt_id=None,
            next_realization_sequence=sequence + 1,
            reserved_realization_id=realization_id,
        )
        _write_json(self.operation.control_path, reservation.to_json())
        state = RealizationState(
            realization_id=realization_id,
            declaration_fingerprint=self.operation.declaration_fingerprint,
            status="running",
            resumable=resumable,
            attempt_ids=(attempt_id,),
            current_attempt_id=attempt_id,
            sequence=sequence,
            consumed_records=tuple(consumed_records),
            consumed_record_links=tuple(consumed_record_links),
        )
        # A write failure may occur after atomic replacement but before the
        # parent directory fsync. Keep the reservation so reacquisition can
        # distinguish an absent publication from an exact recoverable one.
        self.operation._write_realization(state)
        self._write_control_for(state, reset_progress=True)
        action = "rerun" if rerun else "start"
        return OperationDecision(action, state)

    @_serialized_mutation
    def interrupt(
        self,
        realization_id: str,
        *,
        checkpoint_head: str | None = None,
        diagnostic: str | None = None,
        resumable: bool | None = None,
    ) -> RealizationState:
        """Mark current work interrupted while retaining it for resume/rerun."""

        return self._mark_incomplete(
            realization_id,
            "interrupted",
            checkpoint_head=checkpoint_head,
            diagnostic=diagnostic,
            resumable=resumable,
        )

    @_serialized_mutation
    def fail(
        self,
        realization_id: str,
        *,
        checkpoint_head: str | None = None,
        diagnostic: str | None = None,
        resumable: bool | None = None,
    ) -> RealizationState:
        """Mark current work failed while preserving prior active selection."""

        return self._mark_incomplete(
            realization_id,
            "failed",
            checkpoint_head=checkpoint_head,
            diagnostic=diagnostic,
            resumable=resumable,
        )

    @_serialized_mutation
    def checkpoint(self, realization_id: str, checkpoint_head: str) -> RealizationState:
        """Commit an immutable framework-owned checkpoint head under the fence."""

        self._assert_current()
        _control, state = self._require_pending(realization_id)
        if state.status != "running":
            raise ManagedStateError("checkpoint publication requires running work")
        state = replace(state, checkpoint_head=checkpoint_head)
        self.operation._write_realization(state)
        self._write_control_for(state)
        return state

    @_serialized_mutation
    def update_progress(
        self,
        realization_id: str,
        progress: ProgressSnapshot,
    ) -> None:
        """Replace the generation's single bounded progress snapshot."""

        self._assert_current()
        control, state = self._require_pending(realization_id)
        if state.status != "running":
            raise ManagedStateError("progress publication requires running work")
        if not isinstance(progress, ProgressSnapshot):
            raise TypeError("progress must be a ProgressSnapshot")
        _write_json(self.operation.control_path, replace(control, progress=progress).to_json())

    @_serialized_mutation
    def complete(
        self,
        realization_id: str,
        *,
        realization_record_id: str,
    ) -> RealizationState:
        """Mark verified required work complete without changing activation."""

        if realization_record_id is None:
            raise ManagedStateError("completion requires an immutable realization record")
        return self._complete_state(realization_id, realization_record_id)

    @_serialized_mutation
    def _complete_control_only(self, realization_id: str) -> RealizationState:
        """Complete U2 control-state fixtures without asserting U3 publication."""

        return self._complete_state(realization_id, None)

    def _complete_state(
        self,
        realization_id: str,
        realization_record_id: str | None,
    ) -> RealizationState:
        self._assert_current()
        control, state = self._require_pending(realization_id)
        if realization_record_id is not None:
            self._validate_realization_record(realization_id, realization_record_id)
        state = replace(
            state,
            status="completed",
            current_attempt_id=None,
            realization_record_id=realization_record_id,
        )
        self.operation._write_realization(state)
        control = replace(
            control,
            pending_realization_id=None,
            current_attempt_id=None,
            checkpoint_head=state.checkpoint_head,
        )
        _write_json(self.operation.control_path, control.to_json())
        return state

    @_serialized_mutation
    def activate(self, realization_id: str) -> RealizationState:
        """Commit an immutable activation event and reconcile its pointer last."""

        return self._activate_state(realization_id, require_record=True)

    @_serialized_mutation
    def _activate_control_only(self, realization_id: str) -> RealizationState:
        """Activate U2 control-state fixtures without asserting U3 publication."""

        return self._activate_state(realization_id, require_record=False)

    def _activate_state(
        self,
        realization_id: str,
        *,
        require_record: bool,
    ) -> RealizationState:
        self._assert_current()
        state = self.operation._read_realization(realization_id)
        if state.status != "completed":
            raise ManagedStateError("only completed realizations may become active")
        if require_record and state.realization_record_id is None:
            raise ManagedStateError("activation requires an immutable realization record")
        if state.realization_record_id is not None:
            self._validate_realization_record(realization_id, state.realization_record_id)
        previous = self.operation.active(missing_ok=True)
        events = self.operation._activation_events()
        event = ActivationEvent(
            activation_id=_new_id("activation"),
            declaration_fingerprint=self.operation.declaration_fingerprint,
            sequence=max((item.sequence for item in events), default=0) + 1,
            realization_id=realization_id,
            previous_realization_id=None if previous is None else previous.realization_id,
            fence_epoch=self.epoch,
            realization_record_id=state.realization_record_id,
        )
        event_path = self.operation._activation_path(event)
        try:
            _write_json(event_path, event.to_json(), immutable=True)
        except BaseException:
            if not event_path.exists():
                raise
            validated, read_error = self.operation._retry_activation_read(
                lambda: self.operation._require_exact_activation_event(event)
            )
            if not validated:
                raise ManagedActivationIndeterminateError(
                    "activation event publication is indeterminate; rebuild the "
                    "active pointer before retrying"
                ) from read_error
            try:
                _fsync_directory(event_path.parent)
            except OSError as reconciliation_error:
                raise ManagedActivationIndeterminateError(
                    "activation event durability is indeterminate; rebuild the "
                    "active pointer before retrying"
                ) from reconciliation_error
        else:
            # A successful immutable write proves the exact event is durable.
            # Retry transient validation reads, but do not let their continued
            # unavailability reclassify the known committed selection.
            self.operation._retry_activation_read(
                lambda: self.operation._require_exact_activation_event(event)
            )

        # The immutable event is the commit point. A pointer error after this
        # point must be reconciled under the same fence, not reported as a
        # failed rerun that a later rebuild would silently activate.
        for attempt in range(2):
            self._assert_current()
            try:
                _write_json(self.operation.active_pointer_path, event.to_json())
            except BaseException as publication_error:
                self.operation._retry_activation_read(
                    lambda: self.operation._require_exact_activation_event(event)
                )
                pointer_read, pointer = self.operation._retry_activation_read(
                    lambda: self.operation.active_event(missing_ok=True)
                )
                if pointer_read and pointer == event:
                    try:
                        _fsync_directory(self.operation.active_pointer_path.parent)
                    except OSError as reconciliation_error:
                        if attempt == 0:
                            continue
                        raise ManagedActivationIndeterminateError(
                            "activation committed but active pointer durability is "
                            "indeterminate; rebuild the pointer before retrying"
                        ) from reconciliation_error
                    break
                if not pointer_read:
                    if attempt == 0:
                        continue
                    raise ManagedActivationIndeterminateError(
                        "activation committed but active pointer publication is "
                        "indeterminate; rebuild the pointer before retrying"
                    ) from pointer
                if attempt == 0:
                    continue
                raise ManagedStateError(
                    "active pointer could not be reconciled with the "
                    "authoritative activation event"
                ) from publication_error
            pointer_read, pointer = self.operation._retry_activation_read(
                self.operation.active_event
            )
            if pointer_read and pointer != event:
                raise ManagedStateError(
                    "active pointer does not match the authoritative activation event"
                )
            break
        return state

    @_serialized_mutation
    def rebuild_active_pointer(self) -> RealizationState:
        """Rebuild the derived pointer under this lease's fence."""

        self._assert_current()
        return self.operation._rebuild_active_pointer()

    def _mark_incomplete(
        self,
        realization_id: str,
        status: str,
        *,
        checkpoint_head: str | None,
        diagnostic: str | None,
        resumable: bool | None,
    ) -> RealizationState:
        self._assert_current()
        _control, state = self._require_pending(realization_id)
        diagnostics = state.diagnostics
        if diagnostic is not None:
            diagnostics = (*diagnostics, diagnostic)[-MAX_DIAGNOSTICS:]
        state = replace(
            state,
            status=status,
            current_attempt_id=None,
            checkpoint_head=checkpoint_head or state.checkpoint_head,
            diagnostics=diagnostics,
            resumable=state.resumable if resumable is None else resumable,
        )
        self.operation._write_realization(state)
        self._write_control_for(state)
        return state

    def _require_pending(self, realization_id: str) -> tuple[GenerationControl, RealizationState]:
        control = self.operation._read_control()
        assert control is not None
        if control.pending_realization_id != realization_id:
            raise ManagedStateError("realization is not the current pending realization")
        return control, self.operation._read_realization(realization_id)

    def _write_control_for(
        self,
        state: RealizationState,
        *,
        reset_progress: bool = False,
    ) -> None:
        control = self.operation._read_control()
        assert control is not None
        latest_realization_id = state.realization_id
        if (
            control.latest_realization_id is not None
            and control.latest_realization_id != state.realization_id
        ):
            latest = self.operation._read_realization(
                control.latest_realization_id
            )
            if (latest.sequence, latest.realization_id) > (
                state.sequence,
                state.realization_id,
            ):
                latest_realization_id = latest.realization_id
        control = replace(
            control,
            fence_epoch=self.epoch,
            pending_realization_id=state.realization_id,
            current_attempt_id=state.current_attempt_id,
            checkpoint_head=state.checkpoint_head,
            diagnostics=state.diagnostics,
            progress=None if reset_progress else control.progress,
            latest_realization_id=latest_realization_id,
            reserved_realization_id=None,
        )
        _write_json(self.operation.control_path, control.to_json())

    def assert_current(self) -> None:
        """Fail unless this lease still owns the current fence and OS lock."""

        self._assert_current()

    def _validate_realization_record(
        self, realization_id: str, realization_record_id: str
    ) -> None:
        from dryml.records import (
            DataRecord,
            ExecutionRecord,
            RealizationRecord,
            StoredStateRecord,
            require_product_integrity,
        )

        record_io = self.operation.managed_store.store.records
        envelope = record_io.read_record(realization_record_id)
        realization = RealizationRecord.from_envelope(envelope)
        if realization.realization_id != realization_id:
            raise ManagedStateError("realization record binds a different realization")
        if realization.producer_cdef_id != self.operation.key.producer_cdef_id:
            raise ManagedStateError("realization record binds a different producer")
        if realization.method != self.operation.key.method:
            raise ManagedStateError("realization record binds a different method")
        if realization.declaration_fingerprint != self.operation.declaration_fingerprint:
            raise ManagedStateError("realization record binds a different declaration")
        for output in realization.outputs:
            record_io.read_spec(output.representation_id, family="representation")
            output_envelope = record_io.read_record(output.record_id)
            if output.record_kind == "data":
                typed_output = DataRecord.from_envelope(output_envelope)
            elif output.record_kind == "stored_state":
                typed_output = StoredStateRecord.from_envelope(output_envelope)
            else:
                raise ManagedStateError("managed realization output kind is unsupported")
            ownership = (
                typed_output.realization_id,
                typed_output.output_slot,
                typed_output.representation_id,
            )
            expected = (realization_id, output.slot, output.representation_id)
            if ownership != expected:
                raise ManagedStateError("managed output ownership does not match realization")
            require_product_integrity(record_io, output_envelope)
        execution = ExecutionRecord.from_envelope(
            record_io.read_record(realization.execution_record_id)
        )
        if execution.realization_id != realization_id:
            raise ManagedStateError("execution record binds a different realization")
        output_ids = {output.record_id for output in realization.outputs}
        if set(execution.produced_record_ids) != output_ids:
            raise ManagedStateError("execution produced lineage does not match realization outputs")
        try:
            execution_consumed = tuple(
                link.to_resolved()
                for link in execution.consumed_records
                if link.producer_cdef_id is not None
            )
        except Exception as exc:
            raise ManagedStateError("execution consumed lineage is not exact") from exc
        if execution_consumed != realization.consumed_records:
            raise ManagedStateError("execution consumed lineage does not match realization")
        for consumed in realization.consumed_records:
            record_io.read_record(consumed.record_id)
        for consumed in execution.consumed_records:
            record_io.read_record(consumed.record_id)

    def _assert_current(self) -> NamespaceState:
        if self._released or not self._lock.held:
            raise StaleManagedLeaseError("managed lease is closed or no longer owns its OS lock")
        if os.getpid() != self._owner_pid:
            raise StaleManagedLeaseError(
                "managed lease mutation is restricted to the acquiring coordinator process"
            )
        namespace = self.operation.managed_store._read_namespace(self.operation.key)
        assert namespace is not None
        if namespace.fence_epoch != self.epoch:
            raise StaleManagedLeaseError("managed lease fence has been superseded")
        if namespace.current_declaration_fingerprint != self.operation.declaration_fingerprint:
            raise StaleManagedLeaseError("managed lease declaration generation is no longer current")
        return namespace

    def _assert_no_unreconciled_reservation(self) -> None:
        self._assert_current()
        control = self.operation._read_control()
        assert control is not None
        if control.reserved_realization_id is not None:
            raise ManagedStateError(
                "managed lease has an unreconciled realization reservation; "
                "release and reacquire it before mutation"
            )

    def _heartbeat(self) -> None:
        self._assert_current()
        self._heartbeat_count += 1
        try:
            _write_json(
                self.operation._owner_path,
                {
                    "schema_version": 1,
                    "status": "owned",
                    "epoch": self.epoch,
                    "pid": self._owner_pid,
                    "heartbeat": self._heartbeat_count,
                },
            )
        except ManagedStateError:
            # Diagnostics never override an already durable control mutation.
            pass


def _new_id(kind: str) -> str:
    return f"{kind}-v1-{uuid.uuid4().hex}"


def _has_managed_capabilities(store: Store) -> bool:
    return all(store.supports_store_capability(name) for name in MANAGED_CAPABILITIES)


def _require_managed_capabilities(store: Store) -> None:
    if not _has_managed_capabilities(store):
        raise ManagedStoreUnsupportedError(
            "live managed operation control requires a capable local DirStore"
        )


def _require_managed_snapshot_capability(store: Store) -> None:
    if not store.supports_store_capability(MANAGED_SNAPSHOT_CAPABILITY):
        raise ManagedStoreUnsupportedError(
            "managed operation reads require a managed snapshot-capable Store"
        )


def _write_json(path: Path, data: Mapping[str, Any], *, immutable: bool = False) -> None:
    payload = canonical_json_bytes(data)
    temp_path = None
    try:
        _mkdir_durable(path.parent)
        if immutable and path.exists():
            if path.read_bytes() == payload:
                return
            raise ManagedStateError(
                "immutable managed state already exists with different bytes"
            )
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_path = Path(tmp.name)
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except ManagedStateError:
        raise
    except OSError as exc:
        raise ManagedStateError(f"managed state could not be durably written: {exc}") from exc
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        data = canonical_json_load_bytes(path.read_bytes())
    except OSError as exc:
        raise _TransientManagedStateReadError(
            f"{name} could not be read: {exc}"
        ) from exc
    except CanonicalJSONError as exc:
        raise ManagedStateError(f"{name} could not be read: {exc}") from exc
    if not isinstance(data, dict):
        raise ManagedStateError(f"{name} JSON root must be an object")
    return data


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _fsync_directory(directory.parent)


__all__ = [
    "MANAGED_CAPABILITIES",
    "MANAGED_SNAPSHOT_CAPABILITY",
    "ManagedOperationStore",
    "OperationControl",
    "OperationLease",
    "resolve_managed_store",
]
