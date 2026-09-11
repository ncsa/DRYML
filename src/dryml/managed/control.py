"""Closed, atomic current-operation control authority for managed operations.

Only the final ``managed/operations`` namespace is authoritative.  This module
uses staging directories for first publication and a bounded pending intent for
every current-file replacement; it intentionally keeps no lifecycle history.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from dataclasses import dataclass, replace
from collections.abc import Mapping
from pathlib import Path
from typing import Callable
from uuid import UUID
from types import MappingProxyType

from dryml.formats import CanonicalJSONError, canonical_json_bytes, canonical_json_load_bytes
from dryml.locking import LockError, interprocess_lock

from .errors import ManagedConflictError, ManagedControlError, ManagedPublicationError, ManagedRecoveryError
from .identity import _operation_digest_from_object_ref_digest
from .storage import _probe_state_ownership, state_ref_for_digest, validate_state_ref

_MAX_SNAPSHOT_BYTES = 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_FAILURE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_STATES = frozenset({"running", "interrupted", "failed", "completed"})
_MAX_GENERATION = 2**63 - 1


@dataclass(frozen=True, slots=True)
class ControlSnapshot:
    """One decoded current-operation authority record.

    Args:
        operation_id: Stable managed operation digest and path identifier.
        object_ref_digest: Exact receiver ObjectRef digest.
        argument_digest: Normalized managed argument digest.
        member: Declared Python member name.
        attempt_id: UUID hex retained across a resumed attempt.
        owner_id: Running invocation UUID hex, otherwise ``None``.
        generation: Monotonic current authority generation.
        state: Closed lifecycle state represented by this current snapshot.
        interrupt_request: Requested ``(attempt_id, owner_id)`` or ``None``.
        checkpoint_digest: Associated exact checkpoint StateRef digest or ``None``.
        final_digest: Exact completed StateRef digest or ``None``.
        failure_code: Static failure code for failed states or ``None``.

    Snapshot construction validates the closed schema and lifecycle cross-fields;
    Store-backed StateRef closure validation occurs when authority is read.
    """

    operation_id: str
    object_ref_digest: str
    argument_digest: str
    member: str
    attempt_id: str
    owner_id: str | None
    generation: int
    state: str
    interrupt_request: tuple[str, str] | None
    checkpoint_digest: str | None
    final_digest: str | None
    failure_code: str | None
    ownership: Mapping[str, object]

    def __post_init__(self) -> None:
        """Reject malformed current authority before it can be published or used."""

        _digest(self.operation_id, "operation_id")
        _digest(self.object_ref_digest, "object_ref_digest")
        _digest(self.argument_digest, "argument_digest")
        if type(self.member) is not str or not self.member or not self.member.isidentifier() or len(self.member.encode("utf-8")) > 255:
            raise ManagedControlError("invalid_current", "member must be a bounded Python identifier")
        if self.operation_id != _operation_digest_from_object_ref_digest(self.object_ref_digest, self.member):
            raise ManagedControlError("invalid_current", "operation_id does not match its object_ref_digest and member")
        _uuid(self.attempt_id, "attempt_id")
        if self.owner_id is not None:
            _uuid(self.owner_id, "owner_id")
        if type(self.generation) is not int or isinstance(self.generation, bool) or not 1 <= self.generation <= _MAX_GENERATION:
            raise ManagedControlError("invalid_current", "generation is outside the supported range")
        if type(self.state) is not str or self.state not in _STATES:
            raise ManagedControlError("invalid_current", "state is unknown")
        if self.interrupt_request is not None:
            if type(self.interrupt_request) is not tuple or len(self.interrupt_request) != 2:
                raise ManagedControlError("invalid_current", "interrupt_request must be an exact attempt/owner pair")
            _uuid(self.interrupt_request[0], "interrupt request attempt")
            _uuid(self.interrupt_request[1], "interrupt request owner")
        for value, name in ((self.checkpoint_digest, "checkpoint_digest"), (self.final_digest, "final_digest")):
            if value is not None:
                _digest(value, name)
        if self.failure_code is not None and (type(self.failure_code) is not str or not _FAILURE.fullmatch(self.failure_code)):
            raise ManagedControlError("invalid_current", "failure_code must be a bounded static code")
        normalized_ownership = _ownership(self.ownership)
        object.__setattr__(self, "ownership", normalized_ownership)
        _validate_cross_fields(self)

    def to_data(self) -> dict[str, object]:
        """Return the closed canonical JSON projection for this authority record."""

        return {
            "schema": "dryml-managed-current",
            "version": 2,
            "operation_id": self.operation_id,
            "object_ref_digest": self.object_ref_digest,
            "argument_digest": self.argument_digest,
            "member": self.member,
            "attempt_id": self.attempt_id,
            "owner_id": self.owner_id,
            "generation": self.generation,
            "state": self.state,
            "interrupt_request": None if self.interrupt_request is None else list(self.interrupt_request),
            "checkpoint_digest": self.checkpoint_digest,
            "final_digest": self.final_digest,
            "failure_code": self.failure_code,
            "ownership": {
                "version": self.ownership["version"],
                "store_keys": list(self.ownership["store_keys"]),
                "object_keys": list(self.ownership["object_keys"]),
            },
        }

    def to_bytes(self) -> bytes:
        """Encode this closed snapshot as bounded canonical JSON bytes."""

        return canonical_json_bytes(self.to_data(), max_depth=8, max_nodes=16384, max_entries=4096, max_string=512, max_int_bits=64)

    @classmethod
    def from_bytes(cls, payload: bytes, *, operation_id: str | None = None) -> "ControlSnapshot":
        """Decode one bounded closed current snapshot and verify its path identity."""

        if len(payload) > _MAX_SNAPSHOT_BYTES:
            raise ManagedControlError("snapshot_too_large", "current snapshot exceeds 1 MiB")
        try:
            data = canonical_json_load_bytes(payload, max_depth=8, max_nodes=16384, max_entries=4096, max_string=512, max_int_bits=64)
        except CanonicalJSONError as error:
            raise ManagedControlError("invalid_current", "current snapshot is not valid canonical JSON") from error
        required = {
            "schema", "version", "operation_id", "object_ref_digest", "argument_digest", "member", "attempt_id", "owner_id", "generation", "state", "interrupt_request", "checkpoint_digest", "final_digest", "failure_code", "ownership",
        }
        if not isinstance(data, dict) and not hasattr(data, "keys"):
            raise ManagedControlError("invalid_current", "current snapshot must be a mapping")
        if set(data) != required or data["schema"] != "dryml-managed-current" or type(data["version"]) is not int or data["version"] != 2:
            raise ManagedControlError("invalid_current", "current snapshot schema is unsupported")
        request = data["interrupt_request"]
        if request is not None:
            if type(request) is not tuple or len(request) != 2 or any(type(item) is not str for item in request):
                raise ManagedControlError("invalid_current", "interrupt_request is malformed")
            request = (request[0], request[1])
        snapshot = cls(
            operation_id=data["operation_id"], object_ref_digest=data["object_ref_digest"], argument_digest=data["argument_digest"], member=data["member"], attempt_id=data["attempt_id"], owner_id=data["owner_id"], generation=data["generation"], state=data["state"], interrupt_request=request, checkpoint_digest=data["checkpoint_digest"], final_digest=data["final_digest"], failure_code=data["failure_code"], ownership=data["ownership"],
        )
        if operation_id is not None and snapshot.operation_id != operation_id:
            raise ManagedControlError("invalid_current", "current snapshot operation_id does not match its path")
        return snapshot


@dataclass(frozen=True, slots=True)
class _PendingIntent:
    """Closed replacement intent retained only until one current replacement settles."""

    operation_id: str
    previous_generation: int | None
    previous_digest: str | None
    proposed_generation: int
    proposed_digest: str

    def __post_init__(self) -> None:
        _digest(self.operation_id, "operation_id")
        if (self.previous_generation is None) != (self.previous_digest is None):
            raise ManagedControlError("invalid_pending", "pending previous authority must be complete or absent")
        if self.previous_generation is not None and (type(self.previous_generation) is not int or not 1 <= self.previous_generation <= _MAX_GENERATION):
            raise ManagedControlError("invalid_pending", "pending previous generation is invalid")
        if self.previous_digest is not None:
            _digest(self.previous_digest, "previous_digest")
        if type(self.proposed_generation) is not int or not 1 <= self.proposed_generation <= _MAX_GENERATION:
            raise ManagedControlError("invalid_pending", "pending proposed generation is invalid")
        _digest(self.proposed_digest, "proposed_digest")
        if self.previous_generation is not None and self.proposed_generation != self.previous_generation + 1:
            raise ManagedControlError("invalid_pending", "pending generations are not adjacent")
        if self.previous_generation is None and self.proposed_generation != 1:
            raise ManagedControlError("invalid_pending", "initial pending authority must propose generation one")

    def to_bytes(self) -> bytes:
        """Encode this bounded pending intent as canonical JSON bytes."""

        return canonical_json_bytes({
            "schema": "dryml-managed-pending", "version": 1, "operation_id": self.operation_id,
            "previous_generation": self.previous_generation, "previous_digest": self.previous_digest,
            "proposed_generation": self.proposed_generation, "proposed_digest": self.proposed_digest,
        }, max_depth=2, max_nodes=16, max_entries=8, max_string=128, max_int_bits=64)

    @classmethod
    def from_bytes(cls, payload: bytes, *, operation_id: str) -> "_PendingIntent":
        """Decode one closed pending intent addressed to an operation path."""

        if len(payload) > 4096:
            raise ManagedControlError("invalid_pending", "pending intent exceeds its bound")
        try:
            data = canonical_json_load_bytes(payload, max_depth=2, max_nodes=16, max_entries=8, max_string=128, max_int_bits=64)
        except CanonicalJSONError as error:
            raise ManagedControlError("invalid_pending", "pending intent is not valid canonical JSON") from error
        required = {"schema", "version", "operation_id", "previous_generation", "previous_digest", "proposed_generation", "proposed_digest"}
        if not hasattr(data, "keys") or set(data) != required or data["schema"] != "dryml-managed-pending" or type(data["version"]) is not int or data["version"] != 1:
            raise ManagedControlError("invalid_pending", "pending intent schema is unsupported")
        intent = cls(data["operation_id"], data["previous_generation"], data["previous_digest"], data["proposed_generation"], data["proposed_digest"])
        if intent.operation_id != operation_id:
            raise ManagedControlError("invalid_pending", "pending intent operation_id does not match its path")
        return intent


class ManagedControlStore:
    """Managed-owned filesystem adapter over one selected control DirStore.

    Args:
        control_store: Explicit DirStore where this adapter owns only the closed
            ``managed/`` namespace.
        state_repo: Borrowed Repo used solely to validate retained exact StateRefs.

    Reads never bootstrap the namespace.  Mutating methods must be called only
    after managed lifecycle code has separately obtained its graph ownership.
    """

    def __init__(self, control_store, state_repo) -> None:
        """Bind control Store and state Repo without changing either resource."""

        from dryml.core.store.dir import DirStore

        from dryml.core.repo import Repo
        if type(control_store) is not DirStore or not isinstance(state_repo, Repo):
            raise ManagedControlError("invalid_store", "managed control requires an exact DirStore and Repo")
        self.control_store = control_store
        self.state_repo = state_repo

    @property
    def root(self) -> str:
        """Return the managed namespace root without creating it."""

        return os.path.join(self.control_store.base_dir, "managed")

    def inspect(self, operation_id: str) -> ControlSnapshot | None:
        """Read exact current authority without bootstrapping a namespace.

        A present namespace is read under its existing short control lock so the
        pending check and current/reference validation observe one generation.

        Raises:
            ManagedControlError: If retained current/pending authority or any
                existing hierarchy component is malformed, unavailable, or requires
                reconciliation.
            ManagedRecoveryError: If a referenced StateRef closure is incomplete.
        """

        _digest(operation_id, "operation_id")
        if not self._namespace_present():
            return None
        operation = self._operation_path(operation_id)
        with self._control_lock():
            if not self._operation_directory_present(operation):
                return None
            if self._path_present(self._pending_path(operation), "pending authority"):
                raise ManagedControlError("pending_reconciliation", "managed current authority has a pending replacement")
            snapshot, _ = self._read_current(operation, operation_id)
            self._validate_references(snapshot)
            return snapshot

    def initialize(self) -> None:
        """Publish the closed managed format gate using the Store writer lock."""

        try:
            self.control_store.preflight_publication("managed namespace bootstrap")
            with self.control_store.writer_lock():
                if self._path_present(self.root, "managed namespace"):
                    self._validate_namespace()
                    return
                parent = self.control_store.base_dir
                staging = tempfile.mkdtemp(prefix=".managed-staging-", dir=parent)
                try:
                    self._write_new(os.path.join(staging, "format.json"), _format_bytes())
                    _sync_directory(staging)
                    os.rename(staging, self.root)
                    _sync_directory(parent)
                except BaseException:
                    _remove_tree_if_present(staging)
                    raise
        except ManagedControlError:
            raise
        except (LockError, OSError) as error:
            raise ManagedControlError("namespace_bootstrap_failed", "could not publish managed namespace") from error

    def create_initial(self, snapshot: ControlSnapshot) -> ControlSnapshot:
        """Stage and commit a new generation-one operation current authority.

        The directory is first installed with an initial-absence pending intent.
        If process death interrupts the final acknowledgement, normal inspection
        blocks until :meth:`reconcile` validates that exact initial generation.

        Raises:
            ManagedControlError: If the selected hierarchy is malformed or cannot
                be safely inspected or created.
            ManagedConflictError: If operation lineage authority already exists.
            ManagedPublicationError: If initial authority cannot be committed.
        """

        if snapshot.generation != 1:
            raise ManagedControlError("invalid_initial_generation", "new operations must start at generation one")
        self._validate_references(snapshot)
        self.initialize()
        operation = self._operation_path(snapshot.operation_id)
        with self._control_lock():
            if self._operation_directory_present(operation):
                raise ManagedConflictError("operation_exists", "operation lineage already exists")
            parent = os.path.dirname(operation)
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as error:
                raise ManagedControlError("operation_path_create_failed", "could not create managed operation hierarchy") from error
            staging = tempfile.mkdtemp(prefix=".operation-staging-", dir=parent)
            published = False
            try:
                current = snapshot.to_bytes()
                digest = _payload_digest(current)
                intent = _PendingIntent(snapshot.operation_id, None, None, 1, digest)
                self._write_new(os.path.join(staging, "current.json"), current)
                self._write_new(os.path.join(staging, "pending.json"), intent.to_bytes())
                _sync_directory(staging)
                os.rename(staging, operation)
                published = True
                _sync_directory(parent)
            except BaseException as error:
                _remove_tree_if_present(staging)
                outcome = "indeterminate" if published or self._path_present(operation, "operation authority") else "not_committed"
                raise ManagedPublicationError(outcome, "could not publish initial managed operation authority") from error
            return self._reconcile_locked(operation, snapshot.operation_id)

    def transition(
            self,
            operation_id: str,
            proposed: ControlSnapshot,
            *,
            expected_generation: int,
            expected_attempt_id: str | None = None,
            expected_owner_id: str | None = None) -> ControlSnapshot:
        """Atomically publish one expected-next current snapshot.

        Args enforce the complete selected current precondition.  The caller must
        supply a generation exactly one greater than authority; no stale cached
        current can overwrite a concurrent request or transition.

        Raises:
            ManagedControlError: If selected authority or its hierarchy is
                malformed, unavailable, or requires reconciliation.
            ManagedConflictError: If the selected current no longer meets an
                expected precondition.
            ManagedPublicationError: If replacement publication is uncertain.
        """

        _digest(operation_id, "operation_id")
        if proposed.operation_id != operation_id:
            raise ManagedControlError("operation_mismatch", "proposed operation_id does not match selected authority")
        operation = self._operation_path(operation_id)
        with self._control_lock():
            current, current_payload = self._read_unpending_current(operation, operation_id)
            if current.generation != expected_generation:
                raise ManagedConflictError("generation_conflict", "current generation changed before publication")
            if expected_attempt_id is not None and current.attempt_id != expected_attempt_id:
                raise ManagedConflictError("attempt_conflict", "current attempt changed before publication")
            if expected_owner_id is not None and current.owner_id != expected_owner_id:
                raise ManagedConflictError("owner_conflict", "current owner changed before publication")
            if current.generation == _MAX_GENERATION:
                raise ManagedControlError("generation_overflow", "managed generation cannot advance beyond 2**63-1")
            if proposed.generation != current.generation + 1:
                raise ManagedConflictError("generation_conflict", "proposed generation is not the next authority generation")
            self._validate_references(proposed)
            return self._publish_replacement(operation, current, current_payload, proposed)

    def transition_running_owner(
            self,
            operation_id: str,
            *,
            attempt_id: str,
            owner_id: str,
            build: Callable[[ControlSnapshot], ControlSnapshot]) -> ControlSnapshot:
        """Publish one owner-fenced running transition from fresh authority.

        Args:
            operation_id: Selected managed operation identity.
            attempt_id: Exact active attempt that must still own current authority.
            owner_id: Exact active invocation owner that must still be running.
            build: Function that receives the freshly reread running snapshot and
                returns its next-generation replacement, or ``None`` to make a
                read-only owner-fenced decision.

        Returns:
            The committed replacement, or ``None`` when ``build`` made no change.

        Raises:
            ManagedConflictError: If the attempt/owner changed or the selected
                authority is no longer running.
            ManagedControlError: If control authority cannot be safely read or
                published.

        Side Effects:
            Holds only the short control lock while rereading, validating, and
            publishing the replacement. A concurrent request is therefore merged
            through ``build`` rather than discarded with a stale generation.
        """

        _digest(operation_id, "operation_id")
        _uuid(attempt_id, "attempt_id")
        _uuid(owner_id, "owner_id")
        operation = self._operation_path(operation_id)
        with self._control_lock():
            current, current_payload = self._read_unpending_current(operation, operation_id)
            if (
                current.state != "running"
                or current.attempt_id != attempt_id
                or current.owner_id != owner_id
            ):
                raise ManagedConflictError("owner_conflict", "selected running owner changed before publication")
            if current.generation == _MAX_GENERATION:
                raise ManagedControlError("generation_overflow", "managed generation cannot advance beyond 2**63-1")
            proposed = build(current)
            if proposed is None:
                return None
            if (
                proposed.operation_id != operation_id
                or proposed.generation != current.generation + 1
            ):
                raise ManagedConflictError("generation_conflict", "owner transition did not produce the next selected generation")
            self._validate_references(proposed)
            return self._publish_replacement(operation, current, current_payload, proposed)

    def observe_running_owner(self, operation_id: str) -> tuple[ControlSnapshot | None, bool, bool]:
        """Observe a running owner through one generation-rechecked lock probe.

        Returns:
            ``(snapshot, ownerless, stable)``. ``ownerless`` is true only when a
            complete lock probe succeeded and the same running generation remained
            selected. ``stable`` is false when authority changed during the probe.

        Raises:
            ManagedControlError: If control or lock-probe authority is unavailable.

        Side Effects:
            Holds the short control lock across the current read, nonblocking state
            lock probe, and generation reread. It never writes control authority.
        """

        _digest(operation_id, "operation_id")
        operation = self._operation_path(operation_id)
        with self._control_lock():
            current, ownerless, stable = self._observe_running_owner_locked(operation, operation_id)
            return current, ownerless, stable

    def request_interrupt_if_running(
            self, operation_id: str, *, expected_attempt_id: str | None = None,
    ) -> tuple[str, ControlSnapshot | None]:
        """Probe and request interruption under one short control-lock critical section.

        Returns:
            A closed internal outcome and its selected snapshot. ``changed`` means
            that a generation changed during probing and callers must retry instead
            of treating stale evidence as owner loss.

        Raises:
            ManagedControlError: If control or probe adapters fail.

        Side Effects:
            May publish exactly one same-owner interruption request. A stale
            attempt precondition is checked before opening state lock probes.
        """

        _digest(operation_id, "operation_id")
        operation = self._operation_path(operation_id)
        with self._control_lock():
            current, ownerless, stable = self._observe_running_owner_locked(
                operation, operation_id, expected_attempt_id=expected_attempt_id,
            )
            if current is None or current.state != "running":
                return "not_running", current
            if expected_attempt_id is not None and current.attempt_id != expected_attempt_id:
                return "stale_attempt", current
            if not stable:
                return "changed", current
            if ownerless:
                return "not_running", current
            if current.interrupt_request == (current.attempt_id, current.owner_id):
                return "already_requested", current
            proposed = replace(
                current, generation=current.generation + 1,
                interrupt_request=(current.attempt_id, current.owner_id),
            )
            return "requested", self._publish_replacement(
                operation, current, current.to_bytes(), proposed,
            )

    def reconcile(self, operation_id: str) -> ControlSnapshot | None:
        """Resolve exactly one retained pending intent under the short control lock.

        Returns:
            The settled snapshot, or ``None`` when the valid namespace or selected
            operation authority is absent.

        Raises:
            ManagedControlError: If retained authority or any existing hierarchy
                component is malformed or unavailable.
            ManagedPublicationError: If a reconciled generation cannot be made
                durable.
        """

        _digest(operation_id, "operation_id")
        if not self._namespace_present():
            return None
        operation = self._operation_path(operation_id)
        with self._control_lock():
            if not self._operation_directory_present(operation):
                return None
            if not self._path_present(self._pending_path(operation), "pending authority"):
                snapshot, _ = self._read_current(operation, operation_id)
                self._validate_references(snapshot)
                return snapshot
            return self._reconcile_locked(operation, operation_id)

    def probe_state_ownership(self, ownership) -> bool:
        """Probe complete selected state-lock availability without changing control.

        Args:
            ownership: Exact retained Store/ObjectId lock evidence from current
                control authority.

        Returns:
            ``True`` only when every lock was observed free and released again;
            ``False`` is contention, not evidence about a snapshot owner's life.

        Raises:
            ManagedStoreError: If state-lock capability or its format gate is not
                available for a reliable probe.

        Side Effects:
            Opens and releases short nonblocking leases only. Callers must re-read the
            selected generation before interpreting this evidence.
        """

        return _probe_state_ownership(self.state_repo, ownership)

    def _read_unpending_current(self, operation: str, operation_id: str) -> tuple[ControlSnapshot, bytes]:
        """Read and validate current authority while the caller holds control lock."""

        if not self._operation_directory_present(operation):
            raise ManagedRecoveryError("missing_current", "managed operation authority is absent")
        if self._path_present(self._pending_path(operation), "pending authority"):
            raise ManagedControlError("pending_reconciliation", "managed current authority has a pending replacement")
        current, payload = self._read_current(operation, operation_id)
        self._validate_references(current)
        return current, payload

    def _observe_running_owner_locked(
            self, operation: str, operation_id: str,
            *, expected_attempt_id: str | None = None,
    ) -> tuple[ControlSnapshot | None, bool, bool]:
        """Return a generation-rechecked owner-loss observation under control lock."""

        if not self._operation_directory_present(operation):
            return None, False, True
        current, _ = self._read_unpending_current(operation, operation_id)
        if current.state != "running" or (
            expected_attempt_id is not None and current.attempt_id != expected_attempt_id
        ):
            return current, False, True
        ownerless = self.probe_state_ownership(current.ownership)
        observed, _ = self._read_unpending_current(operation, operation_id)
        if (
            observed.generation != current.generation
            or observed.state != "running"
            or observed.attempt_id != current.attempt_id
            or observed.owner_id != current.owner_id
        ):
            return observed, False, False
        return observed, ownerless, True

    def _publish_replacement(self, operation: str, current: ControlSnapshot, current_payload: bytes, proposed: ControlSnapshot) -> ControlSnapshot:
        """Publish intent then current replacement, retaining evidence on uncertainty."""

        proposed_payload = proposed.to_bytes()
        intent = _PendingIntent(proposed.operation_id, current.generation, _payload_digest(current_payload), proposed.generation, _payload_digest(proposed_payload))
        pending = self._pending_path(operation)
        if self._path_present(pending, "pending authority"):
            raise ManagedControlError("pending_reconciliation", "managed current authority has a pending replacement")
        try:
            _publish_new_file(pending, intent.to_bytes())
            _sync_directory(operation)
        except BaseException as error:
            outcome = "indeterminate" if self._path_present(pending, "pending authority") else "not_committed"
            raise ManagedPublicationError(outcome, "could not publish managed replacement intent") from error
        try:
            self._replace_file(self._current_path(operation), proposed_payload)
            _sync_directory(operation)
        except BaseException as error:
            raise ManagedPublicationError("indeterminate", "managed current replacement may require reconciliation") from error
        self._clear_intent_and_acknowledge(operation, proposed, _payload_digest(proposed_payload))
        return proposed

    def _reconcile_locked(self, operation: str, operation_id: str) -> ControlSnapshot:
        """Accept only a current snapshot exactly named by its pending intent."""

        intent = _PendingIntent.from_bytes(self._read_regular(self._pending_path(operation)), operation_id=operation_id)
        current, payload = self._read_current(operation, operation_id)
        digest = _payload_digest(payload)
        old = intent.previous_generation == current.generation and intent.previous_digest == digest
        new = intent.proposed_generation == current.generation and intent.proposed_digest == digest
        if old == new:
            raise ManagedControlError("ambiguous_pending", "pending intent does not identify the current authority")
        self._validate_references(current)
        if new:
            try:
                _sync_file(self._current_path(operation))
            except OSError as error:
                raise ManagedPublicationError("indeterminate", "reconciled current authority could not be synchronized") from error
        self._clear_intent_and_acknowledge(operation, current, digest)
        return current

    def _clear_intent_and_acknowledge(self, operation: str, snapshot: ControlSnapshot, digest: str) -> None:
        """Clear one settled intent or confirm its exact committed current after a fault."""

        pending = self._pending_path(operation)
        try:
            os.unlink(pending)
            _sync_directory(operation)
            _publication_acknowledged()
        except BaseException as error:
            if not self._path_present(pending, "pending authority"):
                self._confirm_current(operation, snapshot, digest)
                return
            raise ManagedPublicationError("indeterminate", "managed current committed but acknowledgement is uncertain") from error

    def _confirm_current(self, operation: str, expected: ControlSnapshot, digest: str) -> None:
        """Verify the exact settled generation without overwriting any authority."""

        current, payload = self._read_current(operation, expected.operation_id)
        if current.generation != expected.generation or _payload_digest(payload) != digest:
            raise ManagedControlError("ambiguous_current", "current authority changed while acknowledging publication")
        self._validate_references(current)

    def _namespace_present(self) -> bool:
        """Return whether a valid final managed namespace exists without creating it."""

        if not self._path_present(self.root, "managed namespace"):
            return False
        self._validate_namespace()
        return True

    def _validate_namespace(self) -> None:
        """Reject malformed final format authority instead of treating it as absent."""

        self._require_directory(self.root, "managed namespace")
        gate = os.path.join(self.root, "format.json")
        if not self._path_present(gate, "managed format gate"):
            raise ManagedRecoveryError("invalid_managed_namespace", "managed namespace lacks format.json")
        if _read_format(self._read_regular(gate)) is not True:
            raise ManagedRecoveryError("invalid_managed_namespace", "managed namespace format gate is invalid")

    def _operation_path(self, operation_id: str) -> str:
        return os.path.join(self.root, "operations", "v1", operation_id[:2], operation_id)

    @staticmethod
    def _current_path(operation: str) -> str:
        return os.path.join(operation, "current.json")

    @staticmethod
    def _pending_path(operation: str) -> str:
        return os.path.join(operation, "pending.json")

    def _read_current(self, operation: str, operation_id: str) -> tuple[ControlSnapshot, bytes]:
        """Read an existing complete current file; absence is never a restart."""

        if not self._operation_directory_present(operation):
            raise ManagedRecoveryError("missing_current", "managed operation authority is absent")
        path = self._current_path(operation)
        if not self._path_present(path, "current authority"):
            raise ManagedRecoveryError("missing_current", "existing managed operation lacks current authority")
        payload = self._read_regular(path)
        return ControlSnapshot.from_bytes(payload, operation_id=operation_id), payload

    def _validate_references(self, snapshot: ControlSnapshot) -> None:
        """Validate every retained StateRef digest before it is accepted as control."""

        for digest in (snapshot.checkpoint_digest, snapshot.final_digest):
            if digest is None:
                continue
            try:
                state_ref = state_ref_for_digest(self.state_repo, digest)
                if state_ref.object.digest() != snapshot.object_ref_digest:
                    raise ManagedRecoveryError("state_object_mismatch", "managed current StateRef does not match its object reference")
                validate_state_ref(self.state_repo, state_ref)
            except ManagedRecoveryError:
                raise
            except BaseException as error:
                raise ManagedRecoveryError("invalid_state_reference", "managed current references unreadable StateRef authority") from error

    def _control_lock(self):
        """Return the stable short managed control lock without bootstrap overlap."""

        return interprocess_lock(os.path.join(self.root, "control.lock"))

    def _operation_directory_present(self, operation: str) -> bool:
        """Validate the operation hierarchy without creating missing directories."""

        operations = os.path.join(self.root, "operations")
        version = os.path.join(operations, "v1")
        shard = os.path.dirname(operation)
        for path, label in (
            (operations, "operations directory"),
            (version, "operations version directory"),
            (shard, "operation shard directory"),
            (operation, "operation directory"),
        ):
            if not self._directory_present(path, label):
                return False
        return True

    @staticmethod
    def _path_present(path: str, label: str) -> bool:
        """Return whether one authority path exists without suppressing I/O errors."""

        try:
            os.lstat(path)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ManagedControlError("control_path_unavailable", f"could not inspect {label}") from error
        return True

    @classmethod
    def _directory_present(cls, path: str, label: str) -> bool:
        """Return whether a present authority path is a non-symlink directory."""

        if not cls._path_present(path, label):
            return False
        cls._require_directory(path, label)
        return True

    @staticmethod
    def _read_regular(path: str) -> bytes:
        try:
            mode = os.lstat(path).st_mode
        except OSError as error:
            raise ManagedControlError("control_path_unavailable", "could not inspect managed authority file") from error
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            raise ManagedControlError("invalid_control_file", "managed authority files must be regular files")
        try:
            with open(path, "rb") as source:
                payload = source.read(_MAX_SNAPSHOT_BYTES + 1)
        except OSError as error:
            raise ManagedControlError("control_path_unavailable", "could not read managed authority file") from error
        if len(payload) > _MAX_SNAPSHOT_BYTES:
            raise ManagedControlError("snapshot_too_large", "managed authority file exceeds 1 MiB")
        return payload

    @staticmethod
    def _require_directory(path: str, label: str) -> None:
        try:
            mode = os.lstat(path).st_mode
        except OSError as error:
            raise ManagedControlError("control_path_unavailable", f"could not inspect {label}") from error
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise ManagedControlError("invalid_control_path", f"{label} must be a directory")

    @staticmethod
    def _write_new(path: str, payload: bytes) -> None:
        _write_new(path, payload)

    @staticmethod
    def _replace_file(path: str, payload: bytes) -> None:
        _replace_file(path, payload)


def _validate_cross_fields(snapshot: ControlSnapshot) -> None:
    """Enforce lifecycle-field combinations that prevent fabricated success."""

    if snapshot.state == "running":
        if snapshot.owner_id is None or snapshot.final_digest is not None or snapshot.failure_code is not None:
            raise ManagedControlError("invalid_current", "running authority requires owner and no final/failure")
        if snapshot.interrupt_request is not None and snapshot.interrupt_request != (snapshot.attempt_id, snapshot.owner_id):
            raise ManagedControlError("invalid_current", "running request must address the exact current owner")
    elif snapshot.state == "interrupted":
        if snapshot.owner_id is not None or snapshot.interrupt_request is not None or snapshot.final_digest is not None or snapshot.failure_code is not None:
            raise ManagedControlError("invalid_current", "interrupted authority has no owner/request/final/failure")
    elif snapshot.state == "failed":
        if snapshot.owner_id is not None or snapshot.interrupt_request is not None or snapshot.final_digest is not None or snapshot.failure_code is None:
            raise ManagedControlError("invalid_current", "failed authority requires only a failure code")
    else:
        if snapshot.owner_id is not None or snapshot.interrupt_request is not None or snapshot.failure_code is not None or snapshot.final_digest is None:
            raise ManagedControlError("invalid_current", "completed authority requires only a final StateRef")


def _digest(value: object, name: str) -> None:
    if type(value) is not str or not _HEX.fullmatch(value):
        raise ManagedControlError("invalid_current", f"{name} must be a lowercase SHA-256 digest")


def _ownership(value: object) -> MappingProxyType:
    """Normalize closed v2 hashed ownership evidence without accepting locators."""

    if not isinstance(value, Mapping) or set(value) != {"version", "store_keys", "object_keys"}:
        raise ManagedControlError("invalid_current", "ownership must be a closed mapping")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ManagedControlError("invalid_current", "ownership version is unsupported")
    stores, objects = value["store_keys"], value["object_keys"]
    if type(stores) not in (list, tuple) or type(objects) not in (list, tuple):
        raise ManagedControlError("invalid_current", "ownership keys must be lists")
    if not stores or not objects or len(stores) > 256 or len(objects) > 4096 or len(stores) * len(objects) > 65536:
        raise ManagedControlError("invalid_current", "ownership topology is outside supported bounds")
    for key in (*stores, *objects):
        _digest(key, "ownership key")
    if tuple(stores) != tuple(sorted(set(stores))) or tuple(objects) != tuple(sorted(set(objects))):
        raise ManagedControlError("invalid_current", "ownership keys must be sorted and unique")
    return MappingProxyType({
        "version": 1,
        "store_keys": tuple(stores),
        "object_keys": tuple(objects),
    })


def _uuid(value: object, name: str) -> None:
    if type(value) is not str:
        raise ManagedControlError("invalid_current", f"{name} must be UUID hex")
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as error:
        raise ManagedControlError("invalid_current", f"{name} must be UUID hex") from error
    if parsed.hex != value:
        raise ManagedControlError("invalid_current", f"{name} must be lowercase UUID hex")


def _payload_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _format_bytes() -> bytes:
    return canonical_json_bytes({"schema": "dryml-managed", "version": 1}, max_depth=1, max_nodes=4, max_entries=2, max_string=64, max_int_bits=8)


def _read_format(payload: bytes) -> bool:
    try:
        data = canonical_json_load_bytes(payload, max_depth=1, max_nodes=4, max_entries=2, max_string=64, max_int_bits=8)
    except CanonicalJSONError as error:
        raise ManagedRecoveryError("invalid_managed_namespace", "managed format gate is malformed") from error
    return hasattr(data, "keys") and set(data) == {"schema", "version"} and data["schema"] == "dryml-managed" and type(data["version"]) is int and data["version"] == 1


def _write_new(path: str, payload: bytes) -> None:
    """Create and synchronize a previously absent regular authority file."""

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError("managed authority write was incomplete")
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_file(path: str, payload: bytes) -> None:
    """Synchronize a same-directory temporary file and atomically replace current."""

    parent = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix=".current-", dir=parent)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError("managed current temporary write was incomplete")
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, path)
    except BaseException:
        if fd != -1:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _publish_new_file(path: str, payload: bytes) -> None:
    """Atomically publish a complete previously absent intent from a synced temporary."""

    parent = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=parent)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError("managed pending temporary write was incomplete")
        os.fsync(fd)
        os.close(fd)
        fd = -1
        if os.path.lexists(path):
            raise FileExistsError("managed pending intent already exists")
        os.replace(temporary, path)
    except BaseException:
        if fd != -1:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sync_file(path: str) -> None:
    """Synchronize a regular file through the platform-supported file seam."""

    with open(path, "r+b") as source:
        os.fsync(source.fileno())


def _sync_directory(path: str) -> None:
    """Synchronize directory entries on POSIX without using unsupported Windows FDs."""

    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publication_acknowledged() -> None:
    """Publication receipt seam used to model acknowledgement failure in tests."""


def _remove_tree_if_present(path: str) -> None:
    """Remove only this call's unpublished staging directory when safe."""

    try:
        Path(path).rmdir()
    except OSError:
        # Staging is non-authoritative diagnostic evidence after a partial failure.
        pass


__all__ = ["ControlSnapshot", "ManagedControlStore"]
