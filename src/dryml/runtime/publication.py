"""Transactional process-global runtime publication.

The service in this module is the sole mutable authority for the current
runtime generation.  It intentionally accepts only precomputed effects: no
imports, adapters, loaders, or user callbacks are invoked while a writer owns
the transaction.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from dryml._framework_imports import ImportEpochBusyError, ImportEpochReentryError, coordinator

from .allocation import NoAllocation
from .enforcement import RequirementAxes, RuntimeEnforcement
from .errors import PublicationBusyError, PublicationError, PublicationFailedError, PublicationReentryError
from .modes import RuntimeMode


def _frozen_mapping(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class SessionGeneration:
    """One immutable process runtime generation and its durable health facts."""

    number: int
    runtime: Any
    visibility_epoch: Any = None
    health: str = "healthy"
    restart_guidance: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PublicationCandidate:
    """A candidate normalized against one exact immutable base generation."""

    expected: SessionGeneration
    generation: SessionGeneration


@dataclass(frozen=True, slots=True)
class EffectPlan:
    """Preplanned reversible process effects for a short writer transaction."""

    environment: Mapping[str, str | None] = field(default_factory=dict)
    interceptor: Any = None
    interceptor_position: int | None = None
    cpu_affinity: tuple[int, ...] | None = None
    process_limits: Mapping[int, tuple[int, int]] = field(default_factory=dict)
    dedicated_process: bool = False
    irreversible_outcome: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", _frozen_mapping(self.environment))
        object.__setattr__(self, "process_limits", _frozen_mapping({kind: tuple(value) for kind, value in self.process_limits.items()}))
        if self.cpu_affinity is not None:
            object.__setattr__(self, "cpu_affinity", tuple(sorted({int(cpu) for cpu in self.cpu_affinity})))


@dataclass(frozen=True, slots=True)
class EffectRecord:
    """One verified process effect recorded for ownership-aware rollback."""

    kind: str
    key: Any
    previous: Any
    written: Any


@dataclass(frozen=True, slots=True)
class FrameworkAdmission:
    """Immutable facts admitted for one wrapped framework loader callback."""

    generation: int
    control_epoch: int
    registry_revision: int
    group: str
    lifecycle: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class MaterializationFence:
    """Reader-free bridge between PEP-451 creation and execution callbacks."""

    admission: FrameworkAdmission
    spec_id: int
    module_id: int | None


class PublicationService:
    """Own generation publication, leases, and import reader/writer admission."""

    def __init__(
        self,
        *,
        environ: MutableMapping[str, str] | None = None,
        meta_path: list[Any] | None = None,
        windows: bool | None = None,
        affinity_getter: Any = None,
        affinity_setter: Any = None,
        limit_getter: Any = None,
        limit_setter: Any = None,
    ) -> None:
        """Initialize one isolated publication authority.

        Args:
            environ: Process environment mapping to own during transitions.
            meta_path: Import finder list to update transactionally.
            windows: Whether environment keys are case-insensitive.
            affinity_getter: Optional CPU-affinity reader for tests or platforms.
            affinity_setter: Optional CPU-affinity writer for tests or platforms.
            limit_getter: Optional process-limit reader for tests or platforms.
            limit_setter: Optional process-limit writer for tests or platforms.

        Returns:
            None.
        """
        self._state_lock = threading.Lock()
        self._generation: SessionGeneration | None = None
        self._leases: dict[int, int] = {}
        self._effects: tuple[EffectRecord, ...] = ()
        self._materializations: dict[tuple[str, int], MaterializationFence] = {}
        self._framework_finalizers: dict[tuple[str, str, int], str] = {}
        self._framework_pre_stages: dict[tuple[int, str, str], str] = {}
        self._environ = os.environ if environ is None else environ
        self._meta_path = sys.meta_path if meta_path is None else meta_path
        self._windows = os.name == "nt" if windows is None else windows
        self._affinity_getter = affinity_getter or self._default_affinity_getter
        self._affinity_setter = affinity_setter or self._default_affinity_setter
        self._limit_getter = limit_getter or self._default_limit_getter
        self._limit_setter = limit_setter or self._default_limit_setter

    def initialize(self, runtime: Any) -> SessionGeneration:
        """Install the import-time baseline once and return its generation."""

        with self._state_lock:
            if self._generation is None:
                self._generation = SessionGeneration(0, runtime)
            return self._generation

    def current(self) -> SessionGeneration:
        """Return the current generation without maintaining a second state copy."""

        self._reject_writer_reentry()
        with self._state_lock:
            if self._generation is None:
                raise PublicationError("runtime publication has not been initialized")
            return self._generation

    snapshot = current

    def stage(self, expected: SessionGeneration, generation: SessionGeneration) -> PublicationCandidate:
        """Bind a normalized generation to the exact generation it observed."""

        self._reject_writer_reentry()
        if generation.number <= expected.number:
            raise PublicationError("published generation numbers must increase monotonically")
        return PublicationCandidate(expected, generation)

    @contextmanager
    def reader(self) -> Iterator[SessionGeneration]:
        """Hold bounded import-reader admission while observing one generation."""

        self._reject_writer_reentry()
        try:
            with coordinator.reader():
                with self._state_lock:
                    generation = self._require_generation()
                yield generation
        except ImportEpochReentryError as exc:
            raise PublicationReentryError(str(exc), context={"phase": "reader"}) from exc

    @contextmanager
    def writer(self) -> Iterator[None]:
        """Hold an exclusive, non-upgradable transition writer token."""

        self._reject_writer_reentry()
        try:
            with coordinator.writer():
                yield
        except ImportEpochBusyError as exc:
            reason = "reader_busy" if coordinator.reader_count else "writer_busy"
            raise PublicationBusyError(str(exc), context={"phase": "writer", "reason": reason}) from exc
        except ImportEpochReentryError as exc:
            raise PublicationReentryError(str(exc), context={"phase": "writer"}) from exc

    def commit(
        self,
        candidate: PublicationCandidate,
        effects: EffectPlan | None = None,
        *,
        validator: Any = None,
        validator_rollback: Any = None,
    ) -> SessionGeneration:
        """CAS-commit a preplanned generation and reversible process effects.

        ``validator`` runs under exclusive writer admission, but outside the
        state mutex. ``validator_rollback`` must be supplied before validation
        when validation can provisionally mutate process-global state. It runs
        under the same writer admission if publication fails without poisoning
        the process. Arbitrary callbacks must never run while publication state
        is locked. The caller must resolve all framework and adapter work before
        calling this method.
        """

        self._reject_writer_reentry()
        effects = effects or EffectPlan()
        with self.writer():
            records: list[EffectRecord] = []
            current: SessionGeneration | None = None
            try:
                with self._state_lock:
                    self._validate_candidate_locked(candidate, effects)
                if validator is not None:
                    validator()
                with self._state_lock:
                    current = self._validate_candidate_locked(candidate, effects)
                    journal = list(self._effects)
                    self._validate_effect_plan(effects)
                    self._apply(effects, records, journal)
                    self._publish(candidate.generation)
                    self._effects = tuple(journal)
                    validator_rollback = None
                    return candidate.generation
            except BaseException as exc:
                with self._state_lock:
                    rollback_ok = self._rollback(records)
                    irreversible_applied = any(record.kind == "irreversible" for record in records)
                    if rollback_ok and not irreversible_applied:
                        if current is not None:
                            self._generation = current
                    else:
                        validator_rollback = None
                        failed_from = current or self._require_generation()
                        self._generation = self._terminal_generation(
                            failed_from,
                            restart_guidance="restart the process; a runtime process effect could not be safely restored",
                            metadata={"failure": type(exc).__name__},
                        )
                if not rollback_ok or irreversible_applied:
                    raise PublicationFailedError("runtime publication failed closed; restart the process", context={"cause": type(exc).__name__}) from exc
                if callable(validator_rollback):
                    validator_rollback()
                raise

    @contextmanager
    def lease(self) -> Iterator[SessionGeneration]:
        """Pin the current generation through a direct operation lifecycle."""

        self._reject_writer_reentry()
        with self._state_lock:
            generation = self._require_generation()
            if generation.health == "failed":
                raise PublicationFailedError("runtime publication is failed; restart the process")
            self._leases[generation.number] = self._leases.get(generation.number, 0) + 1
        try:
            yield generation
        finally:
            with self._state_lock:
                remaining = self._leases[generation.number] - 1
                if remaining:
                    self._leases[generation.number] = remaining
                else:
                    del self._leases[generation.number]

    def effect_journal(self) -> tuple[EffectRecord, ...]:
        """Return immutable records for all process effects currently owned."""

        self._reject_writer_reentry()
        with self._state_lock:
            return self._effects

    def admit_framework(self, group: str, lifecycle: str, fingerprint: str, registry_revision: int) -> FrameworkAdmission:
        """Capture immutable generation facts while an import reader is active."""

        self._reject_writer_reentry()
        with self._state_lock:
            generation = self._require_generation()
            return FrameworkAdmission(
                generation.number,
                int(generation.metadata.get("control_epoch", 0)),
                registry_revision,
                group,
                lifecycle,
                fingerprint,
            )

    def store_materialization(self, key: tuple[str, int], fence: MaterializationFence) -> None:
        """Publish one immutable creation fence without retaining a reader."""

        self._reject_writer_reentry()
        with self._state_lock:
            if key in self._materializations:
                raise PublicationError("repeated framework module creation is unsupported", context={"module": key[0]})
            self._materializations[key] = fence

    def materialization(self, key: tuple[str, int]) -> MaterializationFence:
        """Return the persisted creation fence required by ``exec_module``."""

        self._reject_writer_reentry()
        with self._state_lock:
            try:
                return self._materializations[key]
            except KeyError as exc:
                raise PublicationError("framework execution requires prior PEP-451 module creation", context={"module": key[0]}) from exc

    def validate_materialization(self, fence: MaterializationFence, expected: FrameworkAdmission) -> None:
        """Accept only an exact plan/control descendant of a creation fence."""

        self._reject_writer_reentry()
        with self._state_lock:
            current = self._require_generation()
            if current.health == "failed":
                raise PublicationFailedError("runtime publication is failed; restart the process")
            admitted = fence.admission
            if (
                int(current.metadata.get("control_epoch", 0)) != admitted.control_epoch
                or expected.control_epoch != admitted.control_epoch
                or expected.registry_revision != admitted.registry_revision
                or expected.group != admitted.group
                or expected.lifecycle != admitted.lifecycle
                or expected.fingerprint != admitted.fingerprint
            ):
                raise PublicationError(
                    "framework materialization is stale after a control transition; restart the process",
                    context={"group": admitted.group},
                )

    def finalize_framework(self, admission: FrameworkAdmission, statuses: Mapping[str, Any], *, failure: BaseException | None = None) -> SessionGeneration:
        """Merge same-control-epoch framework outcomes or poison monotonically."""

        self._reject_writer_reentry()
        with self._state_lock:
            current = self._require_generation()
            control_epoch = int(current.metadata.get("control_epoch", 0))
            if current.health == "failed":
                raise PublicationFailedError("runtime publication is failed; restart the process", context={"restart_guidance": current.restart_guidance})
            if control_epoch != admission.control_epoch:
                raise PublicationError("framework finalizer is stale after a control transition", context={"group": admission.group})
            fingerprints = dict(current.metadata.get("framework_plan_fingerprints", {}))
            lifecycle_key = f"{admission.group}:{admission.lifecycle}"
            existing_fingerprint = fingerprints.get(lifecycle_key)
            if existing_fingerprint is not None and existing_fingerprint != admission.fingerprint:
                raise PublicationError("framework finalizer is stale after adapter-plan mutation", context={"group": admission.group})
            if failure is not None:
                return self._fail_framework_locked(current, admission, fingerprints, control_epoch, failure)
            outcomes = dict(current.metadata.get("framework_statuses", {}))
            outcomes.update(statuses)
            fingerprints[lifecycle_key] = admission.fingerprint
            self._generation = SessionGeneration(
                current.number + 1,
                current.runtime,
                visibility_epoch=current.visibility_epoch,
                metadata={
                    **current.metadata,
                    "framework_statuses": outcomes,
                    "framework_registry_revision": admission.registry_revision,
                    "framework_plan_fingerprints": fingerprints,
                    "control_epoch": control_epoch,
                },
            )
            return self._generation

    def fail_framework(self, admission: FrameworkAdmission | None, failure: BaseException) -> SessionGeneration:
        """Poison a controlled import when its post-status publication failed."""

        self._reject_writer_reentry()
        with self._state_lock:
            current = self._require_generation()
            if current.health == "failed":
                return current
            if admission is None:
                raise PublicationError("controlled framework failure has no admission token")
            fingerprints = dict(current.metadata.get("framework_plan_fingerprints", {}))
            return self._fail_framework_locked(current, admission, fingerprints, int(current.metadata.get("control_epoch", 0)), failure)

    def claim_framework_pre_stage(self, admission: FrameworkAdmission) -> bool:
        """Claim a non-idempotent group pre-stage without waiting for peers."""

        self._reject_writer_reentry()
        key = (admission.control_epoch, admission.group, admission.fingerprint)
        with self._state_lock:
            if self._require_generation().health == "failed":
                raise PublicationFailedError("runtime publication is failed; restart the process")
            if key in self._framework_pre_stages:
                raise PublicationError(
                    "non-idempotent framework pre-stage is already running; retry import",
                    context={"group": admission.group},
                )
            self._framework_pre_stages[key] = "running"
            return True

    def publish_framework_pre_stage(self, admission: FrameworkAdmission) -> SessionGeneration:
        """Publish one logical pure-validation outcome for an adapter group."""

        self._reject_writer_reentry()
        with self._state_lock:
            current = self._require_generation()
            if current.health == "failed":
                return current
            if int(current.metadata.get("control_epoch", 0)) != admission.control_epoch:
                raise PublicationError("framework pre-stage is stale after a control transition", context={"group": admission.group})
            outcomes = dict(current.metadata.get("framework_pre_stages", {}))
            fingerprint = outcomes.get(admission.group)
            if fingerprint is not None:
                if fingerprint != admission.fingerprint:
                    raise PublicationError("framework pre-stage is stale after adapter-plan mutation", context={"group": admission.group})
                return current
            outcomes[admission.group] = admission.fingerprint
            self._generation = SessionGeneration(
                current.number + 1,
                current.runtime,
                visibility_epoch=current.visibility_epoch,
                metadata={**current.metadata, "framework_pre_stages": outcomes},
            )
            return self._generation

    def complete_framework_pre_stage(self, admission: FrameworkAdmission) -> None:
        """Complete an owned non-idempotent pre-stage after callback return."""

        self._reject_writer_reentry()
        key = (admission.control_epoch, admission.group, admission.fingerprint)
        with self._state_lock:
            if self._framework_pre_stages.get(key) != "running":
                raise PublicationError("framework pre-stage completion did not own its group", context={"group": admission.group})
            self._framework_pre_stages[key] = "complete"

    def claim_framework_finalizer(self, admission: FrameworkAdmission, spec_id: int) -> bool:
        """Claim one non-waiting post stage for a module lifecycle.

        The claim is only bookkeeping.  Callers invoke adapter code after this
        method returns, with no publication mutex held.
        """

        self._reject_writer_reentry()
        key = (admission.group, admission.lifecycle, spec_id)
        with self._state_lock:
            if self._require_generation().health == "failed":
                raise PublicationFailedError("runtime publication is failed; restart the process")
            if key in self._framework_finalizers:
                return False
            self._framework_finalizers[key] = "running"
            return True

    def complete_framework_finalizer(self, admission: FrameworkAdmission, spec_id: int) -> None:
        """Mark a claimed lifecycle post stage complete after publication."""

        self._reject_writer_reentry()
        key = (admission.group, admission.lifecycle, spec_id)
        with self._state_lock:
            current = self._require_generation()
            if current.health == "failed":
                raise PublicationFailedError("runtime publication is failed; restart the process", context={"restart_guidance": current.restart_guidance})
            if self._framework_finalizers.get(key) != "running":
                raise PublicationError("framework finalizer completion did not own its lifecycle", context={"group": admission.group})
            self._framework_finalizers[key] = "complete"

    def framework_finalizer_seen(self, group: str, lifecycle: str, spec_id: int) -> bool:
        """Return whether raw loader lifecycle already owns/completed a post stage."""

        self._reject_writer_reentry()
        with self._state_lock:
            return (group, lifecycle, spec_id) in self._framework_finalizers

    def _require_generation(self) -> SessionGeneration:
        """Return the initialized generation while the caller holds state access.

        Returns:
            The current immutable generation.

        Raises:
            PublicationError: If baseline initialization has not completed.
        """
        if self._generation is None:
            raise PublicationError("runtime publication has not been initialized")
        return self._generation

    def _fail_framework_locked(self, current: SessionGeneration, admission: FrameworkAdmission, fingerprints: Mapping[str, str], control_epoch: int, failure: BaseException) -> SessionGeneration:
        """Publish the terminal state while the caller holds ``_state_lock``."""

        self._generation = self._terminal_generation(
            current,
            restart_guidance="restart the process; framework import did not complete safely",
            metadata={
                "framework_failure": type(failure).__name__,
                "framework_registry_revision": admission.registry_revision,
                "framework_plan_fingerprints": dict(fingerprints),
                "control_epoch": control_epoch,
            },
        )
        return self._generation

    def _validate_candidate_locked(self, candidate: PublicationCandidate, effects: EffectPlan) -> SessionGeneration:
        """Validate one candidate while the caller holds ``_state_lock``."""

        current = self._require_generation()
        if current.health == "failed":
            raise PublicationFailedError("runtime publication is failed; restart the process", context={"restart_guidance": current.restart_guidance})
        if current is not candidate.expected:
            raise PublicationError(
                "publication candidate no longer matches the current generation",
                context={"reason": "stale_candidate", "expected": candidate.expected.number, "current": current.number},
            )
        if self._leases and self._changes_process_effects(effects):
            raise PublicationBusyError("active generation lease prevents process-effect transition", context={"generations": tuple(sorted(self._leases))})
        return current

    @staticmethod
    def _terminal_generation(
        current: SessionGeneration,
        *,
        restart_guidance: str,
        metadata: Mapping[str, Any],
    ) -> SessionGeneration:
        """Project a failed process without retaining managed allocation state."""

        try:
            runtime = replace(
                current.runtime,
                mode=RuntimeMode.ORCHESTRATOR,
                allocation=NoAllocation,
                spec=None,
                enforcement=RuntimeEnforcement.STRICT,
                requirement_axes=RequirementAxes.all(),
            )
        except TypeError:
            # Test doubles may not be RuntimeState instances. Production
            # generations always use RuntimeState and take the strict branch.
            runtime = current.runtime
        return SessionGeneration(
            current.number + 1,
            runtime,
            visibility_epoch=None,
            health="failed",
            restart_guidance=restart_guidance,
            metadata=metadata,
        )

    def _reject_writer_reentry(self) -> None:
        """Reject publication access from an active transition writer owner.

        Returns:
            None.

        Raises:
            PublicationReentryError: If the current thread owns the writer.
        """
        if coordinator.writer_owner == threading.get_ident():
            raise PublicationReentryError("transition writer is active; publication API re-entry is not allowed", context={"phase": "writer"})

    @staticmethod
    def _changes_process_effects(effects: EffectPlan) -> bool:
        """Report whether an effect plan can invalidate an active lease.

        Args:
            effects: Precomputed process-effect plan to inspect.

        Returns:
            ``True`` when committing the plan mutates process-global state.
        """
        return bool(
            effects.environment
            or effects.interceptor is not None
            or effects.cpu_affinity is not None
            or effects.process_limits
            or effects.irreversible_outcome is not None
        )

    def _publish(self, generation: SessionGeneration) -> None:
        """Replace the immutable generation after all effects are verified."""

        self._generation = generation

    def _validate_effect_plan(self, effects: EffectPlan) -> None:
        """Reject non-restorable limits before any process mutation occurs."""

        if not effects.process_limits:
            return
        if effects.dedicated_process:
            return
        for kind, requested in effects.process_limits.items():
            if len(requested) != 2:
                raise PublicationError("process limits require a (soft, hard) pair", context={"limit": kind})
            previous = self._limit_getter(kind)
            soft, hard = requested
            if hard != previous[1] or soft > hard:
                raise PublicationError(
                    "process limits require a dedicated non-restoring worker when the hard limit changes",
                    context={"limit": kind, "previous": previous, "requested": requested},
                )

    def _apply(self, effects: EffectPlan, records: list[EffectRecord], journal: list[EffectRecord]) -> None:
        """Apply and verify preplanned effects while recording rollback ownership.

        Args:
            effects: Reversible effects validated for the transition.
            records: Effects applied by this attempt for immediate rollback.
            journal: Process effects retained across committed generations.

        Returns:
            None.
        """
        seen_environment_keys: set[str] = set()
        for key, value in effects.environment.items():
            target_key = self._environment_key(key)
            identity = self._environment_identity(target_key)
            if identity in seen_environment_keys:
                raise PublicationError("environment plan contains duplicate logical keys", context={"key": key})
            seen_environment_keys.add(identity)
            existing_index = self._journal_index(journal, "environment", target_key)
            existing = journal[existing_index] if existing_index is not None else None
            previous = self._environ.get(target_key)
            if existing is not None and previous != existing.written:
                raise PublicationError("environment ownership was lost before transition", context={"key": target_key})
            record_index = None
            if previous != value:
                record_index = len(records)
                records.append(EffectRecord("environment_pending", target_key, previous, value))
            if value is None:
                self._environ.pop(target_key, None)
            else:
                self._environ[target_key] = value
            if record_index is not None:
                records[record_index] = EffectRecord("environment", target_key, previous, value)
            if self._environ.get(target_key) != value:
                raise PublicationError("environment effect readback failed", context={"key": target_key})
            if previous == value:
                continue
            self._replace_journal_record(journal, existing_index, EffectRecord("environment", target_key, existing.previous if existing else previous, value))
        if effects.interceptor is not None:
            position = effects.interceptor_position if effects.interceptor_position is not None else 0
            if any(item is effects.interceptor for item in self._meta_path):
                raise PublicationError("passive interceptor is already installed", context={"interceptor": repr(effects.interceptor)})
            self._meta_path.insert(position, effects.interceptor)
            records.append(EffectRecord("interceptor", effects.interceptor, position, effects.interceptor))
            if self._meta_path[position] is not effects.interceptor:
                raise PublicationError("interceptor effect readback failed")
            journal.append(records[-1])
        if effects.cpu_affinity is not None:
            previous = tuple(sorted(self._affinity_getter()))
            existing_index = self._journal_index(journal, "cpu_affinity", 0)
            existing = journal[existing_index] if existing_index is not None else None
            if existing is not None and previous != existing.written:
                raise PublicationError("CPU affinity ownership was lost before transition")
            record_index = len(records)
            records.append(EffectRecord("cpu_affinity_pending", 0, previous, effects.cpu_affinity))
            self._affinity_setter(effects.cpu_affinity)
            records[record_index] = EffectRecord("cpu_affinity", 0, previous, effects.cpu_affinity)
            if tuple(sorted(self._affinity_getter())) != effects.cpu_affinity:
                raise PublicationError("CPU affinity effect readback failed")
            self._replace_journal_record(journal, existing_index, EffectRecord("cpu_affinity", 0, existing.previous if existing else previous, effects.cpu_affinity))
        for kind, requested in effects.process_limits.items():
            previous = tuple(self._limit_getter(kind))
            existing_index = self._journal_index(journal, "process_limit", kind)
            existing = journal[existing_index] if existing_index is not None else None
            if existing is not None and previous != existing.written:
                raise PublicationError("process-limit ownership was lost before transition", context={"limit": kind})
            self._limit_setter(kind, requested)
            # A successful setter may already have changed a non-restorable
            # limit when readback fails. Record it before the first readback so
            # rollback and terminal-state handling retain that fact.
            record = EffectRecord("process_limit", kind, previous, requested)
            records.append(record)
            if tuple(self._limit_getter(kind)) != requested:
                raise PublicationError("process-limit effect readback failed", context={"limit": kind})
            self._replace_journal_record(journal, existing_index, EffectRecord("process_limit", kind, existing.previous if existing else previous, requested))
            if effects.dedicated_process:
                irreversible = EffectRecord("irreversible", f"process_limit:{kind}", None, requested)
                records.append(irreversible)
                journal.append(irreversible)
        if effects.irreversible_outcome is not None:
            irreversible = EffectRecord("irreversible", effects.irreversible_outcome, None, effects.irreversible_outcome)
            records.append(irreversible)
            journal.append(irreversible)

    def _journal_index(self, journal: list[EffectRecord], kind: str, key: Any) -> int | None:
        """Find one owned effect record by its logical key.

        Args:
            journal: Retained process-effect ownership records.
            kind: Effect category to locate.
            key: Effect key, case-folded for Windows environments.

        Returns:
            The matching record index, or ``None`` when absent.
        """
        for index, record in enumerate(journal):
            if record.kind != kind:
                continue
            if kind == "environment":
                if self._environment_identity(record.key) == self._environment_identity(key):
                    return index
            elif record.key == key:
                return index
        return None

    @staticmethod
    def _replace_journal_record(journal: list[EffectRecord], index: int | None, record: EffectRecord) -> None:
        """Insert, replace, or remove one semantically neutral journal record.

        Args:
            journal: Mutable effect journal being prepared for publication.
            index: Existing matching record index, if present.
            record: Verified ownership record to apply.

        Returns:
            None.
        """
        if record.previous == record.written:
            if index is not None:
                del journal[index]
        elif index is None:
            journal.append(record)
        else:
            journal[index] = record

    def _environment_key(self, key: str) -> str:
        """Resolve the owned spelling for an environment key on this platform."""

        if not self._windows:
            return key
        folded = key.casefold()
        for existing in self._environ:
            if existing.casefold() == folded:
                return existing
        return key

    def _environment_identity(self, key: str) -> str:
        """Return the platform-specific logical identity for an environment key.

        Args:
            key: Environment variable spelling to normalize.

        Returns:
            Case-folded identity on Windows, otherwise the original key.
        """
        return key.casefold() if self._windows else key

    @staticmethod
    def _default_affinity_getter() -> tuple[int, ...]:
        if not hasattr(os, "sched_getaffinity"):
            raise PublicationError("CPU affinity is unsupported on this platform")
        return tuple(os.sched_getaffinity(0))

    @staticmethod
    def _default_affinity_setter(cpus: tuple[int, ...]) -> None:
        if not hasattr(os, "sched_setaffinity"):
            raise PublicationError("CPU affinity is unsupported on this platform")
        os.sched_setaffinity(0, set(cpus))

    @staticmethod
    def _default_limit_getter(kind: int) -> tuple[int, int]:
        try:
            import resource
        except ModuleNotFoundError as exc:
            raise PublicationError("process limits are unsupported on this platform") from exc
        return tuple(resource.getrlimit(kind))

    @staticmethod
    def _default_limit_setter(kind: int, value: tuple[int, int]) -> None:
        try:
            import resource
        except ModuleNotFoundError as exc:
            raise PublicationError("process limits are unsupported on this platform") from exc
        resource.setrlimit(kind, value)

    def _rollback(self, records: list[EffectRecord]) -> bool:
        """Restore only effects still owned by this failed transition.

        Args:
            records: Attempt-local records in application order.

        Returns:
            ``True`` when every reversible effect was restored and verified.
        """
        complete = True
        for record in reversed(records):
            try:
                if record.kind in {"environment", "environment_pending"}:
                    current = self._environ.get(record.key)
                    if record.kind == "environment_pending" and current == record.previous:
                        continue
                    if current != record.written:
                        complete = False
                    elif record.previous is None:
                        self._environ.pop(record.key, None)
                    else:
                        self._environ[record.key] = record.previous
                elif record.kind == "interceptor":
                    position = record.previous
                    if position < len(self._meta_path) and self._meta_path[position] is record.written:
                        del self._meta_path[position]
                    else:
                        complete = False
                elif record.kind in {"cpu_affinity", "cpu_affinity_pending"}:
                    current = tuple(sorted(self._affinity_getter()))
                    if record.kind == "cpu_affinity_pending" and current == record.previous:
                        continue
                    if current != record.written:
                        complete = False
                    else:
                        self._affinity_setter(record.previous)
                        if tuple(sorted(self._affinity_getter())) != record.previous:
                            complete = False
                elif record.kind == "process_limit":
                    if tuple(self._limit_getter(record.key)) != record.written:
                        complete = False
                    else:
                        self._limit_setter(record.key, record.previous)
                        if tuple(self._limit_getter(record.key)) != record.previous:
                            complete = False
                elif record.kind == "irreversible":
                    complete = False
            except BaseException:
                complete = False
        return complete


publication = PublicationService()


__all__ = ["EffectPlan", "EffectRecord", "FrameworkAdmission", "MaterializationFence", "PublicationCandidate", "PublicationService", "SessionGeneration", "publication"]
