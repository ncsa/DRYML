"""PID-bound immutable runtime publication with owned reversible effects."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dryml.formats import deep_freeze_json
from dryml.worlds import LocalResourceInventory, local_inventory

from .errors import ForkSafetyError, PublicationBusyError, PublicationError, PublicationFailedError
from .modes import RuntimeMode

_CONTROL_STATUSES = {
    "undeclared", "not-applicable", "pending-import", "visibility-enforced",
    "framework-configured", "enforced", "declarative", "unsupported", "failed",
}


def _freeze_mapping(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class SessionGeneration:
    """One immutable process runtime publication and its health."""

    number: int
    runtime: Any
    inventory: LocalResourceInventory | None = None
    health: str = "healthy"
    restart_guidance: str | None = None
    statuses: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze detached diagnostic projections."""
        if isinstance(self.number, bool) or not isinstance(self.number, int) or self.number < 0:
            raise PublicationError("publication generation number must be a non-negative integer")
        if self.health not in {"healthy", "failed"}:
            raise PublicationError("publication health must be healthy or failed")
        statuses = {key: getattr(value, "value", value) for key, value in self.statuses.items()}
        if any(not isinstance(key, str) or value not in _CONTROL_STATUSES for key, value in statuses.items()):
            raise PublicationError("publication control statuses use a closed vocabulary")
        object.__setattr__(self, "statuses", _freeze_mapping(statuses))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))


@dataclass(frozen=True, slots=True)
class PublicationCandidate:
    """Normalized state staged against one exact immutable generation."""

    expected: SessionGeneration
    generation: SessionGeneration


@dataclass(frozen=True, slots=True)
class EffectPlan:
    """Preplanned process effects accepted by the short publication transaction."""

    environment: Mapping[str, str | None] = field(default_factory=dict)
    cpu_affinity: tuple[int, ...] | None = None
    process_memory: Any = None
    irreversible_outcome: str | None = None

    def __post_init__(self) -> None:
        """Freeze and normalize process-effect inputs before publication."""
        if not isinstance(self.environment, Mapping) or any(not isinstance(key, str) or (value is not None and not isinstance(value, str)) for key, value in self.environment.items()):
            raise PublicationError("environment effects must be a string-to-string-or-null mapping")
        if self.cpu_affinity is not None:
            cpus = tuple(sorted(self.cpu_affinity))
            if not cpus or len(cpus) != len(set(cpus)) or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in cpus):
                raise PublicationError("CPU affinity effects require unique non-negative CPUs")
            object.__setattr__(self, "cpu_affinity", cpus)
        if self.process_memory is not None and (isinstance(self.process_memory, bool) or not isinstance(self.process_memory, int) or self.process_memory < 0):
            raise PublicationError("process memory effects require a non-negative integer")
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))

    @property
    def changes_process(self) -> bool:
        """Return whether this plan can invalidate an active generation lease."""
        return bool(self.environment or self.cpu_affinity is not None or self.process_memory is not None or self.irreversible_outcome is not None)


@dataclass(frozen=True, slots=True)
class EffectRecord:
    """One applied owned effect, including its reversible prior value."""

    kind: str
    key: Any
    previous: Any
    written: Any


@dataclass(frozen=True, slots=True)
class FrameworkAdmission:
    """Fenced status-only finalization token reserved for U6."""

    generation: int
    control_epoch: int


class PublicationService:
    """Own the sole generation, lease, effect, and fork-safety authority.

    The service never calls Store/index APIs while publishing.  Candidate
    normalization, inventory observation, and effect planning are caller work;
    only verified reversible writes occur under its short transaction.
    """

    def __init__(self, *, environ: MutableMapping[str, str] | None = None, windows: bool | None = None, affinity_getter: Callable[[], tuple[int, ...]] | None = None, affinity_setter: Callable[[tuple[int, ...]], None] | None = None, process_memory_getter: Callable[[], Any] | None = None, process_memory_setter: Callable[[Any], None] | None = None, pid_getter: Callable[[], int] | None = None) -> None:
        """Create an injectable publication authority.

        Args:
            environ: Environment mapping to mutate; tests should inject one.
            windows: Enable case-insensitive environment ownership semantics.
            affinity_getter: Optional platform/fake CPU-affinity reader.
            affinity_setter: Optional platform/fake CPU-affinity writer.
            process_memory_getter: Optional platform/fake process-memory reader.
            process_memory_setter: Optional platform/fake process-memory writer.
            pid_getter: Optional PID seam for deterministic fork tests.
        """
        self._pid_getter = pid_getter or os.getpid
        self._pid = self._pid_getter()
        self._lock = threading.RLock()
        self._generation: SessionGeneration | None = None
        self._leases: dict[int, int] = {}
        self._effects: tuple[EffectRecord, ...] = ()
        self._observed = self._mutated = self._in_flight = self._failed = False
        self._environ = os.environ if environ is None else environ
        self._windows = os.name == "nt" if windows is None else windows
        if self._windows:
            identities = [key.casefold() for key in self._environ]
            if len(identities) != len(set(identities)):
                raise PublicationError("Windows environment contains case-colliding keys")
        self._affinity_getter = affinity_getter or self._default_affinity_getter
        self._affinity_setter = affinity_setter or self._default_affinity_setter
        self._process_memory_getter = process_memory_getter
        self._process_memory_setter = process_memory_setter

    def initialize(self, runtime: Any) -> SessionGeneration:
        """Install the passive generation-zero baseline without process effects."""
        self._check_pid()
        with self._lock:
            if self._generation is None:
                self._generation = SessionGeneration(0, runtime, metadata={"control_epoch": 0})
            return self._generation

    def current(self) -> SessionGeneration:
        """Return the immutable current generation and record an observation."""
        self._check_pid()
        with self._lock:
            self._observed = True
            return self._require_generation()

    snapshot = current

    def stage(self, expected: SessionGeneration, runtime: Any, *, inventory: LocalResourceInventory | None = None, statuses: Mapping[str, str] | None = None) -> PublicationCandidate:
        """Create a monotonic candidate from normalized runtime intent.

        Args:
            expected: Exact generation observed before planning.
            runtime: Validated immutable runtime state.
            inventory: Stable-dimension inventory observation, if applicable.
            statuses: Independent control status mapping.

        Returns:
            A compare-and-swap candidate.
        """
        self._check_pid()
        self._validate_runtime(runtime)
        if not isinstance(expected, SessionGeneration):
            raise PublicationError("publication candidate requires an immutable expected generation")
        next_number = expected.number + 1
        return PublicationCandidate(expected, SessionGeneration(next_number, runtime, inventory, statuses=statuses or {}, metadata={"control_epoch": next_number}))

    def publish(self, runtime: Any, *, inventory: LocalResourceInventory | None = None, inventory_observer: Callable[[], LocalResourceInventory] | None = None, effects: EffectPlan | None = None, statuses: Mapping[str, str] | None = None, restage_retries: int = 2) -> SessionGeneration:
        """Observe, restage, and publish one runtime generation.

        Args:
            runtime: Normalized mode/allocation state to publish.
            inventory: Optional explicit inventory for this operation.
            inventory_observer: Optional deterministic fresh-observation seam.
            effects: Reversible effects planned before transaction entry.
            statuses: Independent pre-import control statuses.
            restage_retries: Operation-local stale-observation retry count, 0-16.

        Returns:
            The current or newly committed immutable generation.

        Raises:
            PublicationError: If observations remain stale or input is invalid.
        """
        if isinstance(restage_retries, bool) or not isinstance(restage_retries, int) or not 0 <= restage_retries <= 16:
            raise PublicationError("restage_retries must be an operation-local integer from 0 through 16")
        self._check_pid()
        self._validate_runtime(runtime)
        planned_effects = effects or EffectPlan()
        if getattr(runtime, "mode", None) is RuntimeMode.NONE:
            if inventory is not None or inventory_observer is not None or planned_effects.changes_process:
                raise PublicationError("NONE runtime has no inventory or process effects")
            return self.commit(self.stage(self.current(), runtime, statuses=statuses), planned_effects)
        observer = inventory_observer or (lambda: inventory if inventory is not None else local_inventory())
        for attempt in range(restage_retries + 1):
            expected = self.current()
            observed = observer()
            if not isinstance(observed, LocalResourceInventory):
                raise PublicationError("inventory observer must return LocalResourceInventory")
            candidate = self.stage(expected, runtime, inventory=observed, statuses=statuses)
            fresh = observer()
            if not isinstance(fresh, LocalResourceInventory):
                raise PublicationError("inventory observer must return LocalResourceInventory")
            if fresh.visibility_identity != observed.visibility_identity:
                if attempt == restage_retries:
                    raise PublicationError("runtime inventory remained stale after restaging", context={"attempts": attempt + 1})
                continue
            return self.commit(candidate, planned_effects)
        raise AssertionError("restaging loop must return or raise")

    def commit(self, candidate: PublicationCandidate, effects: EffectPlan | None = None, *, validator: Callable[[], None] | None = None) -> SessionGeneration:
        """Atomically apply a staged candidate and ownership-tracked effects.

        ``validator`` is called before effects and outside the lock. It exists
        for precondition seams only; it may not perform publication reentry.
        """
        self._check_pid()
        effects = effects or EffectPlan()
        if validator is not None:
            validator()
        records: list[EffectRecord] = []
        with self._framework_registration_guard(candidate.generation):
            with self._lock:
                self._in_flight = True
                current = self._require_generation()
                try:
                    if current is not candidate.expected:
                        raise PublicationError("publication candidate is stale", context={"expected": candidate.expected.number, "current": current.number})
                    if current.health == "failed":
                        raise PublicationFailedError("runtime publication is terminal; restart the process")
                    if getattr(current.runtime, "mode", None) is RuntimeMode.NONE and getattr(candidate.generation.runtime, "mode", None) is not RuntimeMode.NONE:
                        loaded = tuple(root for root in ("tensorflow", "torch", "jax", "jaxlib") if root in sys.modules)
                        if loaded:
                            raise PublicationError("framework was imported before managed visibility control; restart the process", context={"loaded": loaded})
                    equivalent = self._equivalent(current, candidate.generation, effects)
                    if self._leases and not equivalent:
                        raise PublicationBusyError("active generation lease prevents an incompatible process transition")
                    if equivalent:
                        return current
                    self._apply(effects, records)
                    self._generation = candidate.generation
                    self._mutated = self._mutated or effects.changes_process
                    self._effects = self._merge_effects(records)
                    return candidate.generation
                except BaseException as exc:
                    rollback_ok = self._rollback(records)
                    irreversible_applied = any(record.kind == "irreversible" for record in records)
                    terminal = irreversible_applied or not rollback_ok or "ownership was lost" in str(exc)
                    if terminal:
                        self._generation = SessionGeneration(current.number + 1, current.runtime, health="failed", restart_guidance="restart the process; runtime effects could not be restored safely", statuses=current.statuses, metadata={"failure": type(exc).__name__})
                        self._failed = True
                        raise PublicationFailedError("runtime publication failed closed; restart the process") from exc
                    raise
                finally:
                    self._in_flight = False

    @contextmanager
    def lease(self) -> Iterator[SessionGeneration]:
        """Pin a healthy generation until normal, exceptional, or interrupted exit."""
        self._check_pid()
        with self._lock:
            generation = self._require_generation()
            if generation.health == "failed":
                raise PublicationFailedError("runtime publication is terminal; restart the process")
            self._leases[generation.number] = self._leases.get(generation.number, 0) + 1
        try:
            yield generation
        finally:
            self._check_pid()
            with self._lock:
                count = self._leases.get(generation.number, 0) - 1
                if count > 0:
                    self._leases[generation.number] = count
                else:
                    self._leases.pop(generation.number, None)

    def admit_status_finalization(self) -> FrameworkAdmission:
        """Return a fenced status-only admission seam for the U6 import epoch."""
        generation = self.current()
        return FrameworkAdmission(generation.number, int(generation.metadata.get("control_epoch", generation.number)))

    def finalize_statuses(self, admission: FrameworkAdmission, statuses: Mapping[str, str]) -> SessionGeneration:
        """Publish same-epoch status changes without a general reader upgrade."""
        self._check_pid()
        with self._lock:
            current = self._require_generation()
            if current.number < admission.generation or int(current.metadata.get("control_epoch", current.number)) != admission.control_epoch:
                raise PublicationError("framework status finalization is stale")
            if current.health == "failed":
                raise PublicationFailedError("runtime publication is terminal; restart the process")
            merged = {**current.statuses, **statuses}
            self._generation = SessionGeneration(current.number + 1, current.runtime, current.inventory, statuses=merged, metadata=current.metadata)
            return self._generation

    def apply_framework_preimport(self, admission: FrameworkAdmission, environment: Mapping[str, str]) -> None:
        """Apply owned framework environment controls before watched module code.

        Args:
            admission: Same-control-epoch token retained by the loader lease.
            environment: Exact string environment updates required before import.

        Raises:
            PublicationError: If the admission is stale or an owned effect fails.

        Side Effects:
            Records reversible environment ownership for later reset.
        """
        self._check_pid()
        effects = EffectPlan(environment=environment)
        records: list[EffectRecord] = []
        with self._lock:
            current = self._require_generation()
            if current.health == "failed":
                raise PublicationFailedError("runtime publication is terminal; restart the process")
            if int(current.metadata.get("control_epoch", current.number)) != admission.control_epoch:
                raise PublicationError("framework pre-import controls are stale")
            try:
                self._apply(effects, records)
                self._effects = self._merge_effects(records)
                self._mutated = self._mutated or effects.changes_process
            except BaseException:
                if not self._rollback(records):
                    self._generation = SessionGeneration(current.number + 1, current.runtime, current.inventory, health="failed", restart_guidance="restart the process; framework pre-import controls could not be restored", statuses=current.statuses, metadata=current.metadata)
                    self._failed = True
                    raise PublicationFailedError("framework pre-import controls failed closed; restart the process")
                raise

    def fail_status_finalization(self, admission: FrameworkAdmission | None, failure: BaseException) -> SessionGeneration:
        """Publish terminal health after uncertain watched-framework execution.

        Args:
            admission: Loader admission, if execution was controlled.
            failure: Original post-execution exception retained as diagnostic type.

        Returns:
            The terminal immutable generation.
        """
        self._check_pid()
        with self._lock:
            current = self._require_generation()
            if admission is not None and int(current.metadata.get("control_epoch", current.number)) != admission.control_epoch:
                raise PublicationError("framework failure finalization is stale")
            self._generation = SessionGeneration(current.number + 1, current.runtime, current.inventory, health="failed", restart_guidance="restart the process; framework execution may have changed native runtime state", statuses=current.statuses, metadata=current.metadata)
            self._failed = True
            return self._generation

    def reset(self, runtime: Any) -> SessionGeneration:
        """Restore only still-owned reversible effects and publish a fresh baseline."""
        self._check_pid()
        self._validate_runtime(runtime)
        with self._lock:
            current = self._require_generation()
            if self._leases:
                raise PublicationBusyError("active generation lease prevents reset")
            if current.health == "failed":
                raise PublicationFailedError(current.restart_guidance or "restart the process")
            if not self._rollback(list(self._effects)):
                self._generation = SessionGeneration(current.number + 1, runtime, health="failed", restart_guidance="restart the process; session-owned effects were not safely restored")
                self._failed = True
                raise PublicationFailedError("runtime reset could not restore owned effects; restart the process")
            self._effects = ()
            next_number = current.number + 1
            self._generation = SessionGeneration(next_number, runtime, metadata={"control_epoch": next_number})
            return self._generation

    def effect_journal(self) -> tuple[EffectRecord, ...]:
        """Return the immutable session-owned reversible-effect journal."""
        self._check_pid()
        with self._lock:
            return self._effects

    def _check_pid(self) -> None:
        """Reject inherited activated state before touching any inherited lock."""
        if self._pid_getter() == self._pid:
            return
        generation = self._generation
        pristine = generation is not None and generation.number == 0 and getattr(generation.runtime, "mode", None) is RuntimeMode.NONE and not (self._observed or self._mutated or self._leases or self._in_flight or self._failed or self._effects)
        if not pristine:
            raise ForkSafetyError("runtime state was inherited after fork; use spawn or a fresh interpreter")
        self._pid = self._pid_getter()
        self._lock = threading.RLock()
        self._generation = SessionGeneration(0, generation.runtime, metadata={"control_epoch": 0})

    @staticmethod
    def _framework_registration_guard(generation: SessionGeneration):
        """Return the lazy registry guard for an active candidate generation."""
        if getattr(generation.runtime, "mode", None) is RuntimeMode.NONE:
            return nullcontext()
        module = sys.modules.get("dryml.runtime.frameworks")
        if module is None:
            return nullcontext()
        return module.framework_registry._publication_guard()

    def _require_generation(self) -> SessionGeneration:
        if self._generation is None:
            raise PublicationError("runtime publication has not been initialized")
        return self._generation

    def _validate_runtime(self, runtime: Any) -> None:
        mode = getattr(runtime, "mode", None)
        allocation = getattr(runtime, "allocation", None)
        if mode not in set(RuntimeMode):
            raise PublicationError("runtime candidate has an unsupported mode")
        if mode is RuntimeMode.INLINE and allocation is None:
            raise PublicationError("INLINE runtime requires one exact allocation")
        if mode in {RuntimeMode.NONE, RuntimeMode.ORCHESTRATOR} and allocation is not None and allocation.__class__.__name__ != "_NoAllocation":
            raise PublicationError("NONE and ORCHESTRATOR runtimes cannot hold workload allocation")
        spec = getattr(runtime, "spec", None)
        if mode is RuntimeMode.NONE and spec is not None and (spec.visibility or spec.framework or spec.limits or spec.env):
            raise PublicationError("NONE runtime inherits visibility and cannot publish managed controls")

    @staticmethod
    def _equivalent(current: SessionGeneration, candidate: SessionGeneration, effects: EffectPlan) -> bool:
        return not effects.changes_process and current.runtime == candidate.runtime and (current.inventory is None or candidate.inventory is None or current.inventory.visibility_identity == candidate.inventory.visibility_identity) and current.statuses == candidate.statuses

    def _apply(self, effects: EffectPlan, records: list[EffectRecord]) -> None:
        for key, value in effects.environment.items():
            target = self._environment_key(key)
            owned = next((record for record in self._effects if record.kind == "environment" and self._environment_identity(record.key) == self._environment_identity(target)), None)
            previous = self._environ.get(target)
            if owned is not None and previous != owned.written:
                raise PublicationError("environment effect ownership was lost", context={"key": target})
            if previous == value:
                continue
            records.append(EffectRecord("environment", target, previous, value))
            if value is None:
                self._environ.pop(target, None)
            else:
                self._environ[target] = value
            if self._environ.get(target) != value:
                raise PublicationError("environment effect readback failed", context={"key": target})
        if effects.cpu_affinity is not None:
            previous = tuple(sorted(self._affinity_getter()))
            owned = next((record for record in self._effects if record.kind == "affinity"), None)
            if owned is not None and previous != owned.written:
                raise PublicationError("CPU affinity effect ownership was lost")
            records.append(EffectRecord("affinity", 0, previous, effects.cpu_affinity))
            self._affinity_setter(effects.cpu_affinity)
            if tuple(sorted(self._affinity_getter())) != effects.cpu_affinity:
                raise PublicationError("CPU affinity effect readback failed")
        if effects.process_memory is not None:
            if self._process_memory_getter is None or self._process_memory_setter is None:
                raise PublicationError("process-memory enforcement is unsupported on this platform")
            previous = self._process_memory_getter()
            owned = next((record for record in self._effects if record.kind == "process_memory"), None)
            if owned is not None and previous != owned.written:
                raise PublicationError("process memory effect ownership was lost")
            records.append(EffectRecord("process_memory", 0, previous, effects.process_memory))
            self._process_memory_setter(effects.process_memory)
            if self._process_memory_getter() != effects.process_memory:
                raise PublicationError("process memory effect readback failed")
        if effects.irreversible_outcome is not None:
            records.append(EffectRecord("irreversible", effects.irreversible_outcome, None, effects.irreversible_outcome))

    def _merge_effects(self, records: list[EffectRecord]) -> tuple[EffectRecord, ...]:
        """Retain one ownership record per reversible logical effect.

        A later transition may update a session-owned value, but reset must
        restore the value from before the first session-owned write.
        """
        journal = list(self._effects)
        for record in records:
            if record.kind == "irreversible":
                journal.append(record)
                continue
            index = next((index for index, old in enumerate(journal) if old.kind == record.kind and old.key == record.key), None)
            if index is None:
                journal.append(record)
            else:
                old = journal[index]
                journal[index] = EffectRecord(record.kind, record.key, old.previous, record.written)
        return tuple(journal)

    def _rollback(self, records: list[EffectRecord]) -> bool:
        complete = True
        for record in reversed(records):
            try:
                if record.kind == "environment":
                    if self._environ.get(record.key) != record.written:
                        complete = False
                    elif record.previous is None:
                        self._environ.pop(record.key, None)
                    else:
                        self._environ[record.key] = record.previous
                elif record.kind == "affinity":
                    if tuple(sorted(self._affinity_getter())) != record.written:
                        complete = False
                    else:
                        self._affinity_setter(record.previous)
                        complete = tuple(sorted(self._affinity_getter())) == record.previous and complete
                elif record.kind == "process_memory":
                    if self._process_memory_getter is None or self._process_memory_setter is None or self._process_memory_getter() != record.written:
                        complete = False
                    else:
                        self._process_memory_setter(record.previous)
                        complete = self._process_memory_getter() == record.previous and complete
                else:
                    complete = False
            except BaseException:
                complete = False
        return complete

    def _environment_key(self, key: str) -> str:
        if self._windows:
            folded = key.casefold()
            return next((existing for existing in self._environ if existing.casefold() == folded), key)
        return key

    def _environment_identity(self, key: str) -> str:
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


# Bound by ``context`` after it constructs the one process-global baseline.
# Keeping this name here preserves one import path without constructing a rival.
publication: PublicationService | None = None


__all__ = ["EffectPlan", "EffectRecord", "FrameworkAdmission", "PublicationCandidate", "PublicationService", "SessionGeneration", "publication"]
