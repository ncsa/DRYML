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
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dryml._framework_imports import ImportEpochBusyError, ImportEpochReentryError, coordinator

from .errors import PublicationBusyError, PublicationError, PublicationFailedError, PublicationReentryError


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
        self._state_lock = threading.Lock()
        self._generation: SessionGeneration | None = None
        self._leases: dict[int, int] = {}
        self._effects: tuple[EffectRecord, ...] = ()
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
            raise PublicationBusyError(str(exc), context={"phase": "writer"}) from exc
        except ImportEpochReentryError as exc:
            raise PublicationReentryError(str(exc), context={"phase": "writer"}) from exc

    def commit(self, candidate: PublicationCandidate, effects: EffectPlan | None = None) -> SessionGeneration:
        """CAS-commit a preplanned generation and reversible process effects.

        The state lock is intentionally held only for validation, direct effect
        writes/readback, and immutable generation replacement.  The caller must
        resolve all framework and adapter work before calling this method.
        """

        self._reject_writer_reentry()
        effects = effects or EffectPlan()
        with self.writer():
            with self._state_lock:
                current = self._require_generation()
                if current.health == "failed":
                    raise PublicationFailedError("runtime publication is failed; restart the process", context={"restart_guidance": current.restart_guidance})
                if current is not candidate.expected:
                    raise PublicationError("stale publication candidate", context={"expected": candidate.expected.number, "current": current.number})
                if self._leases.get(current.number, 0) and self._changes_process_effects(effects):
                    raise PublicationBusyError("active generation lease prevents process-effect transition", context={"generation": current.number})
                records: list[EffectRecord] = []
                journal = list(self._effects)
                try:
                    self._validate_effect_plan(effects)
                    self._apply(effects, records, journal)
                    self._publish(candidate.generation)
                    self._effects = tuple(journal)
                    return candidate.generation
                except BaseException as exc:
                    rollback_ok = self._rollback(records)
                    irreversible_applied = any(record.kind == "irreversible" for record in records)
                    if rollback_ok and not irreversible_applied:
                        self._generation = current
                        raise
                    self._generation = SessionGeneration(
                        current.number + 1,
                        current.runtime,
                        visibility_epoch=current.visibility_epoch,
                        health="failed",
                        restart_guidance="restart the process; a runtime process effect could not be safely restored",
                        metadata={"failure": type(exc).__name__},
                    )
                    raise PublicationFailedError("runtime publication failed closed; restart the process", context={"cause": type(exc).__name__}) from exc

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

    def _require_generation(self) -> SessionGeneration:
        if self._generation is None:
            raise PublicationError("runtime publication has not been initialized")
        return self._generation

    def _reject_writer_reentry(self) -> None:
        if coordinator.writer_owner == threading.get_ident():
            raise PublicationReentryError("transition writer is active; publication API re-entry is not allowed", context={"phase": "writer"})

    @staticmethod
    def _changes_process_effects(effects: EffectPlan) -> bool:
        return bool(effects.environment or effects.interceptor is not None or effects.cpu_affinity is not None or effects.process_limits)

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
            if value is None:
                self._environ.pop(target_key, None)
            else:
                self._environ[target_key] = value
            if previous != value:
                records.append(EffectRecord("environment", target_key, previous, value))
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
            self._affinity_setter(effects.cpu_affinity)
            records.append(EffectRecord("cpu_affinity", 0, previous, effects.cpu_affinity))
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
            if tuple(self._limit_getter(kind)) != requested:
                raise PublicationError("process-limit effect readback failed", context={"limit": kind})
            record = EffectRecord("process_limit", kind, previous, requested)
            records.append(record)
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
                elif record.kind == "interceptor":
                    position = record.previous
                    if position < len(self._meta_path) and self._meta_path[position] is record.written:
                        del self._meta_path[position]
                    else:
                        complete = False
                elif record.kind == "cpu_affinity":
                    if tuple(sorted(self._affinity_getter())) != record.written:
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


__all__ = ["EffectPlan", "EffectRecord", "PublicationCandidate", "PublicationService", "SessionGeneration", "publication"]
