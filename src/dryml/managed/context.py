"""Operation-owned local execution context and durable effect protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dryml.formats.refs import format_cdef_id
from dryml.records import (
    DataRecord,
    DurableProductWriter,
    RepresentationSpec,
    StorageRef,
    StoredStateRecord,
)

from .callbacks import CallbackCoordinator, ControlRequest
from .declarations import ManagedOutputs, resolve_definition_path
from .errors import ManagedOutputError
from .events import EventBuffer, OperationEvent, ProgressSnapshot


@dataclass(frozen=True, slots=True)
class OperationPreflight:
    """Dynamic whole-pipeline capabilities reported before Store mutation.

    ``None`` delegates a capability to the immutable managed declaration.
    Explicit false values safely narrow declaration capabilities.
    """

    resumable: bool | None = None
    checkpoint_schema: str | None = None
    early_completion: bool | None = None

    def __post_init__(self) -> None:
        if self.resumable is not None and not isinstance(self.resumable, bool):
            raise TypeError("preflight resumable must be bool or None")
        if self.early_completion is not None and not isinstance(self.early_completion, bool):
            raise TypeError("preflight early_completion must be bool or None")
        if self.checkpoint_schema is not None and (
            not isinstance(self.checkpoint_schema, str) or not self.checkpoint_schema
        ):
            raise ValueError("preflight checkpoint_schema must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class OperationResult:
    """Dispatch-neutral completion signal returned by managed implementation code."""

    early_completed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.early_completed, bool):
            raise TypeError("early_completed must be a bool")


@dataclass(frozen=True, slots=True)
class OutputEffect:
    """Validated metadata for bytes written to one declared output slot."""

    slot: str
    representation_id: str
    record_kind: str
    subject_cdef_id: str | None


class _OperationInterrupted(Exception):
    pass


_current_context: ContextVar["OperationContext | None"] = ContextVar(
    "dryml_managed_operation_context",
    default=None,
)


def current_operation_context() -> "OperationContext":
    """Return the active managed implementation context.

    Managed implementations use this accessor rather than adding framework-only
    parameters to their public Python signatures.
    """

    context = _current_context.get()
    if context is None:
        raise RuntimeError("no managed operation context is active")
    return context


class OperationContext:
    """Own durable checkpoint/output effects and callback-safe event polling."""

    def __init__(
        self,
        *,
        producer: Any,
        method: str,
        outputs: ManagedOutputs,
        lease: Any,
        realization_id: str,
        coordinator: CallbackCoordinator,
        checkpoint_schema: str | None,
        early_completion: bool,
        is_resume: bool,
        max_events: int = 32,
    ):
        self.producer = producer
        self.method = method
        self.outputs = outputs
        self.lease = lease
        self.realization_id = realization_id
        self.coordinator = coordinator
        self.checkpoint_schema = checkpoint_schema
        self.early_completion = early_completion
        self.is_resume = is_resume
        self.writer = DurableProductWriter(
            lease.operation.managed_store.store.records,
            lease,
            realization_id,
        )
        self.events = EventBuffer(max_events=max_events)
        self._sequence = 0
        self._effects: dict[str, OutputEffect] = {}
        self._last_checkpoint = lease.operation._read_realization(realization_id).checkpoint_head
        self._checkpoint_provider: Callable[[], Any] | None = None
        self._graceful_stop_requested = False

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return bounded callback diagnostics accumulated by the coordinator."""

        return self.coordinator.diagnostics

    @property
    def store(self):
        """Return the selected Store authority for this invocation."""

        return self.lease.operation.managed_store.store

    @property
    def checkpoint_head(self) -> str | None:
        """Return the latest compatible committed checkpoint ID."""

        return self._last_checkpoint

    @property
    def checkpoint_path(self) -> Path | None:
        """Return the verified prior checkpoint root for a resumed operation."""

        if self._last_checkpoint is None:
            return None
        return self.writer.checkpoint_path(self._last_checkpoint)

    @property
    def output_effects(self) -> Mapping[str, OutputEffect]:
        """Return immutable exact output metadata collected during execution."""

        return MappingProxyType(dict(self._effects))

    def progress(
        self,
        current: int | float,
        *,
        total: int | float | None = None,
        message: str | None = None,
        metrics: Mapping[str, int | float] | None = None,
    ) -> ProgressSnapshot:
        """Replace durable progress and emit one coalesced callback event."""

        progress = ProgressSnapshot(current, total, message, metrics or {})
        self.lease.update_progress(self.realization_id, progress)
        self._publish(
            OperationEvent.progress(
                self._next_sequence(),
                current=current,
                total=total,
                message=message,
                metrics=metrics,
            )
        )
        return progress

    def safe_point(
        self,
        *,
        checkpoint: Callable[[], Any] | None = None,
    ) -> ControlRequest:
        """Emit a safe point, service coalesced checkpoint/control, and poll.

        The operation supplies its own checkpoint producer because the framework
        cannot capture arbitrary Python execution state. Interrupt and strict
        callback failure are raised only after any compatible checkpoint commits.
        """

        if checkpoint is not None:
            self._checkpoint_provider = checkpoint
        self._publish(OperationEvent(self._next_sequence(), "safe_point"))
        return self._service_control(checkpoint)

    def completion_point(self) -> None:
        """Run completion callbacks before immutable publication and activation."""

        self._publish(OperationEvent(self._next_sequence(), "completed"))
        self._service_control(self._checkpoint_provider)

    def _service_control(
        self,
        checkpoint: Callable[[], Any] | None,
    ) -> ControlRequest:
        control = self.coordinator.poll()
        needs_checkpoint = control in {
            ControlRequest.CHECKPOINT,
            ControlRequest.INTERRUPT,
            ControlRequest.FAIL,
        } and self.checkpoint_schema is not None
        before = self._last_checkpoint
        if needs_checkpoint:
            if checkpoint is None:
                if control is ControlRequest.FAIL:
                    raise ManagedOutputError(
                        "strict callback failure reached a safe point without a checkpoint producer"
                    )
            else:
                checkpoint()
                if self._last_checkpoint == before:
                    self.commit_checkpoint()
        if control is ControlRequest.CHECKPOINT:
            self.coordinator.consume_checkpoint()
            return ControlRequest.CHECKPOINT
        if control is ControlRequest.FAIL:
            self.coordinator.raise_failure()
        if control is ControlRequest.INTERRUPT:
            raise _OperationInterrupted()
        if control is ControlRequest.GRACEFUL_STOP:
            self._graceful_stop_requested = True
        return control

    def write_checkpoint(self, path: str, chunks: Any) -> None:
        """Stage one opaque operation-owned checkpoint file."""

        if self.checkpoint_schema is None:
            raise ManagedOutputError("operation did not declare checkpoint capability")
        self.writer.write_checkpoint_stream(path, chunks)

    def commit_checkpoint(self, *, metadata: Mapping[str, Any] | None = None) -> str:
        """Commit staged checkpoint files and advance the framework-owned head."""

        if self.checkpoint_schema is None:
            raise ManagedOutputError("operation did not declare checkpoint capability")
        committed = self.writer.commit_checkpoint(self.checkpoint_schema, metadata=metadata)
        self._last_checkpoint = committed.checkpoint_id
        self._publish(
            OperationEvent(
                self._next_sequence(),
                "checkpoint",
                payload={"checkpoint_id": committed.checkpoint_id},
            )
        )
        return committed.checkpoint_id

    def write_output(
        self,
        slot: str,
        path: str,
        chunks: Any,
        *,
        representation: Mapping[str, Any] | RepresentationSpec | str,
        record_kind: str | None = None,
        subject_cdef_id: str | None = None,
    ) -> None:
        """Stream bytes for one declared slot and register its typed effect."""

        declaration = self.outputs.get(slot)
        if declaration is None:
            raise ManagedOutputError(f"operation wrote undeclared output slot {slot!r}")
        representation_id = self._publish_representation(representation)
        kind = record_kind or declaration.kind
        if kind == "object":
            kind = "stored_state"
        if kind not in {"data", "stored_state"}:
            raise ManagedOutputError(f"managed output kind {kind!r} is unsupported")
        if subject_cdef_id is None and declaration.subject_path is not None:
            subject = resolve_definition_path(self.producer.definition, declaration.subject_path)
            subject_cdef_id = format_cdef_id(subject.stable_hash())
        if kind == "stored_state" and subject_cdef_id is None:
            subject_cdef_id = format_cdef_id(self.producer.definition.stable_hash())
        effect = OutputEffect(slot, representation_id, kind, subject_cdef_id)
        prior = self._effects.get(slot)
        if prior is not None and prior != effect:
            raise ManagedOutputError(f"output slot {slot!r} changed representation or ownership")
        self.writer.write_stream(slot, path, chunks)
        self._effects[slot] = effect

    def output_records(self) -> dict[str, DataRecord | StoredStateRecord]:
        """Build typed output records and reject missing required declarations."""

        declared = set(self.outputs.slots)
        missing = declared - set(self._effects)
        if missing:
            raise ManagedOutputError(
                f"required output slots were not produced: {', '.join(sorted(missing))}"
            )
        records = {}
        for slot, effect in self._effects.items():
            common = {
                "representation_id": effect.representation_id,
                "storage": (StorageRef.self_product(role=slot),),
                "realization_id": self.realization_id,
                "output_slot": slot,
            }
            if effect.record_kind == "data":
                records[slot] = DataRecord(
                    subject_cdef_id=effect.subject_cdef_id,
                    **common,
                )
            else:
                records[slot] = StoredStateRecord(
                    subject_cdef_id=effect.subject_cdef_id,
                    **common,
                )
        return records

    def apply_worker_event(self, event: OperationEvent) -> None:
        """Apply one validated worker intent through coordinator authority."""

        if not isinstance(event, OperationEvent):
            raise TypeError("worker event must be an OperationEvent")
        if event.progress_snapshot is not None:
            self.lease.update_progress(self.realization_id, event.progress_snapshot)
        self._publish(event)

    def register_output_effect(self, effect: OutputEffect) -> None:
        """Register validated worker output metadata without publishing it."""

        if not isinstance(effect, OutputEffect):
            raise TypeError("worker output effect must be an OutputEffect")
        declaration = self.outputs.get(effect.slot)
        if declaration is None:
            raise ManagedOutputError(
                f"operation wrote undeclared output slot {effect.slot!r}"
            )
        expected_kind = "stored_state" if declaration.kind == "object" else declaration.kind
        if effect.record_kind != expected_kind:
            raise ManagedOutputError(
                f"output slot {effect.slot!r} changed its declared record kind"
            )
        prior = self._effects.get(effect.slot)
        if prior is not None and prior != effect:
            raise ManagedOutputError(
                f"output slot {effect.slot!r} changed representation or ownership"
            )
        self._effects[effect.slot] = effect

    def commit_worker_checkpoint(
        self,
        checkpoint_schema: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Commit worker-staged bytes and advance the coordinator-owned head."""

        if checkpoint_schema != self.checkpoint_schema:
            raise ManagedOutputError(
                "worker checkpoint schema does not match operation capability"
            )
        committed = self.writer.commit_checkpoint(
            checkpoint_schema,
            metadata=metadata,
        )
        self._last_checkpoint = committed.checkpoint_id
        return committed.checkpoint_id

    def validate_result(self, value: Any) -> OperationResult:
        """Normalize completion and require explicit valid early completion."""

        result = OperationResult() if value is None else value
        if not isinstance(result, OperationResult):
            raise ManagedOutputError("managed implementation must return OperationResult or None")
        if result.early_completed and not self.early_completion:
            raise ManagedOutputError("operation returned unsupported early completion")
        if self._graceful_stop_requested and not result.early_completed:
            raise ManagedOutputError("graceful stop requires an explicit early-completed result")
        return result

    def publish_terminal(self, kind: str) -> None:
        """Emit one terminal event without changing durable authority."""

        self._publish(OperationEvent(self._next_sequence(), kind))

    def activate(self):
        """Install this context for one direct implementation invocation."""

        return _current_context.set(self)

    @staticmethod
    def deactivate(token) -> None:
        """Restore the prior process-local operation context."""

        _current_context.reset(token)

    def _publish_representation(
        self,
        representation: Mapping[str, Any] | RepresentationSpec | str,
    ) -> str:
        record_io = self.lease.operation.managed_store.store.records
        if isinstance(representation, RepresentationSpec):
            envelope = representation.to_envelope()
            record_io.write_spec(envelope, family="representation")
            return envelope["id"]
        if isinstance(representation, Mapping):
            located = record_io.write_spec(representation, family="representation")
            return located.spec_id
        if isinstance(representation, str):
            record_io.read_spec(representation, family="representation")
            return representation
        raise TypeError("representation must be a representation spec or ID")

    def _publish(self, event: OperationEvent) -> None:
        self.events.append(event)
        self.coordinator.publish(event)

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence


__all__ = [
    "OperationContext",
    "OperationPreflight",
    "OperationResult",
    "OutputEffect",
    "current_operation_context",
]
