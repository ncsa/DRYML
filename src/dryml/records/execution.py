"""Typed execution provenance records and query helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar

from dryml.formats import CanonicalJSONError, deep_freeze_json, json_ready
from dryml.formats.errors import ContentIDError, ReferenceParseError
from dryml.formats.ids import parse_content_id
from dryml.formats.refs import parse_cdef_id

from .errors import RecordValidationError
from .records import make_record, validate_record
from .refs import LocatedRecordRef
from .storage import StorageRef
from .realizations import ResolvedRecord, validate_output_slot, validate_realization_id


EXECUTION_STATUSES = frozenset({"ok", "failed", "cancelled", "timeout", "unsupported", "skipped", "degraded"})
EXECUTION_KINDS = frozenset({"python", "probe", "adapter", "compiler", "lowering", "internal", "unknown"})
PERSISTENCE_SAFE_FAILURE_CODE = "execution_failed"
_RFC3339_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")

_COMMON_ID_FIELDS = {
    "dispatch_id": ("dispatch",),
    "recipe_id": ("recipe",),
    "environment_id": ("envrec", "env"),
    "environment_record_id": ("envrec",),
    "environment_spec_id": ("envspec",),
    "environment_requirement_id": ("envreq",),
    "world_requirement_id": ("worldreq",),
    "world_id": ("world",),
    "world_allocation_id": ("worldalloc",),
    "runtime_id": ("runtime",),
}


@dataclass(frozen=True, slots=True)
class ExecutionRecordLink:
    """Structured consumed/produced record link in an execution payload."""

    record_id: str
    role: str | None = None
    representation_id: str | None = None
    subject_cdef_id: str | None = None
    required: bool = True
    producer_cdef_id: str | None = None
    method: str | None = None
    declaration_fingerprint: str | None = None
    activation_generation: int | None = None
    realization_id: str | None = None
    output_slot: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_id(self.record_id, ("record",), "record_id")
        if self.role is not None and not isinstance(self.role, str):
            raise RecordValidationError("execution record link role must be a string", context={"type": type(self.role).__name__})
        if self.representation_id is not None:
            _validate_id(self.representation_id, ("repr",), "representation_id")
        if self.subject_cdef_id is not None:
            _validate_cdef_id(self.subject_cdef_id, "subject_cdef_id")
        if not isinstance(self.required, bool):
            raise RecordValidationError("execution record link required must be boolean", context={"type": type(self.required).__name__})
        exact_fields = (
            self.producer_cdef_id,
            self.method,
            self.declaration_fingerprint,
            self.activation_generation,
        )
        if any(value is not None for value in exact_fields):
            if any(value is None for value in (*exact_fields, self.realization_id, self.output_slot)):
                raise RecordValidationError(
                    "exact consumed vector requires producer, method, declaration, activation, realization, output slot, and record"
                )
            ResolvedRecord(
                producer_cdef_id=self.producer_cdef_id,
                method=self.method,
                declaration_fingerprint=self.declaration_fingerprint,
                activation_generation=self.activation_generation,
                realization_id=self.realization_id,
                output_slot=self.output_slot,
                record_id=self.record_id,
            )
        else:
            if self.realization_id is not None:
                validate_realization_id(self.realization_id)
            if self.output_slot is not None:
                validate_output_slot(self.output_slot)
            if (self.realization_id is None) != (self.output_slot is None):
                raise RecordValidationError(
                    "execution produced-record ownership requires realization_id and output_slot together"
                )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    @classmethod
    def from_json(cls, value: Any, *, default_required: bool) -> "ExecutionRecordLink":
        """Build a link from JSON-ready persisted data."""

        if isinstance(value, str):
            return cls(record_id=value, required=default_required)
        if not isinstance(value, Mapping):
            raise RecordValidationError("execution record link must be a mapping", context={"type": type(value).__name__})
        unknown = set(value) - {
            "record_id",
            "role",
            "representation_id",
            "subject_cdef_id",
            "required",
            "producer_cdef_id",
            "method",
            "declaration_fingerprint",
            "activation_generation",
            "realization_id",
            "output_slot",
            "metadata",
        }
        if unknown:
            raise RecordValidationError("execution record link contains unknown fields", context={"fields": sorted(unknown)})
        if "record_id" not in value:
            raise RecordValidationError("execution record link requires record_id")
        return cls(
            record_id=value["record_id"],
            role=value.get("role"),
            representation_id=value.get("representation_id"),
            subject_cdef_id=value.get("subject_cdef_id"),
            required=value.get("required", default_required),
            producer_cdef_id=value.get("producer_cdef_id"),
            method=value.get("method"),
            declaration_fingerprint=value.get("declaration_fingerprint"),
            activation_generation=value.get("activation_generation"),
            realization_id=value.get("realization_id"),
            output_slot=value.get("output_slot"),
            metadata=value.get("metadata") or {},
        )

    def to_json(self) -> dict[str, Any]:
        """Return the canonical JSON form of this link."""

        data: dict[str, Any] = {"record_id": self.record_id, "required": self.required}
        _put_optional(data, "role", self.role)
        _put_optional(data, "representation_id", self.representation_id)
        _put_optional(data, "subject_cdef_id", self.subject_cdef_id)
        for field_name in (
            "producer_cdef_id",
            "method",
            "declaration_fingerprint",
            "activation_generation",
            "realization_id",
            "output_slot",
        ):
            _put_optional(data, field_name, getattr(self, field_name))
        if self.metadata:
            data["metadata"] = json_ready(self.metadata)
        return data

    @classmethod
    def from_resolved(
        cls,
        resolved: ResolvedRecord,
        *,
        role: str | None = None,
        representation_id: str | None = None,
        subject_cdef_id: str | None = None,
        required: bool = True,
    ) -> "ExecutionRecordLink":
        """Create an execution link from one exact consumed vector."""

        if not isinstance(resolved, ResolvedRecord):
            raise TypeError("resolved must be a ResolvedRecord")
        return cls(
            record_id=resolved.record_id,
            role=role,
            representation_id=representation_id,
            subject_cdef_id=subject_cdef_id,
            required=required,
            producer_cdef_id=resolved.producer_cdef_id,
            method=resolved.method,
            declaration_fingerprint=resolved.declaration_fingerprint,
            activation_generation=resolved.activation_generation,
            realization_id=resolved.realization_id,
            output_slot=resolved.output_slot,
        )

    def to_resolved(self) -> ResolvedRecord:
        """Return the exact consumed vector or reject a non-exact link."""

        try:
            return ResolvedRecord(
                producer_cdef_id=self.producer_cdef_id,
                method=self.method,
                declaration_fingerprint=self.declaration_fingerprint,
                activation_generation=self.activation_generation,
                realization_id=self.realization_id,
                output_slot=self.output_slot,
                record_id=self.record_id,
            )
        except RecordValidationError as exc:
            raise RecordValidationError("execution link is not an exact consumed vector") from exc


@dataclass(frozen=True, slots=True)
class ExecutionLogRef:
    """Store-relative stdout/stderr/log product reference."""

    stream: str
    storage: StorageRef | Mapping[str, Any]
    content_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stream, str) or not self.stream:
            raise RecordValidationError("execution log stream must be a non-empty string")
        try:
            storage = self.storage if isinstance(self.storage, StorageRef) else StorageRef.from_json(self.storage)
        except Exception as exc:
            raise RecordValidationError("invalid execution log storage ref", context=getattr(exc, "context", {})) from exc
        if self.content_type is not None and not isinstance(self.content_type, str):
            raise RecordValidationError("execution log content_type must be a string", context={"type": type(self.content_type).__name__})
        object.__setattr__(self, "storage", storage)

    @classmethod
    def from_json(cls, value: Any) -> "ExecutionLogRef":
        """Build a log ref from JSON-ready persisted data."""

        if not isinstance(value, Mapping):
            raise RecordValidationError("execution log ref must be a mapping", context={"type": type(value).__name__})
        unknown = set(value) - {"stream", "storage", "content_type"}
        if unknown:
            raise RecordValidationError("execution log ref contains unknown fields", context={"fields": sorted(unknown)})
        if "storage" not in value:
            raise RecordValidationError("execution log ref requires storage")
        return cls(value.get("stream"), value["storage"], value.get("content_type"))

    def to_json(self) -> dict[str, Any]:
        """Return the canonical JSON form of this log ref."""

        data = {"stream": self.stream, "storage": self.storage.to_json()}
        _put_optional(data, "content_type", self.content_type)
        return data


@dataclass(frozen=True, slots=True)
class ExecutionErrorInfo:
    """Normalized error information for failed executions."""

    type: str | None = None
    message: str | None = None
    traceback: str | None = None
    exit_code: int | None = None
    signal: str | int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("type", "message", "traceback"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise RecordValidationError(f"execution error {name} must be a string", context={"type": type(value).__name__})
        if self.exit_code is not None and not isinstance(self.exit_code, int):
            raise RecordValidationError("execution error exit_code must be an int", context={"type": type(self.exit_code).__name__})
        if self.signal is not None and not isinstance(self.signal, (str, int)):
            raise RecordValidationError("execution error signal must be string, int, or null", context={"type": type(self.signal).__name__})
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    @classmethod
    def from_json(cls, value: Any) -> "ExecutionErrorInfo":
        """Build error info from a JSON-ready mapping."""

        if not isinstance(value, Mapping):
            raise RecordValidationError("execution error must be a mapping", context={"type": type(value).__name__})
        unknown = set(value) - {"type", "message", "traceback", "exit_code", "signal", "metadata"}
        if unknown:
            raise RecordValidationError("execution error contains unknown fields", context={"fields": sorted(unknown)})
        return cls(value.get("type"), value.get("message"), value.get("traceback"), value.get("exit_code"), value.get("signal"), value.get("metadata") or {})

    def to_json(self) -> dict[str, Any]:
        """Return the canonical JSON form."""

        data: dict[str, Any] = {}
        for key in ("type", "message", "traceback", "exit_code", "signal"):
            _put_optional(data, key, getattr(self, key))
        if self.metadata:
            data["metadata"] = json_ready(self.metadata)
        return data


def persistence_safe_execution_error(
    error: BaseException | Mapping[str, Any],
) -> dict[str, Any]:
    """Project a failure without persisting exception-controlled text."""

    candidate = type(error).__name__ if isinstance(error, BaseException) else error.get("type")
    error_type = (
        candidate
        if isinstance(candidate, str)
        and len(candidate) <= 128
        and candidate.isidentifier()
        else "Error"
    )
    return {
        "type": error_type,
        "metadata": {"code": PERSISTENCE_SAFE_FAILURE_CODE},
    }


@dataclass(frozen=True, slots=True)
class ExecutionCancellationInfo:
    """Normalized cancellation facts for cancelled executions."""

    requested: bool = True
    method: str | None = None
    escalated: bool = False
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.requested, bool):
            raise RecordValidationError("cancellation requested must be boolean")
        if not isinstance(self.escalated, bool):
            raise RecordValidationError("cancellation escalated must be boolean")
        for name in ("method", "reason"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise RecordValidationError(f"cancellation {name} must be a string")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    @classmethod
    def from_json(cls, value: Any) -> "ExecutionCancellationInfo":
        """Build cancellation info from a JSON-ready mapping."""

        if not isinstance(value, Mapping):
            raise RecordValidationError("execution cancellation must be a mapping", context={"type": type(value).__name__})
        unknown = set(value) - {"requested", "method", "escalated", "reason", "metadata"}
        if unknown:
            raise RecordValidationError("execution cancellation contains unknown fields", context={"fields": sorted(unknown)})
        return cls(value.get("requested", True), value.get("method"), value.get("escalated", False), value.get("reason"), value.get("metadata") or {})

    def to_json(self) -> dict[str, Any]:
        """Return the canonical JSON form."""

        data = {"requested": self.requested, "escalated": self.escalated}
        _put_optional(data, "method", self.method)
        _put_optional(data, "reason", self.reason)
        if self.metadata:
            data["metadata"] = json_ready(self.metadata)
        return data


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Typed wrapper for immutable ``kind='execution'`` provenance records."""

    execution_kind: str
    operation_id: str
    backend: Mapping[str, Any]
    status: str
    dispatch_id: str | None = None
    recipe_id: str | None = None
    environment_id: str | None = None
    environment_record_id: str | None = None
    environment_spec_id: str | None = None
    environment_requirement_id: str | None = None
    world_requirement_id: str | None = None
    world_id: str | None = None
    world_allocation_id: str | None = None
    runtime_id: str | None = None
    realization_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_ms: int | float | None = None
    input_cdef_ids: tuple[str, ...] = ()
    output_cdef_ids: tuple[str, ...] = ()
    consumed_cdef_ids: tuple[str, ...] = ()
    produced_cdef_ids: tuple[str, ...] = ()
    consumed_records: tuple[ExecutionRecordLink | str | Mapping[str, Any], ...] = ()
    produced_records: tuple[ExecutionRecordLink | str | Mapping[str, Any], ...] = ()
    probe_report_ids: tuple[str, ...] = ()
    adapter_record_ids: tuple[str, ...] = ()
    program_record_ids: tuple[str, ...] = ()
    logs: tuple[ExecutionLogRef | Mapping[str, Any], ...] = ()
    error: ExecutionErrorInfo | Mapping[str, Any] | None = None
    cancellation: ExecutionCancellationInfo | Mapping[str, Any] | None = None
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "execution"
    _known_payload_keys: ClassVar[frozenset[str]] = frozenset(
        {
            "execution_kind",
            "operation_id",
            "backend",
            "status",
            "dispatch_id",
            "recipe_id",
            "environment_id",
            "environment_record_id",
            "environment_spec_id",
            "environment_requirement_id",
            "world_requirement_id",
            "world_id",
            "world_allocation_id",
            "runtime_id",
            "realization_id",
            "started_at",
            "ended_at",
            "duration_ms",
            "input_cdef_ids",
            "output_cdef_ids",
            "consumed_cdef_ids",
            "produced_cdef_ids",
            "consumed_records",
            "produced_records",
            "probe_report_ids",
            "adapter_record_ids",
            "program_record_ids",
            "logs",
            "error",
            "cancellation",
            "diagnostics",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_kind", normalize_execution_kind(self.execution_kind))
        object.__setattr__(self, "status", normalize_execution_status(self.status))
        _validate_id(self.operation_id, ("op",), "operation_id")
        object.__setattr__(self, "backend", _backend_identity(self.backend))
        for field_name, prefixes in _COMMON_ID_FIELDS.items():
            value = getattr(self, field_name)
            if value is not None:
                _validate_id(value, prefixes, field_name)
        if self.realization_id is not None:
            validate_realization_id(self.realization_id)
        for field_name in ("started_at", "ended_at"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_timestamp(value, field_name)
        _validate_duration(self.duration_ms)
        for field_name in ("input_cdef_ids", "output_cdef_ids", "consumed_cdef_ids", "produced_cdef_ids"):
            object.__setattr__(self, field_name, _cdef_id_tuple(getattr(self, field_name), field_name))
        for field_name in ("probe_report_ids", "adapter_record_ids", "program_record_ids"):
            object.__setattr__(self, field_name, _record_id_tuple(getattr(self, field_name), field_name))
        object.__setattr__(self, "consumed_records", _link_tuple(self.consumed_records, default_required=True, field_name="consumed_records"))
        object.__setattr__(self, "produced_records", _link_tuple(self.produced_records, default_required=False, field_name="produced_records"))
        object.__setattr__(self, "produced_records", _with_specialized_outputs(self.produced_records, self.probe_report_ids, self.adapter_record_ids, self.program_record_ids))
        object.__setattr__(self, "logs", _log_tuple(self.logs))
        if self.error is not None:
            object.__setattr__(self, "error", self.error if isinstance(self.error, ExecutionErrorInfo) else ExecutionErrorInfo.from_json(self.error))
        if self.cancellation is not None:
            object.__setattr__(self, "cancellation", self.cancellation if isinstance(self.cancellation, ExecutionCancellationInfo) else ExecutionCancellationInfo.from_json(self.cancellation))
        object.__setattr__(self, "diagnostics", tuple(_freeze_mapping(item, "diagnostics") for item in _json_sequence_value(self.diagnostics, "diagnostics")))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "extra", _freeze_mapping(self.extra, "extra"))
        _validate_status_context(self)

    @property
    def consumed_record_ids(self) -> tuple[str, ...]:
        """Return record IDs from structured consumed-record links."""

        return tuple(link.record_id for link in self.consumed_records)

    @property
    def produced_record_ids(self) -> tuple[str, ...]:
        """Return record IDs from structured produced-record links."""

        return tuple(link.record_id for link in self.produced_records)

    @classmethod
    def from_envelope(cls, record: Mapping[str, Any]) -> "ExecutionRecord":
        """Validate and wrap a generic execution record envelope."""

        validate_record(record, kind=cls.kind)
        payload = _payload(record)
        extra = {key: payload[key] for key in payload if key not in cls._known_payload_keys}
        for required in ("execution_kind", "operation_id", "backend", "status"):
            if required not in payload:
                raise RecordValidationError("execution payload missing required field", context={"field": required})
        return cls(
            execution_kind=payload["execution_kind"],
            operation_id=payload["operation_id"],
            backend=payload["backend"],
            status=payload["status"],
            dispatch_id=payload.get("dispatch_id"),
            recipe_id=payload.get("recipe_id"),
            environment_id=payload.get("environment_id"),
            environment_record_id=payload.get("environment_record_id"),
            environment_spec_id=payload.get("environment_spec_id"),
            environment_requirement_id=payload.get("environment_requirement_id"),
            world_requirement_id=payload.get("world_requirement_id"),
            world_id=payload.get("world_id"),
            world_allocation_id=payload.get("world_allocation_id"),
            runtime_id=payload.get("runtime_id"),
            realization_id=payload.get("realization_id"),
            started_at=payload.get("started_at"),
            ended_at=payload.get("ended_at"),
            duration_ms=payload.get("duration_ms"),
            input_cdef_ids=_json_sequence(payload, "input_cdef_ids"),
            output_cdef_ids=_json_sequence(payload, "output_cdef_ids"),
            consumed_cdef_ids=_json_sequence(payload, "consumed_cdef_ids"),
            produced_cdef_ids=_json_sequence(payload, "produced_cdef_ids"),
            consumed_records=_json_sequence(payload, "consumed_records"),
            produced_records=_json_sequence(payload, "produced_records"),
            probe_report_ids=_json_sequence(payload, "probe_report_ids"),
            adapter_record_ids=_json_sequence(payload, "adapter_record_ids"),
            program_record_ids=_json_sequence(payload, "program_record_ids"),
            logs=_json_sequence(payload, "logs"),
            error=payload.get("error"),
            cancellation=payload.get("cancellation"),
            diagnostics=_json_sequence(payload, "diagnostics"),
            metadata=record.get("metadata") or {},
            extra=extra,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical generic execution-record payload."""

        payload = dict(json_ready(self.extra))
        payload.update({"execution_kind": self.execution_kind, "operation_id": self.operation_id, "backend": json_ready(self.backend), "status": self.status})
        for field_name in _COMMON_ID_FIELDS:
            _put_optional(payload, field_name, getattr(self, field_name))
        _put_optional(payload, "realization_id", self.realization_id)
        for field_name in ("started_at", "ended_at", "duration_ms"):
            _put_optional(payload, field_name, getattr(self, field_name))
        for field_name in ("input_cdef_ids", "output_cdef_ids", "consumed_cdef_ids", "produced_cdef_ids", "probe_report_ids", "adapter_record_ids", "program_record_ids"):
            values = getattr(self, field_name)
            if values:
                payload[field_name] = list(values)
        if self.consumed_records:
            payload["consumed_records"] = [link.to_json() for link in self.consumed_records]
        if self.produced_records:
            payload["produced_records"] = [link.to_json() for link in self.produced_records]
        if self.logs:
            payload["logs"] = [log.to_json() for log in self.logs]
        if self.error is not None:
            payload["error"] = self.error.to_json()
        if self.cancellation is not None:
            payload["cancellation"] = self.cancellation.to_json()
        if self.diagnostics:
            payload["diagnostics"] = [json_ready(item) for item in self.diagnostics]
        return payload

    def to_envelope(self) -> dict[str, Any]:
        """Return a validated generic record envelope."""

        return make_record(kind=self.kind, payload=self.to_payload(), metadata=self.metadata)


def execution_record_for_result(**kwargs: Any) -> ExecutionRecord:
    """Build a generic execution record from normalized result metadata."""

    return ExecutionRecord(**kwargs)


def execution_record_for_probe_report(probe_report: Any, *, operation_id: str | None = None, probe_report_id: str | None = None, **kwargs: Any) -> ExecutionRecord:
    """Create optional execution provenance for a provider probe report.

    Probe report objects, payload mappings, and full record envelopes are
    accepted. Status, diagnostics, and error details are copied when present so
    failed probe reports produce valid execution provenance without callers
    repeating the failure context.
    """

    op_id = operation_id or _source_field(probe_report, "operation_id")
    status = _source_field(probe_report, "status") or "ok"
    record_id = probe_report_id or _source_field(probe_report, "report_id") or _source_field(probe_report, "id")
    produced = tuple(kwargs.pop("produced_records", ()))
    probe_ids = tuple(kwargs.pop("probe_report_ids", ()))
    diagnostics = _diagnostics_tuple(kwargs.pop("diagnostics", _source_field(probe_report, "diagnostics") or ()))
    error = kwargs.pop("error", _source_field(probe_report, "error"))
    if record_id:
        probe_ids = tuple(dict.fromkeys((*probe_ids, record_id)))
    return ExecutionRecord(execution_kind="probe", operation_id=op_id, backend=kwargs.pop("backend", {"name": "dryml.provider_probe", "kind": "probe"}), status=status, probe_report_ids=probe_ids, produced_records=produced, diagnostics=diagnostics, error=error, **kwargs)


def execution_record_for_adapter_result(adapter_result: Any, *, operation_id: str, backend: Mapping[str, Any] | None = None, **kwargs: Any) -> ExecutionRecord:
    """Create optional execution provenance for an adapter execution result.

    Adapter result objects, payload mappings, and full record envelopes are
    accepted. Diagnostics, errors, adapter records, and target records are
    copied when present so failed adapter results retain their failure context.
    """

    status = _source_field(adapter_result, "status") or "ok"
    produced = tuple(kwargs.pop("produced_records", ()))
    adapter_ids = tuple(kwargs.pop("adapter_record_ids", ()))
    diagnostics_value = kwargs.pop("diagnostics", _source_field(adapter_result, "diagnostics"))
    if diagnostics_value is None:
        diagnostics_value = _source_field(adapter_result, "issues") or ()
    diagnostics = _diagnostics_tuple(diagnostics_value)
    error = kwargs.pop("error", _source_field(adapter_result, "error"))
    for ref in _source_sequence(adapter_result, "adapter_records"):
        adapter_ids = (*adapter_ids, getattr(ref, "record_id", ref))
    for ref in _source_sequence(adapter_result, "target_records"):
        produced = (*produced, getattr(ref, "record_id", ref))
    return ExecutionRecord(execution_kind="adapter", operation_id=operation_id, backend=backend or {"name": "dryml.adapter", "kind": "adapter"}, status=status, produced_records=produced, adapter_record_ids=tuple(dict.fromkeys(adapter_ids)), diagnostics=diagnostics, error=error, **kwargs)


def unsupported_compiler_execution_record(*, operation_id: str, backend: Mapping[str, Any] | None = None, execution_kind: str = "compiler", **kwargs: Any) -> ExecutionRecord:
    """Create an unsupported compiler/lowering provenance record."""

    return ExecutionRecord(execution_kind=execution_kind, operation_id=operation_id, backend=backend or {"name": "dryml.compiler", "kind": execution_kind}, status="unsupported", **kwargs)


def write_execution_record(record_io: Any, execution: ExecutionRecord | Mapping[str, Any], *, overwrite: bool = False) -> LocatedRecordRef:
    """Write an execution record through ``RecordStoreIO`` with validation."""

    envelope = execution.to_envelope() if isinstance(execution, ExecutionRecord) else ExecutionRecord.from_envelope(execution).to_envelope()
    return record_io.write_record(envelope, overwrite=overwrite)


def find_execution_records(repo_or_store: Any, **filters: Any) -> tuple[LocatedRecordRef, ...]:
    """Find execution provenance records on a store, repo, or federation."""

    records = getattr(repo_or_store, "records", repo_or_store)
    if hasattr(records, "find_execution_records"):
        return records.find_execution_records(**filters)
    raise RecordValidationError("object does not expose execution-record queries", context={"type": type(repo_or_store).__name__})


def find_execution_records_for_operation(repo_or_store: Any, operation_id: str, **filters: Any) -> tuple[LocatedRecordRef, ...]:
    """Find execution records for one operation ID."""

    return find_execution_records(repo_or_store, operation_id=operation_id, **filters)


def find_execution_records_consuming(repo_or_store: Any, record_id: str, **filters: Any) -> tuple[LocatedRecordRef, ...]:
    """Find execution records consuming a record ID."""

    return find_execution_records(repo_or_store, consumed_record_id=record_id, **filters)


def find_execution_records_producing(repo_or_store: Any, record_id: str, **filters: Any) -> tuple[LocatedRecordRef, ...]:
    """Find execution records producing a record ID."""

    return find_execution_records(repo_or_store, produced_record_id=record_id, **filters)


def execution_record_matches(record: Mapping[str, Any], **filters: Any) -> bool:
    """Return whether an execution record envelope matches query filters."""

    execution = ExecutionRecord.from_envelope(record)
    if filters.get("operation_id") is not None and execution.operation_id != filters["operation_id"]:
        return False
    for name in ("dispatch_id", "recipe_id", "status", "execution_kind"):
        if filters.get(name) is not None and getattr(execution, name) != filters[name]:
            return False
    if filters.get("consumed_record_id") is not None and filters["consumed_record_id"] not in execution.consumed_record_ids:
        return False
    if filters.get("produced_record_id") is not None and filters["produced_record_id"] not in execution.produced_record_ids:
        return False
    return True


def normalize_execution_status(value: Any) -> str:
    """Return a normalized execution status token."""

    return _normalize_token(value, EXECUTION_STATUSES, "status")


def normalize_execution_kind(value: Any) -> str:
    """Return a normalized execution-kind token."""

    return _normalize_token(value, EXECUTION_KINDS, "execution_kind")


def _payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise RecordValidationError("typed record payload must be a mapping", context={"type": type(payload).__name__})
    return payload


def _normalize_token(value: Any, allowed: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str):
        raise RecordValidationError(f"{field_name} must be a string", context={"type": type(value).__name__})
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise RecordValidationError(f"invalid {field_name}", context={"value": value, "allowed": sorted(allowed)})
    return normalized


def _backend_identity(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordValidationError("backend must be a mapping", context={"type": type(value).__name__})
    unknown = set(value) - {"name", "kind", "version", "provider", "metadata"}
    if unknown:
        raise RecordValidationError("backend contains unknown fields", context={"fields": sorted(unknown)})
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise RecordValidationError("backend.name must be a non-empty string")
    for key in ("kind", "version", "provider"):
        if key in value and value[key] is not None and not isinstance(value[key], str):
            raise RecordValidationError(f"backend.{key} must be a string", context={"type": type(value[key]).__name__})
    if "metadata" in value and value["metadata"] is not None and not isinstance(value["metadata"], Mapping):
        raise RecordValidationError("backend.metadata must be a mapping", context={"type": type(value['metadata']).__name__})
    return _freeze_mapping(value, "backend")


def _with_specialized_outputs(
    produced_records: tuple[ExecutionRecordLink, ...],
    probe_report_ids: tuple[str, ...],
    adapter_record_ids: tuple[str, ...],
    program_record_ids: tuple[str, ...],
) -> tuple[ExecutionRecordLink, ...]:
    result = list(produced_records)
    seen = {link.record_id for link in result}
    for role, record_ids in (
        ("probe-report", probe_report_ids),
        ("adapter-record", adapter_record_ids),
        ("program-record", program_record_ids),
    ):
        for record_id in record_ids:
            if record_id not in seen:
                result.append(ExecutionRecordLink(record_id, role=role, required=False))
                seen.add(record_id)
    return tuple(result)


def _validate_status_context(execution: ExecutionRecord) -> None:
    if execution.status == "ok" and execution.error is not None:
        raise RecordValidationError("ok execution records must not include error")
    if execution.status == "ok" and execution.cancellation is not None:
        raise RecordValidationError("ok execution records must not include cancellation")
    if execution.status in {"failed", "timeout"} and execution.error is None and not execution.diagnostics:
        raise RecordValidationError("failed and timeout execution records require error or diagnostics", context={"status": execution.status})
    if execution.error is not None and not execution.diagnostics and _error_info_empty(execution.error):
        raise RecordValidationError("execution error requires details or diagnostics", context={"status": execution.status})
    if execution.status == "cancelled" and execution.cancellation is None:
        raise RecordValidationError("cancelled execution records require cancellation")
    if execution.cancellation is not None and execution.status != "cancelled":
        raise RecordValidationError("cancellation is only valid for cancelled execution records", context={"status": execution.status})


def _error_info_empty(error: ExecutionErrorInfo) -> bool:
    return not any((error.type, error.message, error.traceback, error.exit_code is not None, error.signal is not None, error.metadata))


def _validate_timestamp(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise RecordValidationError(f"{field_name} must be a string", context={"type": type(value).__name__})
    if _RFC3339_UTC_RE.fullmatch(value) is None:
        raise RecordValidationError(f"{field_name} must be an RFC3339 UTC timestamp", context={"value": value})
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecordValidationError(f"{field_name} must be an RFC3339 UTC timestamp", context={"value": value}) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RecordValidationError(f"{field_name} must be an RFC3339 UTC timestamp", context={"value": value})


def _validate_duration(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 or not math.isfinite(value):
        raise RecordValidationError("duration_ms must be a finite non-negative number", context={"duration_ms": value})


def _link_tuple(value: Sequence[Any], *, default_required: bool, field_name: str) -> tuple[ExecutionRecordLink, ...]:
    items = _json_sequence_value(value, field_name)
    return tuple(item if isinstance(item, ExecutionRecordLink) else ExecutionRecordLink.from_json(item, default_required=default_required) for item in items)


def _log_tuple(value: Sequence[Any]) -> tuple[ExecutionLogRef, ...]:
    items = _json_sequence_value(value, "logs")
    return tuple(item if isinstance(item, ExecutionLogRef) else ExecutionLogRef.from_json(item) for item in items)


def _record_id_tuple(value: Sequence[str], field_name: str) -> tuple[str, ...]:
    items = _json_sequence_value(value, field_name)
    result = tuple(items)
    for item in result:
        _validate_id(item, ("record",), field_name)
    return result


def _cdef_id_tuple(value: Sequence[str], field_name: str) -> tuple[str, ...]:
    items = _json_sequence_value(value, field_name)
    return tuple(_validate_cdef_id(item, field_name) for item in items)


def _json_sequence(payload: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in payload or payload[field_name] is None:
        return ()
    return _json_sequence_value(payload[field_name], field_name)


def _json_sequence_value(value: Any, field_name: str) -> Any:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise RecordValidationError(f"{field_name} must be a JSON array, not a string" if isinstance(value, str) else f"{field_name} must be a JSON array", context={"type": type(value).__name__})
    return value


def _source_payload(source: Any) -> Mapping[str, Any] | None:
    if not isinstance(source, Mapping):
        return None
    payload = source.get("payload")
    return payload if isinstance(payload, Mapping) else None


def _source_field(source: Any, field_name: str) -> Any:
    value = getattr(source, field_name, None)
    if value is not None:
        return value
    if isinstance(source, Mapping):
        if field_name in source:
            return source.get(field_name)
        payload = _source_payload(source)
        if payload is not None:
            return payload.get(field_name)
    return None


def _source_sequence(source: Any, field_name: str) -> tuple[Any, ...]:
    value = _source_field(source, field_name)
    if value is None:
        return ()
    return tuple(_json_sequence_value(value, field_name))


def _diagnostics_tuple(value: Any) -> tuple[Mapping[str, Any], ...]:
    result = []
    for item in _json_sequence_value(value, "diagnostics"):
        if isinstance(item, Mapping):
            result.append(item)
        elif hasattr(item, "to_data"):
            result.append(item.to_data())
        elif hasattr(item, "to_json"):
            result.append(item.to_json())
        else:
            result.append(item)
    return tuple(result)


def _validate_cdef_id(value: Any, field_name: str) -> str:
    try:
        return parse_cdef_id(value).raw
    except ReferenceParseError as exc:
        raise RecordValidationError(f"invalid {field_name}", context=exc.context) from exc


def _validate_id(value: Any, prefixes: tuple[str, ...], field_name: str) -> None:
    if not isinstance(value, str):
        raise RecordValidationError(f"{field_name} must be a string", context={"type": type(value).__name__})
    try:
        parts = parse_content_id(value)
    except ContentIDError as exc:
        raise RecordValidationError(f"invalid {field_name}", context=exc.context) from exc
    if parts.prefix not in prefixes or parts.schema_version != 1:
        raise RecordValidationError(f"{field_name} prefix mismatch", context={"value": value, "expected": prefixes})


def _freeze_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordValidationError("execution record JSON field must be a mapping", context={"path": path, "type": type(value).__name__})
    try:
        frozen = deep_freeze_json(value)
    except CanonicalJSONError as exc:
        raise RecordValidationError("execution record JSON field is not canonical JSON", context={"path": path, **exc.context}) from exc
    assert isinstance(frozen, Mapping)
    return frozen


def _put_optional(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


__all__ = [
    "EXECUTION_KINDS",
    "EXECUTION_STATUSES",
    "ExecutionCancellationInfo",
    "ExecutionErrorInfo",
    "ExecutionLogRef",
    "ExecutionRecord",
    "ExecutionRecordLink",
    "execution_record_for_adapter_result",
    "execution_record_for_probe_report",
    "execution_record_for_result",
    "execution_record_matches",
    "find_execution_records",
    "find_execution_records_consuming",
    "find_execution_records_for_operation",
    "find_execution_records_producing",
    "normalize_execution_kind",
    "normalize_execution_status",
    "unsupported_compiler_execution_record",
    "write_execution_record",
]
