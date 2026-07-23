"""Typed wrappers over generic DRYML record envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from dryml.formats import CanonicalJSONError, deep_freeze_json, json_ready
from dryml.formats.errors import ContentIDError, ReferenceParseError
from dryml.formats.ids import parse_content_id
from dryml.formats.refs import parse_cdef_id

from .errors import RecordValidationError, StorageRefError
from .records import make_record, validate_record
from .storage import StorageRef
from .execution import ExecutionRecord
from .realizations import (
    RealizationRecord,
    validate_output_slot,
    validate_realization_id,
)


_COMMON_IDS = {
    "environment_id": ("envrec", "env"),
    "environment_record_id": ("envrec",),
    "environment_requirement_id": ("envreq",),
    "world_requirement_id": ("worldreq",),
    "world_id": ("world",),
    "runtime_id": ("runtime",),
}


@dataclass(frozen=True, slots=True)
class StoredStateRecord:
    """Typed wrapper for ``kind='stored_state'`` record payloads."""

    subject_cdef_id: str
    representation_id: str
    storage: tuple[StorageRef | Mapping[str, Any], ...]
    owner_cdef_id: str | None = None
    owner_path: str | None = None
    environment_id: str | None = None
    environment_record_id: str | None = None
    environment_requirement_id: str | None = None
    world_requirement_id: str | None = None
    world_id: str | None = None
    runtime_id: str | None = None
    state_role: str | None = None
    manifest: Mapping[str, Any] | None = None
    derived_from: tuple[str, ...] = ()
    realization_id: str | None = None
    output_slot: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "stored_state"
    _known_payload_keys: ClassVar[frozenset[str]] = frozenset(
        {
            "subject_cdef_id",
            "owner_cdef_id",
            "owner_path",
            "representation_id",
            "environment_id",
            "environment_record_id",
            "environment_requirement_id",
            "world_requirement_id",
            "world_id",
            "runtime_id",
            "storage",
            "state_role",
            "manifest",
            "derived_from",
            "realization_id",
            "output_slot",
        }
    )

    def __post_init__(self) -> None:
        _validate_cdef_id(self.subject_cdef_id, "subject_cdef_id")
        if self.owner_cdef_id is not None:
            _validate_cdef_id(self.owner_cdef_id, "owner_cdef_id")
        _validate_id(self.representation_id, ("repr",), "representation_id")
        _validate_optional_common_ids(self)
        object.__setattr__(self, "storage", _storage_tuple(self.storage, require_non_empty=True))
        object.__setattr__(self, "derived_from", _record_id_tuple(self.derived_from, "derived_from"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "extra", _freeze_mapping(self.extra, "extra"))
        object.__setattr__(self, "manifest", None if self.manifest is None else _freeze_mapping(self.manifest, "manifest"))
        _validate_managed_ownership(self.realization_id, self.output_slot)

    @classmethod
    def from_envelope(cls, record: Mapping[str, Any]) -> "StoredStateRecord":
        """Validate and wrap a generic stored-state record envelope."""

        validate_record(record, kind=cls.kind)
        payload = _payload(record)
        extra = {key: payload[key] for key in payload if key not in cls._known_payload_keys}
        return cls(
            subject_cdef_id=payload.get("subject_cdef_id"),
            representation_id=payload.get("representation_id"),
            storage=_json_sequence(payload, "storage"),
            owner_cdef_id=payload.get("owner_cdef_id"),
            owner_path=payload.get("owner_path"),
            environment_id=payload.get("environment_id"),
            environment_record_id=payload.get("environment_record_id"),
            environment_requirement_id=payload.get("environment_requirement_id"),
            world_requirement_id=payload.get("world_requirement_id"),
            world_id=payload.get("world_id"),
            runtime_id=payload.get("runtime_id"),
            state_role=payload.get("state_role"),
            manifest=payload.get("manifest"),
            derived_from=_json_sequence(payload, "derived_from"),
            realization_id=payload.get("realization_id"),
            output_slot=payload.get("output_slot"),
            metadata=record.get("metadata") or {},
            extra=extra,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical generic record payload."""

        payload = dict(json_ready(self.extra))
        payload.update(
            {
                "subject_cdef_id": self.subject_cdef_id,
                "representation_id": self.representation_id,
                "storage": [ref.to_json() for ref in self.storage],
            }
        )
        _put_optional(payload, "owner_cdef_id", self.owner_cdef_id)
        _put_optional(payload, "owner_path", self.owner_path)
        _put_optional(payload, "environment_id", self.environment_id)
        _put_optional(payload, "environment_record_id", self.environment_record_id)
        _put_optional(payload, "environment_requirement_id", self.environment_requirement_id)
        _put_optional(payload, "world_requirement_id", self.world_requirement_id)
        _put_optional(payload, "world_id", self.world_id)
        _put_optional(payload, "runtime_id", self.runtime_id)
        _put_optional(payload, "state_role", self.state_role)
        if self.manifest is not None:
            payload["manifest"] = json_ready(self.manifest)
        if self.derived_from:
            payload["derived_from"] = list(self.derived_from)
        _put_optional(payload, "realization_id", self.realization_id)
        _put_optional(payload, "output_slot", self.output_slot)
        return payload

    def to_envelope(self) -> dict[str, Any]:
        """Return a validated generic record envelope."""

        return make_record(kind=self.kind, payload=self.to_payload(), metadata=self.metadata)


@dataclass(frozen=True, slots=True)
class DataRecord:
    """Typed wrapper for data product records."""

    representation_id: str
    storage: tuple[StorageRef | Mapping[str, Any], ...]
    subject_cdef_id: str | None = None
    operation_id: str | None = None
    data_role: str | None = None
    manifest: Mapping[str, Any] | None = None
    preview: Mapping[str, Any] | None = None
    derived_from: tuple[str, ...] = ()
    realization_id: str | None = None
    output_slot: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "data"
    _known_payload_keys: ClassVar[frozenset[str]] = frozenset({"subject_cdef_id", "operation_id", "representation_id", "storage", "data_role", "manifest", "preview", "derived_from", "realization_id", "output_slot"})

    def __post_init__(self) -> None:
        if self.subject_cdef_id is not None:
            _validate_cdef_id(self.subject_cdef_id, "subject_cdef_id")
        if self.operation_id is not None:
            _validate_id(self.operation_id, ("op",), "operation_id")
        _validate_id(self.representation_id, ("repr",), "representation_id")
        object.__setattr__(self, "storage", _storage_tuple(self.storage, require_non_empty=True))
        object.__setattr__(self, "derived_from", _record_id_tuple(self.derived_from, "derived_from"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "extra", _freeze_mapping(self.extra, "extra"))
        object.__setattr__(self, "manifest", None if self.manifest is None else _freeze_mapping(self.manifest, "manifest"))
        object.__setattr__(self, "preview", None if self.preview is None else _freeze_mapping(self.preview, "preview"))
        _validate_managed_ownership(self.realization_id, self.output_slot)

    @classmethod
    def from_envelope(cls, record: Mapping[str, Any]) -> "DataRecord":
        """Validate and wrap a generic data record envelope."""

        validate_record(record, kind=cls.kind)
        payload = _payload(record)
        extra = {key: payload[key] for key in payload if key not in cls._known_payload_keys}
        return cls(
            subject_cdef_id=payload.get("subject_cdef_id"),
            operation_id=payload.get("operation_id"),
            representation_id=payload.get("representation_id"),
            storage=_json_sequence(payload, "storage"),
            data_role=payload.get("data_role"),
            manifest=payload.get("manifest"),
            preview=payload.get("preview"),
            derived_from=_json_sequence(payload, "derived_from"),
            realization_id=payload.get("realization_id"),
            output_slot=payload.get("output_slot"),
            metadata=record.get("metadata") or {},
            extra=extra,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical generic record payload."""

        payload = dict(json_ready(self.extra))
        payload.update({"representation_id": self.representation_id, "storage": [ref.to_json() for ref in self.storage]})
        _put_optional(payload, "subject_cdef_id", self.subject_cdef_id)
        _put_optional(payload, "operation_id", self.operation_id)
        _put_optional(payload, "data_role", self.data_role)
        if self.manifest is not None:
            payload["manifest"] = json_ready(self.manifest)
        if self.preview is not None:
            payload["preview"] = json_ready(self.preview)
        if self.derived_from:
            payload["derived_from"] = list(self.derived_from)
        _put_optional(payload, "realization_id", self.realization_id)
        _put_optional(payload, "output_slot", self.output_slot)
        return payload

    def to_envelope(self) -> dict[str, Any]:
        """Return a validated generic record envelope."""

        return make_record(kind=self.kind, payload=self.to_payload(), metadata=self.metadata)


@dataclass(frozen=True, slots=True)
class ProgramRecord:
    """Typed wrapper for compiler/JIT-like product records."""

    representation_id: str
    storage: tuple[StorageRef | Mapping[str, Any], ...]
    operation_id: str | None = None
    target: Mapping[str, Any] = field(default_factory=dict)
    entrypoints: Mapping[str, Any] = field(default_factory=dict)
    provider: Mapping[str, Any] = field(default_factory=dict)
    toolchain: Mapping[str, Any] = field(default_factory=dict)
    manifest: Mapping[str, Any] | None = None
    derived_from: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "program"
    _known_payload_keys: ClassVar[frozenset[str]] = frozenset({"operation_id", "representation_id", "storage", "target", "entrypoints", "provider", "toolchain", "manifest", "derived_from"})

    def __post_init__(self) -> None:
        if self.operation_id is not None:
            _validate_id(self.operation_id, ("op",), "operation_id")
        _validate_id(self.representation_id, ("repr",), "representation_id")
        object.__setattr__(self, "storage", _storage_tuple(self.storage, require_non_empty=True))
        object.__setattr__(self, "derived_from", _record_id_tuple(self.derived_from, "derived_from"))
        for name in ("target", "entrypoints", "provider", "toolchain", "metadata", "extra"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name), name))
        object.__setattr__(self, "manifest", None if self.manifest is None else _freeze_mapping(self.manifest, "manifest"))

    @classmethod
    def from_envelope(cls, record: Mapping[str, Any]) -> "ProgramRecord":
        """Validate and wrap a generic program record envelope."""

        validate_record(record, kind=cls.kind)
        payload = _payload(record)
        extra = {key: payload[key] for key in payload if key not in cls._known_payload_keys}
        return cls(
            operation_id=payload.get("operation_id"),
            representation_id=payload.get("representation_id"),
            storage=_json_sequence(payload, "storage"),
            target=payload.get("target") or {},
            entrypoints=payload.get("entrypoints") or {},
            provider=payload.get("provider") or {},
            toolchain=payload.get("toolchain") or {},
            manifest=payload.get("manifest"),
            derived_from=_json_sequence(payload, "derived_from"),
            metadata=record.get("metadata") or {},
            extra=extra,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical generic record payload."""

        payload = dict(json_ready(self.extra))
        payload.update({"representation_id": self.representation_id, "storage": [ref.to_json() for ref in self.storage]})
        _put_optional(payload, "operation_id", self.operation_id)
        if self.target:
            payload["target"] = json_ready(self.target)
        if self.entrypoints:
            payload["entrypoints"] = json_ready(self.entrypoints)
        if self.provider:
            payload["provider"] = json_ready(self.provider)
        if self.toolchain:
            payload["toolchain"] = json_ready(self.toolchain)
        if self.manifest is not None:
            payload["manifest"] = json_ready(self.manifest)
        if self.derived_from:
            payload["derived_from"] = list(self.derived_from)
        return payload

    def to_envelope(self) -> dict[str, Any]:
        """Return a validated generic record envelope."""

        return make_record(kind=self.kind, payload=self.to_payload(), metadata=self.metadata)


@dataclass(frozen=True, slots=True)
class AdapterRecord:
    """Typed lineage record for representation adapter execution."""

    source_record_id: str
    source_representation_id: str
    target_representation_id: str
    adapter: Mapping[str, Any]
    target_record_id: str | None = None
    operation_id: str | None = None
    produced_records: tuple[str, ...] = ()
    derived_from: tuple[str, ...] = ()
    status: str = "ok"
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "adapter"
    _known_payload_keys: ClassVar[frozenset[str]] = frozenset(
        {"adapter", "operation_id", "source_record_id", "source_representation_id", "target_record_id", "target_representation_id", "produced_records", "derived_from", "status", "diagnostics"}
    )

    def __post_init__(self) -> None:
        _validate_id(self.source_record_id, ("record",), "source_record_id")
        if self.target_record_id is not None:
            _validate_id(self.target_record_id, ("record",), "target_record_id")
        if self.operation_id is not None:
            _validate_id(self.operation_id, ("op",), "operation_id")
        _validate_id(self.source_representation_id, ("repr",), "source_representation_id")
        _validate_id(self.target_representation_id, ("repr",), "target_representation_id")
        produced = _record_id_tuple(self.produced_records, "produced_records")
        derived = _record_id_tuple(self.derived_from, "derived_from")
        if self.target_record_id is not None and self.target_record_id not in produced:
            raise RecordValidationError("adapter produced_records must include target_record_id")
        if self.source_record_id not in derived:
            raise RecordValidationError("adapter derived_from must include source_record_id")
        if self.status not in {"ok", "unsupported", "failed", "degraded"}:
            raise RecordValidationError("adapter status is invalid", context={"status": self.status})
        object.__setattr__(self, "adapter", _freeze_mapping(self.adapter, "adapter"))
        object.__setattr__(self, "produced_records", produced)
        object.__setattr__(self, "derived_from", derived)
        object.__setattr__(self, "diagnostics", tuple(_freeze_mapping(item, "diagnostics") for item in self.diagnostics))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "extra", _freeze_mapping(self.extra, "extra"))

    @classmethod
    def from_envelope(cls, record: Mapping[str, Any]) -> "AdapterRecord":
        """Validate and wrap a generic adapter record envelope."""

        validate_record(record, kind=cls.kind)
        payload = _payload(record)
        extra = {key: payload[key] for key in payload if key not in cls._known_payload_keys}
        return cls(
            adapter=payload.get("adapter") or {},
            operation_id=payload.get("operation_id"),
            source_record_id=payload.get("source_record_id"),
            source_representation_id=payload.get("source_representation_id"),
            target_record_id=payload.get("target_record_id"),
            target_representation_id=payload.get("target_representation_id"),
            produced_records=_json_sequence(payload, "produced_records"),
            derived_from=_json_sequence(payload, "derived_from"),
            status=payload.get("status", "ok"),
            diagnostics=_json_sequence(payload, "diagnostics"),
            metadata=record.get("metadata") or {},
            extra=extra,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical generic record payload."""

        payload = dict(json_ready(self.extra))
        payload.update(
            {
                "adapter": json_ready(self.adapter),
                "source_record_id": self.source_record_id,
                "source_representation_id": self.source_representation_id,
                "target_representation_id": self.target_representation_id,
                "produced_records": list(self.produced_records),
                "derived_from": list(self.derived_from),
                "status": self.status,
                "diagnostics": [json_ready(item) for item in self.diagnostics],
            }
        )
        _put_optional(payload, "target_record_id", self.target_record_id)
        _put_optional(payload, "operation_id", self.operation_id)
        return payload

    def to_envelope(self) -> dict[str, Any]:
        """Return a validated generic record envelope."""

        return make_record(kind=self.kind, payload=self.to_payload(), metadata=self.metadata)


TypedRecord = StoredStateRecord | DataRecord | ProgramRecord | AdapterRecord | ExecutionRecord | RealizationRecord


def typed_record_from_envelope(record: Mapping[str, Any]) -> TypedRecord:
    """Dispatch a generic record envelope to its typed wrapper."""

    validate_record(record)
    kind = record.get("kind")
    if kind == StoredStateRecord.kind:
        return StoredStateRecord.from_envelope(record)
    if kind == DataRecord.kind:
        return DataRecord.from_envelope(record)
    if kind == ProgramRecord.kind:
        return ProgramRecord.from_envelope(record)
    if kind == AdapterRecord.kind:
        return AdapterRecord.from_envelope(record)
    if kind == ExecutionRecord.kind:
        return ExecutionRecord.from_envelope(record)
    if kind == RealizationRecord.kind:
        return RealizationRecord.from_envelope(record)
    raise RecordValidationError("record kind has no typed wrapper", context={"kind": kind})


def _payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise RecordValidationError("typed record payload must be a mapping", context={"type": type(payload).__name__})
    return payload


def _storage_tuple(value: tuple[StorageRef | Mapping[str, Any], ...], *, require_non_empty: bool) -> tuple[StorageRef, ...]:
    if not isinstance(value, (list, tuple)):
        raise RecordValidationError("record storage must be a list", context={"type": type(value).__name__})
    try:
        result = tuple(ref if isinstance(ref, StorageRef) else StorageRef.from_json(ref) for ref in value)
    except StorageRefError as exc:
        raise RecordValidationError("invalid record storage ref", context=exc.context) from exc
    if require_non_empty and not result:
        raise RecordValidationError("record storage must be non-empty")
    return result


def _record_id_tuple(value: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RecordValidationError(f"{field_name} must be a list", context={"type": type(value).__name__})
    result = tuple(value)
    for item in result:
        _validate_id(item, ("record",), field_name)
    return result


def _json_sequence(payload: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in payload:
        return ()
    value = payload[field_name]
    if value is None:
        return ()
    if isinstance(value, str):
        raise RecordValidationError(f"{field_name} must be a JSON array, not a string")
    return value


def _validate_cdef_id(value: Any, field_name: str) -> None:
    try:
        parse_cdef_id(value)
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


def _validate_optional_common_ids(record: StoredStateRecord) -> None:
    for field_name, prefixes in _COMMON_IDS.items():
        value = getattr(record, field_name)
        if value is not None:
            _validate_id(value, prefixes, field_name)


def _freeze_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordValidationError("typed record JSON field must be a mapping", context={"path": path, "type": type(value).__name__})
    try:
        frozen = deep_freeze_json(value)
    except CanonicalJSONError as exc:
        raise RecordValidationError("typed record JSON field is not canonical JSON", context={"path": path, **exc.context}) from exc
    assert isinstance(frozen, Mapping)
    return frozen


def _put_optional(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def _validate_managed_ownership(realization_id: str | None, output_slot: str | None) -> None:
    if (realization_id is None) != (output_slot is None):
        raise RecordValidationError(
            "managed output ownership requires realization_id and output_slot together"
        )
    if realization_id is not None:
        validate_realization_id(realization_id)
        validate_output_slot(output_slot)


__all__ = [
    "AdapterRecord",
    "DataRecord",
    "ExecutionRecord",
    "ProgramRecord",
    "RealizationRecord",
    "StoredStateRecord",
    "TypedRecord",
    "typed_record_from_envelope",
]
