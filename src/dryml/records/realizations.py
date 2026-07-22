"""Immutable managed realization records and exact resolution lineage."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from dryml.formats import CanonicalJSONError, deep_freeze_json
from dryml.formats.errors import ContentIDError, ReferenceParseError
from dryml.formats.ids import parse_content_id
from dryml.formats.refs import parse_cdef_id

from .errors import RecordValidationError
from .records import make_record, validate_record


_METHOD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SLOT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_REALIZATION_RE = re.compile(r"^realization-v1-[0-9a-f]{32}$")
_ATTEMPT_RE = re.compile(r"^attempt-v1-[0-9a-f]{32}$")
_DECLARATION_RE = re.compile(r"^managed-declaration-v1-[0-9a-f]{64}$")
_OUTPUT_KINDS = frozenset({"data", "stored_state"})


def validate_realization_id(value: Any, field_name: str = "realization_id") -> str:
    """Validate one preallocated managed realization ID."""

    if not isinstance(value, str) or _REALIZATION_RE.fullmatch(value) is None:
        raise RecordValidationError(f"invalid {field_name}")
    return value


def validate_attempt_id(value: Any, field_name: str = "attempt_id") -> str:
    """Validate one fence-isolated managed attempt ID."""

    if not isinstance(value, str) or _ATTEMPT_RE.fullmatch(value) is None:
        raise RecordValidationError(f"invalid {field_name}")
    return value


def validate_declaration_fingerprint(value: Any) -> str:
    """Validate one versioned managed declaration fingerprint."""

    if not isinstance(value, str) or _DECLARATION_RE.fullmatch(value) is None:
        raise RecordValidationError("invalid declaration_fingerprint")
    return value


def validate_output_slot(value: Any, field_name: str = "output_slot") -> str:
    """Validate a stable declared output slot."""

    if not isinstance(value, str) or _SLOT_RE.fullmatch(value) is None:
        raise RecordValidationError(f"invalid {field_name}")
    return value


@dataclass(frozen=True, slots=True)
class ResolvedRecord:
    """Exact concurrency-stable logical-output resolution consumed by work."""

    producer_cdef_id: str
    method: str
    declaration_fingerprint: str
    activation_generation: int
    realization_id: str
    output_slot: str
    record_id: str

    def __post_init__(self) -> None:
        _validate_cdef(self.producer_cdef_id, "producer_cdef_id")
        _validate_method(self.method)
        validate_declaration_fingerprint(self.declaration_fingerprint)
        if type(self.activation_generation) is not int or self.activation_generation < 1:
            raise RecordValidationError("activation_generation must be a positive integer")
        validate_realization_id(self.realization_id)
        validate_output_slot(self.output_slot)
        _validate_content_id(self.record_id, "record", "record_id")

    @classmethod
    def from_json(cls, value: Any) -> "ResolvedRecord":
        """Decode a strict exact-resolution vector."""

        fields = {
            "producer_cdef_id",
            "method",
            "declaration_fingerprint",
            "activation_generation",
            "realization_id",
            "output_slot",
            "record_id",
        }
        data = _strict_mapping(value, fields, "resolved record")
        try:
            return cls(**data)
        except KeyError as exc:
            raise RecordValidationError(
                "resolved record is missing a required field",
                context={"field": exc.args[0]},
            ) from exc

    def to_json(self) -> dict[str, Any]:
        """Return the strict machine-readable vector."""

        return {
            "producer_cdef_id": self.producer_cdef_id,
            "method": self.method,
            "declaration_fingerprint": self.declaration_fingerprint,
            "activation_generation": self.activation_generation,
            "realization_id": self.realization_id,
            "output_slot": self.output_slot,
            "record_id": self.record_id,
        }


@dataclass(frozen=True, slots=True)
class RealizationOutput:
    """One typed output record owned by a completed realization."""

    slot: str
    record_id: str
    record_kind: str
    representation_id: str
    required: bool = True
    subject_cdef_id: str | None = None

    def __post_init__(self) -> None:
        validate_output_slot(self.slot, "slot")
        _validate_content_id(self.record_id, "record", "record_id")
        if self.record_kind not in _OUTPUT_KINDS:
            raise RecordValidationError(
                "realization output record_kind is invalid",
                context={"record_kind": self.record_kind},
            )
        _validate_content_id(self.representation_id, "repr", "representation_id")
        if not isinstance(self.required, bool):
            raise RecordValidationError("realization output required must be boolean")
        if self.subject_cdef_id is not None:
            _validate_cdef(self.subject_cdef_id, "subject_cdef_id")

    @classmethod
    def from_json(cls, value: Any) -> "RealizationOutput":
        """Decode one strict realization output link."""

        fields = {
            "slot",
            "record_id",
            "record_kind",
            "representation_id",
            "required",
            "subject_cdef_id",
        }
        data = _strict_mapping(value, fields, "realization output", optional={"subject_cdef_id"})
        try:
            return cls(**data)
        except KeyError as exc:
            raise RecordValidationError(
                "realization output is missing a required field",
                context={"field": exc.args[0]},
            ) from exc

    def to_json(self) -> dict[str, Any]:
        """Return the strict machine-readable output link."""

        data = {
            "slot": self.slot,
            "record_id": self.record_id,
            "record_kind": self.record_kind,
            "representation_id": self.representation_id,
            "required": self.required,
        }
        if self.subject_cdef_id is not None:
            data["subject_cdef_id"] = self.subject_cdef_id
        return data


@dataclass(frozen=True, slots=True)
class RealizationRecord:
    """Immutable completion authority for one independent managed outcome."""

    realization_id: str
    producer_cdef_id: str
    method: str
    declaration_fingerprint: str
    attempt_ids: tuple[str, ...]
    outputs: tuple[RealizationOutput | Mapping[str, Any], ...]
    primary_output_slot: str
    primary_representation_id: str
    execution_record_id: str
    completed_attempt_id: str
    completion_fence: int
    consumed_records: tuple[ResolvedRecord | Mapping[str, Any], ...] = ()
    checkpoint_head: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "realization"

    def __post_init__(self) -> None:
        validate_realization_id(self.realization_id)
        _validate_cdef(self.producer_cdef_id, "producer_cdef_id")
        _validate_method(self.method)
        validate_declaration_fingerprint(self.declaration_fingerprint)
        attempts = _sequence(self.attempt_ids, "attempt_ids")
        if not attempts:
            raise RecordValidationError("realization attempt_ids must be non-empty")
        for attempt_id in attempts:
            validate_attempt_id(attempt_id)
        if len(set(attempts)) != len(attempts):
            raise RecordValidationError("realization attempt_ids must be unique")
        validate_attempt_id(self.completed_attempt_id, "completed_attempt_id")
        if self.completed_attempt_id not in attempts:
            raise RecordValidationError("completed_attempt_id must appear in attempt_ids")
        if type(self.completion_fence) is not int or self.completion_fence < 1:
            raise RecordValidationError("completion_fence must be a positive integer")
        outputs = tuple(
            item if isinstance(item, RealizationOutput) else RealizationOutput.from_json(item)
            for item in _sequence(self.outputs, "outputs")
        )
        if not outputs:
            raise RecordValidationError("realization outputs must be non-empty")
        if len({item.slot for item in outputs}) != len(outputs):
            raise RecordValidationError("realization output slots must be unique")
        if len({item.record_id for item in outputs}) != len(outputs):
            raise RecordValidationError("realization output record IDs must be unique")
        validate_output_slot(self.primary_output_slot, "primary_output_slot")
        primary = next((item for item in outputs if item.slot == self.primary_output_slot), None)
        if primary is None:
            raise RecordValidationError("primary_output_slot must name an output")
        if not primary.required:
            raise RecordValidationError("primary realization output must be required")
        _validate_content_id(self.primary_representation_id, "repr", "primary_representation_id")
        if primary.representation_id != self.primary_representation_id:
            raise RecordValidationError("primary representation does not match primary output")
        _validate_content_id(self.execution_record_id, "record", "execution_record_id")
        consumed = tuple(
            item if isinstance(item, ResolvedRecord) else ResolvedRecord.from_json(item)
            for item in _sequence(self.consumed_records, "consumed_records")
        )
        if self.checkpoint_head is not None:
            if not isinstance(self.checkpoint_head, str) or not self.checkpoint_head.startswith("checkpoint-v1-"):
                raise RecordValidationError("invalid checkpoint_head")
        object.__setattr__(self, "attempt_ids", tuple(attempts))
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "consumed_records", consumed)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    @classmethod
    def from_envelope(cls, record: Mapping[str, Any]) -> "RealizationRecord":
        """Validate and decode a realization record envelope."""

        validate_record(record, kind=cls.kind)
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise RecordValidationError("realization payload must be a mapping")
        fields = {
            "realization_id",
            "producer_cdef_id",
            "method",
            "declaration_fingerprint",
            "attempt_ids",
            "outputs",
            "primary_output_slot",
            "primary_representation_id",
            "execution_record_id",
            "completed_attempt_id",
            "completion_fence",
            "consumed_records",
            "checkpoint_head",
        }
        data = _strict_mapping(
            payload,
            fields,
            "realization payload",
            optional={"consumed_records", "checkpoint_head"},
        )
        data.setdefault("consumed_records", ())
        data.setdefault("checkpoint_head", None)
        return cls(**data, metadata=record.get("metadata") or {})

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical realization payload."""

        payload = {
            "realization_id": self.realization_id,
            "producer_cdef_id": self.producer_cdef_id,
            "method": self.method,
            "declaration_fingerprint": self.declaration_fingerprint,
            "attempt_ids": list(self.attempt_ids),
            "outputs": [item.to_json() for item in self.outputs],
            "primary_output_slot": self.primary_output_slot,
            "primary_representation_id": self.primary_representation_id,
            "execution_record_id": self.execution_record_id,
            "completed_attempt_id": self.completed_attempt_id,
            "completion_fence": self.completion_fence,
            "consumed_records": [item.to_json() for item in self.consumed_records],
        }
        if self.checkpoint_head is not None:
            payload["checkpoint_head"] = self.checkpoint_head
        return payload

    def to_envelope(self) -> dict[str, Any]:
        """Return a validated generic record envelope."""

        return make_record(kind=self.kind, payload=self.to_payload(), metadata=self.metadata)


def _validate_method(value: Any) -> None:
    if not isinstance(value, str) or _METHOD_RE.fullmatch(value) is None:
        raise RecordValidationError("invalid managed method")


def _validate_cdef(value: Any, field_name: str) -> None:
    try:
        cdef = parse_cdef_id(value)
    except ReferenceParseError as exc:
        raise RecordValidationError(f"invalid {field_name}", context=exc.context) from exc
    if len(cdef.digest) != 64:
        raise RecordValidationError(f"{field_name} requires a full CDef digest")


def _validate_content_id(value: Any, prefix: str, field_name: str) -> None:
    try:
        parts = parse_content_id(value)
    except ContentIDError as exc:
        raise RecordValidationError(f"invalid {field_name}", context=exc.context) from exc
    if parts.prefix != prefix or parts.schema_version != 1:
        raise RecordValidationError(
            f"{field_name} prefix mismatch", context={"expected": f"{prefix}-v1"}
        )


def _strict_mapping(
    value: Any,
    fields: set[str],
    name: str,
    *,
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordValidationError(f"{name} must be a mapping")
    unknown = set(value) - fields
    if unknown:
        raise RecordValidationError(f"{name} contains unknown fields", context={"fields": sorted(unknown)})
    missing = fields - set(value) - (optional or set())
    if missing:
        raise RecordValidationError(f"{name} is missing required fields", context={"fields": sorted(missing)})
    return dict(value)


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise RecordValidationError(f"{field_name} must be a JSON array")
    return value


def _freeze_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordValidationError(f"{path} must be a mapping")
    try:
        frozen = deep_freeze_json(value)
    except CanonicalJSONError as exc:
        raise RecordValidationError(f"{path} is not canonical JSON", context=exc.context) from exc
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = [
    "RealizationOutput",
    "RealizationRecord",
    "ResolvedRecord",
    "validate_attempt_id",
    "validate_declaration_fingerprint",
    "validate_output_slot",
    "validate_realization_id",
]
