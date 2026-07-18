"""Versioned JSON models for Store-local managed operation control."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from dryml.formats.canonical import canonical_json_bytes
from dryml.formats.refs import format_cdef_id, parse_cdef_id
from dryml.formats.errors import ContentIDError
from dryml.formats.ids import parse_content_id

from .declarations import ManagedMethodDeclaration
from .errors import ManagedStateError


CONTROL_SCHEMA_VERSION = 1
MAX_DIAGNOSTICS = 32
_METHOD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DECLARATION_RE = re.compile(r"^managed-declaration-v1-[0-9a-f]{64}$")
_REALIZATION_RE = re.compile(r"^realization-v1-[0-9a-f]{32}$")
_ATTEMPT_RE = re.compile(r"^attempt-v1-[0-9a-f]{32}$")
_ACTIVATION_RE = re.compile(r"^activation-v1-[0-9a-f]{32}$")
_STATUSES = frozenset({"running", "interrupted", "failed", "completed", "abandoned"})


@dataclass(frozen=True, slots=True)
class OperationKey:
    """Store-independent identity of one producer definition method."""

    producer_cdef_id: str
    method: str

    def __post_init__(self) -> None:
        try:
            parsed = parse_cdef_id(self.producer_cdef_id)
        except Exception as exc:
            raise ManagedStateError("producer_cdef_id must be a valid CDef ID") from exc
        if len(parsed.digest) != 64:
            raise ManagedStateError("producer_cdef_id requires a full 64-character digest")
        if not isinstance(self.method, str) or _METHOD_RE.fullmatch(self.method) is None:
            raise ManagedStateError("managed method must be a Python identifier")

    @classmethod
    def from_producer(cls, producer: Any, method: str) -> "OperationKey":
        """Build a key from an Object or exact ConcreteDefinition."""

        cdef = getattr(producer, "definition", producer)
        stable_hash = getattr(cdef, "stable_hash", None)
        if not callable(stable_hash):
            raise TypeError("managed operation keys require an Object or ConcreteDefinition")
        return cls(format_cdef_id(stable_hash()), method)

    def to_json(self) -> dict[str, str]:
        """Return the persistent identity fields."""

        return {"method": self.method, "producer_cdef_id": self.producer_cdef_id}


def declaration_fingerprint(
    method: str,
    declaration: ManagedMethodDeclaration,
    *,
    producer: Any | None = None,
) -> str:
    """Hash canonical managed output and capability declarations."""

    if not isinstance(method, str) or _METHOD_RE.fullmatch(method) is None:
        raise ManagedStateError("managed method must be a Python identifier")
    if not isinstance(declaration, ManagedMethodDeclaration):
        raise TypeError("declaration must be a ManagedMethodDeclaration")
    outputs = declaration.output_declarations(producer)
    payload = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "method": method,
        "outputs": [
            {
                "kind": item.kind,
                "primary": item.primary,
                "representations": list(item.representations),
                "slot": item.slot,
                "subject_path": list(item.subject_path) if item.subject_path is not None else None,
            }
            for item in outputs
        ],
        "capabilities": {
            "checkpoint_schema": declaration.checkpoint_schema,
            "early_completion": declaration.early_completion,
            "resumable": declaration.resumable,
        },
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"managed-declaration-v1-{digest}"


@dataclass(frozen=True, slots=True)
class RealizationState:
    """Mutable control summary for one distinct execution realization."""

    realization_id: str
    declaration_fingerprint: str
    status: str
    resumable: bool
    attempt_ids: tuple[str, ...]
    current_attempt_id: str | None
    sequence: int = 1
    checkpoint_head: str | None = None
    diagnostics: tuple[str, ...] = ()
    realization_record_id: str | None = None

    def __post_init__(self) -> None:
        _require_match(self.realization_id, _REALIZATION_RE, "realization_id")
        validate_declaration_fingerprint(self.declaration_fingerprint)
        if self.status not in _STATUSES:
            raise ManagedStateError(f"unsupported realization status {self.status!r}")
        if not isinstance(self.resumable, bool):
            raise ManagedStateError("realization resumable must be a bool")
        attempts = tuple(self.attempt_ids)
        if not attempts:
            raise ManagedStateError("realization attempt_ids must not be empty")
        for attempt_id in attempts:
            _require_match(attempt_id, _ATTEMPT_RE, "attempt_id")
        if len(set(attempts)) != len(attempts):
            raise ManagedStateError("realization attempt_ids must be unique")
        if self.current_attempt_id is not None:
            _require_match(self.current_attempt_id, _ATTEMPT_RE, "current_attempt_id")
            if self.current_attempt_id not in attempts:
                raise ManagedStateError("current_attempt_id must appear in attempt_ids")
        if self.status == "running" and self.current_attempt_id is None:
            raise ManagedStateError("running realization requires current_attempt_id")
        if self.status != "running" and self.current_attempt_id is not None:
            raise ManagedStateError("only a running realization may have a current_attempt_id")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ManagedStateError("realization sequence must be a positive integer")
        _validate_optional_token(self.checkpoint_head, "checkpoint_head")
        diagnostics = tuple(self.diagnostics)
        if len(diagnostics) > MAX_DIAGNOSTICS:
            raise ManagedStateError(f"diagnostics must contain at most {MAX_DIAGNOSTICS} entries")
        if any(not isinstance(item, str) or not item or len(item) > 512 for item in diagnostics):
            raise ManagedStateError("diagnostics must be non-empty strings of at most 512 characters")
        object.__setattr__(self, "attempt_ids", attempts)
        object.__setattr__(self, "diagnostics", diagnostics)
        _validate_optional_record_id(self.realization_record_id, "realization_record_id")

    def to_json(self) -> dict[str, Any]:
        """Return the strict versioned JSON representation."""

        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "realization_id": self.realization_id,
            "declaration_fingerprint": self.declaration_fingerprint,
            "status": self.status,
            "resumable": self.resumable,
            "attempt_ids": list(self.attempt_ids),
            "current_attempt_id": self.current_attempt_id,
            "sequence": self.sequence,
            "checkpoint_head": self.checkpoint_head,
            "diagnostics": list(self.diagnostics),
            "realization_record_id": self.realization_record_id,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "RealizationState":
        """Validate and decode one realization control document."""

        data = _strict_mapping(
            value,
            {
                "schema_version",
                "realization_id",
                "declaration_fingerprint",
                "status",
                "resumable",
                "attempt_ids",
                "current_attempt_id",
                "sequence",
                "checkpoint_head",
                "diagnostics",
                "realization_record_id",
            },
            "realization state",
        )
        _require_schema(data, "realization state")
        try:
            return cls(
                realization_id=data["realization_id"],
                declaration_fingerprint=data["declaration_fingerprint"],
                status=data["status"],
                resumable=data["resumable"],
                attempt_ids=_string_tuple(data["attempt_ids"], "attempt_ids"),
                current_attempt_id=data["current_attempt_id"],
                sequence=data["sequence"],
                checkpoint_head=data["checkpoint_head"],
                diagnostics=_string_tuple(data["diagnostics"], "diagnostics"),
                realization_record_id=data["realization_record_id"],
            )
        except KeyError as exc:
            raise ManagedStateError(f"realization state is missing {exc.args[0]!r}") from exc


@dataclass(frozen=True, slots=True)
class NamespaceState:
    """Direct operation namespace state shared by all declaration generations."""

    key: OperationKey
    current_declaration_fingerprint: str
    generations: tuple[str, ...]
    fence_epoch: int

    def __post_init__(self) -> None:
        validate_declaration_fingerprint(self.current_declaration_fingerprint)
        generations = tuple(self.generations)
        if not generations or self.current_declaration_fingerprint not in generations:
            raise ManagedStateError("current declaration fingerprint must appear in generations")
        for fingerprint in generations:
            validate_declaration_fingerprint(fingerprint)
        if len(set(generations)) != len(generations):
            raise ManagedStateError("operation generations must be unique")
        if type(self.fence_epoch) is not int or self.fence_epoch < 0:
            raise ManagedStateError("fence_epoch must be a non-negative integer")
        object.__setattr__(self, "generations", generations)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            **self.key.to_json(),
            "current_declaration_fingerprint": self.current_declaration_fingerprint,
            "generations": list(self.generations),
            "fence_epoch": self.fence_epoch,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "NamespaceState":
        fields = {
            "schema_version",
            "producer_cdef_id",
            "method",
            "current_declaration_fingerprint",
            "generations",
            "fence_epoch",
        }
        data = _strict_mapping(value, fields, "operation namespace")
        _require_schema(data, "operation namespace")
        try:
            return cls(
                key=OperationKey(data["producer_cdef_id"], data["method"]),
                current_declaration_fingerprint=data["current_declaration_fingerprint"],
                generations=_string_tuple(data["generations"], "generations"),
                fence_epoch=data["fence_epoch"],
            )
        except KeyError as exc:
            raise ManagedStateError(f"operation namespace is missing {exc.args[0]!r}") from exc


@dataclass(frozen=True, slots=True)
class GenerationControl:
    """Bounded direct control state for one declaration generation."""

    declaration_fingerprint: str
    fence_epoch: int
    pending_realization_id: str | None = None
    current_attempt_id: str | None = None
    checkpoint_head: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_declaration_fingerprint(self.declaration_fingerprint)
        if type(self.fence_epoch) is not int or self.fence_epoch < 0:
            raise ManagedStateError("fence_epoch must be a non-negative integer")
        if self.pending_realization_id is not None:
            _require_match(self.pending_realization_id, _REALIZATION_RE, "pending_realization_id")
        if self.current_attempt_id is not None:
            _require_match(self.current_attempt_id, _ATTEMPT_RE, "current_attempt_id")
            if self.pending_realization_id is None:
                raise ManagedStateError("current_attempt_id requires a pending realization")
        _validate_optional_token(self.checkpoint_head, "checkpoint_head")
        diagnostics = tuple(self.diagnostics)
        if len(diagnostics) > MAX_DIAGNOSTICS:
            raise ManagedStateError(f"diagnostics must contain at most {MAX_DIAGNOSTICS} entries")
        if any(not isinstance(item, str) or not item or len(item) > 512 for item in diagnostics):
            raise ManagedStateError("diagnostics must be non-empty strings of at most 512 characters")
        object.__setattr__(self, "diagnostics", diagnostics)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "declaration_fingerprint": self.declaration_fingerprint,
            "fence_epoch": self.fence_epoch,
            "pending_realization_id": self.pending_realization_id,
            "current_attempt_id": self.current_attempt_id,
            "checkpoint_head": self.checkpoint_head,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "GenerationControl":
        fields = {
            "schema_version",
            "declaration_fingerprint",
            "fence_epoch",
            "pending_realization_id",
            "current_attempt_id",
            "checkpoint_head",
            "diagnostics",
        }
        data = _strict_mapping(value, fields, "generation control")
        _require_schema(data, "generation control")
        try:
            return cls(
                declaration_fingerprint=data["declaration_fingerprint"],
                fence_epoch=data["fence_epoch"],
                pending_realization_id=data["pending_realization_id"],
                current_attempt_id=data["current_attempt_id"],
                checkpoint_head=data["checkpoint_head"],
                diagnostics=_string_tuple(data["diagnostics"], "diagnostics"),
            )
        except KeyError as exc:
            raise ManagedStateError(f"generation control is missing {exc.args[0]!r}") from exc


@dataclass(frozen=True, slots=True)
class ActivationEvent:
    """Immutable authoritative selection event for one completed realization."""

    activation_id: str
    declaration_fingerprint: str
    sequence: int
    realization_id: str
    previous_realization_id: str | None
    fence_epoch: int
    realization_record_id: str | None = None

    def __post_init__(self) -> None:
        _require_match(self.activation_id, _ACTIVATION_RE, "activation_id")
        validate_declaration_fingerprint(self.declaration_fingerprint)
        _require_match(self.realization_id, _REALIZATION_RE, "realization_id")
        if self.previous_realization_id is not None:
            _require_match(self.previous_realization_id, _REALIZATION_RE, "previous_realization_id")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ManagedStateError("activation sequence must be a positive integer")
        if type(self.fence_epoch) is not int or self.fence_epoch < 1:
            raise ManagedStateError("activation fence_epoch must be a positive integer")
        _validate_optional_record_id(self.realization_record_id, "realization_record_id")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "activation_id": self.activation_id,
            "declaration_fingerprint": self.declaration_fingerprint,
            "sequence": self.sequence,
            "realization_id": self.realization_id,
            "previous_realization_id": self.previous_realization_id,
            "fence_epoch": self.fence_epoch,
            "realization_record_id": self.realization_record_id,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "ActivationEvent":
        fields = {
            "schema_version",
            "activation_id",
            "declaration_fingerprint",
            "sequence",
            "realization_id",
            "previous_realization_id",
            "fence_epoch",
            "realization_record_id",
        }
        data = _strict_mapping(value, fields, "activation event")
        _require_schema(data, "activation event")
        try:
            return cls(**{key: data[key] for key in fields - {"schema_version"}})
        except KeyError as exc:
            raise ManagedStateError(f"activation event is missing {exc.args[0]!r}") from exc


@dataclass(frozen=True, slots=True)
class OperationDecision:
    """State-only outcome for a normal invocation or explicit rerun request."""

    action: str
    realization: RealizationState

    def __post_init__(self) -> None:
        if self.action not in {"start", "resume", "rerun", "reuse"}:
            raise ManagedStateError(f"unsupported operation decision {self.action!r}")


def validate_declaration_fingerprint(value: str) -> str:
    """Validate and return a declaration fingerprint."""

    _require_match(value, _DECLARATION_RE, "declaration_fingerprint")
    return value


def _strict_mapping(value: Mapping[str, Any], fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManagedStateError(f"{name} must be a JSON object")
    unknown = set(value) - fields
    if unknown:
        raise ManagedStateError(f"{name} contains unknown fields: {sorted(unknown)!r}")
    return dict(value)


def _require_schema(data: Mapping[str, Any], name: str) -> None:
    if data.get("schema_version") != CONTROL_SCHEMA_VERSION:
        raise ManagedStateError(f"{name} schema_version must be {CONTROL_SCHEMA_VERSION}")


def _require_match(value: Any, pattern: re.Pattern[str], field_name: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ManagedStateError(f"invalid {field_name}")


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or any(not isinstance(item, str) for item in value):
        raise ManagedStateError(f"{field_name} must be a list of strings")
    return tuple(value)


def _validate_optional_token(value: Any, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value or len(value) > 256 or "/" in value or "\\" in value:
        raise ManagedStateError(f"invalid {field_name}")


def _validate_optional_record_id(value: Any, field_name: str) -> None:
    if value is None:
        return
    try:
        parts = parse_content_id(value)
    except ContentIDError as exc:
        raise ManagedStateError(f"invalid {field_name}") from exc
    if parts.prefix != "record" or parts.schema_version != 1:
        raise ManagedStateError(f"invalid {field_name}")


__all__ = [
    "ActivationEvent",
    "CONTROL_SCHEMA_VERSION",
    "GenerationControl",
    "MAX_DIAGNOSTICS",
    "NamespaceState",
    "OperationDecision",
    "OperationKey",
    "RealizationState",
    "declaration_fingerprint",
    "validate_declaration_fingerprint",
]
