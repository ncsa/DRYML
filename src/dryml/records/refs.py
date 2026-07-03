"""Reference dataclasses for store-owned record and spec sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dryml.formats.errors import ContentIDError
from dryml.formats.ids import parse_content_id

from .errors import RecordValidationError, SpecValidationError
from .kinds import SPEC_FAMILIES, SPEC_FAMILY_BY_PREFIX


@dataclass(frozen=True, slots=True)
class RecordRef:
    """Reference to a content-addressed record in an unspecified store."""

    record_id: str

    def __post_init__(self) -> None:
        _validate_record_id(self.record_id)

    def __str__(self) -> str:
        """Return the raw record ID."""

        return self.record_id

    def to_json(self) -> str:
        """Return the compact JSON form for this record reference."""

        return self.record_id

    @classmethod
    def from_json(cls, data: Any) -> "RecordRef":
        """Build and validate a record reference from JSON data."""

        if not isinstance(data, str):
            raise RecordValidationError("record ref JSON must be a string", context={"type": type(data).__name__})
        return cls(data)


@dataclass(frozen=True, slots=True)
class LocatedRecordRef:
    """Reference to a record copy in a specific store locator."""

    store_ref: str
    record_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.store_ref, str) or not self.store_ref:
            raise RecordValidationError("located record ref store_ref must be a non-empty string")
        _validate_record_id(self.record_id)

    def to_json(self) -> dict[str, str]:
        """Return the object JSON form for this located record reference."""

        return {"store_ref": self.store_ref, "record_id": self.record_id}

    @classmethod
    def from_json(cls, data: Any) -> "LocatedRecordRef":
        """Build and validate a located record reference from JSON data."""

        if not isinstance(data, dict):
            raise RecordValidationError("located record ref JSON must be an object", context={"type": type(data).__name__})
        return cls(store_ref=data.get("store_ref"), record_id=data.get("record_id"))


@dataclass(frozen=True, slots=True)
class SpecRef:
    """Reference to a content-addressed spec in an unspecified store."""

    spec_id: str
    kind: str | None = None

    def __post_init__(self) -> None:
        _validate_spec_id(self.spec_id, self.kind)

    def __str__(self) -> str:
        """Return the raw spec ID."""

        return self.spec_id

    def to_json(self) -> str | dict[str, str]:
        """Return compact JSON for unqualified refs and object JSON when kind is known."""

        if self.kind is None:
            return self.spec_id
        return {"spec_id": self.spec_id, "kind": self.kind}

    @classmethod
    def from_json(cls, data: Any) -> "SpecRef":
        """Build and validate a spec reference from JSON data."""

        if isinstance(data, str):
            return cls(data)
        if not isinstance(data, dict):
            raise SpecValidationError("spec ref JSON must be a string or object", context={"type": type(data).__name__})
        return cls(spec_id=data.get("spec_id"), kind=data.get("kind"))


@dataclass(frozen=True, slots=True)
class LocatedSpecRef:
    """Reference to a spec copy in a specific store locator."""

    store_ref: str
    spec_id: str
    kind: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.store_ref, str) or not self.store_ref:
            raise SpecValidationError("located spec ref store_ref must be a non-empty string")
        _validate_spec_id(self.spec_id, self.kind)

    def to_json(self) -> dict[str, str]:
        """Return the object JSON form for this located spec reference."""

        data = {"store_ref": self.store_ref, "spec_id": self.spec_id}
        if self.kind is not None:
            data["kind"] = self.kind
        return data

    @classmethod
    def from_json(cls, data: Any) -> "LocatedSpecRef":
        """Build and validate a located spec reference from JSON data."""

        if not isinstance(data, dict):
            raise SpecValidationError("located spec ref JSON must be an object", context={"type": type(data).__name__})
        return cls(store_ref=data.get("store_ref"), spec_id=data.get("spec_id"), kind=data.get("kind"))


def _validate_record_id(record_id: str) -> None:
    try:
        parts = parse_content_id(record_id)
    except ContentIDError as exc:
        raise RecordValidationError("invalid record ID", context=exc.context) from exc
    if parts.prefix != "record" or parts.schema_version != 1:
        raise RecordValidationError(
            "record ID must use record-v1 prefix",
            context={"record_id": record_id, "prefix": parts.prefix, "schema_version": parts.schema_version},
        )


def _validate_spec_id(spec_id: str, kind: str | None = None) -> None:
    try:
        parts = parse_content_id(spec_id)
    except ContentIDError as exc:
        raise SpecValidationError("invalid spec ID", context=exc.context) from exc
    if kind is None:
        return
    if kind not in SPEC_FAMILIES:
        raise SpecValidationError("unknown spec ref kind", context={"kind": kind})
    expected_prefix = SPEC_FAMILIES[kind].prefix
    if parts.prefix != expected_prefix:
        raise SpecValidationError(
            "spec ref kind does not match ID prefix",
            context={"kind": kind, "expected_prefix": expected_prefix, "observed_prefix": parts.prefix},
        )
    expected_kind = SPEC_FAMILY_BY_PREFIX.get(parts.prefix)
    if expected_kind is not None and expected_kind != kind:
        raise SpecValidationError(
            "spec ref kind does not match known family",
            context={"kind": kind, "expected_kind": expected_kind},
        )


__all__ = ["LocatedRecordRef", "LocatedSpecRef", "RecordRef", "SpecRef"]
