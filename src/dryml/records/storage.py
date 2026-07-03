"""Logical storage references used by DRYML record sidecars."""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from typing import Any, Literal

from dryml.formats.errors import ContentIDError, ReferenceParseError
from dryml.formats.ids import parse_content_id
from dryml.formats.refs import parse_cdef_id

from .errors import StorageRefError


StorageKind = Literal["object-dir", "product-dir", "blob"]


@dataclass(frozen=True, slots=True)
class StorageRef:
    """Store-relative logical pointer to object state, products, or blobs."""

    kind: StorageKind
    path: str = "."
    role: str | None = None
    subject_cdef_id: str | None = None
    record_id: str | None = None
    blob_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"object-dir", "product-dir", "blob"}:
            raise StorageRefError("invalid storage ref kind", context={"kind": self.kind})
        object.__setattr__(self, "path", _normalize_path(self.path))
        if self.role is not None and (not isinstance(self.role, str) or not self.role):
            raise StorageRefError("storage ref role must be a non-empty string", context={"role": self.role})
        self._validate_fields()

    @classmethod
    def object_dir(cls, subject_cdef_id: str, *, path: str = ".", role: str = "default-state") -> "StorageRef":
        """Create a reference to an existing object-state directory."""

        return cls("object-dir", path=path, role=role, subject_cdef_id=subject_cdef_id)

    @classmethod
    def product_dir(cls, record_id: str, *, path: str = ".", role: str | None = None) -> "StorageRef":
        """Create a reference under ``products/<record-id>/``."""

        return cls("product-dir", path=path, role=role, record_id=record_id)

    @classmethod
    def blob(cls, blob_id: str, *, role: str | None = None) -> "StorageRef":
        """Create a reference to a content-addressed blob placeholder."""

        return cls("blob", role=role, blob_id=blob_id)

    def to_json(self) -> dict[str, str]:
        """Return a canonical JSON-ready mapping for this storage reference."""

        data: dict[str, str] = {"kind": self.kind, "path": self.path}
        if self.role is not None:
            data["role"] = self.role
        if self.subject_cdef_id is not None:
            data["subject_cdef_id"] = self.subject_cdef_id
        if self.record_id is not None:
            data["record_id"] = self.record_id
        if self.blob_id is not None:
            data["blob_id"] = self.blob_id
        return data

    @classmethod
    def from_json(cls, data: Any) -> "StorageRef":
        """Build and validate a storage reference from JSON data."""

        if not isinstance(data, dict):
            raise StorageRefError("storage ref JSON must be an object", context={"type": type(data).__name__})
        return cls(
            kind=data.get("kind"),
            path=data.get("path", "."),
            role=data.get("role"),
            subject_cdef_id=data.get("subject_cdef_id"),
            record_id=data.get("record_id"),
            blob_id=data.get("blob_id"),
        )

    def _validate_fields(self) -> None:
        fields = {
            "subject_cdef_id": self.subject_cdef_id,
            "record_id": self.record_id,
            "blob_id": self.blob_id,
        }
        present = {name for name, value in fields.items() if value is not None}
        if self.kind == "object-dir":
            if present != {"subject_cdef_id"}:
                raise StorageRefError("object-dir storage refs require only subject_cdef_id", context={"fields": sorted(present)})
            _validate_cdef_id(self.subject_cdef_id)
        elif self.kind == "product-dir":
            if present != {"record_id"}:
                raise StorageRefError("product-dir storage refs require only record_id", context={"fields": sorted(present)})
            _validate_content_id(self.record_id, "record", "record ID")
        else:
            if present != {"blob_id"}:
                raise StorageRefError("blob storage refs require only blob_id", context={"fields": sorted(present)})
            _validate_content_id(self.blob_id, "blob", "blob ID")


def _normalize_path(path: str) -> str:
    if not isinstance(path, str):
        raise StorageRefError("storage ref path must be a string", context={"type": type(path).__name__})
    if path == "":
        raise StorageRefError("storage ref path cannot be empty")
    if "\\" in path:
        raise StorageRefError("storage ref path must use POSIX separators", context={"path": path})
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise StorageRefError("storage ref path must be relative", context={"path": path})
    parts = path.split("/")
    if any(part == "" for part in parts):
        raise StorageRefError("storage ref path has an empty component", context={"path": path})
    if any(part == ".." for part in parts):
        raise StorageRefError("storage ref path cannot contain traversal", context={"path": path})
    normalized = posixpath.normpath(path)
    if normalized in {"", "."}:
        return "."
    if normalized.startswith("../") or normalized == "..":
        raise StorageRefError("storage ref path cannot contain traversal", context={"path": path})
    return normalized


def _validate_cdef_id(value: str | None) -> None:
    try:
        parse_cdef_id(value)  # type: ignore[arg-type]
    except ReferenceParseError as exc:
        raise StorageRefError("invalid subject CDef ID", context=exc.context) from exc


def _validate_content_id(value: str | None, prefix: str, label: str) -> None:
    try:
        parts = parse_content_id(value)  # type: ignore[arg-type]
    except ContentIDError as exc:
        raise StorageRefError(f"invalid {label}", context=exc.context) from exc
    if parts.prefix != prefix or parts.schema_version != 1:
        raise StorageRefError(
            f"{label} must use {prefix}-v1 prefix",
            context={"value": value, "prefix": parts.prefix, "schema_version": parts.schema_version},
        )


__all__ = ["StorageKind", "StorageRef"]
