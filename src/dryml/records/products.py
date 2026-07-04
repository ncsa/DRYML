"""Product manifest and staging helpers for record-owned products."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dryml.formats import json_ready

from .errors import RecordIOError, RecordValidationError, StorageRefError
from .records import attach_record_id, validate_record
from .refs import LocatedRecordRef
from .storage import StorageRef


@dataclass(frozen=True, slots=True)
class ProductManifestEntry:
    """One file entry in a product manifest."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_product_path(self.path)
        if not isinstance(self.size, int) or self.size < 0:
            raise RecordValidationError("product manifest size must be a non-negative integer")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise RecordValidationError("product manifest sha256 must be lowercase hex")

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-ready manifest entry."""

        return {"path": self.path, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ProductManifestEntry":
        """Build a manifest entry from JSON data."""

        if not isinstance(data, Mapping):
            raise RecordValidationError("product manifest entry must be a mapping", context={"type": type(data).__name__})
        return cls(path=data.get("path"), size=data.get("size"), sha256=data.get("sha256"))


@dataclass(frozen=True, slots=True)
class ProductManifest:
    """Deterministically ordered manifest for record product files."""

    entries: tuple[ProductManifestEntry, ...] = ()

    def __post_init__(self) -> None:
        entries = tuple(entry if isinstance(entry, ProductManifestEntry) else ProductManifestEntry.from_json(entry) for entry in self.entries)
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda entry: entry.path)))

    def to_json(self) -> dict[str, Any]:
        """Return JSON-ready manifest data."""

        return {"entries": [entry.to_json() for entry in self.entries]}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ProductManifest":
        """Build a manifest from JSON data."""

        if not isinstance(data, Mapping):
            raise RecordValidationError("product manifest must be a mapping", context={"type": type(data).__name__})
        return cls(tuple(ProductManifestEntry.from_json(entry) for entry in _json_sequence(data, "entries")))


@dataclass(frozen=True, slots=True)
class ProductWriteResult:
    """Result returned after committing a product record."""

    located: LocatedRecordRef
    manifest: ProductManifest
    product_root: Path


@dataclass(frozen=True, slots=True)
class ProductAvailabilityIssue:
    """One missing product path referenced by a record sidecar."""

    code: str
    message: str
    record_id: str
    path: str
    storage_index: int | None = None

    def to_json(self) -> dict[str, Any]:
        """Return JSON-ready issue data."""

        return {
            "code": self.code,
            "message": self.message,
            "record_id": self.record_id,
            "path": self.path,
            "storage_index": self.storage_index,
        }


class ProductWriteSession:
    """Stage product bytes before committing a record-owned product root.

    The record ID is computed after files are staged and a manifest is available.
    Staged files are then moved to ``products/<record-id>/`` and the record
    sidecar is written only after the product root is in place.
    """

    def __init__(self, record_io: Any, *, overwrite: bool = False):
        self.record_io = record_io
        self.overwrite = overwrite
        self._staging_dir: Path | None = None
        self._closed = False

    def __enter__(self) -> "ProductWriteSession":
        self._ensure_staging()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    @property
    def staging_dir(self) -> Path:
        """Return the temporary staging directory, creating it if necessary."""

        return self._ensure_staging()

    def write_bytes(self, path: str, data: bytes) -> Path:
        """Write bytes to a staged relative product path."""

        _validate_product_path(path)
        if not isinstance(data, bytes):
            raise RecordIOError("product bytes must be bytes")
        target = self.staging_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def write_text(self, path: str, text: str, *, encoding: str = "utf-8") -> Path:
        """Write text to a staged relative product path."""

        return self.write_bytes(path, text.encode(encoding))

    def write_json(self, path: str, data: Any) -> Path:
        """Write deterministic JSON to a staged relative product path."""

        return self.write_text(path, json.dumps(json_ready(data), sort_keys=True, separators=(",", ":")))

    def manifest(self) -> ProductManifest:
        """Compute a deterministic manifest from staged files."""

        root = self.staging_dir
        entries: list[ProductManifestEntry] = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rel = path.relative_to(root).as_posix()
            entries.append(ProductManifestEntry(rel, path.stat().st_size, _sha256(path)))
        return ProductManifest(tuple(entries))

    def commit_record(self, record: Mapping[str, Any], *, overwrite: bool | None = None) -> ProductWriteResult:
        """Commit staged products and write the corresponding record sidecar."""

        if self._closed:
            raise RecordIOError("product write session is closed")
        use_overwrite = self.overwrite if overwrite is None else overwrite
        attached = attach_record_id(record)
        validate_record(attached)
        record_id = attached["id"]
        target_root = self.record_io.product_root(record_id)
        if target_root.exists():
            if not use_overwrite:
                raise RecordIOError("product root already exists", context={"record_id": record_id})
            shutil.rmtree(target_root)
        target_root.parent.mkdir(parents=True, exist_ok=True)
        staged = self.staging_dir
        moved = False
        try:
            os.replace(staged, target_root)
            moved = True
            located = self.record_io.write_record(attached, overwrite=use_overwrite)
            self._staging_dir = None
            self._closed = True
            return ProductWriteResult(located=located, manifest=_manifest_for_root(target_root), product_root=target_root)
        except Exception:
            if moved and target_root.exists():
                shutil.rmtree(target_root, ignore_errors=True)
            raise
        finally:
            if self._staging_dir is not None and self._staging_dir.exists():
                shutil.rmtree(self._staging_dir, ignore_errors=True)

    def cleanup(self) -> None:
        """Remove any staged bytes that have not been committed."""

        if self._staging_dir is not None and self._staging_dir.exists():
            shutil.rmtree(self._staging_dir, ignore_errors=True)
        self._staging_dir = None
        self._closed = True

    def _ensure_staging(self) -> Path:
        if self._closed:
            raise RecordIOError("product write session is closed")
        if self._staging_dir is None:
            self.record_io.products_dir.mkdir(parents=True, exist_ok=True)
            self._staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.record_io.products_dir))
        return self._staging_dir


def stage_product_file(session: ProductWriteSession, path: str, data: bytes | str) -> Path:
    """Stage one product file in an existing session."""

    if isinstance(data, bytes):
        return session.write_bytes(path, data)
    if isinstance(data, str):
        return session.write_text(path, data)
    raise RecordIOError("product data must be bytes or text")


def commit_product_record(session: ProductWriteSession, record: Mapping[str, Any], *, overwrite: bool | None = None) -> ProductWriteResult:
    """Commit a staged product record through a session."""

    return session.commit_record(record, overwrite=overwrite)


def validate_product_availability(record_io: Any, record: Mapping[str, Any]) -> tuple[ProductAvailabilityIssue, ...]:
    """Return missing product paths referenced by a record sidecar.

    JSON record/spec sidecars remain authoritative, but product-backed records
    should not be copied or treated as complete when their ``product-dir``
    storage roots or manifest entries are absent.
    """

    attached = attach_record_id(record)
    validate_record(attached)
    record_id = attached["id"]
    payload = attached.get("payload") or {}
    if not isinstance(payload, Mapping):
        raise RecordValidationError("record payload must be a mapping", context={"record_id": record_id})
    issues: list[ProductAvailabilityIssue] = []
    for index, storage_data in enumerate(_json_sequence(payload, "storage")):
        storage_ref = storage_data if isinstance(storage_data, StorageRef) else StorageRef.from_json(storage_data)
        if storage_ref.kind != "product-dir":
            continue
        path = record_io.resolve_storage_ref(storage_ref, record_id=record_id)
        if not path.exists():
            issues.append(ProductAvailabilityIssue("missing_product_path", "record product storage path is missing", record_id, path.as_posix(), index))
    manifest_data = payload.get("manifest")
    if isinstance(manifest_data, Mapping):
        root = record_io.product_root(record_id)
        manifest = ProductManifest.from_json(manifest_data)
        for entry in manifest.entries:
            path = root / entry.path
            if not path.exists():
                issues.append(ProductAvailabilityIssue("missing_manifest_entry", "record product manifest entry is missing", record_id, path.as_posix(), None))
    return tuple(issues)


def _manifest_for_root(root: Path) -> ProductManifest:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append(ProductManifestEntry(path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path)))
    return ProductManifest(tuple(entries))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_product_path(path: str) -> None:
    try:
        from .storage import _normalize_path

        normalized = _normalize_path(path)
    except StorageRefError as exc:
        raise RecordValidationError("invalid product path", context=exc.context) from exc
    if normalized == ".":
        raise RecordValidationError("product file path cannot be the root")


def _json_sequence(data: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in data:
        return ()
    value = data[field_name]
    if value is None:
        return ()
    if isinstance(value, str):
        raise RecordValidationError(f"product manifest {field_name} must be a JSON array, not a string")
    if not isinstance(value, (list, tuple)):
        raise RecordValidationError(f"product manifest {field_name} must be a JSON array", context={"type": type(value).__name__})
    return value


__all__ = [
    "ProductManifest",
    "ProductManifestEntry",
    "ProductAvailabilityIssue",
    "ProductWriteResult",
    "ProductWriteSession",
    "commit_product_record",
    "stage_product_file",
    "validate_product_availability",
]
