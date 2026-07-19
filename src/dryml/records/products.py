"""Product manifest and staging helpers for record-owned products."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dryml.formats import canonical_json_bytes, canonical_json_load_bytes, json_ready
from dryml.formats.ids import content_id

from .errors import RecordIOError, RecordValidationError, StorageRefError
from .records import attach_record_id, validate_record
from .refs import LocatedRecordRef
from .storage import StorageRef


PRODUCT_MANIFEST_SCHEMA = "dryml.product.manifest.v1"
PRODUCT_ROOT_MANIFEST_SCHEMA = "dryml.product.root.v1"
PRODUCT_MANIFEST_FILE = ".dryml-product-manifest-v1.json"
FINALIZATION_INTENT_SCHEMA = "dryml.managed.finalization_intent.v1"


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
        paths = [entry.path for entry in entries]
        if len(set(paths)) != len(paths):
            raise RecordValidationError("product manifest contains duplicate paths")
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda entry: entry.path)))

    def to_json(self) -> dict[str, Any]:
        """Return JSON-ready manifest data."""

        return {"entries": [entry.to_json() for entry in self.entries]}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ProductManifest":
        """Build a manifest from JSON data."""

        if not isinstance(data, Mapping):
            raise RecordValidationError("product manifest must be a mapping", context={"type": type(data).__name__})
        if set(data) != {"entries"}:
            raise RecordValidationError(
                "product manifest contains unknown fields",
                context={"fields": sorted(set(data) - {"entries"})},
            )
        return cls(tuple(ProductManifestEntry.from_json(entry) for entry in _json_sequence(data, "entries")))


@dataclass(frozen=True, slots=True)
class ProductRootManifest:
    """Compact record payload that authenticates a detailed product manifest."""

    manifest_path: str
    manifest_size: int
    manifest_sha256: str
    file_count: int
    total_size: int

    def __post_init__(self) -> None:
        _validate_product_path(self.manifest_path)
        if self.manifest_path != PRODUCT_MANIFEST_FILE:
            raise RecordValidationError("product root manifest uses an unsupported manifest path")
        for name in ("manifest_size", "file_count", "total_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise RecordValidationError(f"product root {name} must be a non-negative integer")
        valid_digest = isinstance(self.manifest_sha256, str)
        if valid_digest:
            valid_digest = len(self.manifest_sha256) == 64
        if valid_digest:
            valid_digest = all(char in "0123456789abcdef" for char in self.manifest_sha256)
        if not valid_digest:
            raise RecordValidationError("product root manifest_sha256 must be lowercase hex")

    def to_json(self) -> dict[str, Any]:
        """Return the strict compact root manifest."""

        return {
            "schema": PRODUCT_ROOT_MANIFEST_SCHEMA,
            "schema_version": 1,
            "manifest_path": self.manifest_path,
            "manifest_size": self.manifest_size,
            "manifest_sha256": self.manifest_sha256,
            "file_count": self.file_count,
            "total_size": self.total_size,
        }

    @classmethod
    def from_json(cls, data: Any) -> "ProductRootManifest":
        """Decode a strict compact root manifest."""

        fields = {
            "schema",
            "schema_version",
            "manifest_path",
            "manifest_size",
            "manifest_sha256",
            "file_count",
            "total_size",
        }
        if not isinstance(data, Mapping) or set(data) != fields:
            raise RecordValidationError("product root manifest fields are malformed")
        if data.get("schema") != PRODUCT_ROOT_MANIFEST_SCHEMA or data.get("schema_version") != 1:
            raise RecordValidationError("product root manifest schema is unsupported")
        return cls(
            manifest_path=data["manifest_path"],
            manifest_size=data["manifest_size"],
            manifest_sha256=data["manifest_sha256"],
            file_count=data["file_count"],
            total_size=data["total_size"],
        )


@dataclass(frozen=True, slots=True)
class ProductWriteResult:
    """Result returned after committing a product record."""

    located: LocatedRecordRef
    manifest: ProductManifest
    product_root: Path


@dataclass(frozen=True, slots=True)
class CheckpointCommit:
    """Reference to one immutable operation-owned checkpoint payload."""

    checkpoint_id: str
    checkpoint_schema: str
    product_root: Path
    manifest: ProductManifest


@dataclass(frozen=True, slots=True)
class RealizationPublicationResult:
    """Immutable records published before optional active promotion."""

    output_records: Mapping[str, LocatedRecordRef]
    execution_record: LocatedRecordRef
    realization_record: LocatedRecordRef
    activated: bool


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
        if manifest_data.get("schema") == PRODUCT_ROOT_MANIFEST_SCHEMA:
            root_manifest = ProductRootManifest.from_json(manifest_data)
            manifest_path = root / root_manifest.manifest_path
            if not manifest_path.is_file():
                issues.append(ProductAvailabilityIssue("missing_detailed_manifest", "record detailed product manifest is missing", record_id, manifest_path.as_posix(), None))
            else:
                manifest_bytes = manifest_path.read_bytes()
                if len(manifest_bytes) != root_manifest.manifest_size:
                    issues.append(ProductAvailabilityIssue("manifest_size_mismatch", "detailed product manifest size does not match", record_id, manifest_path.as_posix(), None))
                if hashlib.sha256(manifest_bytes).hexdigest() != root_manifest.manifest_sha256:
                    issues.append(ProductAvailabilityIssue("manifest_digest_mismatch", "detailed product manifest digest does not match", record_id, manifest_path.as_posix(), None))
                try:
                    manifest = _load_detailed_manifest(manifest_bytes)
                except RecordValidationError:
                    issues.append(ProductAvailabilityIssue("invalid_detailed_manifest", "detailed product manifest is malformed", record_id, manifest_path.as_posix(), None))
                else:
                    if len(manifest.entries) != root_manifest.file_count or sum(entry.size for entry in manifest.entries) != root_manifest.total_size:
                        issues.append(ProductAvailabilityIssue("manifest_summary_mismatch", "compact product summary does not match detailed manifest", record_id, manifest_path.as_posix(), None))
                    issues.extend(_verify_manifest_files(root, manifest, record_id, allowed_extra={root_manifest.manifest_path}))
        else:
            manifest = ProductManifest.from_json(manifest_data)
            issues.extend(_verify_manifest_files(root, manifest, record_id))
    return tuple(issues)


def require_product_integrity(record_io: Any, record: Mapping[str, Any]) -> None:
    """Raise unless every product path, size, digest, and exact file set verifies."""

    issues = validate_product_availability(record_io, record)
    if issues:
        raise RecordValidationError(
            "record product integrity verification failed",
            context={"record_id": attach_record_id(record)["id"], "issues": [item.to_json() for item in issues]},
        )


def require_checkpoint_integrity(root: str | Path, checkpoint_id: str) -> None:
    """Raise unless an immutable managed checkpoint verifies exactly."""

    _validate_checkpoint_root(Path(root), checkpoint_id)


class DurableProductWriter:
    """Fence-checked streaming writer for retained managed attempt workspaces.

    The writer treats operation payload bytes as opaque. It owns durable file
    publication, checkpoint references, immutable manifests, finalization
    intent, record ordering, and optional pointer-last activation.
    """

    def __init__(self, record_io: Any, lease: Any, realization_id: str):
        from .realizations import validate_realization_id

        validate_realization_id(realization_id)
        lease.assert_current()
        state = lease.operation._read_realization(realization_id)
        if state.current_attempt_id is None:
            raise RecordIOError("durable writer requires a running realization attempt")
        self.record_io = record_io
        self.lease = lease
        self.realization_id = realization_id
        self.attempt_id = state.current_attempt_id
        self.workspace = lease.operation.attempts_dir / f"{lease.epoch:020d}-{self.attempt_id}"
        self._mkdir(self.workspace)

    def write_stream(self, slot: str, path: str, chunks: Any) -> ProductManifestEntry:
        """Stream bounded chunks into one declared output file and hash inline."""

        from .realizations import validate_output_slot

        validate_output_slot(slot, "slot")
        return self._write_stream(self.workspace / "outputs" / slot, path, chunks)

    def write_checkpoint_stream(self, path: str, chunks: Any) -> ProductManifestEntry:
        """Stream opaque operation checkpoint bytes into the current staging root."""

        return self._write_stream(self.workspace / "checkpoint-staging", path, chunks)

    def commit_checkpoint(
        self,
        checkpoint_schema: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointCommit:
        """Commit the staged opaque checkpoint and atomically advance its head."""

        if not isinstance(checkpoint_schema, str) or not checkpoint_schema or len(checkpoint_schema) > 128:
            raise RecordValidationError("checkpoint_schema must be a bounded non-empty string")
        metadata_data = json_ready(metadata or {})
        if not isinstance(metadata_data, Mapping):
            raise RecordValidationError("checkpoint metadata must be a mapping")
        staging = self.workspace / "checkpoint-staging"
        if not staging.is_dir():
            raise RecordIOError("checkpoint has no staged payload")
        manifest = _manifest_for_root(staging)
        identity = {
            "checkpoint_schema": checkpoint_schema,
            "manifest": manifest.to_json(),
            "metadata": metadata_data,
        }
        checkpoint_id = content_id("checkpoint", 1, identity)
        descriptor = {
            "schema": "dryml.managed.checkpoint.v1",
            "schema_version": 1,
            "checkpoint_id": checkpoint_id,
            **identity,
        }
        self._write_immutable_bytes(staging / "checkpoint.json", canonical_json_bytes(descriptor))
        target = self.workspace / "checkpoints" / checkpoint_id
        self.lease.assert_current()
        self._mkdir(target.parent)
        if target.exists():
            if not _trees_match(staging, target):
                raise RecordIOError("immutable checkpoint already exists with different bytes")
            retained = self.workspace / f"checkpoint-staging-committed-{uuid.uuid4().hex}"
            os.replace(staging, retained)
            _fsync_directory(retained.parent)
        else:
            os.replace(staging, target)
            _fsync_directory(target.parent)
        self.lease.checkpoint(self.realization_id, checkpoint_id)
        return CheckpointCommit(checkpoint_id, checkpoint_schema, target, manifest)

    def checkpoint_path(self, checkpoint_id: str | None) -> Path:
        """Resolve a committed checkpoint retained by any attempt of this realization."""

        if checkpoint_id is None:
            raise RecordIOError("realization has no committed checkpoint")
        matches = tuple(self.lease.operation.attempts_dir.glob(f"*/checkpoints/{checkpoint_id}"))
        if not matches:
            raise RecordIOError("committed checkpoint payload is missing", context={"checkpoint_id": checkpoint_id})
        if len(matches) > 1 and any(not _trees_match(matches[0], item) for item in matches[1:]):
            raise RecordIOError("checkpoint identity resolves to conflicting retained payloads")
        _validate_checkpoint_root(matches[0], checkpoint_id)
        return matches[0]

    def retained_output_paths(self, slot: str) -> tuple[Path, ...]:
        """Return retained output files for this realization across attempts."""

        from .realizations import validate_output_slot

        validate_output_slot(slot, "slot")
        paths = []
        for workspace in self._realization_workspaces():
            root = workspace / "outputs" / slot
            if root.exists():
                paths.extend(path for path in root.rglob("*") if path.is_file() and path.name != PRODUCT_MANIFEST_FILE)
        return tuple(sorted(paths))

    def retain_output_file(self, slot: str, path: str) -> Path:
        """Link one checkpoint-authenticated prior output into this attempt.

        Managed implementations use this only after validating their own
        checkpoint metadata against the retained bytes. A same-filesystem hard
        link avoids a second payload copy; the streaming fallback remains
        bounded for filesystems that do not support links.
        """

        from .realizations import validate_output_slot

        validate_output_slot(slot, "slot")
        _validate_product_path(path)
        target = self.workspace / "outputs" / slot / path
        if target.exists():
            return target
        candidates = []
        for workspace in self._realization_workspaces():
            candidate = workspace / "outputs" / slot / path
            if candidate.is_file() and candidate != target:
                candidates.append(candidate)
        if not candidates:
            raise RecordIOError(
                "retained output file is missing",
                context={"slot": slot, "path": path},
            )
        if len(candidates) > 1:
            first = ProductManifestEntry(path, candidates[0].stat().st_size, _sha256(candidates[0]))
            if any(
                ProductManifestEntry(path, item.stat().st_size, _sha256(item)) != first
                for item in candidates[1:]
            ):
                raise RecordIOError(
                    "retained output file resolves to conflicting bytes",
                    context={"slot": slot, "path": path},
                )
        self.lease.assert_current()
        self._mkdir(target.parent)
        try:
            os.link(candidates[0], target)
        except OSError:
            temp = target.parent / f".{target.name}.retain-{uuid.uuid4().hex}"
            with candidates[0].open("rb") as source, temp.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            self.lease.assert_current()
            os.replace(temp, target)
        _fsync_directory(target.parent)
        return target

    def finalize(
        self,
        output_records: Mapping[str, Any],
        execution_record: Any,
        *,
        primary_output_slot: str,
        required_output_slots: tuple[str, ...] | list[str],
        activate: bool = False,
    ) -> RealizationPublicationResult:
        """Persist finalization intent, publish immutable records, then optionally activate."""

        intent = self._build_finalization_intent(
            output_records,
            execution_record,
            primary_output_slot=primary_output_slot,
            required_output_slots=required_output_slots,
        )
        intent_path = self.workspace / "finalization-intent.json"
        self._write_immutable_bytes(intent_path, canonical_json_bytes(intent))
        return self._adopt_intent(intent_path, activate=activate)

    @classmethod
    def recover_finalization(
        cls,
        record_io: Any,
        lease: Any,
        realization_id: str,
        *,
        activate: bool = False,
    ) -> RealizationPublicationResult:
        """Idempotently adopt one retained durable finalization intent."""

        from .realizations import validate_realization_id

        validate_realization_id(realization_id)
        lease.assert_current()
        candidates = []
        for path in lease.operation.attempts_dir.glob("*/finalization-intent.json"):
            data = _load_json_file(path, "finalization intent")
            if data.get("realization_id") == realization_id:
                candidates.append((path, data))
        if not candidates:
            raise RecordIOError("no durable finalization intent exists for realization")
        canonical = canonical_json_bytes(candidates[0][1])
        if any(canonical_json_bytes(data) != canonical for _path, data in candidates[1:]):
            raise RecordIOError("conflicting finalization intents exist for realization")
        writer = cls.__new__(cls)
        writer.record_io = record_io
        writer.lease = lease
        writer.realization_id = realization_id
        state = lease.operation._read_realization(realization_id)
        writer.attempt_id = state.current_attempt_id
        writer.workspace = candidates[0][0].parent
        return writer._adopt_intent(candidates[0][0], activate=activate)

    def _write_stream(self, root: Path, path: str, chunks: Any) -> ProductManifestEntry:
        _validate_product_path(path)
        self.lease.assert_current()
        self._mkdir(root / Path(path).parent)
        target = root / path
        partial_root = self.workspace / "partials" / root.relative_to(self.workspace)
        self._mkdir(partial_root / Path(path).parent)
        temp_path = partial_root / f".{Path(path).name}.partial-{uuid.uuid4().hex}"
        digest = hashlib.sha256()
        size = 0
        try:
            with temp_path.open("xb") as handle:
                for chunk in chunks:
                    self.lease.assert_current()
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise RecordIOError("product stream chunks must be bytes-like")
                    view = memoryview(chunk)
                    handle.write(view)
                    digest.update(view)
                    size += len(view)
                handle.flush()
                os.fsync(handle.fileno())
            entry = ProductManifestEntry(path, size, digest.hexdigest())
            self.lease.assert_current()
            if target.exists():
                existing = ProductManifestEntry(path, target.stat().st_size, _sha256(target))
                if existing != entry:
                    raise RecordIOError("immutable attempt file already exists with different bytes")
                temp_path.unlink()
            else:
                os.replace(temp_path, target)
                _fsync_directory(target.parent)
            return entry
        except Exception:
            # A short-write partial is retained for diagnosis and explicit recovery.
            raise

    def _build_finalization_intent(
        self,
        output_records: Mapping[str, Any],
        execution_record: Any,
        *,
        primary_output_slot: str,
        required_output_slots: tuple[str, ...] | list[str],
    ) -> dict[str, Any]:
        from .execution import ExecutionRecord, ExecutionRecordLink
        from .realizations import RealizationOutput, RealizationRecord, validate_output_slot

        validate_output_slot(primary_output_slot, "primary_output_slot")
        required = tuple(required_output_slots)
        if not required or len(set(required)) != len(required):
            raise RecordValidationError("required_output_slots must be unique and non-empty")
        for slot in required:
            validate_output_slot(slot, "required_output_slots")
        if primary_output_slot not in required:
            raise RecordValidationError("primary output must be required")
        if not set(required).issubset(output_records):
            raise RecordValidationError("all required outputs must be finalized together")
        for slot in output_records:
            validate_output_slot(slot, "output slot")
        attached_outputs = {}
        output_links = []
        for slot in sorted(output_records):
            root = self.workspace / "outputs" / slot
            if not root.is_dir():
                raise RecordIOError("required output has no durable bytes", context={"slot": slot})
            manifest = _manifest_for_root(root)
            if any(entry.path == PRODUCT_MANIFEST_FILE for entry in manifest.entries):
                raise RecordIOError("output uses the reserved detailed manifest path")
            detail_bytes = _detailed_manifest_bytes(manifest)
            self._write_immutable_bytes(root / PRODUCT_MANIFEST_FILE, detail_bytes)
            root_manifest = ProductRootManifest(
                PRODUCT_MANIFEST_FILE,
                len(detail_bytes),
                hashlib.sha256(detail_bytes).hexdigest(),
                len(manifest.entries),
                sum(entry.size for entry in manifest.entries),
            )
            value = output_records[slot]
            envelope = value.to_envelope() if hasattr(value, "to_envelope") else dict(value)
            payload = envelope.get("payload")
            if not isinstance(payload, Mapping):
                raise RecordValidationError("managed output record payload must be a mapping")
            payload = dict(payload)
            if payload.get("realization_id") != self.realization_id or payload.get("output_slot") != slot:
                raise RecordValidationError("managed output ownership does not match finalization")
            payload["manifest"] = root_manifest.to_json()
            envelope = dict(envelope)
            envelope["payload"] = payload
            attached = attach_record_id(envelope)
            validate_record(attached)
            if attached["kind"] not in {"data", "stored_state"}:
                raise RecordValidationError("managed output record kind is unsupported")
            representation_id = payload.get("representation_id")
            if not self.record_io.has_spec(representation_id, family="representation"):
                raise RecordValidationError(
                    "managed output representation spec is missing",
                    context={"representation_id": representation_id},
                )
            self.record_io.read_spec(representation_id, family="representation")
            attached_outputs[slot] = attached
            output_links.append(
                RealizationOutput(
                    slot=slot,
                    record_id=attached["id"],
                    record_kind=attached["kind"],
                    representation_id=representation_id,
                    required=slot in required,
                    subject_cdef_id=payload.get("subject_cdef_id"),
                )
            )
        execution_envelope = execution_record.to_envelope() if hasattr(execution_record, "to_envelope") else dict(execution_record)
        execution_payload = execution_envelope.get("payload")
        if not isinstance(execution_payload, Mapping):
            raise RecordValidationError("managed execution payload must be a mapping")
        execution_payload = dict(execution_payload)
        execution_payload["realization_id"] = self.realization_id
        execution_payload["produced_records"] = [
            ExecutionRecordLink(
                item.record_id,
                role=item.slot,
                representation_id=item.representation_id,
                subject_cdef_id=item.subject_cdef_id,
                required=item.required,
                realization_id=self.realization_id,
                output_slot=item.slot,
            ).to_json()
            for item in output_links
        ]
        execution_envelope = dict(execution_envelope)
        execution_envelope["payload"] = execution_payload
        execution = ExecutionRecord.from_envelope(execution_envelope)
        exact_consumed = tuple(
            link.to_resolved()
            for link in execution.consumed_records
            if link.producer_cdef_id is not None
        )
        for link in execution.consumed_records:
            self.record_io.read_record(link.record_id)
        attached_execution = attach_record_id(execution.to_envelope())
        state = self.lease.operation._read_realization(self.realization_id)
        if state.current_attempt_id != self.attempt_id:
            raise RecordIOError("finalization attempt is no longer current")
        primary = next(item for item in output_links if item.slot == primary_output_slot)
        realization = RealizationRecord(
            realization_id=self.realization_id,
            producer_cdef_id=self.lease.operation.key.producer_cdef_id,
            method=self.lease.operation.key.method,
            declaration_fingerprint=self.lease.operation.declaration_fingerprint,
            attempt_ids=state.attempt_ids,
            outputs=tuple(output_links),
            primary_output_slot=primary_output_slot,
            primary_representation_id=primary.representation_id,
            execution_record_id=attached_execution["id"],
            consumed_records=exact_consumed,
            completed_attempt_id=self.attempt_id,
            completion_fence=self.lease.epoch,
            checkpoint_head=state.checkpoint_head,
        )
        attached_realization = attach_record_id(realization.to_envelope())
        return {
            "schema": FINALIZATION_INTENT_SCHEMA,
            "schema_version": 1,
            "realization_id": self.realization_id,
            "attempt_id": self.attempt_id,
            "fence_epoch": self.lease.epoch,
            "outputs": attached_outputs,
            "execution_record": attached_execution,
            "realization_record": attached_realization,
        }

    def _adopt_intent(self, intent_path: Path, *, activate: bool) -> RealizationPublicationResult:
        from .realizations import RealizationRecord

        intent = _load_json_file(intent_path, "finalization intent")
        _validate_finalization_intent(intent, self.realization_id)
        self.lease.assert_current()
        output_refs = {}
        outputs = intent["outputs"]
        for slot in sorted(outputs):
            envelope = outputs[slot]
            record_id = envelope["id"]
            source = intent_path.parent / "outputs" / slot
            target = self.record_io.product_root(record_id)
            self.lease.assert_current()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require_product_integrity(self.record_io, envelope)
            elif source.exists():
                os.replace(source, target)
                _fsync_directory(target.parent)
                require_product_integrity(self.record_io, envelope)
            else:
                raise RecordIOError("finalized product bytes are missing", context={"slot": slot})
        for slot in sorted(outputs):
            self.lease.assert_current()
            output_refs[slot] = self.record_io.write_record(outputs[slot])
        self.lease.assert_current()
        execution_ref = self.record_io.write_record(intent["execution_record"])
        self.lease.assert_current()
        realization_ref = self.record_io.write_record(intent["realization_record"])
        state = self.lease.operation._read_realization(self.realization_id)
        realization = RealizationRecord.from_envelope(intent["realization_record"])
        if state.attempt_ids != realization.attempt_ids:
            raise RecordIOError(
                "finalization intent does not include the current realization attempts"
            )
        if state.status != "completed":
            self.lease.complete(
                self.realization_id,
                realization_record_id=realization_ref.record_id,
            )
        elif state.realization_record_id != realization_ref.record_id:
            raise RecordIOError("completed realization binds a different immutable record")
        if activate:
            self.lease.activate(self.realization_id)
        return RealizationPublicationResult(
            output_records=output_refs,
            execution_record=execution_ref,
            realization_record=realization_ref,
            activated=activate,
        )

    def _realization_workspaces(self) -> tuple[Path, ...]:
        state = self.lease.operation._read_realization(self.realization_id)
        suffixes = set(state.attempt_ids)
        return tuple(
            path
            for path in sorted(self.lease.operation.attempts_dir.iterdir())
            if path.is_dir() and any(path.name.endswith(attempt_id) for attempt_id in suffixes)
        ) if self.lease.operation.attempts_dir.exists() else ()

    def _mkdir(self, path: Path) -> None:
        self.lease.assert_current()
        path.mkdir(parents=True, exist_ok=True)

    def _write_immutable_bytes(self, path: Path, payload: bytes) -> None:
        self.lease.assert_current()
        self._mkdir(path.parent)
        if path.exists():
            if path.read_bytes() != payload:
                raise RecordIOError("immutable durable metadata already exists with different bytes")
            return
        temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        with temp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.lease.assert_current()
        os.replace(temp, path)
        _fsync_directory(path.parent)


def _manifest_for_root(root: Path) -> ProductManifest:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append(ProductManifestEntry(path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path)))
    return ProductManifest(tuple(entries))


def _detailed_manifest_bytes(manifest: ProductManifest) -> bytes:
    return canonical_json_bytes(
        {
            "schema": PRODUCT_MANIFEST_SCHEMA,
            "schema_version": 1,
            "entries": [entry.to_json() for entry in manifest.entries],
        }
    )


def _load_detailed_manifest(payload: bytes) -> ProductManifest:
    try:
        data = canonical_json_load_bytes(payload)
    except Exception as exc:
        raise RecordValidationError("detailed product manifest is invalid JSON") from exc
    fields = {"schema", "schema_version", "entries"}
    if not isinstance(data, Mapping) or set(data) != fields:
        raise RecordValidationError("detailed product manifest fields are malformed")
    if data.get("schema") != PRODUCT_MANIFEST_SCHEMA or data.get("schema_version") != 1:
        raise RecordValidationError("detailed product manifest schema is unsupported")
    return ProductManifest.from_json({"entries": data["entries"]})


def _verify_manifest_files(
    root: Path,
    manifest: ProductManifest,
    record_id: str,
    *,
    allowed_extra: set[str] | None = None,
) -> list[ProductAvailabilityIssue]:
    issues = []
    expected = {entry.path for entry in manifest.entries}
    allowed = set(allowed_extra or ())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    } if root.exists() else set()
    for path_name in sorted(actual - expected - allowed):
        issues.append(
            ProductAvailabilityIssue(
                "unexpected_product_path",
                "record product contains a file absent from its manifest",
                record_id,
                (root / path_name).as_posix(),
            )
        )
    for entry in manifest.entries:
        path = root / entry.path
        if not path.is_file():
            issues.append(
                ProductAvailabilityIssue(
                    "missing_manifest_entry",
                    "record product manifest entry is missing",
                    record_id,
                    path.as_posix(),
                )
            )
            continue
        observed_size = path.stat().st_size
        if observed_size != entry.size:
            issues.append(
                ProductAvailabilityIssue(
                    "product_size_mismatch",
                    "record product file size does not match its manifest",
                    record_id,
                    path.as_posix(),
                )
            )
            continue
        if _sha256(path) != entry.sha256:
            issues.append(
                ProductAvailabilityIssue(
                    "product_digest_mismatch",
                    "record product file digest does not match its manifest",
                    record_id,
                    path.as_posix(),
                )
            )
    return issues


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trees_match(first: Path, second: Path) -> bool:
    if not first.is_dir() or not second.is_dir():
        return False
    first_manifest = _manifest_for_root(first)
    second_manifest = _manifest_for_root(second)
    return first_manifest == second_manifest


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


def _load_json_file(path: Path, name: str) -> dict[str, Any]:
    try:
        data = canonical_json_load_bytes(path.read_bytes())
    except Exception as exc:
        raise RecordIOError(f"{name} could not be read", context={"error": str(exc)}) from exc
    if not isinstance(data, dict):
        raise RecordIOError(f"{name} must be a JSON object")
    return data


def _validate_finalization_intent(data: Mapping[str, Any], realization_id: str) -> None:
    from .execution import ExecutionRecord
    from .realizations import RealizationRecord, validate_attempt_id, validate_output_slot

    fields = {
        "schema",
        "schema_version",
        "realization_id",
        "attempt_id",
        "fence_epoch",
        "outputs",
        "execution_record",
        "realization_record",
    }
    if set(data) != fields:
        raise RecordIOError("finalization intent fields are malformed")
    if data.get("schema") != FINALIZATION_INTENT_SCHEMA or data.get("schema_version") != 1:
        raise RecordIOError("finalization intent schema is unsupported")
    if data.get("realization_id") != realization_id:
        raise RecordIOError("finalization intent binds a different realization")
    validate_attempt_id(data.get("attempt_id"))
    if type(data.get("fence_epoch")) is not int or data["fence_epoch"] < 1:
        raise RecordIOError("finalization intent fence is malformed")
    outputs = data.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise RecordIOError("finalization intent outputs are malformed")
    output_ids = {}
    for slot, envelope in outputs.items():
        validate_output_slot(slot, "output slot")
        validate_record(envelope)
        payload = envelope.get("payload") or {}
        if payload.get("realization_id") != realization_id or payload.get("output_slot") != slot:
            raise RecordIOError("finalization output ownership is malformed")
        output_ids[slot] = envelope["id"]
    execution = ExecutionRecord.from_envelope(data.get("execution_record"))
    realization = RealizationRecord.from_envelope(data.get("realization_record"))
    if execution.realization_id != realization_id or realization.realization_id != realization_id:
        raise RecordIOError("finalization lineage binds a different realization")
    if realization.execution_record_id != data["execution_record"]["id"]:
        raise RecordIOError("finalization execution lineage is inconsistent")
    if {item.slot: item.record_id for item in realization.outputs} != output_ids:
        raise RecordIOError("finalization output lineage is inconsistent")


def _validate_checkpoint_root(root: Path, checkpoint_id: str) -> None:
    descriptor_path = root / "checkpoint.json"
    descriptor = _load_json_file(descriptor_path, "checkpoint descriptor")
    fields = {
        "schema",
        "schema_version",
        "checkpoint_id",
        "checkpoint_schema",
        "manifest",
        "metadata",
    }
    if set(descriptor) != fields:
        raise RecordIOError("checkpoint descriptor fields are malformed")
    if descriptor.get("schema") != "dryml.managed.checkpoint.v1" or descriptor.get("schema_version") != 1:
        raise RecordIOError("checkpoint descriptor schema is unsupported")
    identity = {
        "checkpoint_schema": descriptor.get("checkpoint_schema"),
        "manifest": descriptor.get("manifest"),
        "metadata": descriptor.get("metadata"),
    }
    expected_id = content_id("checkpoint", 1, identity)
    if descriptor.get("checkpoint_id") != checkpoint_id or expected_id != checkpoint_id:
        raise RecordIOError("checkpoint identity does not match its descriptor")
    manifest = ProductManifest.from_json(descriptor["manifest"])
    issues = _verify_manifest_files(root, manifest, checkpoint_id, allowed_extra={"checkpoint.json"})
    if issues:
        raise RecordIOError(
            "checkpoint product integrity verification failed",
            context={"checkpoint_id": checkpoint_id, "issues": [item.to_json() for item in issues]},
        )


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    "CheckpointCommit",
    "DurableProductWriter",
    "FINALIZATION_INTENT_SCHEMA",
    "ProductManifest",
    "ProductManifestEntry",
    "ProductRootManifest",
    "ProductAvailabilityIssue",
    "ProductWriteResult",
    "ProductWriteSession",
    "RealizationPublicationResult",
    "commit_product_record",
    "stage_product_file",
    "require_product_integrity",
    "require_checkpoint_integrity",
    "validate_product_availability",
]
