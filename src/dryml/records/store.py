"""Store-local canonical JSON IO for DRYML record and spec sidecars."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from dryml.formats.canonical import canonical_json_bytes, canonical_json_load_bytes
from dryml.formats.errors import CanonicalJSONError, ContentIDError
from dryml.formats.ids import parse_content_id
from dryml.formats.refs import parse_cdef_id

from .errors import RecordIOError, RecordNotFoundError, RecordValidationError, SpecNotFoundError, SpecValidationError, StorageRefError
from .records import attach_record_id, validate_record
from .refs import LocatedRecordRef, LocatedSpecRef
from .specs import attach_spec_id, spec_dir_name, spec_family_for_id, validate_spec
from .storage import StorageRef


class RecordStoreIO:
    """Facade for store-owned record/spec sidecars under one Store base dir."""

    def __init__(self, store: Any):
        self.store = store

    @property
    def base_dir(self) -> Path:
        """Return the backing store base directory as a path."""

        return Path(os.fspath(self.store.base_dir))

    @property
    def records_dir(self) -> Path:
        """Return ``<base>/records``."""

        return self.base_dir / "records"

    @property
    def items_dir(self) -> Path:
        """Return ``<base>/records/items``."""

        return self.records_dir / "items"

    @property
    def specs_dir(self) -> Path:
        """Return ``<base>/records/specs``."""

        return self.records_dir / "specs"

    @property
    def indexes_dir(self) -> Path:
        """Return optional ``<base>/records/indexes``."""

        return self.records_dir / "indexes"

    @property
    def products_dir(self) -> Path:
        """Return ``<base>/products``."""

        return self.base_dir / "products"

    def spec_family_dir(self, family: str) -> Path:
        """Return the directory for a spec family."""

        return self.specs_dir / spec_dir_name(family)

    def product_root(self, record_id: str, *, create: bool = False) -> Path:
        """Return ``products/<record-id>`` and optionally create it."""

        _validate_content_id(record_id, "record", RecordValidationError)
        path = self.products_dir / record_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_dirs(self) -> None:
        """Create optional record sidecar root directories lazily."""

        self.items_dir.mkdir(parents=True, exist_ok=True)
        self.specs_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.products_dir.mkdir(parents=True, exist_ok=True)

    def write_record(self, record: Mapping[str, Any], *, overwrite: bool = False) -> LocatedRecordRef:
        """Validate and atomically write a record JSON sidecar."""

        attached = attach_record_id(record)
        validate_record(attached)
        record_id = attached["id"]
        path = self._record_path(record_id)
        self._write_json(path, attached, overwrite=overwrite, error_cls=RecordIOError)
        return LocatedRecordRef(store_ref=self._store_ref(), record_id=record_id)

    def read_record(self, record_id: str) -> dict[str, Any]:
        """Read and validate one record by ID."""

        _validate_content_id(record_id, "record", RecordValidationError)
        path = self._record_path(record_id)
        if not path.exists():
            raise RecordNotFoundError("record not found", context={"record_id": record_id})
        data = self._read_json(path, RecordIOError)
        validate_record(data)
        if data.get("id") != record_id:
            raise RecordValidationError("record file ID mismatch", context={"expected": record_id, "observed": data.get("id")})
        return data

    def has_record(self, record_id: str) -> bool:
        """Return whether a record sidecar exists for *record_id*."""

        _validate_content_id(record_id, "record", RecordValidationError)
        return self._record_path(record_id).exists()

    def iter_record_ids(self) -> Iterator[str]:
        """Yield record IDs by scanning record JSON source files."""

        if not self.items_dir.exists():
            return
        for path in sorted(self.items_dir.iterdir()):
            if path.suffix != ".json":
                continue
            record_id = path.stem
            _validate_content_id(record_id, "record", RecordValidationError)
            yield record_id

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield validated records by scanning record JSON source files."""

        for record_id in self.iter_record_ids():
            yield self.read_record(record_id)

    def write_spec(self, spec: Mapping[str, Any], *, family: str | None = None, overwrite: bool = False) -> LocatedSpecRef:
        """Validate and atomically write a spec JSON sidecar."""

        attached = attach_spec_id(spec, family=family)
        validate_spec(attached, family=family)
        spec_id = attached["id"]
        resolved_family = family or spec_family_for_id(spec_id)
        if resolved_family is None:
            raise SpecValidationError("spec family is required for unknown spec ID prefix", context={"spec_id": spec_id})
        path = self._spec_path(spec_id, resolved_family)
        self._write_json(path, attached, overwrite=overwrite, error_cls=RecordIOError)
        return LocatedSpecRef(store_ref=self._store_ref(), spec_id=spec_id, kind=resolved_family)

    def read_spec(self, spec_id: str, *, family: str | None = None) -> dict[str, Any]:
        """Read and validate one spec by ID."""

        resolved_family = family or spec_family_for_id(spec_id)
        if resolved_family is None:
            raise SpecValidationError("spec family is required for unknown spec ID prefix", context={"spec_id": spec_id})
        _validate_content_id(spec_id, None, SpecValidationError)
        path = self._spec_path(spec_id, resolved_family)
        if not path.exists():
            raise SpecNotFoundError("spec not found", context={"spec_id": spec_id, "family": resolved_family})
        data = self._read_json(path, RecordIOError)
        validate_spec(data, family=resolved_family)
        if data.get("id") != spec_id:
            raise SpecValidationError("spec file ID mismatch", context={"expected": spec_id, "observed": data.get("id")})
        return data

    def has_spec(self, spec_id: str, *, family: str | None = None) -> bool:
        """Return whether a spec sidecar exists for *spec_id*."""

        resolved_family = family or spec_family_for_id(spec_id)
        if resolved_family is None:
            raise SpecValidationError("spec family is required for unknown spec ID prefix", context={"spec_id": spec_id})
        _validate_content_id(spec_id, None, SpecValidationError)
        return self._spec_path(spec_id, resolved_family).exists()

    def iter_spec_ids(self, *, family: str | None = None) -> Iterator[str]:
        """Yield spec IDs by scanning spec JSON source files."""

        family_dirs: list[tuple[str, Path]]
        if family is None:
            if not self.specs_dir.exists():
                return
            family_dirs = [(path.name, path) for path in sorted(self.specs_dir.iterdir()) if path.is_dir()]
        else:
            family_dirs = [(family, self.spec_family_dir(family))]
        for current_family, directory in family_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.iterdir()):
                if path.suffix != ".json":
                    continue
                spec_id = path.stem
                _validate_content_id(spec_id, None, SpecValidationError)
                known_family = spec_family_for_id(spec_id)
                if known_family is not None and spec_dir_name(known_family) != current_family:
                    raise SpecValidationError(
                        "spec file is stored under the wrong family directory",
                        context={"spec_id": spec_id, "directory": current_family, "expected": known_family},
                    )
                yield spec_id

    def iter_specs(self, *, family: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield validated specs by scanning spec JSON source files."""

        for spec_id in self.iter_spec_ids(family=family):
            yield self.read_spec(spec_id, family=family)

    def resolve_storage_ref(self, ref: StorageRef | Mapping[str, Any], *, create: bool = False) -> Path:
        """Resolve a logical storage reference inside this store."""

        storage_ref = ref if isinstance(ref, StorageRef) else StorageRef.from_json(ref)
        if storage_ref.kind == "object-dir":
            cdef = parse_cdef_id(storage_ref.subject_cdef_id)  # type: ignore[arg-type]
            if len(cdef.digest) != 64:
                raise StorageRefError("object-dir refs require a full CDef digest for direct resolution", context={"subject_cdef_id": cdef.raw})
            root = Path(os.fspath(self.store.object_root_dir)) / cdef.digest[:2] / cdef.digest
            path = root if storage_ref.path == "." else root / storage_ref.path
            if create:
                raise StorageRefError("object-dir storage refs cannot create object directories")
            if not path.exists():
                raise StorageRefError("object-dir storage ref does not exist", context={"subject_cdef_id": cdef.raw})
            return path
        if storage_ref.kind == "product-dir":
            root = self.product_root(storage_ref.record_id, create=create)  # type: ignore[arg-type]
            path = root if storage_ref.path == "." else root / storage_ref.path
            if create:
                path.mkdir(parents=True, exist_ok=True)
            return path
        raise StorageRefError("blob storage refs cannot be resolved in Sprint 1", context={"blob_id": storage_ref.blob_id})

    def _record_path(self, record_id: str) -> Path:
        return self.items_dir / f"{record_id}.json"

    def _spec_path(self, spec_id: str, family: str) -> Path:
        return self.spec_family_dir(family) / f"{spec_id}.json"

    def _store_ref(self) -> str:
        catalog_key = getattr(self.store, "catalog_key", None)
        if callable(catalog_key):
            return str(catalog_key())
        return os.path.abspath(os.fspath(self.store.base_dir))

    @staticmethod
    def _write_json(path: Path, data: Mapping[str, Any], *, overwrite: bool, error_cls: type[RecordIOError]) -> None:
        payload = canonical_json_bytes(data)
        if path.exists() and not overwrite:
            existing = path.read_bytes()
            if existing == payload:
                return
            raise error_cls("sidecar already exists with different canonical bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
                tmp.write(payload)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _read_json(path: Path, error_cls: type[RecordIOError]) -> dict[str, Any]:
        try:
            data = canonical_json_load_bytes(path.read_bytes())
        except (OSError, CanonicalJSONError) as exc:
            raise error_cls("sidecar JSON could not be read", context={"error": str(exc)}) from exc
        if not isinstance(data, dict):
            raise error_cls("sidecar JSON root must be an object", context={"type": type(data).__name__})
        return data


def _validate_content_id(value: str, prefix: str | None, error_cls: type[Exception]) -> None:
    try:
        parts = parse_content_id(value)
    except ContentIDError as exc:
        raise error_cls("invalid content ID", context=exc.context) from exc
    if prefix is not None and (parts.prefix != prefix or parts.schema_version != 1):
        raise error_cls(
            f"content ID must use {prefix}-v1 prefix",
            context={"value": value, "prefix": parts.prefix, "schema_version": parts.schema_version},
        )


__all__ = ["RecordStoreIO"]
