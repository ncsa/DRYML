"""Direct-root filesystem implementation of current Store authority."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import replace
from collections.abc import Mapping
from uuid import uuid4
from pathlib import Path
from typing import Iterable, Literal

from ...locking import interprocess_lock, supports_advisory_locking
from .records import (
    ClaimRecord, DeclarationRecord, DefinitionRecord, LocalStateManifest,
    MainRefRecord, ObjectAliasRecord, StateAliasRecord, StateRefRecord,
    StoredRootRecord, StoreFormatRecord, StoreRecordError,
)
from .store import LocalStateSource, Store, StoreAuthorityError, StoreCapabilityError, StorePublicationCapabilities
from ..query.model import QueryIndexStatus, QueryIndexUnavailable, ReconcileReport
from ..query.sqlite import SQLiteQueryIndexConfig, sqlite_available
from ..query.sqlite.index import SQLiteStoreQueryIndex


QueryIndexPolicy = Literal["auto", "sqlite", "memory", "none"]


class DirStore(Store):
    """Persist current logical authority directly beneath one local filesystem root.

    Immutable records are digest sharded. Each StateRef is authoritative only as
    a complete ``snapshots/<hh>/<digest>/`` directory, which contains metadata,
    placement, and every locally owned payload. All mutable references use a
    sibling temporary file and atomic replacement while the Store writer lock is
    held. The old ``objects/`` generation layout is not a recognized authority
    format. Initial root creation and format-gate publication use a derived
    sibling advisory lock, so simultaneous trusted constructors never mistake a
    live temporary format file for old authority.
    """

    def __init__(
            self,
            base_dir: str | os.PathLike[str],
            *,
            query_index: QueryIndexPolicy | SQLiteQueryIndexConfig = "auto",
            _existing_only: bool = False):
        self._base_dir = os.path.abspath(os.fspath(base_dir))
        if isinstance(query_index, str):
            if query_index not in {"auto", "sqlite", "memory", "none"}:
                raise ValueError("DirStore query_index must be 'auto', 'sqlite', 'memory', 'none', or SQLiteQueryIndexConfig.")
            self._query_index_policy: QueryIndexPolicy = query_index
            self._query_index_config = None
        elif isinstance(query_index, SQLiteQueryIndexConfig):
            self._query_index_policy = "sqlite"
            self._query_index_config = query_index
        else:
            raise ValueError("DirStore query_index must be 'auto', 'sqlite', 'memory', 'none', or SQLiteQueryIndexConfig.")
        self.query_index = query_index
        self._query_index_instance: SQLiteStoreQueryIndex | None = None
        self._initialize_format(existing_only=_existing_only)
        evidence = os.stat(self._base_dir)
        self._authority_evidence = (evidence.st_dev, evidence.st_ino)

    @classmethod
    def open_existing(
            cls,
            base_dir: str | os.PathLike[str],
            *,
            query_index: QueryIndexPolicy | SQLiteQueryIndexConfig = "auto") -> "DirStore":
        """Open validated current authority without creating or repairing it.

        Args:
            base_dir: Existing direct Store root.
            query_index: Derived-index policy for the fresh handle.

        Returns:
            A matching Session-cached Store handle when resource caching is active,
            otherwise a new caller-owned handle over current-format authority.

        Raises:
            StoreAuthorityError: If the root or required format record is absent,
                malformed, or not the required filesystem type.
        """

        path = os.path.abspath(os.fspath(base_dir))
        if cls is not DirStore or not isinstance(query_index, str):
            # Custom SQLite configuration has no portable descriptor grammar and
            # retains the established fresh-handle behavior.
            cls._validate_existing_root(path)
            return cls(path, query_index=query_index, _existing_only=True)

        from ..repo_definition import _open_store_descriptor

        return _open_store_descriptor({
            "kind": "dir",
            "path": path,
            "query_index": query_index,
        })

    @property
    def base_dir(self) -> str:
        """Return the direct-layout Store root directory."""
        return self._base_dir

    @property
    def query_index_policy(self) -> QueryIndexPolicy:
        """Return the configured derived-index policy without making it authority."""
        return self._query_index_policy

    @property
    def dryml_dir(self) -> str:
        """Return the derived-sidecar directory, which is never Store authority."""
        return os.path.join(self.base_dir, ".dryml")

    @property
    def query_index_path(self) -> str:
        """Return the canonical SQLite sidecar path without creating it."""
        return os.path.join(self.dryml_dir, "query-index-v1.sqlite")

    @property
    def query_index_dirty_path(self) -> str:
        """Return the prefix used for durable derived-index dirty markers."""
        return os.path.join(self.dryml_dir, "query-index.dirty")

    def open_query_index(self):
        """Open the configured derived SQLite index lazily, when available.

        The returned index only accelerates queries. DefinitionRecords remain
        authoritative and are scanned to rebuild a missing or stale sidecar.
        """
        if self._query_index_instance is not None:
            return self._query_index_instance
        if self._query_index_policy in {"memory", "none"}:
            return None
        if not sqlite_available():
            if self._query_index_policy == "auto":
                return None
            raise QueryIndexUnavailable("DirStore query_index='sqlite' requires Python's optional sqlite3 module.")
        config = self._query_index_config
        path = self.query_index_path
        if config is None:
            config = SQLiteQueryIndexConfig(path=path)
        elif config.path is None:
            config = replace(config, path=path)
        else:
            path = os.fspath(config.path)
        self._query_index_instance = SQLiteStoreQueryIndex(
            source_key=self.catalog_key(), path=path, config=config, store=self,
            dirty_path=self.query_index_dirty_path,
        )
        return self._query_index_instance

    def mark_query_index_dirty(self, cdef=None, *, metadata_target=None) -> str | None:
        """Publish a durable marker for an authoritative queryable mutation.

        Args:
            cdef: Optional definition whose immutable DefinitionRecord changed.
            metadata_target: Optional ObjectRef or StateRef whose metadata changed,
                mutually exclusive with ``cdef``. Omitted targets mark an
                unscoped mutation requiring full recovery.

        Returns:
            The marker path, or ``None`` when this Store has no SQLite sidecar.

        Raises:
            TypeError: If the metadata target is not an exact reference.
            ValueError: If both scopes are supplied.
            OSError: If durable marker publication fails.

        Side Effects:
            Writes and fsyncs a unique derived token before metadata publication.
        """
        from ..reference_values import ObjectRef, StateRef

        if metadata_target is not None:
            if cdef is not None:
                raise ValueError("Dirty markers require one scope only.")
            if not isinstance(metadata_target, (ObjectRef, StateRef)):
                raise TypeError("Metadata dirty targets must be ObjectRef or StateRef.")
        if self._query_index_policy not in {"auto", "sqlite"}:
            return None
        os.makedirs(self.dryml_dir, exist_ok=True)
        key = "dirty"
        if cdef is not None:
            key = DefinitionRecord(cdef).digest
        elif metadata_target is not None:
            kind = "state" if isinstance(metadata_target, StateRef) else "object"
            key = f"metadata:{kind}:{metadata_target.digest()}"
        marker_path = f"{self.query_index_dirty_path}.{uuid4().hex}"
        fd, temporary_path = tempfile.mkstemp(prefix=".query-index-dirty-", dir=self.dryml_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                target.write(f"{key}\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_path, marker_path)
            self._fsync_directory(self.dryml_dir)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise
        return marker_path

    def _query_index_dirty_markers(self) -> tuple[str, ...]:
        """Return all current dirty tokens without interpreting them as authority."""
        root = Path(self.dryml_dir)
        if not root.exists():
            return ()
        markers = tuple(os.fspath(path) for path in root.glob("query-index.dirty.*"))
        if os.path.exists(self.query_index_dirty_path):
            markers = (*markers, self.query_index_dirty_path)
        return markers

    def clear_query_index_dirty(self) -> None:
        """Remove derived dirty markers without modifying DefinitionRecords."""
        removed = False
        for marker in self._query_index_dirty_markers():
            try:
                os.unlink(marker)
                removed = True
            except FileNotFoundError:
                pass
        if removed:
            self._fsync_directory(self.dryml_dir)

    def query_index_is_dirty(self) -> bool:
        """Return whether a published definition may not be represented by SQLite."""
        return bool(self._query_index_dirty_markers())

    def query_index_status(self) -> QueryIndexStatus:
        """Return the configured derived-index status without reading authority."""
        if self._query_index_policy == "none":
            return QueryIndexStatus("none", self.catalog_key(), None, None, {}, "disabled")
        if self._query_index_policy == "memory":
            return QueryIndexStatus("memory", self.catalog_key(), None, None, {}, "ready")
        index = self.open_query_index()
        if index is None:
            return QueryIndexStatus("memory", self.catalog_key(), None, None, {}, "ready")
        return index.status()

    def _open_rebuildable_query_index(self) -> SQLiteStoreQueryIndex:
        index = self.open_query_index()
        if index is None:
            raise QueryIndexUnavailable(
                f"DirStore query_index={self._query_index_policy!r} does not provide a rebuildable persistent index."
            )
        return index

    def rebuild_query_index(self) -> ReconcileReport:
        """Rebuild SQLite solely by scanning validated authoritative DefinitionRecords."""
        index = self._open_rebuildable_query_index()
        before = index.status()
        index.rebuild()
        after = index.status()
        return ReconcileReport(
            backend=after.backend,
            store_key=after.store_key,
            changed=True,
            action="rebuild",
            generation_before=before.generation,
            generation_after=after.generation,
            definitions_scanned=(after.row_counts or {}).get("stored_roots", 0),
            validated=True,
        )

    def reconcile_query_index(self) -> ReconcileReport:
        """Validate or safely rebuild a missing, stale, or corrupt SQLite sidecar."""
        return self._open_rebuildable_query_index().reconcile()

    def validate_query_index(self, *, thorough: bool = False):
        """Validate the configured index while leaving DefinitionRecord authority intact."""
        index = self.open_query_index()
        if index is None:
            return super().validate_query_index(thorough=thorough)
        return index.validate(thorough=thorough)

    @property
    def publication_capabilities(self) -> StorePublicationCapabilities:
        """Return local-filesystem publication guarantees after format validation."""
        return StorePublicationCapabilities(
            True, True, True, supports_advisory_locking(self._writer_lock_path), True
        )

    @property
    def _writer_lock_path(self) -> str:
        return os.path.join(self.base_dir, ".writer.lock")

    def writer_lock(self):
        """Serialize a multi-record Store authority transition.

        The reentrant adapter lets record writers retain their own defensive
        locking while Repo performs one declaration or claim compare-and-swap.
        """
        return interprocess_lock(self._writer_lock_path)

    @property
    def store_format_path(self) -> str:
        """Return the only Store-wide authority format record path."""
        return os.path.join(self.base_dir, "store-format.record")

    @property
    def _bootstrap_lock_path(self) -> str:
        """Return the derived sibling lock for this canonical Store root.

        The durable lock file is named from the SHA-256 digest of the ``os.fsencode``
        bytes of the normalized real root path and is never authority or removed
        by a Store handle.
        Keeping it outside the root lets bootstrap reject arbitrary nonempty
        roots without first adding a Store-owned entry to them.
        """
        root = os.path.normcase(os.path.realpath(self.base_dir))
        identity = hashlib.sha256(os.fsencode(root)).hexdigest()
        return os.path.join(os.path.dirname(root), f".dryml-bootstrap-{identity}.lock")

    @classmethod
    def _validate_existing_root(cls, base_dir: str) -> None:
        """Validate existing root/type evidence without creating Store state."""

        try:
            root_mode = os.lstat(base_dir).st_mode
        except OSError as error:
            raise StoreAuthorityError("DirStore root is missing or inaccessible.") from error
        if not stat.S_ISDIR(root_mode):
            raise StoreAuthorityError("DirStore root is not a directory.")
        format_path = os.path.join(base_dir, "store-format.record")
        try:
            format_mode = os.lstat(format_path).st_mode
        except OSError as error:
            raise StoreAuthorityError("DirStore lacks current store-format.record.") from error
        if not stat.S_ISREG(format_mode):
            raise StoreAuthorityError("DirStore format record is not a regular file.")
        try:
            with open(format_path, "rb") as source:
                StoreFormatRecord.from_bytes(source.read())
        except (OSError, StoreRecordError) as error:
            raise StoreAuthorityError("DirStore format record is malformed or inaccessible.") from error

    def _initialize_format(self, *, existing_only: bool = False) -> None:
        root = Path(self.base_dir)
        if existing_only:
            self._validate_existing_root(self.base_dir)
            return
        if root.exists() and not root.is_dir():
            raise StoreAuthorityError(f"DirStore root is not a directory: {self.base_dir!r}.")
        if os.path.lexists(self.store_format_path):
            self._read_file(self.store_format_path, StoreFormatRecord)
            return
        # This lease covers root creation through final marker replacement.  It
        # remains outside authority so an unmarked root can still be rejected
        # without treating a Store-created lock as permission to overwrite it.
        with interprocess_lock(self._bootstrap_lock_path):
            if root.exists() and not root.is_dir():
                raise StoreAuthorityError(f"DirStore root is not a directory: {self.base_dir!r}.")
            if not root.exists():
                root.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(self.store_format_path):
                self._read_file(self.store_format_path, StoreFormatRecord)
                return
            # A non-empty root without the current gate is retired/incompatible;
            # do not create a new format marker over unknown authority.
            if any(root.iterdir()):
                raise StoreAuthorityError("Store lacks current store-format.record; old or mixed authority is unsupported.")
            self._atomic_write(self.store_format_path, StoreFormatRecord().to_bytes())

    def _read_file(self, path: str, record_type):
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise StoreAuthorityError(f"Store record is not a regular file: {path!r}.")
        try:
            with open(path, "rb") as source:
                return record_type.from_bytes(source.read())
        except StoreRecordError as error:
            raise StoreAuthorityError(f"Malformed Store record {path!r}: {error}") from error

    def _atomic_write(self, path: str, payload: bytes) -> None:
        parent = os.path.dirname(path)
        self._makedirs_durable(parent)
        fd, temporary = tempfile.mkstemp(prefix=".store-", dir=parent)
        try:
            with os.fdopen(fd, "wb") as target:
                written = target.write(payload)
                if written != len(payload):
                    raise OSError("Store authority temporary write was incomplete.")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
            self._fsync_directory(parent)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _fsync_directory(path: str) -> None:
        """Persist a directory entry or fail before claiming durable publication.

        Args:
            path: Existing directory containing a newly replaced or removed entry.

        Raises:
            StoreCapabilityError: If the active filesystem cannot fsync directory
                metadata required for the direct Store durability contract.

        Side Effects:
            Flushes filesystem metadata for ``path``. It does not read or alter
            Store authority.
        """

        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            fd = os.open(path, flags)
        except OSError as error:
            raise StoreCapabilityError(
                "DirStore requires directory fsync support for durable publication."
            ) from error
        try:
            os.fsync(fd)
        except OSError as error:
            raise StoreCapabilityError(
                "DirStore filesystem cannot persist directory entries."
            ) from error
        finally:
            os.close(fd)

    def _makedirs_durable(self, path: str) -> None:
        """Create a directory chain and persist every new parent entry.

        Args:
            path: Directory that must exist before authority publication.

        Raises:
            StoreCapabilityError: If a new directory entry cannot be persisted.

        Side Effects:
            Creates missing directories and fsyncs each containing directory.
        """

        missing = []
        current = os.path.abspath(path)
        while not os.path.exists(current):
            missing.append(current)
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        os.makedirs(path, exist_ok=True)
        for directory in reversed(missing):
            self._fsync_directory(os.path.dirname(directory))

    def _fsync_tree(self, root: str) -> None:
        """Persist staged files and directories before snapshot activation.

        Args:
            root: Complete staged snapshot directory.

        Raises:
            StoreCapabilityError: If staged content or directory metadata cannot
                be flushed before the final atomic directory replacement.

        Side Effects:
            Flushes every regular staged file and directory from leaves to root.
        """

        for directory, _dirs, files in os.walk(root, topdown=False):
            for name in files:
                path = os.path.join(directory, name)
                try:
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                except OSError as error:
                    raise StoreCapabilityError(
                        "DirStore cannot persist staged snapshot content."
                    ) from error
            self._fsync_directory(directory)

    @staticmethod
    def _digest_path(directory: str, digest: str) -> str:
        return os.path.join(directory, digest[:2], f"{digest}.record")

    def _definition_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "definitions"), digest)

    def _stored_root_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "stored-roots"), digest)

    def _snapshot_path(self, digest: str) -> str:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise StoreAuthorityError("snapshot digest is malformed.")
        return os.path.join(self.base_dir, "snapshots", digest[:2], digest)

    def _declaration_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "declarations"), digest)

    def _claim_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "claims"), digest)

    def _metadata_path(self, target) -> str:
        from ..reference_values import ObjectRef, StateRef

        if isinstance(target, ObjectRef):
            scope = "object"
        elif isinstance(target, StateRef):
            scope = "state"
        else:
            raise TypeError("metadata target must be an ObjectRef or StateRef.")
        return self._digest_path(os.path.join(self.base_dir, "metadata", scope), target.digest())

    def _lineage_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "lineage"), digest)

    @property
    def _staging_root(self) -> str:
        return os.path.join(self.base_dir, ".staging")

    def create_local_state_staging(self) -> str:
        """Create one Store-owned empty staging directory with an empty ``data`` root."""
        self.preflight_publication("create local-state staging", local_state=True)
        path = os.path.join(self._staging_root, uuid4().hex)
        os.makedirs(os.path.join(path, "data"))
        return path

    def _install_immutable(self, path: str, record, record_type):
        self.preflight_publication(f"write {record_type.schema}")
        with interprocess_lock(self._writer_lock_path):
            existing = self._read_file(path, record_type)
            if existing is not None:
                if existing != record:
                    if (
                        isinstance(existing, DefinitionRecord)
                        and isinstance(record, DefinitionRecord)
                        and existing.definition.graph_equal(record.definition)
                    ):
                        # Private CDef node allocations are runtime-local and do
                        # not distinguish immutable graph authority.
                        return existing
                    raise StoreAuthorityError(f"Immutable {record_type.schema} collision at {path!r}.")
                return existing
            self._atomic_write(path, record.to_bytes())
            return record

    def read_definition_record(self, digest: str) -> DefinitionRecord | None:
        """Read a validated DefinitionRecord and recompute its path key."""
        record = self._read_file(self._definition_path(digest), DefinitionRecord)
        if record is not None and record.digest != digest:
            raise StoreAuthorityError("DefinitionRecord digest does not match its direct path.")
        return record

    def write_definition_record(
            self, record: DefinitionRecord, *, stored_root: bool = True
    ) -> DefinitionRecord:
        """Install a definition and optional stored-root membership.

        Args:
            record: Complete immutable graph authority to install.
            stored_root: Whether this definition is independently queryable as a
                stored root rather than closure-only graph authority.

        Returns:
            The installed record. A new publication leaves a durable derived
            index marker; idempotent publication does not rewrite that marker.
        """
        if not isinstance(record, DefinitionRecord):
            raise TypeError("record must be a DefinitionRecord.")
        path = self._definition_path(record.digest)
        existed = self._read_file(path, DefinitionRecord) is not None
        installed = self._install_immutable(path, record, DefinitionRecord)
        root_installed = False
        if stored_root:
            root_record = StoredRootRecord(record.digest)
            root_path = self._stored_root_path(record.digest)
            root_installed = self._read_file(root_path, StoredRootRecord) is None
            self._install_immutable(root_path, root_record, StoredRootRecord)
        if not existed:
            self.mark_query_index_dirty(record.definition)
        elif root_installed:
            self.mark_query_index_dirty(record.definition)
        return installed

    def iter_definition_records(self) -> Iterable[DefinitionRecord]:
        """Yield all validated direct-layout DefinitionRecords in digest order."""
        root = Path(self.base_dir, "definitions")
        if not root.exists():
            return ()
        records = []
        for path in sorted(root.glob("*/*.record")):
            record = self._read_file(os.fspath(path), DefinitionRecord)
            if record is None or path.name != f"{record.digest}.record" or path.parent.name != record.digest[:2]:
                raise StoreAuthorityError(f"DefinitionRecord is stored under an invalid digest path: {path!s}.")
            records.append(record)
        return tuple(records)

    def iter_stored_root_records(self) -> Iterable[StoredRootRecord]:
        """Yield validated stored-root membership in digest order.

        Raises:
            StoreAuthorityError: If a marker path or DefinitionRecord target is
                malformed or missing.
        """
        root = Path(self.base_dir, "stored-roots")
        if not root.exists():
            return ()
        records = []
        for path in sorted(root.glob("*/*.record")):
            record = self._read_file(os.fspath(path), StoredRootRecord)
            if (
                    record is None
                    or path.name != f"{record.definition_digest}.record"
                    or path.parent.name != record.definition_digest[:2]
                    or self.read_definition_record(record.definition_digest) is None
            ):
                raise StoreAuthorityError(
                    f"StoredRootRecord is stored under an invalid digest path: {path!s}."
                )
            records.append(record)
        return tuple(records)

    def read_stored_root_record(self, digest: str) -> StoredRootRecord | None:
        """Read one validated digest-addressed stored-root membership record.

        Args:
            digest: Derived DefinitionRecord digest named by the requested
                StoredRootRecord path.

        Returns:
            The matching StoredRootRecord, or ``None`` when its direct marker
            path is absent.

        Raises:
            StoreAuthorityError: If ``digest`` is malformed, the direct marker is
                malformed or stored at the wrong digest, or its DefinitionRecord
                target is missing or malformed.

        Side Effects:
            None. This reads only the requested membership marker and referenced
            DefinitionRecord; it does not consult the derived query index or
            validate unrelated stored-root authority.
        """
        try:
            expected = StoredRootRecord(digest)
        except Exception as error:
            raise StoreAuthorityError("Stored-root digest is malformed.") from error
        record = self._read_file(self._stored_root_path(digest), StoredRootRecord)
        if record is None:
            return None
        if record != expected:
            raise StoreAuthorityError(
                "StoredRootRecord digest does not match its direct path."
            )
        if self.read_definition_record(digest) is None:
            raise StoreAuthorityError(
                "Stored-root membership targets a missing DefinitionRecord."
            )
        return record

    def read_definition(self, cdef):
        """Return the direct DefinitionRecord definition matching ``cdef``, if present.

        This supports exact query-index activation without consulting retired
        object roots. The deterministic DefinitionRecord digest gives the lookup
        its direct authority path.
        """
        record = self.read_definition_record(DefinitionRecord(cdef).digest)
        if record is not None and record.definition.graph_equal(cdef):
            return record.definition
        return None

    def query_index_record_metadata(self, cdef) -> tuple[str, str, int, int] | None:
        """Return direct-record metadata used only to validate a SQLite sidecar.

        Args:
            cdef: Definition represented by a prospective stored-root row.

        Returns:
            ``(record_digest, relative_path, size, mtime_ns)`` when a
            graph-equivalent DefinitionRecord or declaration, complete snapshot,
            or object-alias record supplies root authority, otherwise ``None``.

        Raises:
            StoreAuthorityError: If a present authority record is invalid.
            OSError: If a selected authority file cannot be inspected.

        Side Effects:
            Reads authority and file metadata without modifying it. Falls back
            to reference enumeration only when the direct definition is absent.
        """
        record = self.read_definition_record(DefinitionRecord(cdef).digest)
        if record is not None and record.definition.graph_equal(cdef):
            path = self._definition_path(record.digest)
        else:
            candidates = (
                (self.iter_declaration_records, lambda item: item.object_ref.definition,
                 lambda item: self._declaration_path(item.digest)),
                (self.iter_state_ref_records, lambda item: item.state_ref.definition,
                 lambda item: os.path.join(self._snapshot_path(item.digest), "state-ref.record")),
                (self.iter_object_alias_records, lambda item: item.object_ref.definition,
                 lambda item: self._ref_path("objects", f"{item.alias}.record")),
            )
            selected = next((
                (item, record_path(item))
                for records, definition, record_path in candidates
                for item in records() if definition(item).graph_equal(cdef)
            ), None)
            if selected is None:
                return None
            record, path = selected
        metadata = os.stat(path)
        return (
            record.object_ref.digest() if isinstance(record, ObjectAliasRecord) else record.digest,
            os.path.relpath(path, self.base_dir).replace(os.sep, "/"),
            metadata.st_size,
            metadata.st_mtime_ns,
        )

    def _validate_local_state_dir(self, directory: str, manifest: LocalStateManifest) -> None:
        expected = {"data", "def.pkl", "manifest.record"}
        try:
            root_mode = os.lstat(directory).st_mode
            entries = {entry.name: entry for entry in os.scandir(directory)}
        except FileNotFoundError as error:
            raise StoreAuthorityError("local state staging directory is missing.") from error
        if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
            raise StoreAuthorityError("local state directory must be a real directory.")
        if set(entries) != expected:
            raise StoreAuthorityError("local state directory must contain exactly data/, def.pkl, and manifest.record.")
        if not entries["data"].is_dir(follow_symlinks=False):
            raise StoreAuthorityError("local state data entry must be a real directory.")
        if not entries["def.pkl"].is_file(follow_symlinks=False) or not entries["manifest.record"].is_file(follow_symlinks=False):
            raise StoreAuthorityError("local state metadata entries must be regular files.")
        stored_manifest = self._read_file(os.path.join(directory, "manifest.record"), LocalStateManifest)
        if stored_manifest != manifest:
            raise StoreAuthorityError("local state manifest bytes do not match requested manifest authority.")
        definition_bytes = Path(directory, "def.pkl").read_bytes()
        if hashlib.sha256(definition_bytes).hexdigest() != manifest.definition_file_digest:
            raise StoreAuthorityError("local state definition file bytes do not match manifest authority.")
        definition = self._read_file(os.path.join(directory, "def.pkl"), DefinitionRecord)
        if definition is None or definition.digest != manifest.definition_digest or definition.graph_hash != manifest.graph_hash:
            raise StoreAuthorityError("local state definition does not match manifest graph and definition digests.")
        try:
            manifest.validate_payload(os.path.join(directory, "data"))
        except StoreRecordError as error:
            raise StoreAuthorityError(f"local state payload is invalid: {error}") from error

    def iter_state_ref_records(self) -> Iterable[StateRefRecord]:
        """Yield only complete v3 snapshot StateRef authority in digest order."""
        root = Path(self.base_dir, "snapshots")
        if not root.exists():
            return ()
        records = []
        for path in sorted(root.glob("*/*")):
            if not path.is_dir() or path.parent.name != path.name[:2]:
                raise StoreAuthorityError(f"Snapshot is stored under an invalid digest path: {path!s}.")
            snapshot = self._read_snapshot(path.name)
            assert snapshot is not None
            records.append(snapshot[0])
        return tuple(records)

    @staticmethod
    def _read_json(path: str):
        from dryml.formats import canonical_json_load_bytes

        try:
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise StoreAuthorityError("snapshot metadata is not a regular file.")
            with open(path, "rb") as source:
                return canonical_json_load_bytes(source.read(), max_depth=64, max_nodes=131072, max_entries=65536, max_string=4096, max_int_bits=4096)
        except FileNotFoundError:
            return None
        except StoreAuthorityError:
            raise
        except Exception as error:
            raise StoreAuthorityError("snapshot JSON authority is malformed.") from error

    @staticmethod
    def _write_json(path: str, value) -> None:
        from dryml.formats import canonical_json_bytes

        payload = canonical_json_bytes(value, max_depth=64, max_nodes=131072, max_entries=65536, max_string=4096, max_int_bits=4096)
        Path(path).write_bytes(payload)

    def read_metadata(self, target):
        """Return detached current metadata for one exact reference target.

        Args:
            target: Exact ObjectRef or StateRef attachment scope.

        Returns:
            Detached mapping, or ``None`` when no current mapping exists.

        Raises:
            TypeError: If ``target`` is unsupported.
            StoreAuthorityError: If a present metadata record is malformed.

        Side Effects:
            Reads only the direct metadata sidecar, not snapshot payloads or
            derived indexes. The result has no open-handle lifetime.
        """

        from ..metadata import decode_current_annotations

        record = self._read_json(self._metadata_path(target))
        return None if record is None else decode_current_annotations(record, target)

    def write_metadata(self, target, values) -> None:
        """Atomically replace one direct current metadata mapping.

        Args:
            target: Exact ObjectRef or StateRef attachment scope.
            values: Valid complete metadata mapping to install.

        Returns:
            ``None``.

        Raises:
            TypeError: If inputs are unsupported.
            ValueError: If metadata values violate codec bounds.
            StoreCapabilityError: If this filesystem cannot provide publication.

        Side Effects:
            Serializes cooperating writers and atomically replaces one sidecar.
            It does not alter snapshots, lineage metadata, or payload files.
        """

        from dryml.formats import canonical_json_bytes
        from ..metadata import encode_current_annotations

        record = encode_current_annotations(target, values)
        payload = canonical_json_bytes(
            record, max_depth=64, max_nodes=131072, max_entries=65536,
            max_string=4096, max_int_bits=4096,
        )
        self.preflight_publication("write current metadata")
        with self.writer_lock():
            # The derived index must become observably stale before authority can
            # expose a new whole mapping.
            self.mark_query_index_dirty(metadata_target=target)
            self._atomic_write(self._metadata_path(target), payload)

    def delete_metadata(self, target) -> bool:
        """Atomically remove one direct current mapping.

        Args:
            target: Exact ObjectRef or StateRef attachment scope.

        Returns:
            ``True`` when the sidecar existed and was removed, otherwise ``False``.

        Raises:
            TypeError: If ``target`` is unsupported.
            StoreCapabilityError: If this filesystem cannot provide publication.

        Side Effects:
            Serializes cooperating writers and preserves all target, snapshot,
            lineage, and payload authority.
        """

        self.preflight_publication("delete current metadata")
        path = self._metadata_path(target)
        with self.writer_lock():
            try:
                os.stat(path)
            except FileNotFoundError:
                return False
            try:
                # Absence is indexed metadata too, so invalidate before removal.
                self.mark_query_index_dirty(metadata_target=target)
                os.unlink(path)
                self._fsync_directory(os.path.dirname(path))
            except FileNotFoundError:
                return False
            return True

    def read_lineage_metadata(self, target):
        """Read detached immutable lineage evidence without treating absence as corruption.

        Args:
            target: Exact ObjectRef whose lineage fact is requested.

        Returns:
            LineageMetadata, or ``None`` when the direct sidecar is absent.

        Raises:
            TypeError: If ``target`` is unsupported.
            StoreAuthorityError: If a present sidecar is malformed.

        Side Effects:
            Reads only the direct lineage sidecar; snapshot and payload authority
            remain unchanged.
        """

        from ..metadata import decode_lineage_metadata

        record = self._read_json(self._lineage_path(target.digest()))
        return None if record is None else decode_lineage_metadata(record, target)

    def write_lineage_metadata(self, value):
        """Install one immutable direct lineage fact.

        Args:
            value: Valid LineageMetadata to install.

        Returns:
            The installed fact, or an equal existing fact.

        Raises:
            TypeError: If ``value`` is unsupported.
            ValueError: If its lineage fields are invalid.
            StoreAuthorityError: If unequal evidence already exists.

        Side Effects:
            Serializes cooperating writers and writes at most once. It does not
            change current annotations, snapshot captures, or payload files.
        """

        from dryml.formats import canonical_json_bytes
        from ..metadata import encode_lineage_metadata

        record = encode_lineage_metadata(value)
        payload = canonical_json_bytes(
            record, max_depth=64, max_nodes=131072, max_entries=65536,
            max_string=4096, max_int_bits=4096,
        )
        self.preflight_publication("write lineage metadata")
        path = self._lineage_path(value.object_ref.digest())
        with self.writer_lock():
            existing = self.read_lineage_metadata(value.object_ref)
            if existing is not None:
                if existing != value:
                    raise StoreAuthorityError("lineage write-once authority conflicts with existing evidence.")
                return existing
            self.mark_query_index_dirty(metadata_target=value.object_ref)
            self._atomic_write(path, payload)
            return value

    def _read_snapshot(
            self, digest: str, *, payloads: bool = False,
            directory: str | None = None):
        """Validate one v3 snapshot association without opening payloads by default."""

        from dryml.core.metadata import decode_snapshot_metadata
        from dryml.records import decode_record
        from ..utils.graph.path import GraphPath

        directory = self._snapshot_path(digest) if directory is None else directory
        try:
            mode = os.lstat(directory).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise StoreAuthorityError("snapshot authority is not a real directory.")
            entries = {entry.name: entry for entry in os.scandir(directory)}
        except FileNotFoundError:
            return None
        except StoreAuthorityError:
            raise
        except OSError as error:
            raise StoreAuthorityError("snapshot authority is inaccessible.") from error
        required = {"state-ref.record", "placement.json", "metadata.json", "snapshot.json"}
        if not required.issubset(entries) or any(
                not entries[name].is_file(follow_symlinks=False) for name in required):
            raise StoreAuthorityError("snapshot lacks required complete metadata siblings.")
        record = self._read_file(os.path.join(directory, "state-ref.record"), StateRefRecord)
        if record is None or record.digest != digest:
            raise StoreAuthorityError("snapshot StateRef record does not match its directory.")
        placement_envelope = self._read_json(os.path.join(directory, "placement.json"))
        metadata_envelope = self._read_json(os.path.join(directory, "metadata.json"))
        snapshot_envelope = self._read_json(os.path.join(directory, "snapshot.json"))
        try:
            placement_record = decode_record(placement_envelope)
            snapshot_record = decode_record(snapshot_envelope)
            if placement_record.kind != "dryml.core.snapshot_placement" or placement_record.version != 1:
                raise ValueError("unsupported placement record")
            if snapshot_record.kind != "dryml.core.snapshot" or snapshot_record.version != 1:
                raise ValueError("unsupported snapshot association")
            placement = placement_record.data
            association = snapshot_record.data
            if set(placement) != {"state_ref_digest", "local", "children"} or set(association) != {"state_ref_digest", "placement_id", "metadata_id"}:
                raise ValueError("snapshot records are not closed")
            if placement["state_ref_digest"] != digest or association["state_ref_digest"] != digest:
                raise ValueError("snapshot records target another StateRef")
            if association["placement_id"] != placement_envelope.get("id") or association["metadata_id"] != metadata_envelope.get("id"):
                raise ValueError("snapshot association ids do not match sibling records")
            metadata = decode_snapshot_metadata(metadata_envelope, record.state_ref)
        except Exception as error:
            raise StoreAuthorityError("snapshot metadata association is malformed.") from error
        if metadata.state_ref != record.state_ref:
            raise StoreAuthorityError("snapshot metadata does not match StateRef authority.")
        if not isinstance(placement["local"], list) or not isinstance(placement["children"], list):
            raise StoreAuthorityError("snapshot placement is malformed.")
        local = {}
        for item in placement["local"]:
            if not isinstance(item, dict) or set(item) != {"path", "state_hash", "directory"}:
                raise StoreAuthorityError("snapshot local placement entry is malformed.")
            path = GraphPath.from_data(item["path"])
            if path in local or record.state_ref.states.get(path) != item["state_hash"]:
                raise StoreAuthorityError("snapshot local placement does not match StateRef states.")
            directory_name = item["directory"]
            if not isinstance(directory_name, str) or not directory_name.startswith("local-state/") or ".." in directory_name.split("/"):
                raise StoreAuthorityError("snapshot local placement directory is malformed.")
            local[path] = directory_name
        covered = dict(local)
        for item in placement["children"]:
            if not isinstance(item, dict) or set(item) != {"path", "state_ref_digest"}:
                raise StoreAuthorityError("snapshot child placement entry is malformed.")
            try:
                path = GraphPath.from_data(item["path"])
                child = record.state_ref.at(path)
            except Exception as error:
                raise StoreAuthorityError("snapshot child placement path is malformed.") from error
            if child.digest() != item["state_ref_digest"]:
                raise StoreAuthorityError("snapshot child placement does not match its projection.")
            for child_path, state_hash in child.states.items():
                target_path = path.join(child_path)
                if target_path in covered or record.state_ref.states.get(target_path) != state_hash:
                    raise StoreAuthorityError("snapshot placement overlaps or does not match StateRef states.")
                covered[target_path] = None
        if set(covered) != set(record.state_ref.states):
            raise StoreAuthorityError("snapshot placement does not cover every stateful path.")
        if payloads:
            for path, relative in local.items():
                manifest = self._read_file(os.path.join(directory, relative, "manifest.record"), LocalStateManifest)
                if manifest is None or manifest.state_hash != record.state_ref.states[path]:
                    raise StoreAuthorityError("snapshot payload manifest does not match placement.")
                self._validate_local_state_dir(os.path.join(directory, relative), manifest)
                expected = record.state_ref.object.at(path).definition
                definition = self._read_file(os.path.join(directory, relative, "def.pkl"), DefinitionRecord)
                if definition is None or not definition.definition.graph_equal(expected):
                    raise StoreAuthorityError("snapshot payload definition does not match StateRef path.")
        return record, metadata, local

    def read_state_ref_record(self, digest: str) -> StateRefRecord | None:
        """Read one StateRef only when its complete v3 snapshot validates."""

        snapshot = self._read_snapshot(digest)
        return None if snapshot is None else snapshot[0]

    def get_snapshot_directory(self, target):
        """Return the validated direct directory for one exact snapshot.

        Args:
            target: Exact StateRef to locate.

        Returns:
            Persistent absolute Path for the complete snapshot directory.

        Raises:
            TypeError: If ``target`` is not a StateRef.
            KeyError: If the snapshot is absent.
            StoreAuthorityError: If its association is malformed or incompatible.

        Side Effects:
            Validates metadata and placement records without opening payload bytes,
            materializing Objects, or changing authority.
        """

        from ..reference_values import StateRef

        if not isinstance(target, StateRef):
            raise TypeError("target must be a StateRef.")
        snapshot = self._read_snapshot(target.digest())
        if snapshot is None:
            raise KeyError(target.digest())
        if snapshot[0].state_ref != target:
            raise StoreAuthorityError("snapshot directory has incompatible StateRef authority.")
        return Path(self._snapshot_path(target.digest()))

    def read_snapshot_metadata(self, digest: str):
        """Return captured metadata for one validated complete snapshot.

        Args:
            digest: Exact StateRef digest naming the snapshot directory.

        Returns:
            Detached SnapshotMetadata, or ``None`` when the directory is absent.

        Raises:
            StoreAuthorityError: If the snapshot association is malformed.

        Side Effects:
            Reads metadata association without opening payload bytes or changing
            Store authority.
        """

        snapshot = self._read_snapshot(digest)
        return None if snapshot is None else snapshot[1]

    def discard_local_state_staging(self, handle: object) -> None:
        """Idempotently remove only Store-owned unpublished local-state staging."""

        try:
            path = os.path.abspath(os.fspath(handle))
            root = os.path.realpath(self._staging_root)
            if os.path.commonpath((root, os.path.realpath(path))) != root:
                raise StoreAuthorityError("local-state staging is not owned by this Store.")
            shutil.rmtree(path)
        except FileNotFoundError:
            return

    def prepare_local_state(self, source: object, manifest: LocalStateManifest) -> LocalStateSource:
        """Validate owned completed staging before it is copied into a snapshot."""

        path = os.path.abspath(os.fspath(source))
        root = os.path.realpath(self._staging_root)
        if os.path.commonpath((root, os.path.realpath(path))) != root:
            raise StoreAuthorityError("local-state staging must be created by the selected Store.")
        self._validate_local_state_dir(path, manifest)
        return LocalStateSource(self, path, manifest)

    def prepare_rebound_local_state(self, source: LocalStateSource, target_definition) -> LocalStateSource:
        """Copy validated bytes into this Store's staging and bind a fork definition."""

        if not isinstance(source, LocalStateSource):
            raise TypeError("source must be a LocalStateSource.")
        stage = self.create_local_state_staging()
        try:
            shutil.rmtree(stage)
            shutil.copytree(os.fspath(source.handle), stage)
            record = DefinitionRecord(target_definition)
            definition_bytes = record.to_bytes()
            Path(stage, "def.pkl").write_bytes(definition_bytes)
            manifest = LocalStateManifest(source.manifest.codec, record.graph_hash, record.digest, hashlib.sha256(definition_bytes).hexdigest(), source.manifest.files)
            if manifest.state_hash != source.manifest.state_hash:
                raise StoreAuthorityError("rebound local state changed its payload identity.")
            Path(stage, "manifest.record").write_bytes(manifest.to_bytes())
            return self.prepare_local_state(stage, manifest)
        except BaseException:
            self.discard_local_state_staging(stage)
            raise

    def validate_local_state(self, reference, path) -> LocalStateManifest:
        """Fully validate one exact snapshot-local payload manifest and bytes."""

        from ..reference_values import StateRef
        from ..utils.graph.path import normalize_path

        if not isinstance(reference, StateRef):
            raise TypeError("reference must be a StateRef.")
        path = normalize_path(path)
        snapshot = self._read_snapshot(reference.digest(), payloads=True)
        if snapshot is None or snapshot[0].state_ref != reference:
            raise KeyError(reference.digest())
        relative = snapshot[2].get(path)
        if relative is None:
            raise KeyError(path)
        manifest = self._read_file(os.path.join(self._snapshot_path(reference.digest()), relative, "manifest.record"), LocalStateManifest)
        assert manifest is not None
        return manifest

    def open_local_state(self, reference, path) -> LocalStateSource:
        """Open one validated snapshot-local payload source for restoration/copy."""

        manifest = self.validate_local_state(reference, path)
        snapshot = self._read_snapshot(reference.digest())
        assert snapshot is not None
        return LocalStateSource(self, os.path.join(self._snapshot_path(reference.digest()), snapshot[2][path]), manifest)

    def publish_snapshot(
            self, reference, *, evidence, annotations=None, local_states,
            children=None, _before_annotation_write=None):
        """Install a complete v3 snapshot directory under the writer fence.

        Args:
            reference: Exact StateRef to install.
            evidence: Fresh SnapshotCapture or matching copied SnapshotMetadata.
            annotations: Optional SaveAnnotations current-map replacements for root
                ObjectRef and StateRef scopes.
            local_states: Mapping of locally owned paths to validated sources.
            children: Optional mapping of child projection paths to StateRefs.
            _before_annotation_write: Internal phase callback invoked immediately
                before each requested current-metadata write.

        Returns:
            Detached SnapshotMetadata for the installed or equal existing snapshot.

        Raises:
            TypeError: If publication inputs are unsupported.
            StoreAuthorityError: If records, sources, coverage, or immutable
                existing snapshot evidence conflict.
            StoreCapabilityError: If direct publication guarantees are unavailable.

        Side Effects:
            Atomically installs a complete immutable directory. Explicit
            annotations are applied as current LWW sidecars after installation;
            existing captured annotations are never refreshed. Cooperating writers
            are serialized and staging is removed after success or failure.
        """

        from ..metadata import SaveAnnotations, SnapshotCapture, SnapshotMetadata, encode_snapshot_metadata
        from ..reference_values import StateRef
        from ..utils.graph.path import GraphPath
        from dryml.records import GenericRecord, encode_record

        if children is None:
            children = {}
        if not isinstance(reference, StateRef) or not isinstance(local_states, dict) or not isinstance(children, Mapping):
            raise TypeError("snapshot publication requires a StateRef, local-state mapping, and child projection mapping.")
        if annotations is not None and not isinstance(annotations, SaveAnnotations):
            raise TypeError("annotations must be a SaveAnnotations or None.")
        self.preflight_publication("publish snapshot", local_state=True)
        covered = set(local_states)
        child_entries = []
        for path, child in children.items():
            if not isinstance(path, GraphPath) or child != reference.at(path):
                raise StoreAuthorityError("snapshot child projection does not match its parent StateRef.")
            for child_path in child.states:
                target_path = path.join(child_path)
                if target_path in covered:
                    raise StoreAuthorityError("snapshot child projection overlaps local payload authority.")
                covered.add(target_path)
            child_entries.append({"path": path.to_data(), "state_ref_digest": child.digest()})
        if covered != set(reference.states):
            raise StoreAuthorityError("snapshot publication must cover every state path locally or by child projection.")
        os.makedirs(self._staging_root, exist_ok=True)
        stage = tempfile.mkdtemp(prefix="snapshot-", dir=self._staging_root)
        try:
            placement_entries = []
            for path, source in sorted(local_states.items(), key=lambda item: str(item[0])):
                if not isinstance(path, GraphPath) or not isinstance(source, LocalStateSource):
                    raise TypeError("snapshot local states must map GraphPath to LocalStateSource.")
                if source.manifest.state_hash != reference.states[path]:
                    raise StoreAuthorityError("snapshot source does not match StateRef state hash.")
                target_definition = reference.object.at(path).definition
                source_definition = self._read_file(os.path.join(os.fspath(source.handle), "def.pkl"), DefinitionRecord)
                if source_definition is None or not source_definition.definition.graph_equal(target_definition):
                    raise StoreAuthorityError("snapshot source definition does not match StateRef path.")
                relative = f"local-state/{source.manifest.graph_hash}/{source.manifest.state_hash}"
                destination = os.path.join(stage, relative)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                if not os.path.exists(destination):
                    shutil.copytree(os.fspath(source.handle), destination)
                self._validate_local_state_dir(destination, source.manifest)
                placement_entries.append({"path": path.to_data(), "state_hash": source.manifest.state_hash, "directory": relative})
            target = self._snapshot_path(reference.digest())
            with self.writer_lock():
                if isinstance(evidence, SnapshotCapture):
                    object_values = self.read_metadata(reference.object)
                    state_values = self.read_metadata(reference)
                    if annotations is not None:
                        if annotations.object is not None:
                            object_values = annotations.object
                        if annotations.state is not None:
                            state_values = annotations.state
                    metadata = SnapshotMetadata(
                        reference, evidence.lineages, evidence.saved_at,
                        evidence.environment, evidence.environment_status,
                        evidence.requirements, evidence.requirements_status,
                        evidence.requirements_coverage, evidence.diagnostics,
                        object_values, state_values,
                    )
                elif isinstance(evidence, SnapshotMetadata) and evidence.state_ref == reference:
                    metadata = evidence
                else:
                    raise StoreAuthorityError("snapshot evidence does not match its StateRef.")
                Path(stage, "state-ref.record").write_bytes(StateRefRecord(reference).to_bytes())
                placement = encode_record(GenericRecord("dryml.core.snapshot_placement", 1, {"state_ref_digest": reference.digest(), "local": placement_entries, "children": child_entries}))
                metadata_record = encode_snapshot_metadata(metadata)
                association = encode_record(GenericRecord("dryml.core.snapshot", 1, {"state_ref_digest": reference.digest(), "placement_id": placement["id"], "metadata_id": metadata_record["id"]}))
                self._write_json(os.path.join(stage, "placement.json"), placement)
                self._write_json(os.path.join(stage, "metadata.json"), metadata_record)
                self._write_json(os.path.join(stage, "snapshot.json"), association)
                self._read_snapshot_from_directory(stage, reference, payloads=True)
                existing = self._read_snapshot(reference.digest())
                if existing is not None:
                    if existing[0].state_ref != reference:
                        raise StoreAuthorityError("snapshot write-once authority conflicts with existing evidence.")
                    if isinstance(evidence, SnapshotMetadata) and existing[1] != metadata:
                        raise StoreAuthorityError("snapshot write-once authority conflicts with existing evidence.")
                    if annotations is not None:
                        if annotations.object is not None:
                            if _before_annotation_write is not None:
                                _before_annotation_write("object")
                            self.write_metadata(reference.object, annotations.object)
                        if annotations.state is not None:
                            if _before_annotation_write is not None:
                                _before_annotation_write("state")
                            self.write_metadata(reference, annotations.state)
                    return existing[1]
                self.mark_query_index_dirty(metadata_target=reference)
                self._makedirs_durable(os.path.dirname(target))
                self._fsync_tree(stage)
                os.replace(stage, target)
                self._fsync_directory(os.path.dirname(target))
                installed = self._read_snapshot(reference.digest(), payloads=True)
                if installed is None:
                    raise StoreAuthorityError("snapshot install did not survive read-back.")
                if annotations is not None:
                    if annotations.object is not None:
                        if _before_annotation_write is not None:
                            _before_annotation_write("object")
                        self.write_metadata(reference.object, annotations.object)
                    if annotations.state is not None:
                        if _before_annotation_write is not None:
                            _before_annotation_write("state")
                        self.write_metadata(reference, annotations.state)
                return installed[1]
        finally:
            if os.path.isdir(stage):
                shutil.rmtree(stage, ignore_errors=True)

    def _read_snapshot_from_directory(self, directory: str, reference, *, payloads: bool):
        """Validate candidate staging using the same path-independent snapshot checks."""

        return self._read_snapshot(
            reference.digest(), payloads=payloads, directory=directory,
        )

    def read_declaration_record(self, digest: str) -> DeclarationRecord | None:
        record = self._read_file(self._declaration_path(digest), DeclarationRecord)
        if record is not None and record.digest != digest:
            raise StoreAuthorityError("DeclarationRecord digest does not match its direct path.")
        return record

    def write_declaration_record(self, record: DeclarationRecord) -> DeclarationRecord:
        """Install one immutable DeclarationRecord."""
        if not isinstance(record, DeclarationRecord):
            raise TypeError("record must be a DeclarationRecord.")
        path = self._declaration_path(record.digest)
        existed = self._read_file(path, DeclarationRecord) is not None
        installed = self._install_immutable(path, record, DeclarationRecord)
        if not existed:
            self.mark_query_index_dirty()
        return installed

    def iter_declaration_records(self) -> Iterable[DeclarationRecord]:
        """Yield validated declaration records in deterministic direct-path order."""
        root = Path(self.base_dir, "declarations")
        if not root.exists():
            return ()
        records = []
        for path in sorted(root.glob("*/*.record")):
            record = self._read_file(os.fspath(path), DeclarationRecord)
            if record is None or path.name != f"{record.digest}.record" or path.parent.name != record.digest[:2]:
                raise StoreAuthorityError(f"DeclarationRecord is stored under an invalid digest path: {path!s}.")
            records.append(record)
        return tuple(records)

    def read_claim_record(self, digest: str) -> ClaimRecord | None:
        record = self._read_file(self._claim_path(digest), ClaimRecord)
        if record is not None and record.object_digest != digest:
            raise StoreAuthorityError("ClaimRecord object digest does not match its direct path.")
        return record

    def write_claim_record(self, record: ClaimRecord) -> ClaimRecord:
        """Atomically replace a ClaimRecord under the Store writer lock."""
        if not isinstance(record, ClaimRecord):
            raise TypeError("record must be a ClaimRecord.")
        self.preflight_publication("write claim")
        with interprocess_lock(self._writer_lock_path):
            self._atomic_write(self._claim_path(record.object_digest), record.to_bytes())
        return record

    def _ref_path(self, *parts: str) -> str:
        return os.path.join(self.base_dir, "refs", *parts)

    def read_main_ref(self) -> MainRefRecord | None:
        """Read the current main-definition mutable reference."""
        return self._read_file(self._ref_path("main.record"), MainRefRecord)

    def write_main_ref(self, record: MainRefRecord) -> MainRefRecord:
        """Atomically replace the current main-definition mutable reference."""
        if not isinstance(record, MainRefRecord):
            raise TypeError("record must be a MainRefRecord.")
        self.preflight_publication("write main reference")
        with interprocess_lock(self._writer_lock_path):
            self._atomic_write(self._ref_path("main.record"), record.to_bytes())
        return record

    def read_object_alias(self, alias: str) -> ObjectAliasRecord | None:
        """Read one direct mutable object alias record."""
        return self._read_file(self._ref_path("objects", f"{alias}.record"), ObjectAliasRecord)

    def write_object_alias(self, record: ObjectAliasRecord) -> ObjectAliasRecord:
        """Atomically replace one direct mutable object alias record."""
        if not isinstance(record, ObjectAliasRecord):
            raise TypeError("record must be an ObjectAliasRecord.")
        self.preflight_publication("write object alias")
        with interprocess_lock(self._writer_lock_path):
            self._atomic_write(self._ref_path("objects", f"{record.alias}.record"), record.to_bytes())
        self.mark_query_index_dirty()
        return record

    def read_state_alias(self, object_digest: str, alias: str) -> StateAliasRecord | None:
        """Read one direct mutable StateRef alias record."""
        return self._read_file(self._ref_path("states", object_digest[:2], object_digest, f"{alias}.record"), StateAliasRecord)

    def write_state_alias(self, record: StateAliasRecord) -> StateAliasRecord:
        """Atomically replace one direct mutable StateRef alias record."""
        if not isinstance(record, StateAliasRecord):
            raise TypeError("record must be a StateAliasRecord.")
        self.preflight_publication("write state alias")
        digest = record.object_ref.digest()
        with interprocess_lock(self._writer_lock_path):
            self._atomic_write(self._ref_path("states", digest[:2], digest, f"{record.alias}.record"), record.to_bytes())
        self.mark_query_index_dirty()
        return record

    def iter_object_alias_records(self) -> Iterable[ObjectAliasRecord]:
        """Yield validated object aliases in deterministic path order."""
        root = Path(self.base_dir, "refs", "objects")
        if not root.exists():
            return ()
        return tuple(
            record for path in sorted(root.glob("*.record"))
            if (record := self._read_file(os.fspath(path), ObjectAliasRecord)) is not None
        )

    def iter_state_alias_records(self) -> Iterable[StateAliasRecord]:
        """Yield validated scoped state aliases in deterministic path order."""
        root = Path(self.base_dir, "refs", "states")
        if not root.exists():
            return ()
        return tuple(
            record for path in sorted(root.glob("*/*/*.record"))
            if (record := self._read_file(os.fspath(path), StateAliasRecord)) is not None
        )

    def catalog_key(self) -> str:
        """Return the stable path-backed identity used only by derived indexes."""
        return f"{type(self).__module__}.{type(self).__qualname__}:{self.base_dir}"

    def close(self) -> None:
        """Release this handle's SQLite connections without touching authority.

        Raises:
            RuntimeError: If an active Session resource cache retains this Store.
        """
        super().close()
        if self._query_index_instance is not None:
            self._query_index_instance.close()
            self._query_index_instance = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.base_dir!r})"


DirStore.publish_snapshot._dryml_annotation_phase_callback = True
