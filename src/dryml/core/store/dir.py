"""Direct-root filesystem implementation of current Store authority."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import replace
from uuid import uuid4
from pathlib import Path
from typing import Iterable, Literal

from .locking import interprocess_lock, supports_advisory_locking
from .records import (
    ClaimRecord, DeclarationRecord, DefinitionRecord, LocalStateManifest,
    MainRefRecord, ObjectAliasRecord, StateAliasRecord, StateRefRecord,
    StoreFormatRecord, StoreRecordError,
)
from .store import Store, StoreAuthorityError, StorePublicationCapabilities
from ..query.model import QueryIndexStatus, QueryIndexUnavailable, ReconcileReport
from ..query.sqlite import SQLiteQueryIndexConfig, sqlite_available
from ..query.sqlite.index import SQLiteStoreQueryIndex


QueryIndexPolicy = Literal["auto", "sqlite", "memory", "none"]


class DirStore(Store):
    """Persist current logical authority directly beneath one local filesystem root.

    Immutable records are digest sharded.  Local state lives at
    ``local-state/<hh>/<graph-hash>/<codec>-<manifest-digest>/``.  All mutable
    references use a sibling temporary file and atomic replacement while the
    Store writer lock is held.  The old ``objects/`` generation layout is not a
    recognized authority format.
    """

    def __init__(
            self,
            base_dir: str | os.PathLike[str],
            *,
            query_index: QueryIndexPolicy | SQLiteQueryIndexConfig = "auto"):
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
        self._initialize_format()

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

    def mark_query_index_dirty(self, cdef=None) -> str | None:
        """Publish a durable marker after an authoritative definition mutation.

        Args:
            cdef: Optional definition whose immutable DefinitionRecord changed.

        Returns:
            The marker path, or ``None`` when this Store has no SQLite sidecar.
        """
        if self._query_index_policy not in {"auto", "sqlite"}:
            return None
        os.makedirs(self.dryml_dir, exist_ok=True)
        key = "dirty"
        if cdef is not None:
            key = DefinitionRecord(cdef).digest
        marker_path = f"{self.query_index_dirty_path}.{uuid4().hex}"
        fd, temporary_path = tempfile.mkstemp(prefix=".query-index-dirty-", dir=self.dryml_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                target.write(f"{key}\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_path, marker_path)
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
        for marker in self._query_index_dirty_markers():
            try:
                os.unlink(marker)
            except FileNotFoundError:
                pass

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

    def _initialize_format(self) -> None:
        root = Path(self.base_dir)
        if root.exists() and not root.is_dir():
            raise StoreAuthorityError(f"DirStore root is not a directory: {self.base_dir!r}.")
        if not root.exists():
            root.mkdir(parents=True)
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
        os.makedirs(parent, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".store-", dir=parent)
        try:
            with os.fdopen(fd, "wb") as target:
                written = target.write(payload)
                if written != len(payload):
                    raise OSError("Store authority temporary write was incomplete.")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _digest_path(directory: str, digest: str) -> str:
        return os.path.join(directory, digest[:2], f"{digest}.record")

    def _definition_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "definitions"), digest)

    def _state_ref_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "state-refs"), digest)

    def _declaration_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "declarations"), digest)

    def _claim_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "claims"), digest)

    def _local_state_path(self, graph_hash: str, state_hash: str) -> str:
        try:
            codec, digest = state_hash.split("-", 1)
        except ValueError as error:
            raise StoreAuthorityError("local state hash must be '<codec>-<digest>'.") from error
        if len(graph_hash) != 64 or len(digest) != 64 or not codec.isalnum() or not codec:
            raise StoreAuthorityError("local state graph hash or state hash is malformed.")
        return os.path.join(self.base_dir, "local-state", graph_hash[:2], graph_hash, state_hash)

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

    def write_definition_record(self, record: DefinitionRecord) -> DefinitionRecord:
        """Install an immutable DefinitionRecord and dirty SQLite after publication.

        Args:
            record: Complete immutable graph authority to install.

        Returns:
            The installed record. A new publication leaves a durable derived
            index marker; idempotent publication does not rewrite that marker.
        """
        if not isinstance(record, DefinitionRecord):
            raise TypeError("record must be a DefinitionRecord.")
        path = self._definition_path(record.digest)
        existed = self._read_file(path, DefinitionRecord) is not None
        installed = self._install_immutable(path, record, DefinitionRecord)
        if not existed:
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

    def read_definition(self, cdef):
        """Return the direct DefinitionRecord definition matching ``cdef``, if present.

        This supports exact query-index activation without consulting retired
        object roots. The deterministic DefinitionRecord digest gives the lookup
        its direct authority path.
        """
        try:
            record = self.read_definition_record(DefinitionRecord(cdef).digest)
        except Exception:
            return None
        if record is not None and record.definition == cdef:
            return record.definition
        return None

    def query_index_record_metadata(self, cdef) -> tuple[str, str, int, int] | None:
        """Return direct-record metadata used only to validate a SQLite sidecar.

        Args:
            cdef: Definition represented by a prospective stored-root row.

        Returns:
            ``(record_digest, relative_path, size, mtime_ns)`` when the exact
            immutable DefinitionRecord exists, otherwise ``None``.
        """
        record = self.read_definition_record(DefinitionRecord(cdef).digest)
        if record is None or record.definition != cdef:
            return None
        path = self._definition_path(record.digest)
        metadata = os.stat(path)
        return (
            record.digest,
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

    def install_local_state(self, source_dir: object, manifest: LocalStateManifest) -> LocalStateManifest:
        """Atomically install a complete same-filesystem immutable local state."""
        if not isinstance(manifest, LocalStateManifest):
            raise TypeError("manifest must be a LocalStateManifest.")
        self.preflight_publication("install local state", local_state=True)
        source_dir = os.path.abspath(os.fspath(source_dir))
        if os.stat(source_dir).st_dev != os.stat(self.base_dir).st_dev:
            raise StoreAuthorityError("local state staging must be on the Store filesystem.")
        staging_root = os.path.realpath(self._staging_root)
        if os.path.commonpath((staging_root, os.path.realpath(source_dir))) != staging_root:
            raise StoreAuthorityError("local state staging must be created by the selected Store.")
        self._validate_local_state_dir(source_dir, manifest)
        destination = self._local_state_path(manifest.graph_hash, manifest.state_hash)
        with interprocess_lock(self._writer_lock_path):
            if os.path.lexists(destination):
                try:
                    self._validate_local_state_dir(destination, manifest)
                except Exception:
                    # A partial immutable destination is never authority and may
                    # only be discarded while the Store writer is serialized.
                    shutil.rmtree(destination)
                else:
                    return manifest
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            os.replace(source_dir, destination)
            try:
                self._validate_local_state_dir(destination, manifest)
            except BaseException:
                # The destination is still incomplete/non-authoritative; remove
                # it before releasing the writer lock.
                shutil.rmtree(destination, ignore_errors=True)
                raise
        return manifest

    def open_local_state(self, graph_hash: str, state_hash: str) -> str:
        """Validate and return the direct local-state directory for restoration/copy."""
        path = self._local_state_path(graph_hash, state_hash)
        try:
            manifest = self._read_file(os.path.join(path, "manifest.record"), LocalStateManifest)
        except StoreAuthorityError:
            raise
        if manifest is None or manifest.graph_hash != graph_hash or manifest.state_hash != state_hash:
            raise StoreAuthorityError("local state directory is missing or does not match its requested identity.")
        self._validate_local_state_dir(path, manifest)
        return path

    def validate_local_state(self, definition, state_hash: str) -> str:
        """Verify one state directory has the requested graph definition and roles."""
        path = self.open_local_state(definition.graph_hash(), state_hash)
        stored = self._read_file(os.path.join(path, "def.pkl"), DefinitionRecord)
        if stored is None or not stored.definition.graph_equal(definition):
            raise StoreAuthorityError("local state definition does not match the requested graph.")
        if stored.roles != DefinitionRecord(definition).roles:
            raise StoreAuthorityError("local state stateful roles do not match the requested graph.")
        return path

    def copy_local_state_from(self, source: Store, definition, state_hash: str) -> LocalStateManifest:
        """Copy a source-validated immutable state through target-owned staging."""
        self.preflight_publication("copy local state", local_state=True)
        source_path = source.validate_local_state(definition, state_hash)
        if not isinstance(source_path, (str, os.PathLike)):
            raise StoreAuthorityError("source Store does not expose a copyable local-state handle.")
        stage = self.create_local_state_staging()
        try:
            shutil.rmtree(stage)
            shutil.copytree(os.fspath(source_path), stage)
            manifest = self._read_file(os.path.join(stage, "manifest.record"), LocalStateManifest)
            if manifest is None:
                raise StoreAuthorityError("source local state lacks a manifest.")
            self.install_local_state(stage, manifest)
            return manifest
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    def read_state_ref_record(self, digest: str) -> StateRefRecord | None:
        record = self._read_file(self._state_ref_path(digest), StateRefRecord)
        if record is not None and record.digest != digest:
            raise StoreAuthorityError("StateRefRecord digest does not match its direct path.")
        return record

    def write_state_ref_record(self, record: StateRefRecord) -> StateRefRecord:
        """Install an immutable StateRefRecord after caller closure preflight."""
        if not isinstance(record, StateRefRecord):
            raise TypeError("record must be a StateRefRecord.")
        return self._install_immutable(self._state_ref_path(record.digest), record, StateRefRecord)

    def iter_state_ref_records(self) -> Iterable[StateRefRecord]:
        """Yield validated StateRef records in deterministic direct-path order."""
        root = Path(self.base_dir, "state-refs")
        if not root.exists():
            return ()
        records = []
        for path in sorted(root.glob("*/*.record")):
            record = self._read_file(os.fspath(path), StateRefRecord)
            if record is None or path.name != f"{record.digest}.record" or path.parent.name != record.digest[:2]:
                raise StoreAuthorityError(f"StateRefRecord is stored under an invalid digest path: {path!s}.")
            records.append(record)
        return tuple(records)

    def read_declaration_record(self, digest: str) -> DeclarationRecord | None:
        record = self._read_file(self._declaration_path(digest), DeclarationRecord)
        if record is not None and record.digest != digest:
            raise StoreAuthorityError("DeclarationRecord digest does not match its direct path.")
        return record

    def write_declaration_record(self, record: DeclarationRecord) -> DeclarationRecord:
        """Install one immutable DeclarationRecord."""
        if not isinstance(record, DeclarationRecord):
            raise TypeError("record must be a DeclarationRecord.")
        return self._install_immutable(self._declaration_path(record.digest), record, DeclarationRecord)

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
        return record

    def catalog_key(self) -> str:
        """Return the stable path-backed identity used only by derived indexes."""
        return f"{type(self).__module__}.{type(self).__qualname__}:{self.base_dir}"

    def close(self) -> None:
        """Release this handle's SQLite connections without touching authority."""
        if self._query_index_instance is not None:
            self._query_index_instance.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.base_dir!r})"
