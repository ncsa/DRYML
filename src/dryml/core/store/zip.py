"""Buffered Zip implementation of the current logical Store interface."""

from __future__ import annotations

import hashlib
import os
import stat
from io import IOBase
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
from threading import RLock
import zipfile
from contextlib import contextmanager

from .dir import DirStore, _REMOVED_ENTRY_PREFIX
from .records import StoreFormatRecord, StoreRecordError
from ...locking import interprocess_lock
from .store import StoreAuthorityError, StorePublicationCapabilities


def _is_file_like(value) -> bool:
    return isinstance(value, IOBase) or all(callable(getattr(value, name, None)) for name in ("read", "write", "seek", "truncate"))


class ZipStoreConflictError(StoreAuthorityError):
    """Raised when a buffered path-backed archive changed before commit."""


class ZipStore(DirStore):
    """Expose current logical records through one buffered archive transaction.

    Path-backed archives publish by atomically replacing a complete sibling zip
    after comparing the archive bytes observed at open.  File-like destinations
    remain readable but explicitly reject writable authority because they cannot
    make the required replacement guarantee.
    """

    def __init__(
            self, zip_dest: str | Path | IOBase, *, _existing_only: bool = False,
            _authority_prevalidated: bool = False):
        self.zip_dest = zip_dest
        self._archive_path_value = (
            None if _is_file_like(zip_dest) else os.path.abspath(os.fspath(zip_dest))
        )
        self._tmp = tempfile.TemporaryDirectory()
        self._archive_dirty = False
        # One handle owns one extracted transaction.  This lock is acquired
        # before the inherited Store writer/archive locks so commit cannot clear
        # dirty state while another same-handle mutation is still publishing.
        self._transaction_lock = RLock()
        self._initializing = True
        self._file_like = _is_file_like(zip_dest)
        try:
            if _existing_only:
                if self._file_like:
                    raise StoreAuthorityError("Existing-only ZipStore opening requires a path-backed archive.")
                if not _authority_prevalidated:
                    self._validate_existing_archive(self._archive_path)
            self._extract_if_present()
            super().__init__(self._tmp.name, query_index="memory", _existing_only=_existing_only)
        except BaseException:
            self._tmp.cleanup()
            raise
        self._initializing = False
        self._archive_baseline = None if self._file_like else self._archive_identity()
        self._archive_evidence = None if self._file_like else self._physical_archive_evidence()
        self._closed_handle = False

    @classmethod
    def open_existing(cls, zip_dest: str | Path) -> "ZipStore":
        """Open a committed current-format archive without creating an archive.

        Args:
            zip_dest: Existing path-backed archive location.

        Returns:
            A matching Session-cached buffered handle when resource caching is
            active, otherwise a fresh caller-owned handle over committed authority.

        Raises:
            StoreAuthorityError: If the archive is missing, malformed, or lacks
                current Store authority.
        """

        if cls is not ZipStore:
            return cls(os.path.abspath(os.fspath(zip_dest)), _existing_only=True)

        from ..repo_definition import _open_store_descriptor

        return _open_store_descriptor({
            "kind": "zip", "path": os.path.abspath(os.fspath(zip_dest)),
        })

    @property
    def publication_capabilities(self) -> StorePublicationCapabilities:
        """Return buffered-transaction guarantees or explicit file-like refusal."""
        if self._file_like:
            return StorePublicationCapabilities(False, False, False, False, False)
        return StorePublicationCapabilities(True, True, True, True, True)

    @property
    def _archive_path(self) -> str:
        return self._archive_path_value

    @property
    def archive_path(self) -> str | None:
        """Return the immutable absolute path of a path-backed archive.

        Returns:
            The construction-time absolute archive path, or ``None`` for a
            file-like read-only Store.

        Side Effects:
            None.  In particular, this value is unaffected by later working
            directory changes and is the persistent identity/commit target.
        """

        return self._archive_path_value

    @property
    def _archive_lock_path(self) -> str:
        return f"{self._archive_path}.dryml.lock"

    @staticmethod
    def _validate_archive_members(archive: zipfile.ZipFile) -> None:
        """Reject archive entries that cannot represent one Store root."""

        for info in archive.infolist():
            name = info.filename
            posix_path = PurePosixPath(name)
            windows_path = PureWindowsPath(name)
            if (
                    not name or "\\" in name or posix_path.is_absolute()
                    or windows_path.is_absolute() or ".." in posix_path.parts
                    or (posix_path.parts and ":" in posix_path.parts[0])):
                raise StoreAuthorityError(f"ZipStore archive member escapes its root: {name!r}.")

    @classmethod
    def _validate_existing_archive(cls, path: str) -> None:
        """Validate committed archive and its format gate before extraction."""

        try:
            mode = os.lstat(path).st_mode
        except OSError as error:
            raise StoreAuthorityError("ZipStore archive is missing or inaccessible.") from error
        if not os.path.isfile(path) or not stat.S_ISREG(mode) or os.path.getsize(path) == 0:
            raise StoreAuthorityError("ZipStore archive is not a nonempty regular file.")
        try:
            with zipfile.ZipFile(path, "r") as archive:
                cls._validate_archive_members(archive)
                if archive.testzip() is not None:
                    raise StoreAuthorityError("ZipStore archive is malformed.")
                try:
                    format_bytes = archive.read("store-format.record")
                except KeyError as error:
                    raise StoreAuthorityError("ZipStore archive lacks current Store authority.") from error
                StoreFormatRecord.from_bytes(format_bytes)
        except StoreAuthorityError:
            raise
        except (OSError, zipfile.BadZipFile, StoreRecordError) as error:
            raise StoreAuthorityError("ZipStore archive is malformed or incompatible.") from error

    def _extract_if_present(self) -> None:
        if self._file_like:
            self.zip_dest.seek(0)
            present = bool(self.zip_dest.read(1))
            self.zip_dest.seek(0)
            source = self.zip_dest
        else:
            source = self._archive_path
            present = os.path.exists(source) and os.path.getsize(source) > 0
        if not present:
            return
        try:
            with zipfile.ZipFile(source, "r") as archive:
                self._validate_archive_members(archive)
                archive.extractall(self._tmp.name)
        except zipfile.BadZipFile as error:
            raise StoreAuthorityError("ZipStore archive is malformed.") from error

    def _atomic_write(self, path: str, payload: bytes) -> None:
        """Publish one extracted record and mark this transaction dirty.

        Args:
            path: Final path within this handle's extracted Store.
            payload: Complete record bytes to publish.

        Raises:
            RuntimeError: If this archive handle is closed.
            OSError: If extracted publication fails.
            StoreCapabilityError: If inherited publication guarantees fail.

        Side Effects:
            Serializes same-handle mutation and marks initialized archive state
            dirty after inherited durable publication succeeds.
        """

        self._assert_open()
        with self.transaction_fence():
            super()._atomic_write(path, payload)
            if not self._initializing:
                self._archive_dirty = True

    def mark_query_index_dirty(self, cdef=None, *, metadata_target=None) -> str | None:
        """Publish a scoped dirty marker within this archive transaction.

        Accepts the definition or exact metadata target documented by DirStore
        and returns its marker path, or None without SQLite. Raises the same
        validation/I/O errors, or StoreAuthorityError for a closed/stale archive.
        Acquires the archive transaction fence before publishing the token.
        """

        self._assert_open()
        with self.transaction_fence():
            return super().mark_query_index_dirty(cdef, metadata_target=metadata_target)

    def clear_query_index_dirty(self) -> None:
        """Remove extracted dirty markers and retain that buffered mutation.

        Returns:
            ``None`` after inherited durable marker removal succeeds.

        Raises:
            RuntimeError: If this archive handle is closed.
            OSError: If an extracted marker cannot be removed.
            StoreCapabilityError: If inherited removal guarantees are unavailable.

        Side Effects:
            Serializes same-handle removal and marks an initialized transaction
            dirty so a later commit does not lose the removal.
        """

        self._assert_open()
        with self.transaction_fence():
            had_markers = self.query_index_is_dirty()
            super().clear_query_index_dirty()
            if had_markers and not self._initializing:
                self._archive_dirty = True

    @contextmanager
    def transaction_fence(self):
        """Serialize same-handle buffered mutations and archive commits.

        Returns:
            A reentrant context manager covering this handle's extracted
            transaction state.

        Raises:
            RuntimeError: If the inherited lock implementation cannot acquire its
                required transaction state.

        Side Effects:
            Acquires this handle's in-process transaction lock until context exit.
            It does not read, commit, or modify archive contents by itself.

        The fence is reentrant so inherited record methods may retain their
        writer lock while a graph save spans staging, installation, and commit.
        Separate handles still use the archive baseline conflict check.
        """

        with self._transaction_lock:
            yield

    def authority_fence_key(self) -> str:
        """Return this live transaction's distinct metadata-read fence key.

        Returns:
            A stable key for this ZipStore handle's buffered authority view.

        Raises:
            StoreCapabilityError: If inherited Store capability validation fails.

        Side Effects:
            None. The key does not read, commit, or modify the archive.

        Distinct open handles can hold buffered views with the same archive path
        but different baselines. They must therefore be fenced independently.
        """

        self._assert_open()
        return f"{super().authority_fence_key()}:transaction:{id(self)}"

    def writer_lock(self):
        """Acquire the transaction fence before the inherited Store writer lock."""

        @contextmanager
        def locked():
            self._assert_open()
            with self.transaction_fence():
                with super(ZipStore, self).writer_lock():
                    yield

        return locked()

    def delete_metadata(self, target) -> bool:
        """Remove current metadata and retain the buffered archive dirty marker.

        Args:
            target: Exact ObjectRef or StateRef attachment scope.

        Returns:
            ``True`` when a mapping was removed, otherwise ``False``.

        Raises:
            TypeError: If ``target`` is unsupported.
            OSError: If extracted logical removal fails.
            StoreCapabilityError: If the archive is file-like or cannot publish.

        Side Effects:
            Serializes this handle's transaction, changes only extracted current
            metadata, and marks the archive dirty when a mapping was removed. The
            mutation reaches the archive only at a normal commit boundary.
        """

        self._assert_open()
        with self.transaction_fence():
            result = super().delete_metadata(target)
            if result:
                self._archive_dirty = True
            return result

    def publish_snapshot(
            self, reference, *, evidence, annotations=None, local_states,
            children=None, _before_annotation_write=None):
        """Publish a v3 snapshot in this extraction and mark the archive dirty.

        Args:
            reference: Exact StateRef to install.
            evidence: Fresh or copied snapshot metadata evidence.
            annotations: Optional current-metadata replacements for root scopes.
            local_states: Mapping of local paths to validated payload sources.
            children: Optional child snapshot projections.
            _before_annotation_write: Internal phase callback invoked immediately
                before each requested current-metadata write.

        Returns:
            Detached SnapshotMetadata for the installed or equal existing snapshot.

        Raises:
            TypeError: If inputs are unsupported.
            StoreAuthorityError: If snapshot authority is inconsistent.
            StoreCapabilityError: If the archive cannot publish safely.

        Side Effects:
            Serializes same-handle publication, modifies only the extraction, and
            marks the transaction dirty. Archive replacement is deferred to
            ``commit()``; current annotations retain Store-local LWW behavior.
        """

        self._assert_open()
        with self.transaction_fence():
            result = super().publish_snapshot(
                reference, evidence=evidence, annotations=annotations,
                local_states=local_states, children=children,
                _before_annotation_write=_before_annotation_write,
            )
            self._archive_dirty = True
            return result

    def _archive_identity(self, path: str | None = None) -> str | None:
        target = self._archive_path if path is None else path
        try:
            digest = hashlib.sha256()
            with open(target, "rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except FileNotFoundError:
            return None

    def _physical_archive_evidence(self) -> tuple[int, int] | None:
        """Return the archive inode evidence observed by this transaction.

        Returns:
            The path-backed archive's device/inode pair, or ``None`` when it is
            absent or inaccessible.

        Side Effects:
            None. The value lets resource caching distinguish this transaction's
            own atomic commit from a replacement made by another handle.
        """

        try:
            evidence = os.stat(self._archive_path)
        except OSError:
            return None
        return (evidence.st_dev, evidence.st_ino)

    def commit(self) -> None:
        """Atomically publish the complete buffered archive or reject stale bytes.

        Raises:
            StoreAuthorityError: If the buffered archive cannot be validated or
                the destination changed since this transaction opened.
            OSError: If staging, flushing, or durable archive replacement fails.
            StoreCapabilityError: If archive or parent-directory durability cannot
                be established before reporting commit completion.

        Side Effects:
            Replaces the path-backed archive only when this transaction is dirty,
            then persists the replacement through the active platform primitive.
            File-like archives retain their existing unsupported publication
            behavior.
        """
        with self.transaction_fence():
            self._assert_open()
            if not self._archive_dirty:
                return
            self.preflight_publication("commit ZipStore")
            destination = self._archive_path
            directory = os.path.dirname(destination) or "."
            fd, temporary = tempfile.mkstemp(prefix=".dryml-store-", suffix=".zip", dir=directory)
            os.close(fd)
            try:
                with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                    for root, dirs, files in os.walk(self.base_dir):
                        dirs[:] = sorted(dirs)
                        for name in sorted(files):
                            path = os.path.join(root, name)
                            relative_parts = Path(os.path.relpath(path, self.base_dir)).parts
                            is_tombstone = (
                                name.startswith(_REMOVED_ENTRY_PREFIX)
                                and relative_parts[0] in {".dryml", "metadata"}
                            )
                            if path == self._writer_lock_path or is_tombstone:
                                continue
                            archive.write(path, os.path.relpath(path, self.base_dir))
                with zipfile.ZipFile(temporary, "r") as archive:
                    if archive.testzip() is not None:
                        raise StoreAuthorityError("Buffered ZipStore archive validation failed.")
                with open(temporary, "r+b") as staged_file:
                    os.fsync(staged_file.fileno())
                staged = self._archive_identity(temporary)
                with interprocess_lock(self._archive_lock_path):
                    if self._archive_identity() != self._archive_baseline:
                        raise ZipStoreConflictError("ZipStore archive changed since open; reopen and reapply the mutation.")
                    self._replace_durable(temporary, destination)
                    self._archive_baseline = staged
                    self._archive_evidence = self._physical_archive_evidence()
                    self._archive_dirty = False
            except BaseException:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                raise

    def _assert_open(self) -> None:
        """Reject authority access after this buffered transaction is discarded."""

        if getattr(self, "_closed_handle", False):
            raise RuntimeError("ZipStore transaction is closed.")

    def preflight_publication(self, operation: str, *, local_state: bool = False) -> None:
        """Validate inherited publication capabilities on an open transaction.

        Args:
            operation: Human-readable operation used in capability failures.
            local_state: Whether local payload publication is required.

        Returns:
            ``None`` when publication is supported.

        Raises:
            RuntimeError: If this transaction is closed.
            StoreCapabilityError: If inherited capabilities are insufficient.

        Side Effects:
            None.
        """

        self._assert_open()
        super().preflight_publication(operation, local_state=local_state)

    def _read_file(self, path, record_type):
        """Read one record only while this extracted transaction remains live."""

        self._assert_open()
        return super()._read_file(path, record_type)

    def _read_json(self, path):
        """Read one JSON sidecar only while this transaction remains live."""

        self._assert_open()
        return super()._read_json(path)

    def _read_snapshot(self, digest, *, payloads=False, directory=None):
        """Read one snapshot only while this transaction remains live."""

        self._assert_open()
        return super()._read_snapshot(digest, payloads=payloads, directory=directory)

    def _query_index_dirty_markers(self):
        """Read dirty tokens only while this transaction remains live."""

        self._assert_open()
        return super()._query_index_dirty_markers()

    def open_query_index(self):
        """Return the inherited memory index for an open transaction.

        Returns:
            The transaction-local query index, or ``None`` when disabled.

        Raises:
            RuntimeError: If this transaction is closed.

        Side Effects:
            May initialize the inherited transaction-local derived index.
        """

        self._assert_open()
        return super().open_query_index()

    def query_index_status(self):
        """Return inherited derived-index status for an open transaction.

        Returns:
            Current backend-neutral query-index status.

        Raises:
            RuntimeError: If this transaction is closed.

        Side Effects:
            None.
        """

        self._assert_open()
        return super().query_index_status()

    def iter_definition_records(self):
        """Return validated definitions from an open transaction.

        Returns:
            Deterministically ordered immutable definition records.

        Raises:
            RuntimeError: If this transaction is closed.
            StoreAuthorityError: If retained authority is malformed.

        Side Effects:
            None.
        """

        self._assert_open()
        return super().iter_definition_records()

    def iter_stored_root_records(self):
        """Return validated stored-root memberships from an open transaction.

        Returns:
            Deterministically ordered stored-root records.

        Raises:
            RuntimeError: If this transaction is closed.
            StoreAuthorityError: If retained authority is malformed.

        Side Effects:
            None.
        """

        self._assert_open()
        return super().iter_stored_root_records()

    def iter_state_ref_records(self):
        """Return validated StateRef records from an open transaction.

        Returns:
            Deterministically ordered exact snapshot records.

        Raises:
            RuntimeError: If this transaction is closed.
            StoreAuthorityError: If retained authority is malformed.

        Side Effects:
            None.
        """

        self._assert_open()
        return super().iter_state_ref_records()

    def iter_declaration_records(self):
        """Return validated declarations from an open transaction.

        Returns:
            Deterministically ordered declaration records.

        Raises:
            RuntimeError: If this transaction is closed.
            StoreAuthorityError: If retained authority is malformed.

        Side Effects:
            None.
        """

        self._assert_open()
        return super().iter_declaration_records()

    def iter_object_alias_records(self):
        """Return validated object aliases from an open transaction.

        Returns:
            Deterministically ordered object-alias records.

        Raises:
            RuntimeError: If this transaction is closed.
            StoreAuthorityError: If retained authority is malformed.

        Side Effects:
            None.
        """

        self._assert_open()
        return super().iter_object_alias_records()

    def iter_state_alias_records(self):
        """Return validated state aliases from an open transaction.

        Returns:
            Deterministically ordered state-alias records.

        Raises:
            RuntimeError: If this transaction is closed.
            StoreAuthorityError: If retained authority is malformed.

        Side Effects:
            None.
        """

        self._assert_open()
        return super().iter_state_alias_records()

    def catalog_key(self) -> str:
        """Return a stable archive identity without leaking extraction paths."""
        if self._file_like:
            return f"{type(self).__module__}.{type(self).__qualname__}:buffer:{id(self.zip_dest)}"
        return f"{type(self).__module__}.{type(self).__qualname__}:{self._archive_path}"

    def close(self) -> None:
        """Discard the buffered transaction and invalidate borrowed paths.

        Raises:
            RuntimeError: If an active Session resource cache retains this Store.

        Returns:
            ``None``. Repeated close calls are no-ops.

        Side Effects:
            Acquires the transaction fence, removes the extracted directory, and
            invalidates snapshot and payload paths borrowed from this handle.
        """
        with self.transaction_fence():
            if self._closed_handle:
                return
            super().close()
            self._tmp.cleanup()
            self._closed_handle = True


ZipStore.publish_snapshot._dryml_annotation_phase_callback = True
