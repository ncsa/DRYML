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

from .dir import DirStore
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
        with self.transaction_fence():
            super()._atomic_write(path, payload)
            if not self._initializing:
                self._archive_dirty = True

    def mark_query_index_dirty(self, cdef=None) -> str | None:
        """Fence derived dirty-marker publication with this archive transaction."""

        with self.transaction_fence():
            return super().mark_query_index_dirty(cdef)

    def clear_query_index_dirty(self) -> None:
        """Persist derived-marker removal rather than losing it after a commit."""

        with self.transaction_fence():
            had_markers = self.query_index_is_dirty()
            super().clear_query_index_dirty()
            if had_markers and not self._initializing:
                self._archive_dirty = True

    @contextmanager
    def transaction_fence(self):
        """Serialize same-handle buffered mutations and archive commits.

        The fence is reentrant so inherited record methods may retain their
        writer lock while a graph save spans staging, installation, and commit.
        Separate handles still use the archive baseline conflict check.
        """

        with self._transaction_lock:
            yield

    def writer_lock(self):
        """Acquire the transaction fence before the inherited Store writer lock."""

        @contextmanager
        def locked():
            with self.transaction_fence():
                with super(ZipStore, self).writer_lock():
                    yield

        return locked()

    def install_local_state(self, source_dir: object, manifest):
        """Install local-state authority into this archive's buffered transaction."""
        with self.transaction_fence():
            result = super().install_local_state(source_dir, manifest)
            self._archive_dirty = True
            return result

    def copy_local_state_from(self, source, definition, state_hash: str):
        """Copy immutable payload authority under the same archive transaction fence."""

        with self.transaction_fence():
            return super().copy_local_state_from(source, definition, state_hash)

    def rebind_local_state_from(self, source, source_definition, target_definition, state_hash: str):
        """Rebind copied payload authority without racing this handle's commit."""

        with self.transaction_fence():
            return super().rebind_local_state_from(
                source, source_definition, target_definition, state_hash,
            )

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
        """Atomically publish the complete buffered archive or reject stale bytes."""
        with self.transaction_fence():
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
                            if path == self._writer_lock_path:
                                continue
                            archive.write(path, os.path.relpath(path, self.base_dir))
                with zipfile.ZipFile(temporary, "r") as archive:
                    if archive.testzip() is not None:
                        raise StoreAuthorityError("Buffered ZipStore archive validation failed.")
                staged = self._archive_identity(temporary)
                with interprocess_lock(self._archive_lock_path):
                    if self._archive_identity() != self._archive_baseline:
                        raise ZipStoreConflictError("ZipStore archive changed since open; reopen and reapply the mutation.")
                    os.replace(temporary, destination)
                    self._archive_baseline = staged
                    self._archive_evidence = self._physical_archive_evidence()
                    self._archive_dirty = False
            except BaseException:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                raise

    def catalog_key(self) -> str:
        """Return a stable archive identity without leaking extraction paths."""
        if self._file_like:
            return f"{type(self).__module__}.{type(self).__qualname__}:buffer:{id(self.zip_dest)}"
        return f"{type(self).__module__}.{type(self).__qualname__}:{self._archive_path}"

    def close(self) -> None:
        """Discard the buffered transaction without publishing it.

        Raises:
            RuntimeError: If an active Session resource cache retains this Store.
        """
        if self._closed_handle:
            return
        super().close()
        self._tmp.cleanup()
        self._closed_handle = True
