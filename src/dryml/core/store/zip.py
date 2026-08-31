"""Buffered Zip implementation of the current logical Store interface."""

from __future__ import annotations

import hashlib
import os
from io import IOBase
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
import zipfile

from .dir import DirStore
from .locking import interprocess_lock
from .store import StoreAuthorityError, StoreCapabilityError, StorePublicationCapabilities


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

    def __init__(self, zip_dest: str | Path | IOBase):
        self.zip_dest = zip_dest
        self._tmp = tempfile.TemporaryDirectory()
        self._archive_dirty = False
        self._initializing = True
        self._file_like = _is_file_like(zip_dest)
        try:
            self._extract_if_present()
            super().__init__(self._tmp.name, query_index="memory")
        except BaseException:
            self._tmp.cleanup()
            raise
        self._initializing = False
        self._archive_baseline = None if self._file_like else self._archive_identity()

    @property
    def publication_capabilities(self) -> StorePublicationCapabilities:
        """Return buffered-transaction guarantees or explicit file-like refusal."""
        if self._file_like:
            return StorePublicationCapabilities(False, False, False, False, False)
        return StorePublicationCapabilities(True, True, True, True, True)

    @property
    def _archive_path(self) -> str:
        return os.path.abspath(os.fspath(self.zip_dest))

    @property
    def _archive_lock_path(self) -> str:
        return f"{self._archive_path}.dryml.lock"

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
                for info in archive.infolist():
                    name = info.filename
                    posix_path = PurePosixPath(name)
                    windows_path = PureWindowsPath(name)
                    if (
                            not name or "\\" in name or posix_path.is_absolute()
                            or windows_path.is_absolute() or ".." in posix_path.parts
                            or (posix_path.parts and ":" in posix_path.parts[0])):
                        raise StoreAuthorityError(f"ZipStore archive member escapes its root: {name!r}.")
                archive.extractall(self._tmp.name)
        except zipfile.BadZipFile as error:
            raise StoreAuthorityError("ZipStore archive is malformed.") from error

    def _atomic_write(self, path: str, payload: bytes) -> None:
        super()._atomic_write(path, payload)
        if not self._initializing:
            self._archive_dirty = True

    def install_local_state(self, source_dir: object, manifest):
        """Install local-state authority into this archive's buffered transaction."""
        result = super().install_local_state(source_dir, manifest)
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

    def commit(self) -> None:
        """Atomically publish the complete buffered archive or reject stale bytes."""
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
        """Discard the buffered transaction without publishing it."""
        self._tmp.cleanup()
