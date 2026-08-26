from __future__ import annotations

import zipfile
import hashlib
from pathlib import Path
from io import IOBase
import glob
import tempfile
import os
from io import BytesIO

from .store import Store, StoreAuthorityError, _STATE_GENERATIONS_DIR
from .locking import interprocess_lock
from ..utils.general import atomic_pickle_save, pickle_load


class ZipStoreConflictError(StoreAuthorityError):
    """Report rejected archive publication from a stale ZipStore handle.

    A path-backed Store raises this error instead of replacing roots, aliases, or
    main-reference bytes published after the handle extracted its baseline.
    Reopen the Store, reapply the desired mutation, and commit the new handle.
    """


class ZipStore(Store):
    """Expose one Zip archive as an extracted, hash-addressed object Store.

    Args:
        zip_dest: Existing or destination archive path, or a seekable binary
            stream. Opening and hydration are read-only; explicit mutations are
            buffered in an extracted view until ``commit`` publishes the archive.

    Filesystem-backed commits construct and validate a complete sibling archive
    before atomic replacement. Cooperating path-backed handles serialize
    publication and reject a dirty archive whose bytes changed since extraction;
    reopen the Store and reapply the intended mutation before retrying. File-like
    destinations cannot provide that filesystem atomicity guarantee. State
    readers and writers coordinate through extracted-view leases; commits retain
    only the pointer-reachable state generation, including after an interruption
    leaves an inactive extracted generation behind.
    """

    def __init__(self, zip_dest: str | Path | IOBase):
        self.zip_dest = zip_dest
        self._tmp = tempfile.TemporaryDirectory()
        self._base_dir = self._tmp.name
        self.obj_dir = os.path.join(self.base_dir, "objects")
        os.makedirs(self.obj_dir, exist_ok=True)

        self._main_def = None
        self._main_def_dirty = False
        self._archive_dirty = False

        if isinstance(self.zip_dest, IOBase):
            self._extract_if_nonempty()
            self._archive_baseline = None
        else:
            with interprocess_lock(self._archive_lock_path):
                self._extract_if_nonempty()
                self._archive_baseline = self._archive_identity()

        # Hydration is read-only: retain the exact archive unless an explicit
        # save, alias, or main-definition mutation marks it dirty.
        self._main_def = self.read_main_def()

    @property
    def base_dir(self) -> str:
        """Base directory"""
        return self._base_dir

    @property
    def object_root_dir(self) -> str:
        """Base directory"""
        return self.obj_dir

    def _extract_if_nonempty(self):
        def _load():
            with zipfile.ZipFile(self.zip_dest, 'r') as zf:
                zf.extractall(self.base_dir)

        if isinstance(self.zip_dest, IOBase):
            buf = self.zip_dest
            buf.seek(0)
            if buf.read(1):
                buf.seek(0)
                _load()
                buf.seek(0)
        else:
            path = os.fspath(self.zip_dest)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    if f.read(1):
                        _load()

    @property
    def _archive_path(self) -> str:
        """Return the normalized filesystem destination for path-backed archives."""

        return os.path.abspath(os.fspath(self.zip_dest))

    @property
    def _archive_lock_path(self) -> str:
        """Return the path-scoped advisory lock used for archive publication."""

        return f"{self._archive_path}.dryml.lock"

    def _archive_identity(self, path: str | None = None) -> str | None:
        """Return an archive's SHA-256 digest, or ``None`` when absent."""

        try:
            digest = hashlib.sha256()
            with open(self._archive_path if path is None else path, "rb") as archive:
                for block in iter(lambda: archive.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except FileNotFoundError:
            return None

    # The DirStore-style helpers:
    def _object_dir(self, cdef: "ConcreteDefinition") -> str:
        digest = cdef.stable_hash()
        sub = digest[:2]
        return os.path.join(self.obj_dir, sub, digest)

    def _state_lock_path(self, object_dir: str) -> str:
        """Return an extracted-view-only state lease path for one object root.

        Args:
            object_dir: Stable extracted path for an object root.

        Returns:
            A lock path excluded from archive publication.
        """

        return os.path.join(self.base_dir, ".dryml", "state-locks", f"{os.path.basename(object_dir)}.lock")

    def has(self, cdef: "ConcreteDefinition") -> bool:
        return os.path.exists(self._def_file(cdef))

    def hydrate_index(self) -> list["ConcreteDefinition"]:
        """Return validated authoritative root definitions from the archive.

        Returns:
            Stored root definitions without nested state-generation definitions.

        Raises:
            StoreAuthorityError: If a root path, digest, payload, or duplicate
                stable identity is invalid.
        """

        pattern = os.path.join(self.obj_dir, "*", "*", "def.pkl")
        definitions = []
        seen_hashes: set[str] = set()
        for path in glob.glob(pattern):
            cdef = self._validate_root_definition_path(path)
            digest = cdef.stable_hash()
            if digest in seen_hashes:
                raise StoreAuthorityError(f"Store contains duplicate root definitions for digest {digest!r}.")
            seen_hashes.add(digest)
            definitions.append(cdef)
        return definitions

    def commit(self) -> None:
        """Publish staged references and extracted changes to the archive.

        Raises:
            StoreAuthorityError: If a staged main reference is malformed.
            ZipStoreConflictError: If a path-backed archive changed since this
                handle extracted it; reopen and reapply the mutation to retry.
            OSError: If archive construction or replacement fails.

        Side Effects:
            Writes a dirty main reference into the extracted view, then replaces
            a path-backed archive atomically when that view is dirty. A no-op
            commit preserves the existing archive bytes. A dirty path-backed
            commit compares its extraction baseline while holding the archive
            publication lock before replacing matching archive bytes.
        """
        if self._main_def_dirty and self._main_def is not None:
            self.write_main_def(self._main_def)
        if not self._archive_dirty:
            return
        self._write_archive_atomically()
        self._archive_dirty = False

    def _mark_authority_dirty(self, cdef: ConcreteDefinition | None = None) -> bool:
        """Mark the extracted archive view for explicit publication on commit."""

        was_dirty = self._archive_dirty
        self._archive_dirty = True
        return was_dirty

    def _discard_authority_dirty(self, token) -> None:
        self._archive_dirty = bool(token)

    def _write_archive_atomically(self) -> None:
        """Validate and publish a complete archive without overwriting peer authority.

        Raises:
            ZipStoreConflictError: If the path-backed archive changed since this
                Store extracted it. Reopen the Store and reapply changes to retry.
            OSError: If staging, validation, locking, or replacement fails.

        Side Effects:
            For path-backed archives, holds a path-scoped interprocess lock while
            comparing the baseline digest and atomically replacing matching bytes.
            File-like targets retain their previous in-place stream behavior.
        """

        if isinstance(self.zip_dest, IOBase):
            # File-like targets cannot offer filesystem replacement; construct
            # the complete archive first so a serialization failure leaves them
            # untouched.
            payload = BytesIO()
            self._write_archive(payload)
            self.zip_dest.seek(0)
            self.zip_dest.truncate(0)
            self.zip_dest.write(payload.getvalue())
            self.zip_dest.seek(0)
            return
        path = self._archive_path
        directory = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".dryml-", suffix=".zip", dir=directory)
        os.close(fd)
        try:
            self._write_archive(tmp_path)
            with zipfile.ZipFile(tmp_path, "r") as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise OSError(f"Staged ZipStore archive is corrupt at {bad_member!r}.")
            staged_identity = self._archive_identity(tmp_path)
            with interprocess_lock(self._archive_lock_path):
                if self._archive_identity() != self._archive_baseline:
                    raise ZipStoreConflictError(
                        "ZipStore archive changed since this handle was opened; "
                        "reopen the Store and reapply the intended mutation before retrying."
                    )
                os.replace(tmp_path, path)
                self._archive_baseline = staged_identity
        except BaseException:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def _write_archive(self, destination) -> None:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(self.base_dir):
                relative_root = os.path.relpath(root, self.base_dir)
                if relative_root == ".dryml":
                    dirs[:] = [name for name in dirs if name != "state-locks"]
                object_relative_root = os.path.relpath(root, self.obj_dir).split(os.sep)
                if len(object_relative_root) == 3 and object_relative_root[-1] == _STATE_GENERATIONS_DIR:
                    object_dir = os.path.dirname(root)
                    active_state_dir = self._active_state_dir(object_dir)
                    if active_state_dir == object_dir:
                        dirs.clear()
                        files.clear()
                    else:
                        dirs[:] = [os.path.basename(active_state_dir)]
                        files.clear()
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, self.base_dir)
                    zf.write(full, rel)

    def close(self) -> None:
        self._tmp.cleanup()

    def catalog_key(self) -> str:
        if isinstance(self.zip_dest, IOBase):
            return f"{type(self).__module__}.{type(self).__qualname__}:buffer:{id(self.zip_dest)}"
        return f"{type(self).__module__}.{type(self).__qualname__}:{self._archive_path}"

    def _main_def_path(self) -> str:
        return os.path.join(self.base_dir, "def.pkl")

    def _aliases_path(self) -> str:
        return os.path.join(self.base_dir, "aliases.pkl")

    def read_main_def(self) -> ConcreteDefinition | None:
        """Read this archive Store's validated main reference without publishing.

        Returns:
            The cached or extracted ``ConcreteDefinition``, or ``None`` when no
            main reference exists.

        Raises:
            StoreAuthorityError: If the extracted reference payload is malformed.
        """
        # prefer cached version; fall back to on-disk if needed
        if self._main_def is not None:
            return self._main_def
        path = self._main_def_path()
        if os.path.exists(path):
            return self._read_concrete_definition(path)
        else:
            return None

    def write_main_def(self, main_def: ConcreteDefinition) -> None:
        """Validate and stage a main reference in the extracted archive view.

        Args:
            main_def: ``ConcreteDefinition`` to publish on a later ``commit``.

        Raises:
            StoreAuthorityError: If ``main_def`` is malformed.
            OSError: If extracted reference replacement fails.

        Side Effects:
            Replaces only the extracted reference after validation, updates its
            cache, and marks the archive dirty without changing archive bytes.
        """
        self._validate_main_definition(main_def)
        atomic_pickle_save(main_def, self._main_def_path())
        self._main_def = main_def
        self._main_def_dirty = False
        self._mark_authority_dirty()

    def read_aliases(self) -> dict[str, ConcreteDefinition]:
        """Read a validated copy of aliases from the extracted archive view.

        Returns:
            The persisted non-empty alias mapping, or an empty mapping.

        Raises:
            StoreAuthorityError: If the mapping, alias key, or target is
                malformed.
        """
        path = self._aliases_path()
        if os.path.exists(path):
            aliases = pickle_load(path)
            self._validate_aliases(aliases)
            return dict(aliases)
        return {}

    def write_aliases(self, aliases: dict[str, ConcreteDefinition]) -> dict[str, ConcreteDefinition]:
        """Validate and stage a replacement alias mapping for archive commit.

        Args:
            aliases: Mapping of non-empty string aliases to concrete definitions.

        Returns:
            A copy of the mapping staged in the extracted view.

        Raises:
            StoreAuthorityError: If the mapping or any payload is malformed.
            OSError: If extracted reference replacement fails.

        Side Effects:
            Replaces extracted alias bytes only after validation and marks the
            archive dirty; path-backed archive bytes change on ``commit``.
        """
        self._validate_aliases(aliases)
        atomic_pickle_save(dict(aliases), self._aliases_path())
        self._mark_authority_dirty()
        return dict(aliases)

    def set_main_def(self, main_def: ConcreteDefinition) -> None:
        """Validate and cache a main reference for a later archive commit.

        Args:
            main_def: ``ConcreteDefinition`` to make the archive default.

        Raises:
            StoreAuthorityError: If ``main_def`` is malformed.

        Side Effects:
            Updates the in-memory main-reference cache and dirty flag only;
            neither extracted nor archive bytes change until ``commit``.
        """
        self._validate_main_definition(main_def)
        if main_def != self._main_def:
            self._main_def = main_def
            self._main_def_dirty = True


class ZipExportStore(Store):
    """
    Sink-only Store that creates a zip from an existing directory tree.

    - It NEVER participates in graph IO (no per-object save/load).
    - It just zips a subset of paths from `src_dir` into `zip_dest`.
    - Optionally embeds a `main_def` as def.pkl at the root of the zip.
    """

    def __init__(self, zip_dest, src_dir: str, include_paths: set[str]):
        """
        Parameters
        ----------
        zip_dest : str | Path | IOBase
            Destination zip path or file-like object.
        src_dir : str
            Existing directory to read from (e.g., a DirStore.base_dir).
        include_paths : set[str]
            Relative paths (from src_dir) to include in the export.
            Directories are included recursively.
        """
        self.zip_dest = zip_dest
        self.src_dir = os.fspath(src_dir)
        self.include_paths = set(include_paths)
        self._main_def: ConcreteDefinition | None = None

    # --- Store interface: membership / index ---

    def has(self, cdef: ConcreteDefinition) -> bool:
        # Exporter does not own any cdefs; it's sink-only.
        return False

    def hydrate_index(self) -> Iterable[ConcreteDefinition]:
        # Nothing to hydrate; exporter doesn't enumerate objects.
        return ()

    # --- Store interface: per-object IO ---

    def save_object(self, obj: Object) -> None:
        # No-op: export uses existing files in src_dir only.
        return

    def load_object(self, obj: Object) -> bool:
        # Cannot load from an export-only store.
        return False

    # --- Store interface: main_def ---

    def read_main_def(self) -> ConcreteDefinition | None:
        # We only know about the main_def that was explicitly written to us.
        return self._main_def

    def set_main_def(self, main_def: ConcreteDefinition) -> None:
        # Remember it so commit() can write def.pkl into the zip.
        self._main_def = main_def

    # --- Store interface: lifecycle ---

    def commit(self) -> None:
        """
        Create/overwrite the zip:

        - If _main_def is set, write it as def.pkl at the root of the zip.
        - Then write all include_paths (files or directories) from src_dir.
        """
        if isinstance(self.zip_dest, IOBase):
            buf = self.zip_dest
            buf.seek(0)
            buf.truncate(0)
            zf_ctx = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
        else:
            zf_ctx = zipfile.ZipFile(os.fspath(self.zip_dest),
                                     "w", zipfile.ZIP_DEFLATED)

        with zf_ctx as zf:
            # Optional main def at root
            if self._main_def is not None:
                data = pickle.dumps(self._main_def)
                zf.writestr("def.pkl", data)

            # Stream included paths from src_dir
            for rel in self.include_paths:
                full = os.path.join(self.src_dir, rel)
                if os.path.isdir(full):
                    for root, _, files in os.walk(full):
                        for name in files:
                            ffull = os.path.join(root, name)
                            frel = os.path.relpath(ffull, self.src_dir)
                            zf.write(ffull, frel)
                elif os.path.isfile(full):
                    zf.write(full, rel)
                else:
                    # Missing path: silently skip or raise, your choice.
                    # raise FileNotFoundError(full)
                    pass

        if isinstance(self.zip_dest, IOBase):
            self.zip_dest.seek(0)

    def close(self) -> None:
        # Nothing to clean up (no temp dir).
        pass

    def __repr__(self) -> str:
        return f"{type(self)}(base_dir: {self._base_dir} dest: {self.zip_dest})"
