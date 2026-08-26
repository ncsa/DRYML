from __future__ import annotations

import zipfile
from pathlib import Path
from io import IOBase
import glob
import tempfile
import os
from io import BytesIO

from .store import Store, StoreAuthorityError
from ..utils.general import atomic_pickle_save, pickle_load


class ZipStore(Store):
    def __init__(self, zip_dest: str | Path | IOBase):
        self.zip_dest = zip_dest
        self._tmp = tempfile.TemporaryDirectory()
        self._base_dir = self._tmp.name
        self.obj_dir = os.path.join(self.base_dir, "objects")
        os.makedirs(self.obj_dir, exist_ok=True)

        self._main_def = None
        self._main_def_dirty = False
        self._archive_dirty = False

        self._extract_if_nonempty()

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

    # The DirStore-style helpers:
    def _object_dir(self, cdef: "ConcreteDefinition") -> str:
        digest = cdef.stable_hash()
        sub = digest[:2]
        return os.path.join(self.obj_dir, sub, digest)

    def has(self, cdef: "ConcreteDefinition") -> bool:
        return os.path.exists(self._def_file(cdef))

    def hydrate_index(self) -> list["ConcreteDefinition"]:
        pattern = os.path.join(self.obj_dir, "**", "def.pkl")
        definitions = []
        seen_hashes: set[str] = set()
        for path in glob.glob(pattern, recursive=True):
            cdef = self._validate_root_definition_path(path)
            digest = cdef.stable_hash()
            if digest in seen_hashes:
                raise StoreAuthorityError(f"Store contains duplicate root definitions for digest {digest!r}.")
            seen_hashes.add(digest)
            definitions.append(cdef)
        return definitions

    def commit(self) -> None:
        if self._main_def_dirty and self._main_def is not None:
            self.write_main_def(self._main_def)
        if not self._archive_dirty:
            return
        self._write_archive_atomically()
        self._archive_dirty = False

    def _mark_authority_dirty(self) -> None:
        """Mark the extracted archive view for explicit publication on commit."""

        self._archive_dirty = True

    def _write_archive_atomically(self) -> None:
        """Validate and publish a complete path-backed archive by replacement."""

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
        path = os.fspath(self.zip_dest)
        directory = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".dryml-", suffix=".zip", dir=directory)
        os.close(fd)
        try:
            self._write_archive(tmp_path)
            with zipfile.ZipFile(tmp_path, "r") as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise OSError(f"Staged ZipStore archive is corrupt at {bad_member!r}.")
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def _write_archive(self, destination) -> None:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(self.base_dir):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, self.base_dir)
                    zf.write(full, rel)

    def close(self) -> None:
        self._tmp.cleanup()

    def catalog_key(self) -> str:
        if isinstance(self.zip_dest, IOBase):
            return f"{type(self).__module__}.{type(self).__qualname__}:buffer:{id(self.zip_dest)}"
        return f"{type(self).__module__}.{type(self).__qualname__}:{os.path.abspath(os.fspath(self.zip_dest))}"

    def _main_def_path(self) -> str:
        return os.path.join(self.base_dir, "def.pkl")

    def _aliases_path(self) -> str:
        return os.path.join(self.base_dir, "aliases.pkl")

    def read_main_def(self) -> ConcreteDefinition | None:
        # prefer cached version; fall back to on-disk if needed
        if self._main_def is not None:
            return self._main_def
        path = self._main_def_path()
        if os.path.exists(path):
            return self._read_concrete_definition(path)
        else:
            return None

    def write_main_def(self, main_def: ConcreteDefinition) -> None:
        atomic_pickle_save(main_def, self._main_def_path())
        self._main_def = main_def
        self._main_def_dirty = False
        self._mark_authority_dirty()

    def read_aliases(self) -> dict[str, ConcreteDefinition]:
        path = self._aliases_path()
        if os.path.exists(path):
            aliases = pickle_load(path)
            if not isinstance(aliases, dict):
                raise StoreAuthorityError("Store aliases payload is not a dictionary.")
            for alias, cdef in aliases.items():
                if not isinstance(alias, str):
                    raise StoreAuthorityError("Store aliases contain a non-string alias.")
                self._validate_alias_definition(cdef, alias)
            return aliases
        return {}

    def write_aliases(self, aliases: dict[str, ConcreteDefinition]) -> None:
        atomic_pickle_save(dict(aliases), self._aliases_path())
        self._mark_authority_dirty()

    def set_main_def(self, main_def: ConcreteDefinition) -> None:
        if main_def != self._main_def:
            self._main_def = main_def
            self._main_def_dirty = True

    @staticmethod
    def _validate_alias_definition(cdef, alias: str) -> None:
        from ..definition import ConcreteDefinition

        if not isinstance(cdef, ConcreteDefinition):
            raise StoreAuthorityError(
                f"Store alias {alias!r} points to {type(cdef).__name__}, not a ConcreteDefinition."
            )


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
