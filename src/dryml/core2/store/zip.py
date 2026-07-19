from __future__ import annotations

import zipfile
from pathlib import Path
from io import IOBase
import glob
import tempfile
import os
import pickle
import shutil

from .store import Store
from ..utils.general import pickle_save, pickle_load


def is_binary_file_like(value) -> bool:
    """Return whether *value* supports the binary seekable zip protocol.

    Windows ``NamedTemporaryFile`` returns a wrapper that delegates these
    methods but does not inherit :class:`io.IOBase`.
    """

    return all(callable(getattr(value, name, None)) for name in ("read", "write", "seek", "truncate"))


class ZipStore(Store):
    file_like_commit_is_atomic = False

    def __init__(self, zip_dest: str | Path | IOBase):
        self.zip_dest = zip_dest
        self._tmp = tempfile.TemporaryDirectory()
        self._base_dir = self._tmp.name
        self.obj_dir = os.path.join(self.base_dir, "objects")
        os.makedirs(self.obj_dir, exist_ok=True)

        # cache for main_def
        self._main_def: ConcreteDefinition | None = None

        self._extract_if_nonempty()

        # if an existing main def is present, cache it
        self.set_main_def(self.read_main_def())

    def managed_control_root(self) -> str:
        """Reject live managed mutation for archive-backed Stores."""

        raise NotImplementedError("ZipStore does not support live managed control")

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

        if is_binary_file_like(self.zip_dest):
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
        pattern = os.path.join(self.obj_dir, "[0-9a-f][0-9a-f]", "*", "def.pkl")
        return [pickle_load(p) for p in glob.glob(pattern)]

    def commit(self) -> None:
        self.write_main_def()

        def write_archive(zf):
            for root, _, files in os.walk(self.base_dir):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, self.base_dir)
                    zf.write(full, rel)

        _commit_zip_destination(self.zip_dest, write_archive)

    def close(self) -> None:
        self._tmp.cleanup()

    def catalog_key(self) -> str:
        if is_binary_file_like(self.zip_dest):
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
            return pickle_load(path)
        else:
            return None

    def write_main_def(self) -> None:
        if self._main_def is not None:
            pickle_save(self._main_def, self._main_def_path())

    def read_aliases(self) -> dict[str, ConcreteDefinition]:
        path = self._aliases_path()
        if os.path.exists(path):
            return pickle_load(path)
        return {}

    def write_aliases(self, aliases: dict[str, ConcreteDefinition]) -> None:
        pickle_save(dict(aliases), self._aliases_path())

    def set_main_def(self, main_def: ConcreteDefinition) -> None:
        # just cache it; actual file write happens in commit()
        self._main_def = main_def


class ZipExportStore(Store):
    """
    Sink-only Store that creates a zip from an existing directory tree.

    - It NEVER participates in graph IO (no per-object save/load).
    - It just zips a subset of paths from `src_dir` into `zip_dest`.
    - Optionally embeds a `main_def` as def.pkl at the root of the zip.
    """

    file_like_commit_is_atomic = False

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

    def managed_control_root(self) -> str:
        """Reject live managed mutation for export-only Stores."""

        raise NotImplementedError("ZipExportStore does not support live managed control")

    @property
    def base_dir(self) -> str:
        """Return the source directory being exported."""

        return self.src_dir

    @property
    def object_root_dir(self) -> str:
        """Return the object root in the source directory."""

        return os.path.join(self.src_dir, "objects")

    # --- Store interface: membership / index ---

    def has(self, cdef: ConcreteDefinition) -> bool:
        # Exporter does not own any cdefs; it's sink-only.
        return False

    def hydrate_index(self) -> Iterable[ConcreteDefinition]:
        # Nothing to hydrate; exporter doesn't enumerate objects.
        return ()

    def _object_dir(self, cdef: ConcreteDefinition) -> str:
        """Return the source object directory path for interface completeness."""

        digest = cdef.stable_hash()
        return os.path.join(self.object_root_dir, digest[:2], digest)

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
        files = self._included_files()

        def write_archive(zf):
            # Optional main def at root
            if self._main_def is not None:
                data = pickle.dumps(self._main_def)
                zf.writestr("def.pkl", data)

            for rel, full in files:
                zf.write(full, rel)

        _commit_zip_destination(self.zip_dest, write_archive)

    def _included_files(self) -> tuple[tuple[str, str], ...]:
        """Validate every requested path before returning exact archive files."""

        source = Path(self.src_dir).resolve()
        files = {}
        for requested in sorted(self.include_paths):
            rel_path = Path(requested)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise ValueError(f"zip include path must stay within source: {requested!r}")
            full = (source / rel_path).resolve()
            if full != source and source not in full.parents:
                raise ValueError(f"zip include path escapes source: {requested!r}")
            if not full.exists():
                raise FileNotFoundError(f"missing zip include path: {requested}")
            if full.is_dir():
                for path in sorted(item for item in full.rglob("*") if item.is_file()):
                    files[path.relative_to(source).as_posix()] = os.fspath(path)
            elif full.is_file():
                files[full.relative_to(source).as_posix()] = os.fspath(full)
            else:
                raise ValueError(f"zip include path is not a regular file or directory: {requested!r}")
        return tuple(sorted(files.items()))

    def close(self) -> None:
        # Nothing to clean up (no temp dir).
        pass

    def __repr__(self) -> str:
        return f"{type(self)}(base_dir: {self.src_dir} dest: {self.zip_dest})"


def _commit_zip_destination(zip_dest, write_archive) -> None:
    """Build a complete archive before publishing it to its destination.

    Filesystem paths use same-directory atomic replacement. Generic file-like
    destinations cannot provide an atomic replacement primitive; they are only
    mutated after archive construction succeeds, but a failure while copying the
    completed archive into the caller-owned object can still leave partial data.
    """

    file_like = is_binary_file_like(zip_dest)
    if file_like:
        parent = None
    else:
        destination = Path(os.fspath(zip_dest))
        parent = destination.parent
        if not parent.exists():
            raise FileNotFoundError(f"zip destination parent does not exist: {parent}")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w+b",
            dir=None if parent is None else parent,
            prefix=".dryml-zip-",
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            write_archive(zf)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        if file_like:
            with temp_path.open("rb") as source:
                zip_dest.seek(0)
                zip_dest.truncate(0)
                shutil.copyfileobj(source, zip_dest, length=1024 * 1024)
            flush = getattr(zip_dest, "flush", None)
            if callable(flush):
                flush()
            zip_dest.seek(0)
        else:
            os.replace(temp_path, destination)
            temp_path = None
            if os.name == "posix":
                descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
