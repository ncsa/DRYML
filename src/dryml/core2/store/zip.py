from __future__ import annotations

import zipfile
from pathlib import Path
from io import IOBase
import tempfile
import os

from .store import Store
from ..utils.general import pickle_save, pickle_load


class ZipStore(Store):
    def __init__(self, zip_dest: str | Path | IOBase):
        self.zip_dest = zip_dest
        self._tmp = tempfile.TemporaryDirectory()
        self.base_dir = self._tmp.name
        self.obj_dir = os.path.join(self.base_dir, "objects")
        os.makedirs(self.obj_dir, exist_ok=True)
        self.main_def_path = os.path.join(self.base_dir, "def.pkl")

        # hydrate from existing zip, if any
        self._extract_if_nonempty()

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

    def _def_file(self, cdef: "ConcreteDefinition") -> str:
        return os.path.join(self._object_dir(cdef), "def.pkl")

    def has_cdef(self, cdef: "ConcreteDefinition") -> bool:
        return os.path.exists(self._def_file(cdef))

    def hydrate_index(self) -> list["ConcreteDefinition"]:
        pattern = os.path.join(self.obj_dir, "[0-9a-f][0-9a-f]", "*", "def.pkl")
        return [pickle_load(p) for p in glob.glob(pattern)]

    def save_object(self, obj: "Object") -> None:
        cdef = obj.definition
        obj_dir = self._object_dir(cdef)
        os.makedirs(obj_dir, exist_ok=True)
        pickle_save(cdef, os.path.join(obj_dir, "def.pkl"))
        obj.save_to_dir(obj_dir)

    def load_object(self, obj: "Object") -> bool:
        cdef = obj.definition
        def_path = self._def_file(cdef)
        if not os.path.exists(def_path):
            return False
        stored_cdef = pickle_load(def_path)
        if stored_cdef.stable_hash() != cdef.stable_hash():
            raise RepoLoadError("Definition hash mismatch in ZipStore.load_object")
        obj_dir = os.path.dirname(def_path)
        obj.load_from_dir(obj_dir)
        return True

    def commit(self) -> None:
        # write base_dir back into zip_dest
        if isinstance(self.zip_dest, IOBase):
            buf = self.zip_dest
            buf.seek(0)
            buf.truncate(0)
            zf_ctx = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
        else:
            zf_ctx = zipfile.ZipFile(os.fspath(self.zip_dest), "w", zipfile.ZIP_DEFLATED)

        with zf_ctx as zf:
            for root, _, files in os.walk(self.base_dir):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, self.base_dir)
                    zf.write(full, rel)

        if isinstance(self.zip_dest, IOBase):
            self.zip_dest.seek(0)

    def close(self) -> None:
        self._tmp.cleanup()


class ZipExportStore(Store):
    def __init__(self, zip_dest, src_dir: str, include_paths: set[str]):
        self.zip_dest = zip_dest
        self.src_dir = os.fspath(src_dir)
        self.include_paths = include_paths  # relative paths to include

    def has_cdef(self, cdef): return False
    def hydrate_index(self): return ()
    def save_object(self, obj): pass          # no-op; we export existing data
    def load_object(self, obj): return False  # cannot load

    def commit(self):
        if isinstance(self.zip_dest, IOBase):
            buf = self.zip_dest
            buf.seek(0)
            buf.truncate(0)
            zf_ctx = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
        else:
            zf_ctx = zipfile.ZipFile(os.fspath(self.zip_dest), "w", zipfile.ZIP_DEFLATED)

        with zf_ctx as zf:
            for rel in self.include_paths:
                full = os.path.join(self.src_dir, rel)
                if os.path.isdir(full):
                    for root, _, files in os.walk(full):
                        for name in files:
                            ffull = os.path.join(root, name)
                            frel = os.path.relpath(ffull, self.src_dir)
                            zf.write(ffull, frel)
                else:
                    zf.write(full, rel)

        if isinstance(self.zip_dest, IOBase):
            self.zip_dest.seek(0)
