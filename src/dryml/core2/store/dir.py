from __future__ import annotations

import os
import glob

from .store import Store
from ..utils.general import pickle_save, pickle_load

class DirStore(Store):
    """
    Store that keeps objects in a directory tree:
      base_dir/objects/<hh>/<full_hash>/...
    """
    def __init__(self, base_dir: str):
        self._base_dir = os.fspath(base_dir)
        self.obj_dir = os.path.join(self.base_dir, "objects")
        os.makedirs(self.obj_dir, exist_ok=True)
        self.set_main_def(self.read_main_def())

    def _object_dir(self, cdef: "ConcreteDefinition") -> str:
        digest = cdef.stable_hash()
        sub = digest[:2]
        return os.path.join(self.obj_dir, sub, digest)

    @property
    def base_dir(self) -> str:
        """Base directory"""
        return self._base_dir

    @property
    def object_root_dir(self) -> str:
        """Base directory"""
        return self.obj_dir

    def has(self, cdef: "ConcreteDefinition") -> bool:
        return os.path.exists(self._def_file(cdef))

    def hydrate_index(self) -> Iterable["ConcreteDefinition"]:
        # Walk the objects tree, load def.pkl for each, and yield cdef
        pattern = os.path.join(self.obj_dir, "[0-9a-f][0-9a-f]", "*", "def.pkl")
        for def_path in glob.glob(pattern):
            cdef = pickle_load(def_path)
            # TODO: Optional: sanity check hash matches directory name
            yield cdef

    def _main_def_path(self) -> str:
        return os.path.join(self.base_dir, "def.pkl")

    def read_main_def(self) -> ConcreteDefinition | None:
        path = self._main_def_path()
        if os.path.exists(path):
            return pickle_load(path)
        return None

    def set_main_def(self, main_def: ConcreteDefinition) -> None:
        self._main_def = main_def

    def write_main_def(self, main_def: ConcreteDefinition) -> None:
        path = self._main_def_path()
        pickle_save(main_def, path)

    def commit(self) -> None:
        if self._main_def is not None:
            self.write_main_def(self._main_def)

    def __repr__(self) -> str:
        return f"{type(self)}(dir: {self.base_dir})"
