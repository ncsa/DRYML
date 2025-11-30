from __future__ import annotations

from .store import Store
from ..utils.general import pickle_save, pickle_load

class DirStore(Store):
    """
    Store that keeps objects in a directory tree:
      base_dir/objects/<hh>/<full_hash>/...
    """
    def __init__(self, base_dir: str):
        self.base_dir = os.fspath(base_dir)
        self.obj_dir = os.path.join(self.base_dir, "objects")
        os.makedirs(self.obj_dir, exist_ok=True)

    def _object_dir(self, cdef: "ConcreteDefinition") -> str:
        digest = cdef.stable_hash()
        sub = digest[:2]
        return os.path.join(self.obj_dir, sub, digest)

    def _def_file(self, cdef: "ConcreteDefinition") -> str:
        return os.path.join(self._object_dir(cdef), "def.pkl")

    def has_cdef(self, cdef: "ConcreteDefinition") -> bool:
        return os.path.exists(self._def_file(cdef))

    def hydrate_index(self) -> Iterable["ConcreteDefinition"]:
        # Walk the objects tree, load def.pkl for each, and yield cdef
        pattern = os.path.join(self.obj_dir, "[0-9a-f][0-9a-f]", "*", "def.pkl")
        for def_path in glob.glob(pattern):
            obj_dir = os.path.dirname(def_path)
            cdef = pickle_load(def_path)
            # Optional: sanity check hash matches directory name
            yield cdef

    def save_object(self, obj: "Object") -> None:
        cdef = obj.definition
        obj_dir = self._object_dir(cdef)
        os.makedirs(obj_dir, exist_ok=True)

        # Save definition for checking later / quick load
        def_path = os.path.join(obj_dir, "def.pkl")
        pickle_save(cdef, def_path)

        # Let the object serialize itself
        obj.save_to_dir(obj_dir)

    def load_object(self, obj: "Object") -> bool:
        cdef = obj.definition
        def_path = self._def_file(cdef)
        if not os.path.exists(def_path):
            return False

        obj_dir = os.path.dirname(def_path)

        # Confirm definition matches
        stored_cdef = pickle_load(def_path)
        if stored_cdef.stable_hash() != cdef.stable_hash():
            raise RepoLoadError("Definition hash mismatch while loading object.")

        obj.load_from_dir(obj_dir)
        return True

