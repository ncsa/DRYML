from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
import os

class Store(ABC):
    @property
    def base_dir(self) -> str:
        """Base directory"""
        ...

    @property
    def object_root_dir(self) -> str:
        """Base directory"""
        ...

    @abstractmethod
    def has(self, cdef: "ConcreteDefinition") -> bool:
        """Lightweight membership test: do you have data for this cdef?"""
        ...

    @abstractmethod
    def hydrate_index(self) -> Iterable["ConcreteDefinition"]:
        """
        Full hydration: scan underlying storage and yield all cdefs
        that have data here.
        """
        ...

    @abstractmethod
    def _object_dir(self, cdef: "ConcreteDefinition") -> str:
        """
        Method to get the object directory for this cdef
        """

    def _def_file(self, cdef: "ConcreteDefinition") -> str:
        return os.path.join(self._object_dir(cdef), "def.pkl")

    def save_object(self, obj: Object, *, revision: str|None = None) -> None:
        """
        Save an individual object.
        Store is responsible for creating/using a directory (dir store),
        or temp dir (S3/HDF5/etc.), and calling obj.save_to_dir().
        """

        obj_dir = self._object_dir(obj.definition)
        os.makedirs(obj_dir, exist_ok=True)

        # Let the object serialize itself
        if revision is not None:
            if not isinstance(revision, str):
                raise ValueError("revision must be a string or None at the Store.")
        obj.save_state_to_dir(obj_dir, revision=revision)

    def restore_object(self, obj: Object, *, revision: str|None = None) -> None:
        """
        Load data for this object, if present. Returns True if loaded,
        False if this store doesn't have data for it.
        Store is responsible for creating/using a directory and
        calling obj.load_from_dir().
        """

        cdef = obj.definition
        def_path = self._def_file(cdef)
        if not os.path.exists(def_path):
            return

        obj_dir = self._object_dir(cdef)

        obj.restore_state_from_dir(obj_dir, revision=revision)

    def read_main_def(self) -> "ConcreteDefinition" | None:
        """Return the stored main ConcreteDefinition, or None if not present."""
        return None

    def write_main_def(self, main_def: "ConcreteDefinition") -> None:
        """Persist the given main ConcreteDefinition (no-op by default)."""
        pass

    def set_main_def(self, main_def: "ConcreteDefinition") -> None:
        """Set the store's main def"""
        pass

    def read_aliases(self) -> dict[str, "ConcreteDefinition"]:
        """Return stored object aliases, or an empty mapping if unsupported."""
        return {}

    def write_aliases(self, aliases: dict[str, "ConcreteDefinition"]) -> None:
        """Persist object aliases. Stores may no-op if aliases are unsupported."""
        pass

    def commit(self) -> None:
        """Optional; useful for zips, S3, HDF5, etc."""
        ...

    def close(self) -> None:
        """Cleanup (temp dirs, handles, etc.)"""
        ...
