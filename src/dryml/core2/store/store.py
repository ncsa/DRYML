from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

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
    def save_object(self, obj: Object, *, revision: str|None = None) -> None:
        """
        Save an individual object.
        Store is responsible for creating/using a directory (dir store),
        or temp dir (S3/HDF5/etc.), and calling obj.save_to_dir().
        """
        ...

    @abstractmethod
    def restore_object(self, obj: Object, *, revision: str|None = None) -> None:
        """
        Load data for this object, if present. Returns True if loaded,
        False if this store doesn't have data for it.
        Store is responsible for creating/using a directory and
        calling obj.load_from_dir().
        """
        ...

    def read_main_def(self) -> "ConcreteDefinition" | None:
        """Return the stored main ConcreteDefinition, or None if not present."""
        return None

    def write_main_def(self, main_def: "ConcreteDefinition") -> None:
        """Persist the given main ConcreteDefinition (no-op by default)."""
        pass

    def set_main_def(self, main_def: "ConcreteDefinition") -> None:
        """Set the store's main def"""
        pass

    def commit(self) -> None:
        """Optional; useful for zips, S3, HDF5, etc."""
        ...

    def close(self) -> None:
        """Cleanup (temp dirs, handles, etc.)"""
        ...
