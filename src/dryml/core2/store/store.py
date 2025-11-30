from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

class Store(ABC):
    @abstractmethod
    def has_cdef(self, cdef: "ConcreteDefinition") -> bool:
        """Lightweight membership test: do you have data for this cdef?"""
        ...

    @abstractmethod
    def hydrate_index(self) -> Iterable["ConcreteDefinition"]:
        """
        Full hydration: scan underlying storage and yield all cdefs
        that have data here. Repo will populate obj_cache[cdef] = None.
        """
        ...

    @abstractmethod
    def save_object(self, obj: "Object") -> None:
        """
        Save an individual object.
        Store is responsible for creating/using a directory (dir store),
        or temp dir (S3/HDF5/etc.), and calling obj.save_to_dir().
        """
        ...

    @abstractmethod
    def load_object(self, obj: "Object") -> bool:
        """
        Load data for this object, if present. Returns True if loaded,
        False if this store doesn't have data for it.
        Store is responsible for creating/using a directory and
        calling obj.load_from_dir().
        """
        ...

    def commit(self) -> None:
        """Optional; useful for zips, S3, HDF5, etc."""
        ...

    def close(self) -> None:
        """Cleanup (temp dirs, handles, etc.)"""
        ...
