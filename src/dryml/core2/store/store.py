from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
import os

from ..utils.general import pickle_load
from ..query.model import QueryIndexStatus, QueryIndexUnavailable, ReconcileReport, ValidationReport

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

    def object_dir(self, cdef: "ConcreteDefinition") -> str:
        """Return the store-local directory for an object's definition."""
        return self._object_dir(cdef)

    @property
    def records(self) -> "RecordStoreIO":
        """Return the optional record/spec sidecar IO facade for this store."""

        from dryml.records.store import RecordStoreIO

        return RecordStoreIO(self)

    def _def_file(self, cdef: "ConcreteDefinition") -> str:
        return os.path.join(self.object_dir(cdef), "def.pkl")

    def save_object(self, obj: Object, *, revision: str|None = None) -> None:
        """
        Save an individual object.
        Store is responsible for creating/using a directory (dir store),
        or temp dir (S3/HDF5/etc.), and calling obj.save_to_dir().
        """

        obj_dir = self.object_dir(obj.definition)
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

        obj_dir = self.object_dir(cdef)

        obj.restore_state_from_dir(obj_dir, revision=revision)

    def read_definition(self, cdef: "ConcreteDefinition") -> "ConcreteDefinition | None":
        def_path = self._def_file(cdef)
        if not os.path.exists(def_path):
            return None
        return pickle_load(def_path)

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

    def catalog_key(self) -> str:
        """Stable logical identity for query-catalog replica deduplication."""
        try:
            base_dir = getattr(self, "base_dir", None)
        except Exception:
            base_dir = None
        if base_dir is not None and base_dir is not Ellipsis:
            return f"{type(self).__module__}.{type(self).__qualname__}:{os.path.abspath(os.fspath(base_dir))}"
        return f"{type(self).__module__}.{type(self).__qualname__}:id:{id(self)}"

    def open_query_index(self):
        """Return this Store's optional query index, or None for memory/no index modes."""
        return None

    def query_index_status(self) -> QueryIndexStatus:
        """Return backend-neutral status for this Store's own query index.

        Stores that do not own a persistent query index report a disabled index.
        Concrete Store implementations may return richer backend-specific status.
        """
        return QueryIndexStatus(
            backend="none",
            store_key=self.catalog_key(),
            generation=None,
            schema_version=None,
            semantic_versions={},
            state="disabled",
        )

    def rebuild_query_index(self) -> ReconcileReport:
        """Rebuild this Store's query index from authoritative object state.

        The base Store has no persistent query index, so callers receive a clear
        unavailability error instead of a silent no-op.
        """
        raise QueryIndexUnavailable(f"Store {self!r} does not own a rebuildable query index.")

    def reconcile_query_index(self) -> ReconcileReport:
        """Repair this Store's query index against authoritative object state."""
        return self.rebuild_query_index()

    def validate_query_index(self, *, thorough: bool = False) -> ValidationReport:
        """Validate this Store's query index without exposing backend internals."""
        return ValidationReport("none", self.catalog_key(), True)

    def close(self) -> None:
        """Cleanup (temp dirs, handles, etc.)"""
        ...
