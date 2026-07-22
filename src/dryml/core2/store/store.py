from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
import os
import tempfile

from ..utils.general import pickle_load
from ..query.model import QueryIndexStatus, QueryIndexUnavailable, ReconcileReport, ValidationReport

class Store(ABC):
    @property
    def store_capabilities(self) -> frozenset[str]:
        """Return explicit optional Store capabilities.

        Optional higher layers must check these names before using Store-owned
        sidecars. The base Store supports no managed snapshot or live-write
        capabilities.
        """

        return frozenset()

    def supports_store_capability(self, capability: str) -> bool:
        """Return whether this Store explicitly advertises *capability*."""

        return capability in self.store_capabilities

    def managed_control_root(self) -> str:
        """Return the Store-local managed control root when supported."""

        raise NotImplementedError("Store does not support live managed control")

    def managed_snapshot_root(self) -> str:
        """Return the Store-local read-only managed snapshot root when supported."""

        raise NotImplementedError("Store does not support managed snapshot reads")

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

    def object_dir_for_cdef_id(self, cdef_id: str) -> str:
        """Return the store-local object directory for a full CDef ID string.

        Stores with non-standard object layouts may override this hook. The base
        implementation matches the current shard layout and requires a full
        64-character CDef digest.
        """

        from dryml.formats.refs import parse_cdef_id

        parsed = parse_cdef_id(cdef_id)
        if len(parsed.digest) != 64:
            raise ValueError("object-dir refs require a full CDef digest for direct resolution")
        return os.path.join(self.object_root_dir, parsed.digest[:2], parsed.digest)

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

    def save_definition(self, cdef: "ConcreteDefinition") -> None:
        """Persist an Object definition without constructing or saving its state."""

        from ..definition import ConcreteDefinition

        if not isinstance(cdef, ConcreteDefinition):
            raise TypeError("Store.save_definition requires a ConcreteDefinition.")
        obj_dir = self.object_dir(cdef)
        os.makedirs(obj_dir, exist_ok=True)
        existing = self.read_definition(cdef)
        if existing is not None and existing != cdef:
            raise ValueError("Stored definition does not match its stable-hash location.")
        if existing == cdef:
            return
        from ..utils.general import pickler

        path = self._def_file(cdef)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=obj_dir, prefix=".def.pkl.", delete=False
            ) as handle:
                handle.write(pickler(cdef))
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = handle.name
            os.replace(temp_path, path)
            if os.name == "posix":
                directories = dict.fromkeys(
                    (
                        obj_dir,
                        os.path.dirname(obj_dir),
                        os.fspath(self.object_root_dir),
                    )
                )
                for directory in directories:
                    descriptor = os.open(
                        directory,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    )
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)

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
