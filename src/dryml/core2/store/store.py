from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
import os
import shutil
import tempfile

from ..utils.general import is_regular_file, pickle_load
from ..query.model import QueryIndexStatus, QueryIndexUnavailable, ReconcileReport, ValidationReport

class StoreAuthorityError(RuntimeError):
    """Raised when an authoritative Store definition is malformed or misplaced."""


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

    def _def_file(self, cdef: "ConcreteDefinition") -> str:
        return os.path.join(self.object_dir(cdef), "def.pkl")

    def save_object(self, obj: Object, *, revision: str|None = None) -> None:
        """
        Save an individual object.
        Store is responsible for creating/using a directory (dir store),
        or temp dir (S3/HDF5/etc.), and calling obj.save_to_dir().
        """

        if revision is not None:
            if not isinstance(revision, str):
                raise ValueError("revision must be a string or None at the Store.")
        obj_dir = self.object_dir(obj.definition)
        parent = os.path.dirname(obj_dir)
        os.makedirs(parent, exist_ok=True)
        existing_root = os.path.exists(obj_dir)
        if existing_root:
            existing_def = self._validate_root_definition_path(os.path.join(obj_dir, "def.pkl"))
            if existing_def != obj.definition:
                raise StoreAuthorityError(
                    "Existing Store root identity does not match the object being saved."
                )
        stage_dir = tempfile.mkdtemp(prefix=f".{obj.definition.stable_hash()}-", dir=parent)
        try:
            # A root is undiscoverable until its complete staged definition has
            # been verified and the directory is atomically renamed.
            obj.save_state_to_dir(stage_dir, revision=revision)
            staged_def = self._read_concrete_definition(os.path.join(stage_dir, "def.pkl"))
            if staged_def != obj.definition:
                raise StoreAuthorityError("Staged object definition does not match the object identity.")
            if existing_root:
                self._publish_existing_root(stage_dir, obj_dir)
            else:
                os.replace(stage_dir, obj_dir)
                stage_dir = None
            self._mark_authority_dirty()
        finally:
            if stage_dir is not None:
                shutil.rmtree(stage_dir, ignore_errors=True)

    @staticmethod
    def _publish_existing_root(stage_dir: str, object_dir: str) -> None:
        """Atomically replace explicit state files without reserializing def.pkl."""

        for root, dirs, files in os.walk(stage_dir):
            rel_root = os.path.relpath(root, stage_dir)
            target_root = object_dir if rel_root == "." else os.path.join(object_dir, rel_root)
            for name in dirs:
                os.makedirs(os.path.join(target_root, name), exist_ok=True)
            for name in files:
                if rel_root == "." and name == "def.pkl":
                    continue
                os.replace(os.path.join(root, name), os.path.join(target_root, name))

    def _mark_authority_dirty(self) -> None:
        """Record an explicit authority mutation for buffered Store backends."""

    @staticmethod
    def _read_concrete_definition(path: str) -> "ConcreteDefinition":
        """Decode one regular persisted definition without class resolution."""

        if not is_regular_file(path):
            raise StoreAuthorityError(f"Store definition is not a regular file: {path!r}.")
        value = pickle_load(path)
        from ..definition import ConcreteDefinition

        if not isinstance(value, ConcreteDefinition):
            raise StoreAuthorityError(
                f"Store definition is {type(value).__name__}, not a ConcreteDefinition: {path!r}."
            )
        return value

    def _validate_root_definition_path(self, path: str) -> "ConcreteDefinition":
        """Decode and validate a complete object-root relative path and digest."""

        relative = os.path.relpath(path, self.object_root_dir)
        parts = relative.split(os.sep)
        if len(parts) != 3 or parts[2] != "def.pkl":
            raise StoreAuthorityError(f"Store root must be objects/<fanout>/<digest>/def.pkl: {path!r}.")
        fanout, digest, _ = parts
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise StoreAuthorityError(f"Store root has an invalid stable-hash digest: {path!r}.")
        if fanout != digest[:2]:
            raise StoreAuthorityError(f"Store root fanout does not match its digest: {path!r}.")
        cdef = self._read_concrete_definition(path)
        actual_digest = cdef.stable_hash()
        if actual_digest != digest:
            raise StoreAuthorityError(
                "Store def.pkl is stored under the wrong stable-hash directory. "
                f"path={path!r}, expected={digest!r}, actual={actual_digest!r}"
            )
        return cdef

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
        stored = self._validate_root_definition_path(def_path)
        if stored != cdef:
            raise StoreAuthorityError(
                "Store definition identity does not match the requested identity "
                f"at {def_path!r}."
            )
        return stored

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
