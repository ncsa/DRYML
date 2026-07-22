from __future__ import annotations

import os
import glob
import tempfile
import time
from dataclasses import replace
from typing import Literal

from .store import Store
from ..query.model import QueryIndexDirty, QueryIndexError, QueryIndexStatus, QueryIndexUnavailable, ReconcileReport
from ..query.sqlite import SQLiteQueryIndexConfig, sqlite_available
from ..query.sqlite.index import SQLiteStoreQueryIndex
from ..utils.general import pickle_save, pickle_load


QueryIndexPolicy = Literal["auto", "sqlite", "memory", "none"]

class DirStore(Store):
    """
    Store that keeps objects in a directory tree:
      base_dir/objects/<hh>/<full_hash>/...
    """
    def __init__(self, base_dir: str, *, query_index: QueryIndexPolicy | SQLiteQueryIndexConfig | object = "auto"):
        self._base_dir = os.fspath(base_dir)
        self.query_index = query_index
        self._query_index_policy, self._query_index_config, self._query_index_instance = self._normalize_query_index(query_index)
        self.obj_dir = os.path.join(self.base_dir, "objects")
        os.makedirs(self.obj_dir, exist_ok=True)
        self.set_main_def(self.read_main_def())

    @property
    def store_capabilities(self) -> frozenset[str]:
        """Advertise local managed control, locking, and activation support."""

        return super().store_capabilities | frozenset({
            "managed-snapshot-v1",
            "managed-control-v1",
            "managed-locking-v1",
            "managed-activation-v1",
            "managed-durable-products-v1",
        })

    def managed_control_root(self) -> str:
        """Return this DirStore's versioned live managed-control root."""

        return os.path.join(self.dryml_dir, "managed-v1")

    def managed_snapshot_root(self) -> str:
        """Return this DirStore's managed root for read-only inspection."""

        return self.managed_control_root()

    @property
    def dryml_dir(self) -> str:
        return os.path.join(self.base_dir, ".dryml")

    @property
    def query_index_path(self) -> str:
        return os.path.join(self.dryml_dir, "query-index-v1.sqlite")

    @property
    def query_index_dirty_path(self) -> str:
        return os.path.join(self.dryml_dir, "query-index.dirty")

    @property
    def query_index_registration_path(self) -> str:
        return os.path.join(self.dryml_dir, "query-index-v1.registered")

    @property
    def query_index_policy(self) -> QueryIndexPolicy | str:
        return self._query_index_policy

    def open_query_index(self):
        if self._query_index_instance is not None:
            if self._query_index_policy == "custom" and callable(self._query_index_instance):
                return self._call_query_index_factory(self._query_index_instance)
            return self._query_index_instance
        if self._query_index_policy in {"memory", "none"}:
            return None
        if self._query_index_policy == "auto" and not sqlite_available():
            return None
        if self._query_index_policy not in {"auto", "sqlite"}:
            return self._query_index_instance
        if self._query_index_policy == "sqlite" and not sqlite_available():
            raise QueryIndexUnavailable("DirStore query_index='sqlite' requires Python's optional sqlite3 module.")
        config = self._query_index_config
        path = self.query_index_path
        if config is None:
            config = SQLiteQueryIndexConfig(path=path)
        elif config.path is None:
            config = replace(config, path=path)
        else:
            path = os.fspath(config.path)
        if os.path.exists(path) or os.path.exists(f"{path}.building"):
            self.register_query_index_sidecar()
        self._query_index_instance = SQLiteStoreQueryIndex(
            source_key=self.catalog_key(),
            path=path,
            config=config,
            store=self,
            dirty_path=self.query_index_dirty_path,
        )
        return self._query_index_instance

    def save_object(self, obj, *, revision: str | None = None) -> None:
        """Publish an object and invalidate an existing persistent sidecar."""

        super().save_object(obj, revision=revision)
        if self._uses_nonpersistent_query_index():
            self._mark_existing_query_index_dirty()

    def save_definition(self, cdef) -> None:
        """Publish a definition and invalidate an existing persistent sidecar."""

        super().save_definition(cdef)
        if self._uses_nonpersistent_query_index():
            self._mark_existing_query_index_dirty()

    def mark_query_index_dirty(self) -> None:
        """Atomically publish a new dirty-marker generation.

        Transient destination replacement conflicts are retried for up to five
        seconds. A conflict that outlasts that bound is propagated.
        """

        os.makedirs(self.dryml_dir, exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=self.dryml_dir,
                    prefix=".query-index.dirty.",
                    suffix=".tmp",
                    delete=False) as f:
                tmp_path = f.name
                f.write(f"{os.urandom(16).hex()}\n")
                f.flush()
                os.fsync(f.fileno())
            deadline = time.monotonic() + 5.0
            delay = 0.01
            while True:
                try:
                    os.replace(tmp_path, self.query_index_dirty_path)
                    tmp_path = None
                    break
                except PermissionError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    time.sleep(min(delay, remaining))
                    delay = min(delay * 2, 0.25)
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    pass

    def clear_query_index_dirty(self) -> None:
        try:
            os.remove(self.query_index_dirty_path)
        except FileNotFoundError:
            pass

    def query_index_is_dirty(self) -> bool:
        return os.path.exists(self.query_index_dirty_path)

    def register_query_index_sidecar(self) -> None:
        """Record that this Store has a persistent SQLite query sidecar."""

        os.makedirs(self.dryml_dir, exist_ok=True)
        try:
            fd = os.open(
                self.query_index_registration_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            return
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("sqlite-v1\n")
            f.flush()
            os.fsync(f.fileno())

    def query_index_status(self) -> QueryIndexStatus:
        """Return status for this directory Store's configured query index.

        The method is Store-owned so callers do not need to know the sidecar
        path, backend policy, or SQLite implementation type.
        """
        if self._query_index_policy == "none":
            return QueryIndexStatus(
                backend="none",
                store_key=self.catalog_key(),
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="disabled",
            )
        if self._query_index_policy == "memory":
            return QueryIndexStatus(
                backend="memory",
                store_key=self.catalog_key(),
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="ready",
            )
        try:
            index = self.open_query_index()
        except QueryIndexUnavailable:
            if self._query_index_policy == "auto":
                return QueryIndexStatus(
                    backend="memory",
                    store_key=self.catalog_key(),
                    generation=None,
                    schema_version=None,
                    semantic_versions={},
                    state="ready",
                )
            raise
        if index is None:
            return QueryIndexStatus(
                backend="memory" if self._query_index_policy == "auto" else str(self._query_index_policy),
                store_key=self.catalog_key(),
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="ready" if self._query_index_policy == "auto" else "disabled",
            )
        status = getattr(index, "status", None)
        if status is None:
            return QueryIndexStatus(
                backend=type(index).__name__,
                store_key=self.catalog_key(),
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="ready",
            )
        return status()

    def rebuild_query_index(self) -> ReconcileReport:
        """Rebuild this Store's persistent query index from stored roots."""
        index = self._open_rebuildable_query_index()
        before = index.status()
        index.rebuild()
        after = index.status()
        if after.state != "ready":
            raise QueryIndexDirty(
                f"Query-index rebuild completed with post-rebuild state {after.state!r}."
            )
        return ReconcileReport(
            backend=after.backend,
            store_key=after.store_key,
            changed=True,
            action="rebuild",
            generation_before=before.generation,
            generation_after=after.generation,
            definitions_scanned=(after.row_counts or {}).get("stored_roots", 0),
            validated=True,
        )

    def reconcile_query_index(self) -> ReconcileReport:
        """Reconcile this Store's query index with authoritative object files.

        The current v1 reconciliation policy validates ready indexes and uses an
        exclusive rebuild for missing, dirty, corrupt, or incompatible indexes.
        """
        index = self._open_rebuildable_query_index()
        reconcile = getattr(index, "reconcile", None)
        if reconcile is not None:
            return reconcile()
        before = index.status()
        if before.state in {"missing", "dirty", "incompatible", "corrupt"}:
            return self.rebuild_query_index()
        validate = getattr(index, "validate", None)
        if validate is None:
            return ReconcileReport(
                backend=before.backend,
                store_key=before.store_key,
                changed=False,
                action="none",
                generation_before=before.generation,
                generation_after=before.generation,
            )
        report = validate(thorough=False)
        if not report.ok:
            rebuilt = self.rebuild_query_index()
            return ReconcileReport(
                backend=rebuilt.backend,
                store_key=rebuilt.store_key,
                changed=True,
                action="rebuild",
                generation_before=before.generation,
                generation_after=rebuilt.generation_after,
                definitions_scanned=rebuilt.definitions_scanned,
                validated=True,
                issues=report.issues,
            )
        return ReconcileReport(
            backend=before.backend,
            store_key=before.store_key,
            changed=False,
            action="validate",
            generation_before=before.generation,
            generation_after=before.generation,
            validated=True,
            issues=report.issues,
        )

    def validate_query_index(self, *, thorough: bool = False):
        """Validate this Store's configured query index."""
        index = self.open_query_index()
        if index is None or not hasattr(index, "validate"):
            return super().validate_query_index(thorough=thorough)
        return index.validate(thorough=thorough)

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
        """Yield stored root definitions after validating their hash paths.

        The directory name is part of the Store's stable-hash layout. A loaded
        `def.pkl` whose CDef hash does not match that directory is treated as
        Store corruption and rejected instead of being indexed silently.
        """

        # Walk the objects tree, load def.pkl for each, and yield cdef
        pattern = os.path.join(self.obj_dir, "[0-9a-f][0-9a-f]", "*", "def.pkl")
        for def_path in glob.glob(pattern):
            cdef = pickle_load(def_path)
            expected_hash = os.path.basename(os.path.dirname(def_path))
            actual_hash = cdef.stable_hash()
            if actual_hash != expected_hash:
                raise QueryIndexError(
                    "Store def.pkl is stored under the wrong stable-hash directory."
                    f" path={def_path!r}, expected={expected_hash!r}, actual={actual_hash!r}"
                )
            yield cdef

    def _main_def_path(self) -> str:
        return os.path.join(self.base_dir, "def.pkl")

    def _aliases_path(self) -> str:
        return os.path.join(self.base_dir, "aliases.pkl")

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

    def read_aliases(self) -> dict[str, ConcreteDefinition]:
        path = self._aliases_path()
        if os.path.exists(path):
            return pickle_load(path)
        return {}

    def write_aliases(self, aliases: dict[str, ConcreteDefinition]) -> None:
        pickle_save(dict(aliases), self._aliases_path())

    def commit(self) -> None:
        if self._main_def is not None:
            self.write_main_def(self._main_def)

    def __repr__(self) -> str:
        return f"{type(self)}(dir: {self.base_dir})"

    def catalog_key(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}:{os.path.abspath(self.base_dir)}"

    @staticmethod
    def _normalize_query_index(query_index):
        if isinstance(query_index, str):
            if query_index not in {"auto", "sqlite", "memory", "none"}:
                raise ValueError("DirStore query_index must be 'auto', 'sqlite', 'memory', 'none', or a query-index config/object.")
            return query_index, None, None
        if isinstance(query_index, SQLiteQueryIndexConfig):
            return "sqlite", query_index, None
        return "custom", None, query_index

    def _open_rebuildable_query_index(self):
        index = self.open_query_index()
        if index is None or not hasattr(index, "rebuild"):
            raise QueryIndexUnavailable(f"DirStore query_index={self._query_index_policy!r} does not provide a rebuildable persistent index.")
        return index

    def _mark_existing_query_index_dirty(self) -> None:
        claim_path = f"{self.query_index_path}.building"
        if (
                os.path.exists(self.query_index_registration_path)
                or os.path.exists(self.query_index_path)
                or os.path.exists(claim_path)):
            self.mark_query_index_dirty()

    def _uses_nonpersistent_query_index(self) -> bool:
        return (
            self._query_index_policy in {"memory", "none"}
            or (
                self._query_index_policy == "auto"
                and not sqlite_available()
            )
        )

    def _call_query_index_factory(self, factory):
        try:
            return factory(self)
        except TypeError as original:
            try:
                return factory()
            except TypeError:
                raise original
