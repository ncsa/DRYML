from __future__ import annotations

import os
import glob
from dataclasses import replace
from typing import Literal

from .store import Store
from ..query.model import QueryIndexUnavailable
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
    def dryml_dir(self) -> str:
        return os.path.join(self.base_dir, ".dryml")

    @property
    def query_index_path(self) -> str:
        return os.path.join(self.dryml_dir, "query-index-v1.sqlite")

    @property
    def query_index_dirty_path(self) -> str:
        return os.path.join(self.dryml_dir, "query-index.dirty")

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
        return SQLiteStoreQueryIndex(
            source_key=self.catalog_key(),
            path=path,
            config=config,
            store=self,
            dirty_path=self.query_index_dirty_path,
        )

    def mark_query_index_dirty(self) -> None:
        os.makedirs(self.dryml_dir, exist_ok=True)
        tmp_path = f"{self.query_index_dirty_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("dirty\n")
        os.replace(tmp_path, self.query_index_dirty_path)

    def clear_query_index_dirty(self) -> None:
        try:
            os.remove(self.query_index_dirty_path)
        except FileNotFoundError:
            pass

    def query_index_is_dirty(self) -> bool:
        return os.path.exists(self.query_index_dirty_path)

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

    def _call_query_index_factory(self, factory):
        try:
            return factory(self)
        except TypeError as original:
            try:
                return factory()
            except TypeError:
                raise original
