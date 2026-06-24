from __future__ import annotations

import os
import glob
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager
from io import IOBase
from pathlib import Path
import weakref
from contextvars import ContextVar
from collections.abc import Iterable, Mapping
import numpy as np
import atexit

from .definition import Definition, ConcreteDefinition
from .object import Object, Serializable
from .store.store import Store
from .policies import InstancePolicy, CachePolicy, RepoGraphOptions, RepoLoadOptions, RepoSaveOptions
from .repo_graph import (
    RepoGraphApplyVisitor,
    RepoGraphCollectVisitor,
    RepoSaveVisitor,
    RepoAddObjectsVisitor,
    manage_revision,
)
from .canonical import from_canonical
from .config import CONFIG_MISSING, ConfigError, ConfigRef
from .query.index import DefinitionCatalog
from .query.result import ObjectResultSet


class RepoSaveError(Exception):
    pass


class RepoLoadError(Exception):
    pass


class RepoGraphError(Exception):
    pass


SelectorType = Callable | Definition | ConcreteDefinition
RevisionType = dict[ConcreteDefinition, str]


class Repo:
    # Trackers
    _num_saves: int
    _num_constructions: int

    # Caches
    # Links particular concrete definition with particular object
    weak_obj_cache: weakref.WeakValueDictionary[ConcreteDefinition, Object]
    strong_obj_cache: dict[ConcreteDefinition, Object]
    obj_default_store: dict[ConcreteDefinition, Store]

    # known to exist in stores
    light_index: set[ConcreteDefinition]

    # Links particular Definition object with a concrete definition (Definitions are resolved )
    cdef_cache: weakref.WeakValueDictionary[str, ConcreteDefinition]

    # Main definition
    main_def: ConcreteDefinition | None

    # Backing stores
    stores: list[Store]

    # Object config store
    obj_config: dict[ConcreteDefinition, Any]

    # Runtime config values, intentionally not persisted by stores.
    config: dict[str, Any]

    # User-facing alias index
    alias_index: dict[str, ConcreteDefinition]

    # Settings
    save_objs_on_deletion: bool = False


    # Helper class for saving objects
    def __init__(self, stores=None, config: Mapping[str, Any] | None = None):
        # Initialize caches
        self.weak_obj_cache = weakref.WeakValueDictionary()
        self.strong_obj_cache = {}
        self.obj_default_store = {}
        self.light_index = set()
        self.cdef_cache = weakref.WeakValueDictionary()
        self.obj_config = {}
        self.config = dict(config or {})
        self.alias_index = {}
        self._query_catalog = DefinitionCatalog(self)

        # Some helper variables for monitoring
        self._num_saves = 0
        self._num_constructions = 0

        # Initialize the main def
        self.main_def = None

        # Multiple stores, optional
        self.stores = []
        if stores is not None:
            if not isinstance(stores, (tuple, list)):
                stores = [stores]

            for store in stores:
                if not isinstance(store, Store):
                    self.stores.append(make_store(store))
                else:
                    self.stores.append(store)

        # Try to read main def from first store
        if len(self.stores) > 0:
            # Attempt to read from the default store first.
            main_def = None
            if self.default_store is not None:
                main_def = self.default_store.read_main_def()
            if main_def is None:
                for store in self.stores:
                    main_def = store.read_main_def()
                    if main_def is not None:
                        break
            if main_def is not None:
                self.set_main_def(main_def)

            self._load_aliases_from_stores()

    # Store Methods

    @property
    def default_store(self):
        return self.stores[0] if len(self.stores) > 0 else None

    def set_default_store(self, store: "Store"):
        if not isinstance(store, Store):
            store = make_store(store)
        if store not in self.stores:
            self.stores.insert(0, store)
        else:
            # Find and move to front
            store_idx = self.stores.index(store)
            self.stores.insert(0, self.stores.pop(store_idx))

    def add_store(self, store: "Store", make_default=False):
        if not isinstance(store, Store):
            store = make_store(store)
        if make_default or self.default_store is None:
            self.stores.insert(0, store)
        else:
            self.stores.append(store)

    def _ensure_store(self, store):
        if store is None:
            return None
        store = make_store(store)
        if store not in self.stores:
            self.add_store(store)
        return store

    def cache_strong(self, obj: Object) -> None:
        self.strong_obj_cache[obj.__cdef__] = obj
        self._query_catalog.register_cached(obj.__cdef__)

    def cache_weak(self, obj: Object) -> None:
        self.weak_obj_cache[obj.__cdef__] = obj
        self._query_catalog.register_cached(obj.__cdef__)

    # --- helpers you already have ---
    def get_cached(self, cdef, *, reuse_weak: bool = True):
        obj = self.strong_obj_cache.get(cdef)
        if obj is not None:
            return obj
        if reuse_weak:
            return self.weak_obj_cache.get(cdef)
        return None

    def pin(self, obj):
        """Promote to strong cache."""
        cdef = obj.__cdef__
        self.strong_obj_cache[cdef] = obj
        self.weak_obj_cache.pop(cdef, None)
        self._query_catalog.register_cached(cdef)

    def unpin(self, obj_or_cdef):
        """Demote to weak cache."""
        cdef = obj_or_cdef if isinstance(obj_or_cdef, ConcreteDefinition) else obj_or_cdef.__cdef__
        obj = self.strong_obj_cache.pop(cdef, None)
        if obj is not None:
            self.weak_obj_cache[cdef] = obj
            self._query_catalog.register_cached(cdef)

    def _load_aliases_from_stores(self) -> None:
        for store in self.stores:
            for alias, cdef in store.read_aliases().items():
                self._validate_alias(alias)
                if not isinstance(cdef, ConcreteDefinition):
                    raise RepoLoadError(
                        f"Alias {alias!r} points to {type(cdef).__name__}, not a ConcreteDefinition."
                    )
                existing = self.alias_index.get(alias)
                if existing is not None and existing != cdef:
                    raise RepoLoadError(f"Conflicting definitions found for alias {alias!r}.")
                self.alias_index[alias] = cdef

    @staticmethod
    def _validate_alias(alias: str) -> None:
        if not isinstance(alias, str):
            raise TypeError("Object aliases must be strings.")
        if alias == "":
            raise ValueError("Object aliases cannot be empty strings.")

    def _alias_target_cdef(self, target: Object | Definition | ConcreteDefinition) -> ConcreteDefinition:
        if isinstance(target, Object):
            return target.definition
        if isinstance(target, ConcreteDefinition):
            return target
        if isinstance(target, Definition):
            return target.concretize(repo=self)
        raise TypeError(
            "Alias target must be an Object, Definition, or ConcreteDefinition."
        )

    def _object_target_cdef(self, target: Object | Definition | ConcreteDefinition) -> ConcreteDefinition:
        if isinstance(target, Object):
            return target.definition
        if isinstance(target, ConcreteDefinition):
            return target
        if isinstance(target, Definition):
            return target.concretize(repo=self)
        raise TypeError(
            "Object target must be an Object, Definition, or ConcreteDefinition."
        )

    def set_object_store(self, target: Object | Definition | ConcreteDefinition, store) -> Store:
        store = self._ensure_store(store)
        if store is None:
            raise ValueError("No store provided for object store binding.")
        cdef = self._object_target_cdef(target)
        self.obj_default_store[cdef] = store
        return store

    def location(
            self,
            target: Object | Definition | ConcreteDefinition,
            *,
            store=None,
            require_exists: bool = False) -> str:
        cdef = self._object_target_cdef(target)

        if store is not None:
            store = self.set_object_store(cdef, store)
        else:
            store = self.obj_default_store.get(cdef)
            if store is None:
                store = self._first_store_with(cdef)
                if store is not None:
                    self.obj_default_store[cdef] = store
            if store is None:
                store = self.default_store

        if store is None:
            raise RuntimeError("No store available for object location.")
        if require_exists and not store.has(cdef):
            raise RuntimeError("Object is not saved in the selected store.")
        return store.object_dir(cdef)

    def set_alias(
            self,
            alias: str,
            target: Object | Definition | ConcreteDefinition,
            *,
            store=None,
            save_live: bool = True) -> ConcreteDefinition:
        self._validate_alias(alias)

        store = self._ensure_store(store)

        cdef = self._alias_target_cdef(target)
        if isinstance(target, Object):
            if save_live:
                self.save_object(target, store=store)
            else:
                self.add_objects(target, store=store)

        self.alias_index[alias] = cdef
        return cdef

    def get_alias(self, alias: str) -> ConcreteDefinition:
        self._validate_alias(alias)
        try:
            return self.alias_index[alias]
        except KeyError as e:
            raise KeyError(f"Repo has no alias {alias!r}.") from e

    def delete_alias(self, alias: str) -> ConcreteDefinition:
        self._validate_alias(alias)
        try:
            return self.alias_index.pop(alias)
        except KeyError as e:
            raise KeyError(f"Repo has no alias {alias!r}.") from e

    def aliases(self) -> dict[str, ConcreteDefinition]:
        return dict(self.alias_index)

    def load_alias(self, alias: str, **kwargs):
        return self.load_object(self.get_alias(alias), **kwargs)

    def set_config(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError("Config keys must be strings.")
        if key == "":
            raise ValueError("Config keys cannot be empty.")
        self.config[key] = value

    def update_config(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            self.set_config(key, value)

    def get_config(self, key: str, default=CONFIG_MISSING) -> Any:
        if not isinstance(key, str):
            raise TypeError("Config keys must be strings.")
        if key in self.config:
            return self.config[key]

        cur = self.config
        found_nested = True
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                found_nested = False
                break
        if found_nested:
            return cur

        if default is not CONFIG_MISSING:
            return default
        raise ConfigError(f"Repo config has no value for {key!r}.")

    def resolve_config(self, value: Any) -> Any:
        if isinstance(value, ConfigRef):
            if value.has_default:
                return self.get_config(value.key, default=value.default)
            return self.get_config(value.key)

        if isinstance(value, Mapping):
            return {k: self.resolve_config(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self.resolve_config(v) for v in value)
        if isinstance(value, list):
            return [self.resolve_config(v) for v in value]
        if isinstance(value, (set, frozenset)):
            return type(value)(self.resolve_config(v) for v in value)

        return value

    def has_cdef_light(self, cdef: ConcreteDefinition) -> bool:
        # do any stores have data for this cdef?
        return any(store.has(cdef) for store in self.stores)

    def hydrate_from_stores(self):
        """
        Ask each store to enumerate all cdefs it has.
        Populate obj_cache[cdef] = None for those not already present.
        """
        self._query_catalog.refresh(True)

    def refresh_index(self, *, force: bool = True):
        self._query_catalog.refresh(True if force else "auto")
        return self

    def __len__(self):
        return len(self.strong_obj_cache)

    def _save_options(
            self,
            *,
            options: RepoSaveOptions | None = None,
            main: bool = False,
            store=None,
            revision: RevisionType | str | None = None,
            alias: str | None = None,
            ephemeral_depth: int | None = 0) -> RepoSaveOptions:
        if options is not None:
            return options
        return RepoSaveOptions(
            main=main,
            store=store,
            revision=revision,
            alias=alias,
            ephemeral_depth=ephemeral_depth,
        )

    def save_object(
            self,
            obj,
            main=False,
            store=None,
            revision=None,
            alias: str | None = None,
            ephemeral_depth: int | None = 0,
            options: RepoSaveOptions | None = None):
        save_options = self._save_options(
            options=options,
            main=main,
            store=store,
            revision=revision,
            alias=alias,
            ephemeral_depth=ephemeral_depth,
        )
        store = self._ensure_store(save_options.store)
        revision = manage_revision(obj, save_options.revision)
        self.add_objects(obj, store=store)
        from .repo_plan import build_save_plan, execute_save_plan

        plan = build_save_plan(
            self,
            obj,
            store=store,
            revision=revision,
            ephemeral_depth=save_options.ephemeral_depth,
        )
        execute_save_plan(self, plan)

        if save_options.main:
            self.set_main_def(obj.definition, store=store)
        if save_options.alias is not None:
            self.set_alias(save_options.alias, obj, store=store, save_live=False)
        return True

    def save(
            self,
            obj: Object | None = None,
            store=None,
            revision: RevisionType | str | None = None,
            ephemeral_depth: int | None = 0,
            options: RepoSaveOptions | None = None):
        save_options = self._save_options(
            options=options,
            main=False,
            store=store,
            revision=revision,
            ephemeral_depth=ephemeral_depth,
        )
        if obj is None:
            # Save all loaded objects in the cache
            obj_list = []
            for _, obj in self.strong_obj_cache.items():
                if obj is not None:
                    obj_list.append(obj)
            self.save_object(obj_list, options=save_options)
            self.flush()
        else:
            self.save_object(obj, options=save_options)
            self.flush()

    def _first_store_with(self, cdef):
        for st in self.stores:
            if st.has(cdef):
                return st
        return None

    def _load_options(
            self,
            *,
            options: RepoLoadOptions | None = None,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            build_missing: bool = False,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision: RevisionType | str | None = None) -> RepoLoadOptions:
        if options is not None:
            return options
        return RepoLoadOptions(
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )

    def _candidate_cdefs(self, *, reuse_weak: bool = True) -> set[ConcreteDefinition]:
        cdefs = set(self.strong_obj_cache.keys())
        if reuse_weak:
            cdefs.update(self.weak_obj_cache.keys())
        cdefs.update(self.light_index)
        return cdefs

    @staticmethod
    def _selector_tuple(selector):
        if type(selector) is list:
            return tuple(selector)
        if type(selector) is tuple:
            return selector
        return (selector,)

    # -------------------------------------------------------------------------
    # Core: realize arbitrary structure into runtime Python + Objects
    # -------------------------------------------------------------------------
    def _realize(
        self,
        x: Any,
        *,
        instance: InstancePolicy = "reuse",
        restore_state: bool = True,
        build_missing: bool = False,
        reuse_weak: bool = True,
        cache: CachePolicy = "weak",
        revision: RevisionType | None = None,
        options: RepoLoadOptions | None = None,
        memo: dict | None = None,
        path: list[str | int] | None = None,
    ):
        load_options = self._load_options(
            options=options,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )
        return from_canonical(
            x,
            repo=self,
            options=load_options,
            memo=memo,
            path=path,
        )

    # -------------------------------------------------------------------------
    # Core: turn a ConcreteDefinition into a live Object under load knobs
    # -------------------------------------------------------------------------
    def _materialize_cdef(
        self,
        cdef,
        revision: RevisionType | str | None = None,
        *,
        options: RepoLoadOptions | None = None,
        instance: InstancePolicy = "reuse",
        restore_state: bool = True,
        build_missing: bool = False,
        reuse_weak: bool = True,
        cache: CachePolicy = "weak",
        # internal
        memo: dict | None = None,   # cdef->obj memo for this realization pass
        path: list[str | int] | None = None,
    ):
        if memo is None:
            memo = {}
        if path is None:
            path = ["<root>"]

        load_options = self._load_options(
            options=options,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )
        revision = manage_revision(cdef, load_options.revision)
        from .materialization import build_materialization_plan, execute_materialization_plan

        plan = build_materialization_plan(
            self,
            cdef,
            load_options,
            memo=memo,
            path=path,
        )
        return execute_materialization_plan(
            self,
            plan,
            memo=memo,
            revision=revision,
            root=cdef,
        )


    def load_object(
        self,
        x: object,
        *,
        instance: InstancePolicy = "reuse",
        restore_state: bool = True,
        build_missing: bool = False,
        reuse_weak: bool = True,
        cache: CachePolicy = "weak",
        revision: RevisionType|str | None = None,
        options: RepoLoadOptions | None = None,
    ):
        load_options = self._load_options(
            options=options,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )
        memo: dict[ConcreteDefinition, Object] = {}
        return self._realize(
            x,
            options=load_options,
            path=[""],
            memo=memo,
        )

    def load(self, cdef: ConcreteDefinition, **kwargs) -> Object:
        if not isinstance(cdef, ConcreteDefinition):
            raise TypeError("Repo.load requires an exact ConcreteDefinition.")
        if kwargs.get("options") is not None:
            kwargs["options"] = replace(kwargs["options"], build_missing=False)
        kwargs["build_missing"] = False
        return self.load_object(cdef, **kwargs)

    def load_or_build(self, x: object, **kwargs):
        if isinstance(x, Object):
            x = x.definition
        elif isinstance(x, Definition):
            x = x.concretize(repo=self)
        elif not isinstance(x, ConcreteDefinition):
            raise TypeError("Repo.load_or_build requires a Definition, ConcreteDefinition, or Object.")
        if kwargs.get("options") is not None:
            kwargs["options"] = replace(kwargs["options"], build_missing=True)
        kwargs["build_missing"] = True
        return self.load_object(x, **kwargs)


    def __contains__(
            self, item: Object | ConcreteDefinition, weak=True):
        # if weak is true, check both strong and weak caches
        if isinstance(item, ConcreteDefinition):
            cdef = item
        elif isinstance(item, Object):
            cdef = item.definition
        else:
            raise TypeError(
                f"Unsupported type {type(item)} for Repo.__contains__!")

        # “Strong” membership: known in cache and either loaded or known to exist
        in_cache = cdef in self.strong_obj_cache
        if not in_cache and weak:
            in_cache = cdef in self.weak_obj_cache
        in_store = cdef in self.light_index or bool(self._query_catalog.stores_for_cdef(cdef))
        return in_cache or in_store

    def __getitem__(
            self, key: ConcreteDefinition):
        """
        Easy access to objects within.

        if unpack is true, plain objects are returned
        """
        if not isinstance(key, ConcreteDefinition):
            raise TypeError("Repo.__getitem__ requires a ConcreteDefinition key.")
        result = self.query(key).known().objects()
        if len(result) == 0:
            raise KeyError(f"Repo doesn't contain an object with definition {key}")
        return result.one()

    def query(self, selector=None):
        from .query import DefinitionQuery

        return DefinitionQuery.from_source(self, selector)

    def definition_graph(self, value) -> "ConcreteDefinitionGraph":
        from .cdef_graph import ConcreteDefinitionGraph

        def cdef_from(item):
            if isinstance(item, Object):
                return item.definition
            if isinstance(item, ConcreteDefinition):
                return item
            if isinstance(item, Definition):
                raise TypeError("definition_graph() requires exact ConcreteDefinition values; concretize Definitions first.")
            raise TypeError(f"definition_graph() cannot inspect {type(item).__name__}.")

        if isinstance(value, (Object, ConcreteDefinition, Definition)):
            return ConcreteDefinitionGraph.from_root(cdef_from(value))
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return ConcreteDefinitionGraph.from_roots(cdef_from(item) for item in value)
        raise TypeError(f"definition_graph() cannot inspect {type(value).__name__}.")

    def find_defs(
            self,
            selector=None,
            *,
            scope: str = "stored",
            refresh="auto",
            class_match: str = "selector"):
        q = self.query(selector).class_match(class_match).refresh(refresh)
        if scope == "stored":
            return q.stored().defs()
        if scope == "known":
            return q.known().defs()
        if scope == "cached":
            return q.cached().defs()
        if scope == "nested":
            return q.nested().definitions().defs()
        raise ValueError("scope must be 'stored', 'known', 'cached', or 'nested'.")

    def find_occurrences(
            self,
            selector=None,
            *,
            refresh="auto",
            class_match: str = "selector"):
        return self.query(selector).class_match(class_match).refresh(refresh).nested().execute()

    def find_owner_defs(
            self,
            selector=None,
            *,
            refresh="auto",
            class_match: str = "selector"):
        return self.query(selector).class_match(class_match).refresh(refresh).nested().owners().defs()

    def find(
            self,
            selector=None,
            *,
            scope: str = "stored",
            refresh="auto",
            class_match: str = "selector",
            **load_options):
        q = self.query(selector).class_match(class_match).refresh(refresh)
        if scope == "stored":
            q = q.stored()
        elif scope == "known":
            q = q.known()
        elif scope == "cached":
            q = q.cached()
        else:
            raise ValueError("scope must be 'stored', 'known', or 'cached'.")
        return q.objects(**load_options)

    def find_owners(
            self,
            selector=None,
            *,
            refresh="auto",
            class_match: str = "selector",
            **load_options):
        return (
            self.query(selector)
            .class_match(class_match)
            .refresh(refresh)
            .nested()
            .owners()
            .objects(**load_options)
        )

    def get(self,
            selector:  SelectorType | tuple[SelectorType] | list[SelectorType] | None = None,
            sel_args=None, sel_kwargs=None,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            build_missing: bool = False,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision: RevisionType | str | None = None,
            options: RepoLoadOptions | None = None,
            verbose: bool = True) -> ObjectResultSet:
        if sel_args is None:
            sel_args = []
        if sel_kwargs is None:
            sel_kwargs = {}
        load_options = self._load_options(
            options=options,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )
        if load_options.build_missing:
            raise ValueError("Repo.get selects existing objects only; use Repo.load_or_build for construction.")
        selectors = self._selector_tuple(selector)
        if isinstance(load_options.revision, str):
            raise ValueError("plain string revisions aren't supported in `get`.")
        selected_objects: dict[ConcreteDefinition, Object] = {}
        for sel in selectors:
            if isinstance(sel, Callable) and not isinstance(sel, (Definition, ConcreteDefinition)):
                for cdef, obj in self.strong_obj_cache.items():
                    if sel(obj, *sel_args, **sel_kwargs):
                        selected_objects[cdef] = obj
                continue

            objs = (
                self.query(sel)
                .known()
                .reuse_weak(load_options.reuse_weak)
                .objects(options=load_options)
            )
            selected_objects.update(objs)

        return ObjectResultSet(self, selected_objects, domain="known")

    def apply(self,
              func, func_args=None, func_kwargs=None,
              selector: Optional[Callable] = None,
              sel_args=None, sel_kwargs=None,
              verbose: bool = False,
              options: RepoLoadOptions | None = None,
              **kwargs):
        """
        Apply a function to all objects tracked by the repo.
        We can also use a Selector to apply only to specific models
        **kwargs is passed to self.get
        """
        if func_args is None:
            func_args = []
        if func_kwargs is None:
            func_kwargs = {}

        # Create apply function
        def apply_func(obj):
            return func(obj, *func_args, **func_kwargs)

        # Get object list
        objs = self.get(
            selector=selector,
            sel_args=sel_args, sel_kwargs=sel_kwargs,
            options=options,
            **kwargs)

        obj_iter = objs.items()
        if verbose:
            obj_iter = tqdm(obj_iter)
        return {
            obj_def: apply_func(obj) for obj_def, obj in obj_iter
        }

    def _graph_options(
            self,
            *,
            options: RepoGraphOptions | None = None,
            include_root: bool = True,
            order: str = "post",
            missing: str = "raise",
            dedupe: bool = True,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            build_missing: bool = False,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision: RevisionType | str | None = None) -> RepoGraphOptions:
        if options is not None:
            return options
        load_options = self._load_options(
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )
        return RepoGraphOptions(
            load=load_options,
            include_root=include_root,
            order=order,
            missing=missing,
            dedupe=dedupe,
        )

    def iter_graph(
            self,
            root,
            *,
            options: RepoGraphOptions | None = None,
            include_root: bool = True,
            order: str = "post",
            missing: str = "raise",
            dedupe: bool = True,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            build_missing: bool = False,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision: RevisionType | str | None = None):
        graph_options = self._graph_options(
            options=options,
            include_root=include_root,
            order=order,
            missing=missing,
            dedupe=dedupe,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )
        from .repo_plan import iter_graph_objects

        return iter(iter_graph_objects(self, root, graph_options))

    def apply_graph(
            self,
            root,
            func,
            *,
            options: RepoGraphOptions | None = None,
            include_root: bool = True,
            order: str = "post",
            missing: str = "raise",
            dedupe: bool = True,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            build_missing: bool = False,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision: RevisionType | str | None = None):
        graph_options = self._graph_options(
            options=options,
            include_root=include_root,
            order=order,
            missing=missing,
            dedupe=dedupe,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
        )
        from .repo_plan import apply_graph_objects

        return apply_graph_objects(self, root, func, graph_options)

    def set_main_def(self, main_def: ConcreteDefinition, store=None):
        self.main_def = main_def
        if store is None:
            store = self.default_store
        if store is not None:
            store.set_main_def(self.main_def)
        else:
            raise ValueError("No store available to set main definition!")

    def add_objects(self, *args, store=None):
        store = self._ensure_store(store)
        from .repo_plan import add_objects

        add_objects(self, args, store=store)

    def flush(self):
        # Commit all stores
        for store in self.stores:
            store.write_aliases(self.alias_index)
            store.commit()

    def close(self, flush=True):
        if flush:
            self.flush()

    def __del__(self):
        if self.save_objs_on_deletion:
            self.save()
            self.close(flush=True)

    def clear_cache(self, strong=False, weak=True):
        if strong:
            self.strong_obj_cache.clear()
        if weak:
            self.weak_obj_cache.clear()

    @staticmethod
    def dir_store_inspect(root_path: str):
        files = glob.glob(os.path.join(root_path, '**/def.pkl'), recursive=True)
        # Strip root directory
        return list(map(lambda f: f[len(root_path)+1:], files))


def make_store(store):
    from .store.store import Store
    if isinstance(store, Store):
        return store

    elif isinstance(store, IOBase):
        from .store.zip import ZipStore
        # file-like => zip-backed store in a temp dir
        return ZipStore(store)

    elif isinstance(store, (str, Path)):
        from .store.dir import DirStore
        from .store.zip import ZipStore
        path = os.fspath(store)
        if os.path.isdir(path):
            store = DirStore(store)
        else:
            # treat as zip file path (may or may not exist yet)
            store = ZipStore(store)
        return store
    else:
        raise ValueError(f"Cannot open a store pointing to location {store!r}")


# Context management for default repo
_current_repo: ContextVar["Repo|None"] = ContextVar("_current_repo", default=None)
_global_repo: "Repo" = Repo()


# This cleanup system is required because we use a 'heavy'
# hash function which wants to import types at runtime.
# This causes a crash at cleanup, so we explicitly cleanup
# repos so they aren't left until after the module import
# system is cleaned up.
def global_repo_cleanup():
    global _global_repo
    global _current_repo
    from .session import close_configured_repo

    close_configured_repo()
    _global_repo.close()
    del _global_repo
    r = _current_repo.get()
    if r is not None:
        r.close()
        del r
atexit.register(global_repo_cleanup)


# Get the current default repo
def get_default_repo() -> "Repo":
    r = _current_repo.get()
    if r is not None:
        return r

    from .session import current_repo

    r = current_repo()
    return r if r is not None else _global_repo


# Context manager for isolated repo
@contextmanager
def default_repo(r: Repo|None=None):
    if r is None:
        r = Repo()
    tok = _current_repo.set(r)
    try:
        yield r
    finally:
        _current_repo.reset(tok)


@contextmanager
def manage_repo(repo=None):
    """
    Handle all the following cases:

      * repo is None:
          - create a fresh Repo() with no stores (pure in-memory)
          - auto-close at the end of the context

      * repo is a Repo:
          - use it as-is, do not close it at the end

      * repo is a Store:
          - Use the store as is

      * repo is an IOBase:
          - treat it as a zip container
          - create ZipStore(repo), Repo([ZipStore])
          - auto-close (commit+cleanup) at the end

      * repo is a str/Path:
          - if it points to an existing directory: DirStore(path)
          - else: ZipStore(path)
          - Repo([store])
          - auto-close at the end

      * repo is a list containing the previous types
          - Create a repo backed with multiple stores
    """
    close_repo = False

    if repo is None:
        repo_obj = get_default_repo()

    elif isinstance(repo, Repo):
        # user-supplied repo, don't manage its lifetime
        repo_obj = repo

    else:
        if isinstance(repo, list):
            # Check there are no Repos or Nones.
            for el in repo:
                if el is None or isinstance(el, Repo):
                    raise ValueError("Store list can't contain a None or Repo object.")
            stores = [
                make_store(store)
                for store in repo
            ]
        else:
            stores = [ make_store(repo) ]
            
        repo_obj = Repo(stores=stores)
        close_repo = True

    try:
        yield repo_obj
    finally:
        if close_repo:
            repo_obj.close()


# Saving and Loading
def save_object(
        obj,
        repo=None,
        main=False,
        revision: RevisionType|str|None=None,
        store=None,
        alias: str | None = None,
        ephemeral_depth: int | None = 0,
        options: RepoSaveOptions | None = None):
    with manage_repo(repo=repo) as sub_repo:
        if options is None:
            main = main or ((repo is not sub_repo) and isinstance(obj, Object))
        save_options = sub_repo._save_options(
            options=options,
            main=main,
            store=store,
            revision=revision,
            alias=alias,
            ephemeral_depth=ephemeral_depth,
        )
        sub_repo.save_object(obj, options=save_options)


def load_alias(alias: str, repo=None, **kwargs):
    with manage_repo(repo=repo) as repo:
        return repo.load_alias(alias, **kwargs)


def load_object(
        cdef=None, repo=None,
        revision: RevisionType|str|None=None,
        **kwargs):
    with manage_repo(repo=repo) as repo:
        if cdef is None:
            cdef = repo.main_def
            if cdef is None:
                raise ValueError("When cdef is None, the repo must have a main def, we didn't find one.")
        return repo.load_object(cdef, revision=revision, **kwargs)
