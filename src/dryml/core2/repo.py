from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager
from io import IOBase
from pathlib import Path
import weakref
from contextvars import ContextVar
from collections.abc import Mapping
import numpy as np
import atexit

from .definition import Definition, ConcreteDefinition
from .object import Object
from .store.store import Store
from .policies import InstancePolicy, CachePolicy
from .repo_graph import RepoSaveVisitor, RepoAddObjectsVisitor, manage_revision
from .canonical import from_canonical
from .config import CONFIG_MISSING, ConfigError, ConfigRef


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

    def cache_strong(self, obj: Object) -> None:
        self.strong_obj_cache[obj.__cdef__] = obj

    def cache_weak(self, obj: Object) -> None:
        self.weak_obj_cache[obj.__cdef__] = obj

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

    def unpin(self, obj_or_cdef):
        """Demote to weak cache."""
        cdef = obj_or_cdef if isinstance(obj_or_cdef, ConcreteDefinition) else obj_or_cdef.__cdef__
        obj = self.strong_obj_cache.pop(cdef, None)
        if obj is not None:
            self.weak_obj_cache[cdef] = obj

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

    def set_alias(self, alias: str, target: Object | Definition | ConcreteDefinition, *, store=None) -> ConcreteDefinition:
        self._validate_alias(alias)

        if store is not None:
            store = make_store(store)
            if store not in self.stores:
                self.add_store(store)

        cdef = self._alias_target_cdef(target)
        if isinstance(target, Object):
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
        for store in self.stores:
            for cdef in store.hydrate_index():
                self.light_index.add(cdef)

    def __len__(self):
        return len(self.strong_obj_cache)

    def save_object(self, obj, main=False, store=None, revision=None, alias: str | None = None):
        revision = manage_revision(obj, revision)
        self.add_objects(obj, store=store)
        RepoSaveVisitor(self, store=store, revision=revision).visit(obj)

        if main:
            self.set_main_def(obj.definition, store=store)
        if alias is not None:
            self.set_alias(alias, obj, store=store)
        return True

    def save(self, obj: Object | None = None, store=None, revision: RevisionType|str|None=None):
        if obj is None:
            # Save all loaded objects in the cache
            obj_list = []
            for _, obj in self.strong_obj_cache.items():
                if obj is not None:
                    obj_list.append(obj)
            self.save_object(obj_list, main=False, store=store, revision=revision)
            self.flush()
        else:
            self.save_object(obj, main=False, store=store, revision=revision)
            self.flush()

    def _first_store_with(self, cdef):
        for st in self.stores:
            if st.has(cdef):
                return st
        return None

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
        memo: dict | None = None,
        path: list[str | int] | None = None,
    ):
        return from_canonical(
            x,
            repo=self,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
            memo=memo,
            path=path,
        )

    # -------------------------------------------------------------------------
    # Core: turn a ConcreteDefinition into a live Object under load knobs
    # -------------------------------------------------------------------------
    def _materialize_cdef(
        self,
        cdef,
        revision: RevisionType,
        *,
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

        # If we've already materialized this cdef in this call, return it
        if cdef in memo:
            return memo[cdef]

        revision_str = revision.get(cdef, None)

        # Enforce sane caching semantics for "new"
        if instance == "new" and cache != "none":
            # caches are keyed ONLY by cdef, so caching a "new" instance would overwrite.
            raise ValueError("instance='new' requires cache='none' (caches are keyed by cdef)")

        # Reuse path: consult caches
        if instance == "reuse":
            obj = self.get_cached(cdef, reuse_weak=reuse_weak)
            if obj is not None:
                if restore_state:
                    # Descend into args and kwargs and restore them as well.
                    # TODO: Do we need to change build_missing or other options here?
                    #       I suspect not, but worth double-checking.
                    #       cache options is changed from below for example.
                    _ = self._realize(
                        cdef.args,
                        instance=instance,
                        restore_state=restore_state,
                        build_missing=build_missing,
                        reuse_weak=reuse_weak,
                        cache=cache,
                        revision=revision,
                        memo=memo,
                        path=path + ["args"],
                    )
                    _ = rt_kwargs = self._realize(
                        cdef.kwargs,
                        instance=instance,
                        restore_state=restore_state,
                        build_missing=build_missing,
                        reuse_weak=reuse_weak,
                        cache=cache,
                        revision=revision,
                        memo=memo,
                        path=path + ["kwargs"],
                    )
                    # Restore this object
                    if revision_str is not None:
                        st = self._first_store_with(cdef)
                        if st is None:
                            raise RepoLoadError(f"No store has requested object ({cdef})")
                        try:
                            st.restore_object(obj, revision=revision_str)
                        except Exception as e:
                            raise RepoLoadError(f"Store can't restore requested revision ({revision_str}) for object ({cdef})") from e
                return obj

        # Determine whether state exists somewhere
        in_store = self.has_cdef_light(cdef)

        # If caller expects state and we can't find any, respect build_missing
        if restore_state and (not in_store) and (not build_missing):
            raise RepoLoadError(
                f"Missing stored state for {cdef} at {'/'.join(map(str, path))} "
                f"(set build_missing=True to allow fresh construction)"
            )

        # Realize constructor args/kwargs (objects inside become objects)
        # NOTE: pass cache="none" while constructing dependencies when instance="new"
        # to avoid overwriting caches as well.
        sub_cache = cache if instance == "reuse" else "none"

        rt_args = self._realize(
            cdef.args,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=sub_cache,
            revision=revision,
            memo=memo,
            path=path + ["args"],
        )
        rt_kwargs = self._realize(
            cdef.kwargs,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=sub_cache,
            revision=revision,
            memo=memo,
            path=path + ["kwargs"],
        )

        # Construct a new instance (bypass re-concretization by passing __cdef__)
        try:
            obj = cdef.cls(*rt_args, repo=self, __cdef__=cdef, **rt_kwargs)

            self._num_constructions += 1
        except Exception as e:
            raise RepoLoadError(
                f"Error constructing {cdef.cls.__name__} at {'/'.join(map(str, path))}: {e}"
            ) from e

        # Memo immediately (preserve sharing inside this graph)
        memo[cdef] = obj

        # Optionally restore heavy state from store
        if restore_state and in_store:
            st = self._first_store_with(cdef)
            if st is None:
                # has_cdef_light said True but we didn't find it; treat as missing
                if not build_missing:
                    raise RepoLoadError(f"Inconsistent store index for {cdef}")
            else:
                try:
                    st.restore_object(obj, revision=revision_str)
                except Exception as e:
                    raise RepoLoadError(
                        f"Error restoring state for {cdef} at {'/'.join(map(str, path))}: {e}"
                    ) from e

        # Cache result (only meaningful for instance='reuse')
        if instance == "reuse":
            if cache == "strong":
                self.cache_strong(obj)
            elif cache == "weak":
                self.cache_weak(obj)
            elif cache == "none":
                pass
            else:
                raise ValueError(f"Unknown cache policy: {cache!r}")

        return obj


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
    ):
        memo: dict[ConcreteDefinition, Object] = {}
        return self._realize(
            x,
            instance=instance,
            restore_state=restore_state,
            build_missing=build_missing,
            reuse_weak=reuse_weak,
            cache=cache,
            revision=revision,
            path=[""],
            memo=memo,
        )


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
        in_store = cdef in self.light_index
        return in_cache or in_store

    def __getitem__(
            self, key: ConcreteDefinition):
        """
        Easy access to objects within.

        if unpack is true, plain objects are returned
        """
        result = self.get(key)
        if len(result) == 0:
            raise KeyError(f"Repo doesn't contain an object with definition {key}")
        else:
            return list(self.get(key).values())[0]

    def get(self,
            selector:  SelectorType | tuple[SelectorType] | list[SelectorType] | None = None,
            sel_args=None, sel_kwargs=None,
            instance: InstancePolicy = "reuse",
            restore_state: bool = True,
            build_missing: bool = False,
            reuse_weak: bool = True,
            cache: CachePolicy = "weak",
            revision: RevisionType | None = None,
            verbose: bool = True) -> dict[ConcreteDefinition, Object]:
        if type(selector) is list:
            pass
        elif type(selector) is not tuple:
            selector = (selector,)
        selectors = selector
        if isinstance(revision, str):
            raise ValueError("plain string revisions aren't supported in `get`.")
        revision = manage_revision(None, revision)

        # Build list of all cdefs known in the repo
        cached_cdefs = []
        cached_cdefs.extend(self.strong_obj_cache.keys())
        if reuse_weak:
            cached_cdefs.extend(self.weak_obj_cache.keys())
        cached_cdefs = set(cached_cdefs).union(self.light_index)

        def get_obj(cdef: ConcreteDefinition) -> Object | None:
            return self._materialize_cdef(
                cdef,
                revision,
                instance=instance,
                restore_state=restore_state,
                build_missing=build_missing,
                reuse_weak=reuse_weak,
                cache=cache)

        selected_objects = {}
        for sel in selectors:
            if isinstance(sel, ConcreteDefinition):
                if sel in selected_objects:
                    continue
                obj = get_obj(sel)
                if obj is not None:
                    selected_objects[sel] = obj
            elif isinstance(sel, Definition):
                for cdef in cached_cdefs:
                    if sel(cdef, *sel_args, **sel_kwargs):
                        if cdef in selected_objects:
                            continue
                        obj = get_obj(cdef)
                        if obj is not None:
                            selected_objects[cdef] = obj
            elif isinstance(sel, Callable):
                for cdef in cached_cdefs:
                    if cdef in self.strong_obj_cache:
                        if sel(self.strong_obj_cache[cdef], *sel_args, **sel_kwargs):
                            selected_objects[cdef] = self.strong_obj_cache[cdef]
            elif sel is None:
                for cdef in cached_cdefs:
                    obj = get_obj(cdef)
                    if obj is not None:
                        selected_objects[cdef] = obj
            else:
                raise TypeError("sel is of incorrect type.")

        return selected_objects

    def apply(self,
              func, func_args=None, func_kwargs=None,
              selector: Optional[Callable] = None,
              sel_args=None, sel_kwargs=None,
              verbose: bool = False,
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
            **kwargs)

        obj_iter = objs.items()
        if verbose:
            obj_iter = tqdm(obj_iter)
        return {
            obj_def: apply_func(obj) for obj_def, obj in obj_iter
        }

    def set_main_def(self, main_def: ConcreteDefinition, store=None):
        self.main_def = main_def
        if store is None:
            store = self.default_store
        if store is not None:
            store.set_main_def(self.main_def)
        else:
            raise ValueError("No store available to set main definition!")

    def add_objects(self, *args, store=None):
        vis = RepoAddObjectsVisitor(self, store=store)
        for arg in args:
            vis.visit(arg)

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
def save_object(obj, repo=None, main=False, revision: RevisionType|str|None=None, alias: str | None = None):
    with manage_repo(repo=repo) as sub_repo:
        revision = manage_revision(obj, revision)
        main = main or ((repo is not sub_repo) and isinstance(obj, Object))
        sub_repo.add_objects(obj)
        sub_repo.save_object(obj, main=main, revision=revision, alias=alias)


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
