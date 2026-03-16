from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Callable
from contextlib import contextmanager
from io import IOBase
from pathlib import Path
import weakref
from contextvars import ContextVar
import numpy as np
import atexit

from .definition import Definition, ConcreteDefinition
from .utils.recurse import cycle_detect
from .object import Object
from .store.store import Store
from .types import is_pod
from .policies import InstancePolicy, CachePolicy
from .freeze import FrozenList, FrozenTuple, FrozenSet, FrozenDict, FrozenNDArray


class RepoSaveError(Exception):
    pass


class RepoLoadError(Exception):
    pass


SelectorType = Callable | Definition | ConcreteDefinition


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

    # Settings
    save_objs_on_deletion: bool = False


    # Helper class for saving objects
    def __init__(self, stores=None):
        # Initialize caches
        self.weak_obj_cache = weakref.WeakValueDictionary()
        self.strong_obj_cache = {}
        self.obj_default_store = {}
        self.light_index = set()
        self.cdef_cache = weakref.WeakValueDictionary()

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

    def save_object(self, obj, main=False, store=None, revision_map: dict[ConcreteDefinition,str]|None=None):

        saved_objs: dict[int,set[ConcreteDefinition]] = {}
        if revision_map is None:
            revision_map = {}

        def _save_single_object(obj: Object, store: Store|None=None, revision_map: dict[ConcreteDefinition,str]|None=None):
            """
            Defines how the Repo updates its caches and delegates saving a single object to one of it's stores
            """

            cdef = obj.definition
            if store is None:
                if cdef in self.obj_default_store:
                    store = self.obj_default_store[cdef]
                else:
                    store = self.default_store

            _save_object(cdef.args, store=store, revision_map=revision_map)
            _save_object(cdef.kwargs, store=store, revision_map=revision_map)

            if cdef in self.strong_obj_cache:
                # Check that this was the same object
                if obj is not self.strong_obj_cache[cdef]:
                    raise ValueError("We already have a different object with definition: {cdef}")
            else:
                # Update repo cache
                self.pin(obj)

            if store is None:
                raise RepoSaveError("No store available to save object!")

            if id(store) not in saved_objs:
                saved_objs[id(store)] = set()

            if cdef not in saved_objs[id(store)]:
                # Save the object
                revision = revision_map.get(cdef, None)
                store.save_object(obj, revision=revision)
                saved_objs[id(store)].add(cdef)
                self._num_saves += 1


        @cycle_detect()
        def _save_object(obj: Any, store: Store|None=None, revision_map: dict[ConcreteDefinition,str]|None=None):
            if isinstance(obj, Object):
                # we must descend into args/kwargs first
                _save_single_object(obj, store=store, revision_map=revision_map)
                return

            if isinstance(obj, ConcreteDefinition):
                # We must find the linked object
                linked_obj = self.get_cached(obj)
                if linked_obj is None:
                    raise RepoSaveError(f"Definition of object {obj} is not reachable in this repo!")
                _save_object(linked_obj, store=store, revision_map=revision_map)
                return

            if isinstance(obj, (list, tuple, set, FrozenList, FrozenTuple, FrozenSet)):
                for el in obj:
                    _save_object(el, store=store, revision_map=revision_map)
                return
            if isinstance(obj, (FrozenDict, dict)):
                for el in obj.values():
                    _save_object(el, store=store, revision_map=revision_map)
                return
            if isinstance(obj, Definition):
                raise RepoSaveError("Plain Definitions aren't allowed here.")
            if is_pod(obj) or isinstance(obj, (np.ndarray, FrozenNDArray)):
                return
            else:
                raise RepoSaveError(f"Cannot save object of type {type(obj)}!")

        _save_object(obj, store=store, revision_map=revision_map)

        # Save main object definition
        if main:
            self.set_main_def(obj.definition, store=store)
        return True

    def save(self, obj: Object | None = None, store=None):
        if obj is None:
            # Save all loaded objects in the cache
            obj_list = []
            for _, obj in self.strong_obj_cache.items():
                if obj is not None:
                    obj_list.append(obj)
            self.save_object(obj_list, main=False, store=store)
            self.flush()
        else:
            self.save_object(obj, main=False, store=store)
            self.flush()

    def _first_store_with(self, cdef):
        for st in self.stores:
            if st.has(cdef):
                return st
        return None

    # -------------------------------------------------------------------------
    # Core: realize arbitrary structure into runtime Python + Objects
    # -------------------------------------------------------------------------
    @cycle_detect(arg_pos=1)
    def _realize(
        self,
        x: Any,
        *,
        instance: InstancePolicy = "reuse",
        restore_state: bool = True,
        build_missing: bool = False,
        reuse_weak: bool = True,
        cache: CachePolicy = "weak",
        revision: str | None = None,
        # internal
        memo: dict | None = None,          # memo for cdef->obj within a single call
        path: list[str | int] | None = None,
        _stack: set[int] | None = None,    # cycle guard for containers
    ) -> Any:
        if memo is None:
            memo = {}
        if path is None:
            path = ["<root>"]
        if _stack is None:
            _stack = set()

        # scalars
        if is_pod(x):
            return x

        # ndarray / FrozenNDArray -> writable ndarray
        if isinstance(x, FrozenNDArray):
            # your FrozenNDArray currently doesn't define .thaw() in the snippet;
            # support both styles.
            if hasattr(x, "thaw"):
                return x.thaw()
            return np.array(x, copy=True)

        if isinstance(x, np.ndarray):
            return np.array(x, copy=True)

        # ConcreteDefinition -> object materialization
        from .definition import ConcreteDefinition, Definition  # local to avoid cycles
        from .object import Object

        if isinstance(x, ConcreteDefinition):
            return self._materialize_cdef(
                x,
                instance=instance,
                restore_state=restore_state,
                build_missing=build_missing,
                reuse_weak=reuse_weak,
                cache=cache,
                revision=revision,
                memo=memo,
                path=path,
            )

        # Definition -> concretize then materialize
        if isinstance(x, Definition):
            cdef = x.concretize(repo=self)
            return self._materialize_cdef(
                cdef,
                instance=instance,
                restore_state=restore_state,
                build_missing=build_missing,
                reuse_weak=reuse_weak,
                cache=cache,
                revision=revision,
                memo=memo,
                path=path,
            )

        # Object encountered during realization
        if isinstance(x, Object):
            if instance == "new":
                # interpret "new" as "clone-by-identity": rebuild from its cdef
                return self._materialize_cdef(
                    x.definition,
                    instance="new",
                    restore_state=restore_state,
                    build_missing=build_missing,
                    reuse_weak=reuse_weak,
                    cache=cache,
                    revision=revision,
                    memo=memo,
                    path=path,
                )
            # reuse: return it, but ensure repo can reach it
            if self.get_cached(x.definition, reuse_weak=reuse_weak) is None:
                self.cache_weak(x.definition, x)
            return x

        # Containers (cycle-safe)
        oid = id(x)
        if oid in _stack:
            raise RepoLoadError(f"Cycle detected while realizing at {'/'.join(map(str, path))}")
        _stack.add(oid)
        try:
            # tuple-like
            if isinstance(x, (tuple, FrozenTuple)):
                return tuple(
                    self._realize(
                        v,
                        instance=instance,
                        restore_state=restore_state,
                        build_missing=build_missing,
                        reuse_weak=reuse_weak,
                        cache=cache,
                        revision=revision,
                        memo=memo,
                        path=path + [i],
                        _stack=_stack,
                    )
                    for i, v in enumerate(x)
                )

            # list-like (FrozenList is tuple-tagged)
            if isinstance(x, FrozenList) or isinstance(x, list):
                return [
                    self._realize(
                        v,
                        instance=instance,
                        restore_state=restore_state,
                        build_missing=build_missing,
                        reuse_weak=reuse_weak,
                        cache=cache,
                        revision=revision,
                        memo=memo,
                        path=path + [i],
                        _stack=_stack,
                    )
                    for i, v in enumerate(list(x))
                ]

            # set-like
            if isinstance(x, (set, FrozenSet)):
                out = set()
                for i, v in enumerate(x):
                    out.add(
                        self._realize(
                            v,
                            instance=instance,
                            restore_state=restore_state,
                            build_missing=build_missing,
                            reuse_weak=reuse_weak,
                            cache=cache,
                            revision=revision,
                            memo=memo,
                            path=path + [f"<set:{i}>"],
                            _stack=_stack,
                        )
                    )
                return out

            # mapping-like (FrozenDict implements Mapping)
            if isinstance(x, (dict, FrozenDict)):
                out = {}
                for k, v in (x.items() if hasattr(x, "items") else x):
                    rk = self._realize(
                        k,
                        instance=instance,
                        restore_state=restore_state,
                        build_missing=build_missing,
                        reuse_weak=reuse_weak,
                        cache=cache,
                        revision=revision,
                        memo=memo,
                        path=path + ["<key>"],
                        _stack=_stack,
                    )
                    rv = self._realize(
                        v,
                        instance=instance,
                        restore_state=restore_state,
                        build_missing=build_missing,
                        reuse_weak=reuse_weak,
                        cache=cache,
                        revision=revision,
                        memo=memo,
                        path=path + [str(k)],
                        _stack=_stack,
                    )
                    out[rk] = rv
                return out

        finally:
            _stack.remove(oid)

        raise RepoLoadError(
            f"Cannot realize type {type(x).__name__} at {'/'.join(map(str, path))}"
        )

    # -------------------------------------------------------------------------
    # Core: turn a ConcreteDefinition into a live Object under load knobs
    # -------------------------------------------------------------------------
    def _materialize_cdef(
        self,
        cdef,
        *,
        instance: InstancePolicy = "reuse",
        restore_state: bool = True,
        build_missing: bool = False,
        reuse_weak: bool = True,
        cache: CachePolicy = "weak",
        revision: str | None = None,
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

        # Enforce sane caching semantics for "new"
        if instance == "new" and cache != "none":
            # caches are keyed ONLY by cdef, so caching a "new" instance would overwrite.
            raise ValueError("instance='new' requires cache='none' (caches are keyed by cdef)")

        # Reuse path: consult caches
        if instance == "reuse":
            obj = self.get_cached(cdef, reuse_weak=reuse_weak)
            if obj is not None:
                if restore_state and revision is not None and obj.state_revision != revision:
                    st = self._first_store_with(cdef)
                    if st is None:
                        raise RepoLoadError("No store has requested revision")
                    ok = st.restore_object(obj, revision=revision)
                    if not ok:
                        raise RepoLoadError("Store can't restore requested revision")
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
                    st.restore_object(obj, revision=revision)
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
        revision: str | None = None,
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
            revision: str | None = None,
            verbose: bool = True) -> dict[ConcreteDefinition, Object]:
        if type(selector) is list:
            pass
        elif type(selector) is not tuple:
            selector = (selector,)
        selectors = selector

        # Build list of all cdefs known in the repo
        cached_cdefs = []
        cached_cdefs.extend(self.strong_obj_cache.keys())
        if reuse_weak:
            cached_cdefs.extend(self.weak_obj_cache.keys())
        cached_cdefs = set(cached_cdefs).union(self.light_index)

        def get_obj(cdef: ConcreteDefinition) -> Object | None:
            return self._materialize_cdef(
                cdef,
                instance=instance,
                restore_state=restore_state,
                build_missing=build_missing,
                reuse_weak=reuse_weak,
                cache=cache,
                revision=revision)

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
                            selected_objects[obj_def] = obj
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

    def add_objects(self, *args, store: Store|None=None):
        from dryml.core2.object import Object

        def _add_object_single(obj: Object):
            self.pin(obj)
            if store is not None:
                self.obj_default_store[obj.definition] = store
            else:
                # Set the default store to the current default store of the repo, if it exists
                if self.default_store is not None:
                    self.obj_default_store[obj.definition] = self.default_store

        # Recursively add objects.
        def _add_object(obj: Any):
            if isinstance(obj, Object):
                cdef = obj.definition
                _add_object(cdef.args)
                _add_object(cdef.kwargs)
                if cdef in self.strong_obj_cache and (obj is not self.strong_obj_cache[cdef]):
                    raise KeyError(f"Repo already has a different object matching {cdef}!")
                else:
                    _add_object_single(obj)
                return
            if isinstance(obj, ConcreteDefinition):
                # Find linked object
                linked_obj = self.get_cached(obj)
                if linked_obj is None:
                    # check the general repo
                    if self is not _global_repo: 
                        linked_obj = _global_repo.get_cached(obj)
                if linked_obj is None:    
                    raise KeyError(f"No object linked to definition {obj} found in repo!")
                _add_object(linked_obj)
                return
            if isinstance(obj, (list, FrozenList, tuple, FrozenTuple, set, FrozenSet)):
                for el in obj:
                    _add_object(el)
                return
            if isinstance(obj, (dict, FrozenDict)):
                for el in obj.values():
                    _add_object(el)
                return
            if is_pod(obj):
                return
            if isinstance(obj, Definition):
                raise ValueError("Plain Definitions aren't allowed here.")
            raise TypeError(f"Unsupported type {type(obj)} found when adding objects to repo.")

        for arg in args:
            _add_object(arg)

    def flush(self):
        # Commit all stores
        for store in self.stores:
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
        files = glob.glob(os.path.join(root_path, '**', 'def.pkl'), recursive=True)
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
def isolated_repo():
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
def save_object(obj, repo=None, main=False, revision_map: dict[ConcreteDefinition,str]|None=None):
    with manage_repo(repo=repo) as sub_repo:
        main = main or ((repo is not sub_repo) and isinstance(obj, Object))
        sub_repo.add_objects(obj)
        sub_repo.save_object(obj, main=main, revision_map=revision_map)


def load_object(
        cdef=None, repo=None,
        **kwargs):
    with manage_repo(repo=repo) as repo:
        if cdef is None:
            cdef = repo.main_def
            if cdef is None:
                raise ValueError("When cdef is None, the repo must have a main def, we didn't find one.")
        return repo.load_object(cdef, **kwargs)
