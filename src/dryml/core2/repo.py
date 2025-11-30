from __future__ import annotations

import os
from pathlib import Path
import glob
import zipfile
from typing import Callable
from boltons.iterutils import remap, default_enter, default_visit, default_exit
from collections.abc import ItemsView
from contextlib import contextmanager
from io import IOBase
from pathlib import Path

from boltons.iterutils import remap, default_enter, default_exit

from .definition import Definition, ConcreteDefinition
from .utils.general import zip_directory, \
    pickle_load, pickle_save, get_temp_directory, get_object_view, \
    get_definition_view
from .object import Object
from .store.store import Store


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
    obj_cache: dict[ConcreteDefinition, Object | None]
    # Links particular Definition object with a concrete definition (Definitions are resolved )
    cdef_cache: dict[int, ConcreteDefinition]

    # Main definition
    main_def: ConcreteDefinition | None

    # Backing stores
    stores: list[Store]

    # Helper class for saving objects
    def __init__(self, stores=None):
        # Initialize caches
        self.obj_cache = {}
        self.cdef_cache = {}

        # Some helper variables for monitoring
        self._num_saves = 0
        self._num_constructions = 0

        # Initialize the main def
        self.main_def = None

        # Multiple stores, optional
        self.stores: list[Store] = []
        if isinstance(stores, Store):
            self.stores = [stores]
        elif stores is None:
            self.stores = []
        elif isinstance(stores, (list, tuple)):
            self.stores = []
            for store in stores:
                if not isinstance(store, Store):
                    self.stores.append(make_store(store))
                else:
                    self.stores.append(store)

        # Optional default store for write
        self.default_store: Store | None = self.stores[0] if self.stores else None

        # Try to read main def from first store
        if len(self.stores) > 0:
            # Attempt to read from the default store first.
            if self.default_store is not None:
                main_def = self.default_store.read_main_def()
            if main_def is None:
                for store in self.stores:
                    main_def = store.read_main_def()

            if main_def is not None:
                self.set_main_def(main_def)

    # Store Methods

    def add_store(self, store: "Store", make_default=False):
        self.stores.append(store)
        if make_default or self.default_store is None:
            self.default_store = store

    def has_cdef_light(self, cdef: ConcreteDefinition) -> bool:
        # do any stores have data for this cdef?
        return any(store.has_cdef(cdef) for store in self.stores)

    def hydrate_from_stores(self):
        """
        Ask each store to enumerate all cdefs it has.
        Populate obj_cache[cdef] = None for those not already present.
        """
        for store in self.stores:
            for cdef in store.hydrate_index():
                self.obj_cache.setdefault(cdef, None)

    def __len__(self):
        return len(self.obj_cache)

    # Graph Logic

    def concretize_definition(self, defn):
        """
        Inside a possibly nested structure do these transformations:
            Definition -> ConcreteDefinition
            Object -> ConcreteDefinition
        """

        def _enter(path, key, value):
            if id(value) in self.cdef_cache:
                # We've seen this object before
                return value, False
            elif isinstance(value, ConcreteDefinition):
                # The definition is already concrete. don't enter it.
                return value, False
            elif isinstance(value, Definition):
                return {}, get_definition_view(value)
            elif isinstance(value, Object):
                return {}, get_object_view(value)
            else:
                return default_enter(path, key, value)

        def _visit(path, key, value):
            if id(value) in self.cdef_cache: 
                # We've seen this object before
                return key, self.cdef_cache[id(value)]
            elif type(value) is ConcreteDefinition:
                # Value is already Concrete
                return key, value
            elif isinstance(value, Object):
                # We have an already realized class instance. We shouldn't deep copy it.
                raise TypeError("We shouldn't get an Object object here.")
            elif isinstance(value, Definition):
                raise TypeError("We shouldn't get a Definition object here.")
            else:
                return key, value

        def _create_cdef(new_parent, new_items):
            for k, v in new_items:
                new_parent[k] = v
            try:
                args = new_parent['args']
            except KeyError:
                raise ValueError("Definition {values} which skipped arguments isn't concretizable.")
            kwargs = new_parent['kwargs']
            cls = new_parent['cls']
            # Do argument manipulations
            args, kwargs = cls.__prepare_args__(*args, **kwargs)
            # Create the now concrete definition
            return ConcreteDefinition(cls, *args, **kwargs) 

        def _exit(path, key, values, new_parent, new_items):
            is_obj = isinstance(values, Object)
            if isinstance(values, Definition) or is_obj:
                if is_obj:
                    # Store the object since we know it's cdef
                    self.obj_cache[values.__cdef__] = values
                    # Store the link between this particular value and its cdef.
                    self.cdef_cache[id(values)] = values.__cdef__
                    return values.__cdef__

                # Cache built
                new_cdef = _create_cdef(new_parent, new_items)
                self.cdef_cache[id(values)] = new_cdef
                return new_cdef
            else:
                return default_exit(path, key, values, new_parent, new_items)

        if isinstance(defn, Definition):
            return remap(
                [defn],
                enter=_enter,
                visit=_visit,
                exit=_exit)[0]
        else:
            return remap(
                defn,
                enter=_enter,
                visit=_visit,
                exit=_exit)

    def _save_single_object(self, obj: Object):
        """
        Defines how the Repo updates its caches and delegates saving a single object to one of it's stores
        """

        obj_def = obj.definition
        if obj_def in self.obj_cache:
            if self.obj_cache[obj_def] is None:
                # Update repo cache
                self.obj_cache[obj_def] = obj
            else:
                # Check that this was the same object
                if obj is not self.obj_cache[obj_def]:
                    raise ValueError("We already have a different object with definition: {obj_def}")
        else:
            # Update repo cache
            self.obj_cache[obj_def] = obj

        if self.default_store is not None:
            self.default_store.save_object(obj)
            self._num_saves += 1
            return
        elif len(self.stores) > 0:
            self.stores[0].save_object(obj)
            self._num_saves += 1
            return
        else:
            raise RepoSaveError("No store available to save object!")

    def save_object(self, obj, main=False):
        saved_objs = {}
        def _save_object_enter(path, key, value):
            if isinstance(value, Object):
                result = {'args': value.definition.args, 'kwargs': value.definition.kwargs}
                return {}, ItemsView(result)
            elif isinstance(value, ConcreteDefinition):
                obj = value._obj
                if obj is None:
                    raise ValueError("ConcreteDefinitions must be linked to actual objects through _obj to be savable.")
                result = {'args': obj.definition.args, 'kwargs': obj.definition.kwargs}
                return {}, ItemsView(result)
            elif isinstance(value, Definition):
                raise ValueError("Plain Definitions aren't allowed here.")
            else:
                return default_enter(path, key, value)

        def _save_object_visit(path, key, value):
            return key, value

        def _save_object_exit(path, key, value, new_parent, new_items):
            if isinstance(value, Object):
                obj_def = value.definition
                if obj_def not in saved_objs:
                    self._save_single_object(value)
                    saved_objs[obj_def] = value
                return value
            elif isinstance(value, ConcreteDefinition):
                obj = value._obj
                if obj is None:
                    raise ValueError("ConcreteDefinitions must be linked to actual objects through _obj to be savable.")
                if value not in saved_objs:
                    self._save_single_object(obj)
                    saved_objs[value] = obj
                return value
            else:
                return default_exit(path, key, value, new_parent, new_items)

        # Save the object
        if isinstance(obj, Object):
            remap(
                [obj],
                enter=_save_object_enter,
                visit=_save_object_visit,
                exit=_save_object_exit)
        else:
            remap(
                obj,
                enter=_save_object_enter,
                visit=_save_object_visit,
                exit=_save_object_exit)

        # Save main object definition
        if main:
            self.set_main_def(obj.definition)
        return True

    def _load_single_object(self, cdef: ConcreteDefinition, args, kwargs, build_missing=False) -> Object:
        """
        Defines how the Repo updates its caches and delegates loading a single object from one of it's stores
        """

        # Already loaded?
        if cdef in self.obj_cache and self.obj_cache[cdef] is not None:
            return self.obj_cache[cdef]

        in_store = self.has_cdef_light(cdef)

        if not build_missing and not in_store:
            raise RuntimeError("Asked not to build missing objects")

        # Construct object; graph algorithm makes sure args/kwargs are already
        # realized and passed in
        obj = cdef.cls(*args, repo=self, __cdef__=cdef, **kwargs)
        self._num_constructions += 1

        # Try to hydrate from stores, use first one that has it
        if in_store:
            for store in self.stores:
                if store.has_cdef(cdef):
                    store.load_object(obj)
                    break

        self.obj_cache[cdef] = obj
        return obj

    def load_object(self, obj_def, build_missing=False):

        loaded_objs = {}
        def _load_object_enter(path, key, value):
            if type(value) is Definition:
                raise TypeError("Definition not allowed here!")
            elif type(value) is ConcreteDefinition:
                if value in loaded_objs:
                    # We have already loaded this object
                    return value, False
                else:
                    return {}, ItemsView({'args': value['args'], 'kwargs': value['kwargs']})
            else:
                return default_enter(path, key, value)

        def _load_object_visit(path, key, value):
            # We do nothing here.
            if type(value) is ConcreteDefinition:
                # we already loaded this object, return it.
                return key, loaded_objs[value]
            else:
                return key, value

        def _load_object_exit(path, key, value, new_parent, new_items):
            if isinstance(value, ConcreteDefinition):
                # Check if we already have this object
                if value in self.obj_cache and self.obj_cache[value] is not None:
                    # we found it
                    loaded_objs[value] = self.obj_cache[value]
                    return loaded_objs[value]

                new_values = {} # Need to build a dictionary because we get a list of tuples
                for k, v in new_items:
                    new_values[k] = v
                # Get runtime args that have been built already
                args = new_values['args']
                kwargs = new_values['kwargs']

                obj = self._load_single_object(value, args, kwargs, build_missing=build_missing)

                loaded_objs[value] = obj

                return obj
            else:
                return default_exit(path, key, value, new_parent, new_items)

        if isinstance(obj_def, ConcreteDefinition):
            result = remap(
                [obj_def],
                enter=_load_object_enter,
                visit=_load_object_visit,
                exit=_load_object_exit)[0]
            return result
        else:
            result = remap(
                obj_def,
                enter=_load_object_enter,
                visit=_load_object_visit,
                exit=_load_object_exit)
            return result

    def __contains__(
            self, item: Object | ConcreteDefinition):
        if isinstance(ConcreteDefinition):
            obj_def = item
        elif isinstance(Object):
            obj_def = item.definition
        else:
            raise TypeError(
                f"Unsupported type {type(item)} for Repo.__contains__!")

        # “Strong” membership: known in cache and either loaded or known to exist
        in_cache = obj_def in self.obj_cache
        in_store = self.has_cdef_light(obj_def)
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
            load_objects: bool = False,
            build_missing=False,
            verbose: bool = True) -> dict[ConcreteDefinition,Object]:

        if type(selector) is list:
            pass
        elif type(selector) is not tuple:
            selector = (selector,)
        selectors = selector


        def get_obj(obj_def: Definition) -> Object | None:
            if obj_def in self.obj_cache:
                if self.obj_cache[obj_def] is None:
                    # We have content for this object,
                    # but we haven't loaded it yet.
                    if load_objects or build_missing:
                        return self.load_object(obj_def, build_missing=build_missing)
                else:
                    return self.obj_cache[obj_def]
            if build_missing:
                return self.load_object(obj_def, build_missing=build_missing)

            return None

        selected_objects = {}
        for sel in selectors:
            if isinstance(sel, ConcreteDefinition):
                if sel in selected_objects:
                    continue
                obj = get_obj(sel)
                if obj is not None:
                    selected_objects[sel] = obj
            elif isinstance(sel, Definition):
                added_objs = False
                for obj_def in self.obj_cache:
                    if sel(obj_def):
                        if obj_def in selected_objects:
                            continue
                        obj = get_obj(obj_def)
                        if obj is not None:
                            added_objs = True
                            selected_objects[obj_def] = obj
                if not added_objs and build_missing:
                    cdef = self.concretize_definition(sel)
                    obj = get_obj(cdef)
                    selected_objects[cdef] = obj
            elif isinstance(sel, Callable):
                for obj_def in self.obj_cache:
                    if self.obj_cache[obj_def] is not None:
                        if sel(self.obj_cache[obj_def]):
                            selected_objects[obj_def] = self.obj_cache[obj_def]
            elif sel is None:
                for obj_def in self.obj_cache:
                    obj = get_obj(obj_def)
                    if obj is not None:
                        selected_objects[obj_def] = obj
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
            tqdm(obj_iter)
        return {
            obj_def: apply_func(obj) for obj_def, obj in obj_iter
        }

    def set_main_def(self, main_def: ConcreteDefinition):
        self.main_def = main_def
        if self.default_store is not None:
            self.default_store.set_main_def(self.main_def)
        else:
            if len(stores) > 0:
                stores[0].set_main_def(self.main_def)

    def add_object(self, *args):
        from dryml.core2.object import Object
        for obj in args:
            if not isinstance(obj, Object):
                raise TypeError("Only Object objects can be added to a repository.")

        # Recursively add objects.
        def _add_object(obj: Object):
            if obj.definition in self.obj_cache:
                raise KeyError(f"Repo already has an object matching {obj.definition}!")
            self.obj_cache[obj.definition] = obj


        def _enter(path, key, value):
            if isinstance(value, Object):
                return {}, get_object_view(value)
            elif isinstance(value, ConcreteDefinition):
                return {}, get_definition_view(value)
            else:
                return default_enter(path, key, value)

        def _exit(path, key, value, new_parent, new_items):
            if isinstance(value, Object):
                _add_object(value)
                return value
            elif isinstance(value, ConcreteDefinition):
                if value._obj is None:
                    raise ValueError("Can't use a ConcreteDefinition without _obj pointer..")
                _add_object(value._obj)
                return value._obj
            else:
                return default_exit(path, key, value, new_parent, new_items)

        remap(
             args,
             enter=_enter,
             visit=default_visit,
             exit=_exit)

    def flush(self):
        # Commit all stores
        for store in self.stores:
            store.commit()

    def close(self, flush=True):
        if flush:
            self.flush()


def make_store(store):
    if isinstance(store, IOBase):
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

@contextmanager
def manage_repo(repo=None):
    """
    Handle all the following cases:

      * repo is None:
          - create a fresh Repo() with no stores (pure in-memory)
          - auto-close at the end of the context

      * repo is a Repo:
          - use it as-is, do not close it at the end

      * repo is an IOBase:
          - treat it as a zip container
          - create ZipStore(repo), Repo([ZipStore])
          - auto-close (commit+cleanup) at the end

      * repo is a str/Path:
          - if it points to an existing directory: DirStore(path)
          - else: ZipStore(path)
          - Repo([store])
          - auto-close at the end
    """
    close_repo = False

    if repo is None:
        # in-memory repo, no backing store yet
        repo_obj = Repo()
        close_repo = True

    elif isinstance(repo, Repo):
        # user-supplied repo, don't manage its lifetime
        repo_obj = repo

    else:
        store = make_store(repo)
        repo_obj = Repo(stores=[store])
        close_repo = True

    try:
        yield repo_obj
    finally:
        if close_repo:
            repo_obj.close()


# Saving and Loading
def save_object(obj, repo=None, main=False):
    with manage_repo(repo=repo) as sub_repo:
        main = main or ((repo is not sub_repo) and isinstance(obj, Object))
        sub_repo.save_object(obj, main=main)


def load_object(
        obj_def=None, repo=None,
        cls_remap=None):
    with manage_repo(repo=repo) as repo:
        if obj_def is None:
            obj_def = repo.main_def
        return repo.load_object(obj_def)
