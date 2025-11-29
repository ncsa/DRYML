from __future__ import annotations

import os
import glob
import zipfile
from typing import Callable
from boltons.iterutils import remap, default_enter, default_visit, default_exit
from collections.abc import ItemsView
from contextlib import contextmanager
from io import IOBase
from pathlib import Path
import numpy as np

from dryml.core2.definition import Definition, ConcreteDefinition, deepcopy_skip_definition_object
from dryml.core2.util import zip_directory, hashval_to_digest, \
    pickle_load, pickle_save, get_temp_directory, get_object_view, \
    get_definition_view, is_dictlike

from boltons.iterutils import remap, is_collection, default_enter, default_exit
from copy import deepcopy
from dryml.core2.object import Object


class RepoSaveError(Exception):
    pass


class RepoLoadError(Exception):
    pass


SelectorType = Callable | Definition | ConcreteDefinition


class BaseRepo:
    _num_saves: int
    _num_constructions: int
    # Links particular concrete definition with particular object
    obj_cache: dict[ConcreteDefinition,Object|None]
    # Links particular Definition object with a concrete definition (Definitions are resolved )
    cdef_cache: dict[int,ConcreteDefinition]
    dir: str
    obj_dir: str
    main_def: ConcreteDefinition | None

    # Helper class for saving objects
    def __init__(self, dir=None):
        # Initialize caches
        self.obj_cache = {}
        self.cdef_cache = {}

        # Some helper variables for monitoring
        self._num_saves = 0
        self._num_constructions = 0

        # Initialize the main def
        self.main_def = None

        if dir is not None:
            self.init_from_dir(dir)

    def init_from_dir(self, dir):
        # We expect the directory to exist.
        if not os.path.exists(dir):
            raise ValueError(f"Directory {dir} doesn't exist.")

        self.dir = dir
        self.obj_dir = os.path.join(self.dir, "objects")

        # Load main definition if it exists
        def_file = os.path.join(self.dir, "def.pkl")
        if os.path.exists(def_file):
            self.main_def = pickle_load(def_file)

        # List the directory and find all the object directories
        try:
            obj_dirs = os.listdir(self.obj_dir)
        except FileNotFoundError:
            # The objects directory doesn't exist
            os.mkdir(self.obj_dir)
            obj_dirs = os.listdir(self.obj_dir)

        # TODO: Do I really want to load all object definitions like this?
        obj_dirs = glob.glob(f"{self.obj_dir}/[0-9a-f][0-9a-f]/*")
        for obj_dir in obj_dirs:
            obj_dir = os.path.join(self.obj_dir, obj_dir)
            def_file = os.path.join(obj_dir, 'def.pkl')
            obj_def = pickle_load(def_file)
            self.obj_cache[obj_def] = None

    def __len__(self):
        return len(self.obj_cache)

    def get_object_directory(self, obj_def: ConcreteDefinition):
        if self.obj_dir is None:
            raise ValueError("Repo not linked to a directory.")

        # Get the directory for the object indicated by obj_def
        def_hash_digest = obj_def.stable_hash()

        # get first two letters
        obj_subdir = def_hash_digest[:2]
        return os.path.join(self.obj_dir, obj_subdir, def_hash_digest)

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
                    self.save_object_instance(value)
                    saved_objs[obj_def] = value
                    self._num_saves += 1
                return value
            elif isinstance(value, ConcreteDefinition):
                obj = value._obj
                if obj is None:
                    raise ValueError("ConcreteDefinitions must be linked to actual objects through _obj to be savable.")
                if value not in saved_objs:
                    self.save_object_instance(obj)
                    saved_objs[value] = obj
                    self._num_saves += 1
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
            self.main_def = obj.definition
        return True

    def save_object_instance(self, obj):
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

        # Create directory for object
        object_path = self.get_object_directory(obj_def)
        if not os.path.exists(object_path):
            os.makedirs(object_path)
        obj.save_to_dir(object_path)

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

                if self.obj_dir:
                    object_path = self.get_object_directory(value)
                    def_file = os.path.join(object_path, "def.pkl")
                    if not os.path.exists(def_file):
                        def_file = None
                else:   
                    object_path = None
                    def_file = None

                if not build_missing and not def_file:
                    raise RuntimeError("Asked not to build missing objects")

                new_values = {} # Need to build a dictionary because we get a list of tuples
                for k, v in new_items:
                    new_values[k] = v
                # Get runtime args that have been built already
                args = new_values['args']
                kwargs = new_values['kwargs']

                # Call special constructor since we already have what we need.
                obj = value.cls(*args, repo=self, __cdef__=value, **kwargs)
                self._num_constructions += 1
                self.obj_cache[value] = obj
                loaded_objs[value] = obj

                if def_file is not None:
                    # confirm we have the same definition
                    definition = pickle_load(def_file)
                    check_hash = definition.stable_hash()
                    value_hash = value.stable_hash()
                    if check_hash != value_hash:
                        raise ValueError(f"Hashes don't match. {check_hash} != {value_hash}")
                    # Load the data from the directory
                    obj.load_from_dir(object_path)
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
                f"Unsupported type {type(item)} for repo.contains!")
        return obj_def in self.obj_cache and self.obj_cache[obj_def] is not None

    def __getitem__(
            self, key: ConcreteDefinition):
        """
        Easy access to objects within.

        if unpack is true, plain objects are returned
        """
        return self.get(key)

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

        def get_obj(obj_def: ConcreteDefinition) -> Object | None:
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
                for obj_def in self.obj_cache:
                    if sel(obj_def):
                        if obj_def in selected_objects:
                            continue
                        obj = get_obj(obj_def)
                        if obj is not None:
                            selected_objects[obj_def] = obj
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

    def write_main_def(self):
        if self.main_def is not None:
            def_file = os.path.join(self.dir, "def.pkl")
            pickle_save(self.main_def, def_file)

    def add_object(self, *args):
        ic(args)
        from dryml.core2.object import Object
        for obj in args:
            if not isinstance(obj, Object):
                raise TypeError("Only Object objects can be added to a repository.")

        # Recursively add objects.

        def _enter(path, key, value):
            ic(path, key, value)
            if isinstance(value, Object):
                return {}, get_object_view(value)
            elif isinstance(value, ConcreteDefinition):
                # Enter the definition's object.
                if value._obj is not None:
                    return {}, get_object_view(value._obj)
            else:
                return default_enter(path, key, value)

        def _exit(path, key, value, new_parent, new_items):
            ic(path, key, value, new_parent, new_items)
            if isinstance(value, Object):
                ic("got an object on exit.")
                if value.definition in self.obj_cache:
                    raise KeyError(f"Repo already has an object matching {value.definition}!")
                self.obj_cache[value.definition] = value
                return value
            else:
                return default_exit(path, key, value, new_parent, new_items)

        remap(
             args,
             enter=_enter,
             visit=default_visit,
             exit=_exit)

    def close(self):
        self.write_main_def()


class Repo(BaseRepo):
    def __init__(self, dir=None):
        self._temp_dir = None

        if dir is None:
            # If none, get a temporary directory
            self.prepare_temp_dir()
            super().__init__(self._temp_dir.name)
        else:
            super().__init__(dir)

    def create_temp_dir(self):
        self._temp_dir = get_temp_directory()

    def prepare_temp_dir(self):
        self.create_temp_dir()

    def close_temp_dir(self):
        if self._temp_dir is not None:
            self._temp_dir.__exit__(None, None, None)

    def close(self):
        self.write_main_def()
        self.close_temp_dir()


class ZipRepo(Repo):
    # A class meant to zip files 'directly' to a zipfile.
    def __init__(self, zip_dest):
        # Save destination
        self.zip_dest = zip_dest

        # Initialize the Repo in temporary directory mode
        super().__init__()

    def prepare_temp_dir(self):
        self.create_temp_dir()
        dir = self._temp_dir.name

        # Load the data if it exists
        def _load_data():
            with zipfile.ZipFile(self.zip_dest, 'r') as zf:
                zf.extractall(dir)

        # Input validation
        if isinstance(self.zip_dest, IOBase):
            # handles file-like objects
            # Check if the buffer has content, if so load it.
            self.zip_dest.seek(0)
            if self.zip_dest.read(1):
                self.zip_dest.seek(0)
                _load_data()
                self.zip_dest.seek(0)
        else:
            # detect whether the path exists, and is a zip file
            try:
                os.fspath(self.zip_dest)
            except TypeError:
                raise TypeError("self.zip_dest must be a path or a file-like object.")
            if os.path.exists(self.zip_dest):
                # Load the data if it exists
                empty = False
                with open(self.zip_dest, 'rb') as f:
                    if not f.read(1):
                        empty = True
                if not empty:
                    _load_data()

    def close(self):
        self.write_main_def()
        # Zip up the directory and its content to its final destination
        zip_directory(self.dir, self.zip_dest)
        self.close_temp_dir()


@contextmanager
def manage_repo(repo=None):
    close_repo = False
    if repo is None:
        repo = Repo()
        close_repo = True
    elif isinstance(repo, Repo):
        pass
    elif isinstance(repo, IOBase):
            # This is a file-like object
        repo = ZipRepo(repo)
        close_repo = True
    elif type(repo) in [str, Path]:
        if os.path.isdir(repo):
            repo = Repo(repo)
            close_repo = True
        # Save as a dryml zip
        else:
            repo = ZipRepo(repo)
            close_repo = True
    else:
        raise ValueError(f"Cannot open a repo pointing to location {repo}")
    yield repo
    if close_repo:
        repo.close()


# Saving and Loading
def save_object(obj, repo=None):
    with manage_repo(repo=repo) as sub_repo:
        main = (repo is not sub_repo) and isinstance(obj, Object)
        sub_repo.save_object(obj, main=main)


def load_object(
        obj_def=None, repo=None,
        cls_remap=None):
    with manage_repo(repo=repo) as repo:
        if obj_def is None:
            obj_def = repo.main_def
        return repo.load_object(obj_def)
