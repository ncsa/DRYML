from __future__ import annotations

import os
import glob
import zipfile
from boltons.iterutils import remap, default_enter, default_exit
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
        def_hash_digest = hashval_to_digest(hash(obj_def))

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
            elif (is_dictlike(value) or is_collection(value)) and not isinstance(value, np.ndarray):
                return key, value
            else:
                ic("copying", value)
                return key, deepcopy(value)

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
            # Copy args so modifications to this ConcreteDefinition doesn't change the original
            # Values in the original Definitions
            args = deepcopy_skip_definition_object(args)
            kwargs = deepcopy_skip_definition_object(kwargs)
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
            if isinstance(value, Lazy):
                result = {'args': value.__args__, 'kwargs': value.__kwargs__}
                return {}, ItemsView(result)
            else:
                return default_enter(path, key, value)

        def _save_object_visit(path, key, value):
            return key, value

        def _save_object_exit(path, key, value, new_parent, new_items):
            if isinstance(value, Lazy):
                obj_def = value.definition()
                if obj_def not in saved_objs:
                    self.save_object_instance(value)
                    saved_objs[obj_def] = value
                    self._num_saves += 1
                return value

            else:
                return default_exit(path, key, value, new_parent, new_items)

        # Save the object
        if isinstance(obj, Lazy):
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
            self.main_def = obj.definition()
        return True

    def save_object_instance(self, obj):
        obj_def = obj.definition()
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
        if obj.__initialized__:
            # Save the object
            obj.save_to_dir(object_path)
        else:
            raise RepoSaveError("Cannot save an uninitialized Lazy object.")

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
            def _create_obj():
                # method to actually create an object at this step
                new_values = {}
                for k, v in new_items:
                    new_values[k] = v
                args = new_values['args']
                kwargs = new_values['kwargs']
                self._num_constructions += 1
                return value.cls(*args, repo=self, **kwargs)

            if isinstance(value, ConcreteDefinition):
                # Check if we already have this object
                if value in self.obj_cache:
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

                obj = _create_obj()
                self.obj_cache[value] = obj
                loaded_objs[value] = obj

                if def_file is not None:
                    # confirm we have the same definition
                    definition = pickle_load(def_file)
                    check_hash = hash(definition)
                    if check_hash != value_hash:
                        raise ValueError(f"Hashes don't match. {check_hash} != {value_hash}")
                    # Load the data from the directory
                    obj.load_from_dir(object_path)
                return obj
            else:
                return default_exit(path, key, value, new_parent, new_items)

        if isinstance(obj_def, ConcreteDefinition):
            return remap(
                [obj_def],
                enter=_load_object_enter,
                visit=_load_object_visit,
                exit=_load_object_exit)[0]
        else:
            return remap(
                obj_def,
                enter=_load_object_enter,
                visit=_load_object_visit,
                exit=_load_object_exit)

    def write_main_def(self):
        if self.main_def is not None:
            def_file = os.path.join(self.dir, "def.pkl")
            pickle_save(self.main_def, def_file)

    def add_object(self, *args):
        from dryml.core2.object import Lazy
        for obj in args:
            if not isinstance(obj, Lazy):
                raise TypeError("Only Lazy objects can be added to a repository.")
            self.obj_cache[obj] = obj

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
        main = (repo is not sub_repo) and isinstance(obj, Lazy)
        sub_repo.save_object(obj, main=main)


def load_object(
        obj_def=None, repo=None,
        cls_remap=None):
    with manage_repo(repo=repo) as repo:
        if obj_def is None:
            obj_def = repo.main_def
        return repo.load_object(obj_def)
