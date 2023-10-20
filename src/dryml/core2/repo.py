import tempfile
import os
import zipfile
from boltons.iterutils import remap, default_enter, default_exit
from collections.abc import ItemsView
from contextlib import contextmanager
from typing import Union, List
from io import IOBase

from dryml.core2.util import zip_directory, hashval_to_digest, \
    unpickler, pickle_to_file, get_remember_view, \
    get_temp_directory


class BaseRepo:
    # Helper class for saving objects
    def __init__(self, dir=None):
        self.objs = {}
        # Some helper variables for monitoring
        self._num_saves = 0
        self._num_constructions = 0

        # We expect the directory to exist.
        if not os.path.exists(dir):
            raise ValueError(f"Directory {dir} doesn't exist.")
        self.dir = dir
        self.obj_dir = os.path.join(self.dir, "objects")

        # List the directory and find all the object directories
        try:
            obj_dirs = os.listdir(self.obj_dir)
        except FileNotFoundError:
            # The objects directory doesn't exist
            os.mkdir(self.obj_dir)
            obj_dirs = os.listdir(self.obj_dir)

        # TODO: Do I really want to load all objects like this?
        for obj_dir in obj_dirs:
            obj_dir = os.path.join(self.obj_dir, obj_dir)
            def_file = os.path.join(obj_dir, 'def.pkl')
            with open(def_file, 'rb') as f:
                obj_def = unpickler(f.read())
            self.load_object(obj_def.concretize())

        def_file = os.path.join(self.dir, "def.pkl")
        if os.path.exists(def_file):
            with open(def_file, "rb") as f:
                self.main_def = unpickler(f.read())
        else:
            self.main_def = None

    def save_object(self, obj, main=False):
        from dryml.core2.object import Remember, Serializable
        saved_objs = {}
        def _save_object_enter(path, key, value):
            if isinstance(value, Remember) and not isinstance(value, Serializable):
                raise ValueError(f"Cannot save non-serializable Remember object {value}")
            if isinstance(value, Serializable):
                result = {'args': value.__args__, 'kwargs': value.__kwargs__}
                return {}, ItemsView(result)
            else:
                return default_enter(path, key, value)

        def _save_object_visit(path, key, value):
            return key, value

        def _save_object_exit(path, key, value, new_parent, new_items):
            if isinstance(value, Serializable):
                obj_def = value.definition.concretize()
                if obj_def not in saved_objs:
                    # TODO: Handle status checking here.
                    self.save_object_imp(value)
                    saved_objs[obj_def] = value
                    self._num_saves += 1
                return value

            else:
                return default_exit(path, key, value, new_parent, new_items)

        # Save the object
        if isinstance(obj, Serializable):
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
            self.main_def = obj.definition.concretize()
        return True

    def save_object_imp(self, obj):
        obj_def = obj.definition.concretize()
        if obj_def in self.objs:
            # Check that this was the same object
            if obj is not self.objs[obj_def]:
                raise ValueError("We already have a different object with definition: {obj_def}")
        else:
            # Add to repo-wide cache
            self.objs[obj_def] = obj

        # Create directory for object
        def_hash_digest = hashval_to_digest(hash(obj_def))
        object_path = os.path.join(self.dir, "objects", def_hash_digest)
        os.mkdir(object_path)
        # Save the object
        return obj._save_to_dir(object_path)

    def load_object(self, obj_def, build_missing=False):
        from dryml.core2.definition import Definition, \
            ConcreteDefinition
        loaded_objs = {}
        def _load_object_enter(path, key, value):
            nonlocal loaded_objs
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
            nonlocal loaded_objs
            # We do nothing here.
            if type(value) is ConcreteDefinition:
                # we already loaded this object, return it.
                return key, loaded_objs[value]
            else:
                return key, value

        def _load_object_exit(path, key, value, new_parent, new_items):
            nonlocal loaded_objs
            def _create_obj():
                # method to actually create an object at this step
                new_values = {}
                for k, v in new_items:
                    new_values[k] = v
                args = new_values['args']
                kwargs = new_values['kwargs']
                self._num_constructions += 1
                return value.cls(*args, **kwargs)

            if isinstance(value, ConcreteDefinition):
                # Check if we already have this object
                if value in self.objs:
                    # we found it
                    loaded_objs[value] = self.objs[value]
                    return loaded_objs[value]

                value_hash = hash(value)
                value_hash_digest = hashval_to_digest(value_hash)
                object_path = os.path.join(self.dir, "objects", value_hash_digest)
                if not os.path.exists(object_path):
                    if not build_missing:
                        raise IndexError(f"Object with hash {value_hash} not found.")
                    # We should build the object, but not create or load from a directory
                    obj = _create_obj()
                    self.objs[value] = obj
                    loaded_objs[value] = obj
                    return obj
                else:
                    obj = _create_obj()
                    # confirm we have the same definition
                    def_file = os.path.join(object_path, "def.pkl")
                    with open(def_file, 'rb') as f:
                        definition = unpickler(f.read())
                        check_hash = hash(definition.concretize())
                    if check_hash != value_hash:
                        raise ValueError(f"Hashes don't match. {check_hash} != {value_hash}")
                    # Load the data from the directory
                    obj._load_from_dir(object_path)
                    self.objs[value] = obj
                    loaded_objs[value] = obj
                    return obj
            else:
                return default_exit(path, key, value, new_parent, new_items)
        from dryml.core2.definition import Definition, \
            ConcreteDefinition
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

    def load_object_imp(self, obj_def):
        from dryml.core2.definition import ConcreteDefinition
        # Perform the actual load from a directory
        if type(obj_def) is not ConcreteDefinition:
            raise TypeError("Only ConcreteDefinition is supported")
        # `load_object` should already have created the object if it doesn't exist in the cache yet.
        obj = self.objs[obj_def]

        # Original DirRepo implementation
        def_hash = hash(obj_def)
        def_hash_digest = hashval_to_digest(def_hash)
        object_path = os.path.join(self.dir, "objects", def_hash_digest)
        if not os.path.exists(object_path):
            raise IndexError(f"Object with hash {def_hash} not found.")
        # confirm we have the same definition
        def_file = os.path.join(object_path, "def.pkl")
        with open(def_file, 'rb') as f:
            definition = unpickler(f.read())
            check_hash = hash(definition.concretize())
        if check_hash != def_hash:
            raise ValueError(f"Hashes don't match. {check_hash} != {def_hash}")
        # Load the data from the directory
        obj._load_from_dir(object_path)
        return obj

    def write_main_def(self):
        if self.main_def is not None:
            def_file = os.path.join(self.dir, "def.pkl")
            pickle_to_file(self.main_def, def_file)


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
def manage_repo(dest=None, repo=None):
    close_repo = False
    if repo is None:
        if dest is None:
            repo = Repo()
        elif isinstance(dest, IOBase):
            # This is a file-like object
            repo = ZipRepo(dest)
        else:
            # detect if the path is a zip file
            extension = os.path.splitext(dest)[-1]
            if extension == ".zip" or extension == ".dry":
                # We have a single file repo
                repo = ZipRepo(dest)
            elif os.path.exists(dest) and os.path.isdir(dest):
                # We have a directory repo
                repo = Repo(dest)
            else:
                # We will treat this as a zip repo
                repo = ZipRepo(dest)
        close_repo = True
    yield repo
    if close_repo:
        repo.close()


# Saving and Loading
def save_object(obj, dest=None, repo=None):
    from dryml.core2 import Serializable
    main = (repo is None) and (isinstance(obj, Serializable))
    with manage_repo(dest=dest, repo=repo) as repo:
        repo.save_object(obj, main=main)
        return True


def load_object(
        obj_def=None, dest=None, repo=None,
        cls_remap=None):
    from dryml.core2.definition import Definition, concretize_definition
    with manage_repo(dest=dest, repo=repo) as repo:
        if obj_def is None:
            obj_def = repo.main_def
        return repo.load_object(concretize_definition(obj_def))
