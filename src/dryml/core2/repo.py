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
    def __init__(self):
        self.objs = {}
        # Some helper variables for monitoring
        self._num_saves = 0
        self._num_constructions = 0

    def save_object(self, obj):
        from dryml.core2.object import Serializable
        saved_objs = {}
        def _save_object_enter(path, key, value):
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

        return True

    def save_object_imp(self, obj):
        obj_def = obj.definition.concretize()
        if obj_def in self.objs:
            # Check that this was the same object
            if obj is not self.objs[obj_def]:
                raise ValueError("We already have a different object with definition: {obj_def}")
        else:
            self.objs[obj_def] = obj
        # We return the obj_def to prevent having to recalculate it in other functions
        return obj_def

    def load_object(self, obj_def):
        from dryml.core2.definition import Definition, \
            ConcreteDefinition
        def _load_object_enter(path, key, value):
            if type(value) is Definition:
                raise TypeError("Definition not allowed here!")
            elif type(value) is ConcreteDefinition:
                if value in self.objs:
                    return value, False
                else:
                    return {}, ItemsView({'args': value['args'], 'kwargs': value['kwargs']})
            else:
                return default_enter(path, key, value)

        def _load_object_visit(path, key, value):
            # We do nothing here.
            if type(value) is ConcreteDefinition:
                return key, self.objs[value]
            else:
                return key, value

        def _load_object_exit(path, key, value, new_parent, new_items):
            if isinstance(value, ConcreteDefinition):
                # We need to create this object.
                new_values = {}
                for k, v in new_items:
                    new_values[k] = v
                args = new_values['args']
                kwargs = new_values['kwargs']
                obj = value.cls(*args, **kwargs)
                self.objs[value] = obj
                # Trigger loading of data for this object
                self.load_object_imp(value)
                self._num_constructions += 1
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
        return self.objs[obj_def]

    def close(self):
        pass


class Repo(BaseRepo):
    def __init__(self):
        super().__init__()

    def save_object_imp(self, obj):
        super().save_object_imp(obj)

    def load_object_imp(self, obj_def):
        return super().load_object_imp(obj_def)


class DirRepo(BaseRepo):
    # A class to manage saving objects to a directory.
    def __init__(self, dir):
        super().__init__()
        # We expect the directory to exist.
        if not os.path.exists(dir):
            raise ValueError(f"Directory {dir} doesn't exist.")
        self.dir = dir
        self.obj_dir = os.path.join(self.dir, "objects")
        # List the directory and find all the hashes
        try:
            obj_dirs = os.listdir(self.obj_dir)
        except FileNotFoundError:
            # The objects directory doesn't exist
            os.mkdir(self.obj_dir)
            obj_dirs = os.listdir(self.obj_dir)

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
        from dryml.core2.object import Serializable
        super().save_object(obj)
        if main:
            self.main_def = obj.definition.concretize()
        return True

    def save_object_imp(self, obj):
        obj_def = super().save_object_imp(obj)
        # Create directory for object
        def_hash_digest = hashval_to_digest(hash(obj_def))
        object_path = os.path.join(self.dir, "objects", def_hash_digest)
        os.mkdir(object_path)
        # Save the object
        obj._save_to_dir(object_path)

    def load_object_imp(self, obj_def):
        obj = super().load_object_imp(obj_def)
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

    def close(self):
        super().close()
        if self.main_def is not None:
            def_file = os.path.join(self.dir, "def.pkl")
            pickle_to_file(self.main_def, def_file)


class ZipRepo(DirRepo):
    # A class meant to zip files 'directly' to a zipfile.
    def __init__(self, zip_dest):
        # We load the zip file to a temporary directory
        self.temp_dir = get_temp_directory()

        def _load_data():
            with zipfile.ZipFile(zip_dest, 'r') as zf:
                zf.extractall(self.temp_dir.name)

        # Input validation
        if isinstance(zip_dest, IOBase):
            # handles file-like objects
            # Check if the buffer has content, if so load it.
            zip_dest.seek(0)
            if zip_dest.read(1):
                zip_dest.seek(0)
                _load_data()
                zip_dest.seek(0)
        else:
            # detect whether the path exists, and is a zip file
            try:
                os.fspath(zip_dest)
            except TypeError:
                raise TypeError("zip_dest must be a path or a file-like object.")
            if os.path.exists(zip_dest):
                # Load the data if it exists
                _load_data()

        # Save destination
        self.zip_dest = zip_dest

        super().__init__(self.temp_dir.name)

    def close(self):
        super().close()
        # Zip up the temp directory
        zip_directory(self.temp_dir.name, self.zip_dest)
        # Close the temp directory
        self.temp_dir.__exit__(None, None, None)


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
            if os.path.splitext(dest)[-1] == ".zip":
                repo = ZipRepo(dest)
            else:
                repo = DirRepo(dest)
        close_repo = True
    yield repo
    if close_repo:
        repo.close()


# Saving and Loading
def save_object(obj, dest=None, repo=None):
    main = repo is None
    with manage_repo(dest=dest, repo=repo) as repo:
        repo.save_object(obj, main=main)
        return True


def load_object(obj_def=None, dest=None, repo=None):
    from dryml.core2.definition import Definition
    with manage_repo(dest=dest, repo=repo) as repo:
        if obj_def is None:
            obj_def = repo.main_def
        return repo.load_object(obj_def.concretize())
