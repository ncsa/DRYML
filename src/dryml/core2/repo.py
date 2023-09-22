import tempfile
import os
from boltons.iterutils import remap, default_enter, default_exit
from collections.abc import ItemsView
from contextlib import contextmanager
from typing import Union, List

from dryml.core2.util import zip_directory, hashval_to_digest, \
    unpickler, pickle_to_file


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

    def save_object(self, obj):
        from dryml.core2.object import Serializable
        super().save_object(obj)
        if isinstance(obj, Serializable):
            # Save the definition to the main repo directory
            def_file = os.path.join(self.dir, "def.pkl")
            pickle_to_file(obj.definition, def_file)

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


class ZipRepo(DirRepo):
    # A class meant to zip files 'directly' to a zipfile.
    def __init__(self, path):
        # We load the zip file to a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = path
        super().__init__(self.temp_dir)

    def close(self):
        # Zip up the temp directory
        zip_directory(self.temp_dir, self.path)
        # Close the temp directory
        self.temp_dir.__exit__(None, None, None)


@contextmanager
def manage_repo(path=None, repo=None):
    close_repo = False
    if repo is None:
        if path is None:
            repo = Repo()
        else:
            # detect if the path is a zip file
            if os.path.splitext(path)[-1] == ".zip":
                repo = ZipRepo(path)
            else:
                repo = DirRepo(path)
        close_repo = True
    yield repo
    if close_repo:
        repo.close()


# Saving and Loading
def save_object(obj, path=None, repo=None):
    with manage_repo(path=path, repo=repo) as repo:
        repo.save_object(obj)


def load_object(obj_def, path=None, repo=None):
    from dryml.core2.definition import Definition
    if type(obj_def) is Definition:
        obj_def = obj_def.concretize()
    with manage_repo(path=path, repo=repo) as repo:
        return repo.load_object(obj_def)
