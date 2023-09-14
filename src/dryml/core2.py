import uuid
import time
import hashlib
from inspect import signature, Parameter, isclass
from dryml.utils import is_dictlike, pickle_to_file
from boltons.iterutils import remap, is_collection, get_path, PathAccessError
from functools import cached_property
import numpy as np
import pickle
import os
import tempfile
import zipfile
from typing import Union, List
from copy import deepcopy
from contextlib import contextmanager
from inspect import currentframe


def collide_attributes(obj, attr_list):
    # Check if these attributes are already defined. Throw an error if they are.
    colliding_attrs = []
    for attr in attr_list:
        if hasattr(obj, attr):
            colliding_attrs.append(attr)
    if len(colliding_attrs) > 0:
        raise AttributeError(f"Attributes {colliding_attrs} already exist on object. Cannot create object.")


def cls_super(cls=None):
    if cls is None:
         # We must detect the class ourselves
         frame = currentframe().f_back
         cls = frame.f_locals.get('self', None).__class__
    return ClsSuperManager(0, cls.mro())


class ClsSuperManager:
    # A class to mimic `super()` but without an actual instance
    def __init__(self, mro_i, mro):
        self.mro = mro
        self.mro_i = mro_i

    def __call__(self, *args, **kwargs):
        # Create a new object which will reset the advance counter
        return self

    def __getattribute__(self, name):
        if name in super().__getattribute__('__dict__'):
            # If its a plain attribute just return it
            return super().__getattribute__(name)
        else:
            # We have a method
            # We must advance through the mro until we find a class with the method
            for i in range(self.mro_i, len(self.mro)):
                cls = self.mro[i]
                if name in cls.__dict__:
                    # We found a class with the method
                    # Advance the counter
                    self.mro_i = i
                    # Get the method
                    method = cls.__dict__[name]
                    def exec_method(*args, **kwargs):
                        return method(self, *args, **kwargs)
                    return exec_method
            # We didn't find anymore classes with this method
            return None


class CreationControl(type):
    # Support metaclass to allow more complex metaclass behavior
    def __create_instance__(cls):
        return cls.__new__(cls)

    def __call__(cls, *args, **kwargs):
        obj = cls.__create_instance__()
        args, kwargs = cls_super(cls).__arg_manipulation__(*args, **kwargs)
        obj.__pre_init__(*args, **kwargs)
        obj.__initialize_instance__(*args, **kwargs)
        return obj


class Object(metaclass=CreationControl):
    @staticmethod
    def __arg_manipulation__(cls_super, *args, **kwargs):
        return args, kwargs

    def __pre_init__(self, *args, **kwargs):
        pass

    def __initialize_instance__(self, *args, **kwargs):
        return self.__init__(*args, **kwargs)

    def __init__(self):
        pass


def get_kwarg_defaults(cls):
    kwarg_defaults = {}
    for current_class in reversed(cls.mro()):
        if hasattr(current_class, '__init__'):
            init_signature = signature(current_class.__init__)
            for name, param in init_signature.parameters.items():
                if param.default != Parameter.empty:
                    kwarg_defaults[name] = param.default
    return kwarg_defaults


class Remember(Object):
    # Support class which remembers the arguments used when creating it.
    def __pre_init__(self, *args, **kwargs):
        super().__pre_init__(*args, **kwargs)
        collide_attributes(self, [
            '__args__',
            '__kwargs__',])
        default_kwargs = get_kwarg_defaults(type(self))
        self.__args__ = deepcopy(args)
        # We merge the default kwargs with the kwargs passed in.
        # Defaults are first so they can be overwritten.
        self.__kwargs__ = deepcopy({ **default_kwargs, **kwargs })

    @cached_property
    def definition(self):
        return build_definition(self)


class Defer(Remember):
    # Since methods are part of the class, we only have to remove data from the object. We mark the protected data here. Keep up to date with attributes added 
    def __pre_init__(self, *args, **kwargs):
        super().__pre_init__(*args, **kwargs)
        collide_attributes(self, [
            '__initialized__',
            '__locked__',])
        self.__initialized__ = False
        self.__locked__ = False
        self.__orig_keys__ = None
        self.__orig_keys__ = list(self.__dict__.keys())

    def __initialize_instance__(self, *args, **kwargs):
        # We explicitly don't initialize the instance.
        pass

    def __getattribute__(self, name):
        # First, check if we have this attribute
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # If we don't next check if we're initialized
            if not super().__getattribute__('__initialized__'):
                super().__getattribute__('__initialize__')()
        # Then check again
        return super().__getattribute__(name)

    def __initialize__(self):
        if self.__locked__:
            raise RuntimeError("Cannot initialize object. Object is locked.")
        self.__init__(*self.__args__, **self.__kwargs__)
        self.__initialized__ = True

    def __unload__(self):
        if self.__locked__:
            raise RuntimeError("Cannot unload object. Object is locked.")
        # Remove all attributes besides self._orig_attrs
        for attr in list(self.__dict__.keys()):
            if attr not in self.__orig_keys__:
                delattr(self, attr)
        self.__initialized__ = False


class UniqueID(Object):
    @staticmethod
    def __arg_manipulation__(cls_super, *args, **kwargs):
        args, kwargs = cls_super().__arg_manipulation__(*args, **kwargs)
        if 'uid' not in kwargs:
            kwargs['uid'] = str(uuid.uuid4())
        return args, kwargs

    def __init__(self, *args, uid=None, **kwargs):
        super().__init__(*args, **kwargs)
        # unique ID
        self.uid = uid


class Metadata(Object):
    @staticmethod
    def __arg_manipulation__(cls_super, *args, **kwargs):
        args, kwargs = cls_super().__arg_manipulation__(*args, **kwargs)
        if 'metadata' not in kwargs:
            kwargs['metadata'] = {
            }
        if 'description' not in kwargs['metadata']:
            kwargs['metadata']['description'] = ""
        if 'creation_time' not in kwargs['metadata']:
            kwargs['metadata']['creation_time'] = time.time()
        return args, kwargs

    def __init__(self, *args, metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata


class Serializable(Remember):
    def _save_to_dir(self, dir: str):
        # Directory into which the model should save its 'heavy' content
        # Full save procedure handled elsewhere
        # We expect the directory to exist. Caller should handle this
        if not os.path.exists(dir):
            raise ValueError(f"Path {dir} does not exist. Can't save")

        # Save the definition
        def_file = os.path.join(dir, 'def.pkl')
        pickle_to_file(self.definition, def_file)

        output_file = os.path.join(dir, 'object.pkl')
        with open(output_file, 'wb') as f:
            f.write(pickle.dumps(self))

    def _load_from_dir(self, dir: str):
        # Load 'heavy' content from directory
        # Again directory should exist. Caller will handle it.
        if not os.path.exists(dir):
            raise ValueError(f"Path {dir} does not exist. Can't load")

        def_file = os.path.join(dir, 'def.pkl')
        with open(def_file, 'rb') as f:
            definition = pickle.loads(f.read())

        if definition != self.definition:
            raise ValueError("Definition for data in directory {dir} doesn't match this object. Can't load")

        input_file = os.path.join(dir, 'object.pkl')
        with open(input_file, 'rb') as f:
            obj = pickle.loads(f.read())
        self.__dict__.update(obj.__dict__)


class Definition(dict):
    allowed_keys = ['cls', 'args', 'kwargs']
    def __init__(self, *args, **kwargs):
        # We want to copy the arguments so we don't
        # mutate them
        args = deepcopy(args)
        kwargs = deepcopy(kwargs)
        init = False
        if len(args) > 0:
            if callable(args[0]):
                super().__init__(
                    cls=args[0],
                    args=args[1:],
                    kwargs=kwargs)
                init = True
        if not init:
            super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key not in self.allowed_keys:
            raise KeyError(f"Key {key} not allowed in Definition. Allowed keys are {self.allowed_keys}")
        super().__setitem__(key, value)

    def copy(self):
        return deepcopy(self)

    def __hash__(self):
        return digest_to_hashval(hash_function(self))

    def __eq__(self, rhs):
        return selector_match(self, rhs)

    @property
    def cls(self):
        return self['cls']

    @property
    def args(self):
        return self['args']

    @property
    def kwargs(self):
        return self['kwargs']


def hashval_to_digest(val):
    # Suggestion from gpt-4
    return hex(val & ((1 << 64) - 1))[2:]


def digest_to_hashval(digest):
    return int(digest, 16)


def hash_function(structure):
    # Definition hash support
    class HashHelper(object):
        def __init__(self, the_hash):
            self.hash = the_hash

    def hash_value(value):
        # Hashes for supported values here
        try:
            hash_val = hash(value)
            return hashval_to_digest(hash_val)
        except TypeError:
            pass

        if isclass(value):
            return str(value.__qualname__)
        elif isinstance(value, np.ndarray):
            # Hash for numpy arrays
            return hashlib.sha256(value.tobytes()).hexdigest()
        else:
            raise TypeError(f"Value of type {type(value)} not supported for hashing.")

    def hash_visit(path, key, value):
        # Skip if it's a hashlib hasher
        if isinstance(value, HashHelper):
            return key, value.hash
        elif (is_dictlike(value) or is_collection(value)) and not isinstance(value, np.ndarray):
            return key, value
    
        return key, hash_value(value)

    def hash_exit(path, key, old_parent, new_parent, new_items):
        # At this point, all items should be hashes

        # sort the items. format is [(key, value)]
        new_items = sorted(new_items, key=lambda x: x[0])

        # Combine the hashes
        hasher = hashlib.sha256()
        # Add a string representation of the old parent type
        hasher.update(type(old_parent).__qualname__.encode())
        for _, v in new_items:
            hasher.update(v.encode())
        new_hash = hasher.hexdigest()

        return HashHelper(new_hash)

    return remap(structure, visit=hash_visit, exit=hash_exit).hash


# Creating definitions from objects
def build_definition(obj):
    if isinstance(obj, Remember):
        return remap([obj], visit=build_definition_visit)[0]
    else:
        return remap(obj, visit=build_definition_visit)


def build_definition_visit(_, key, value):
    if isinstance(value, Remember):
        args = build_definition(value.__args__)
        kwargs = build_definition(value.__kwargs__)
        return key, Definition(
            type(value),
            *args,
            **kwargs)
    else:
        return key, value


# Creating objects from definitions
def build_from_definition(definition, path=None, repo=None):
    with manage_repo(path=path, repo=repo) as repo:
        def build_from_definition_visit(_, key, value):
            if isinstance(value, Definition):
                # Delegate to a repo to do the loading
                obj = repo.load_object(value)
                return key, obj
            else:
                return key, value

        if isinstance(definition, Definition):
            return remap([definition], visit=build_from_definition_visit)[0]
        else:
            return remap(definition, visit=build_from_definition_visit)


def is_nonclass_callable(obj):
    return callable(obj) and not isclass(obj)


## Selecting objects
def selector_match(selector, definition):
    def selector_match_visit(path, key, value):
        print(f"selector_match_visit: path: {path}, key: {key}, value: {value}")
        # Try to get the value at the right path from the definition
        try:
            def_val = get_path(definition, path)[key]
        except PathAccessError:
            print("selector_match_visit: definition doesn't have path")
            return key, False

        print(f"selector_match_visit: def_val: {def_val}")

        if isclass(def_val):
            # We have a class in the definition.
            # If the selector value is a class, then the definition value must be a subclass.
            # This must also work for objects with metaclasses which aren't type
            if isclass(value):
                print("selector_match_visit: class 1")
                return key, issubclass(value, def_val)
            elif callable(value):
                print("selector_match_visit: class 2")
                # We use the callable to determine if we match
                return key, value(def_val)
            else:
                print("selector_match_visit: class 3")
                # we don't have the right type in the selector
                return key, False
        elif (is_collection(def_val) or is_dictlike(def_val)) and not isinstance(def_val, np.ndarray):
            print("selector_match_visit: collection 1")
            # We do nothing for these collections. Wait for their elements to be matched
            return key, value
        elif isinstance(def_val, np.ndarray):
            if isinstance(value, np.ndarray):
                print("selector_match_visit: array 1")
                return key, np.all(def_val == value)
            elif is_nonclass_callable(value):
                print("selector_match_visit: array 2")
                return key, value(def_val)
            else:
                # type doesn't match.
                print("selector_match_visit: array 3")
                return key, False
        else:
            # Plain matching branch
            if is_nonclass_callable(value):
                print("selector_match_visit: plain 1")
                return key, value(def_val)
            elif type(value) != type(value):
                print("selector_match_visit: plain 2")
                return key, False
            else:
                print("selector_match_visit: plain 3")
                return key, value == def_val
            

    def selector_match_exit(path, key, old_parent, new_parent, new_items):
        print(f"selector_match_exit: path: {path}, key: {key}, old_parent: {old_parent}, new_parent: {new_parent}, new_items: {new_items}")
        if type(old_parent) != type(new_parent):
            print(f"selector_match_exit: type check")
            return False
        else:
            if is_collection(old_parent) and not is_dictlike(old_parent):
                print(f"collection check definition: {definition}")
                # For tuples and list arguments, the lengths must match.
                if key is None:
                    def_values = get_path(definition, path)
                else:
                    def_values = get_path(definition, path)[key]
                print(f"def_values: {def_values}")
                if len(def_values) != len(new_items):
                    return False
            print(f"selector_match_exit: catchall check")
            final = True
            for val in map(lambda t: t[1], new_items):
                final = final & val
            return final

    # We reduce across the selector because we are only checking the values supplied
    # In the selector.
    return remap(selector, visit=selector_match_visit, exit=selector_match_exit)


def zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Ignoring dirs for now. May need to edit this in the future.
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)


class BaseRepo:
    # Helper class for saving objects
    def __init__(self):
        self.objs = {}
        # Some helper variables for monitoring
        self._num_saves = 0
        self._num_constructions = 0

    def save_object(self, obj: Serializable):
        def_hash = hash(obj.definition)
        if def_hash not in self.objs:
            # store the object
            self.objs[def_hash] = obj
            self._num_saves += 1
            return def_hash
        else:
            return None

    def load_object(self, obj_def: Definition):
        def_hash = hash(obj_def)
        if def_hash in self.objs:
            return None, self.objs[def_hash]
        else:
            # We don't have the object. We must construct it
            args = build_from_definition(obj_def.args, repo=self)
            kwargs = build_from_definition(obj_def.kwargs, repo=self)
            obj = obj_def.cls(*args, **kwargs)
            self.objs[def_hash] = obj
            self._num_constructions += 1
            return def_hash, obj

    def close(self):
        pass


class Repo(BaseRepo):
    def __init__(self):
        super().__init__()

    def save_object(self, obj: Serializable):
        super().save_object(obj)

    def load_object(self, obj_def: Definition):
        _, obj = super().load_object(obj_def)
        return obj


class DirRepo(BaseRepo):
    # A class to manage saving objects to a directory.
    def __init__(self, dir):
        # We expect the directory to exist.
        if not os.path.exists(dir):
            raise ValueError(f"Directory {dir} doesn't exist.")
        self.dir = dir
        self.obj_dir = os.path.join(self.dir, "objects")
        # List the directory and find all the hashes
        try:
            self.saved_objs = set(os.listdir(self.obj_dir))
        except FileNotFoundError:
            # The directory doesn't exist
            os.mkdir(self.obj_dir)
        super().__init__()

    def save_object(self, obj: Serializable):
        def_hash = super().save_object(obj)
        if def_hash is not None:
            # Create directory for object
            def_hash_digest = hashval_to_digest(def_hash)
            object_path = os.mkdir(os.path.join(self.dir, "objects", def_hash_digest))
            # Save the object
            obj._save_to_dir(object_path)

    def load_object(self, obj_def):
        def_hash, obj = super().load_object(obj_def)
        if def_hash is None:
            # We don't have to load any data
            return obj
        else:
            def_hash_digest = hashval_to_digest(def_hash)
            object_path = os.path.join(self.dir, "objects", def_hash_digest)
            if not os.path.exists(object_path):
                raise IndexError(f"Object with hash {def_hash} not found.")
            # confirm we have the same definition
            def_file = os.path.join(object_path, "def.pkl")
            with open(def_file, 'rb') as f:
                definition = pickle.loads(f.read())
                check_hash = hash(definition)
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
def save_object(obj: Union[Serializable,List[Serializable]], path=None, repo=None):
    with manage_repo(path=path, repo=repo) as repo:
        # Save the object
        repo.save_object(obj)


def load_object(obj_def: Definition, path=None, repo=None):
    with manage_repo(path=path, repo=repo) as repo:
        # Save the object
        repo.load_object(obj_def)
