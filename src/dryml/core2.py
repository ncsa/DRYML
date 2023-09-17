import uuid
import time
import hashlib
from inspect import signature, Parameter, isclass
from dryml.utils import is_dictlike, pickle_to_file
from boltons.iterutils import remap, is_collection, get_path, PathAccessError, default_enter, default_exit
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
from collections.abc import ItemsView


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
        self.mro_i += 1
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
        # __arg_manipulation__ should be an idempotent function
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
        # TODO investigate whether we should include a check to make sure the user isn't passing
        # any Definition objects. I think we should probably disallow that.
        self.__args__ = deepcopy_skip_definition_object(args)
        # We merge the default kwargs with the kwargs passed in.
        # Defaults are first so they can be overwritten.
        self.__kwargs__ = deepcopy_skip_definition_object({ **default_kwargs, **kwargs })

    @cached_property
    def definition(self):
        return build_definition(self)

    def __repr__(self):
        return f"<{self.__class__.__name__} at {hex(id(self))}>(args={self.__args__}, kwargs={self.__kwargs__})"


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


def definition_enter(path, key, value):
    if isinstance(value, Definition):
        return {}, ItemsView(value)
    else:
        return default_enter(path, key, value)


def definition_exit(path, key, value, new_parent, new_items):
    if isinstance(value, Definition):
        # We have a definition. we have to be careful how we construct it.
        for k, v in new_items:
            new_parent[k] = v
        args = new_parent['args']
        kwargs = new_parent['kwargs']
        cls = new_parent['cls']
        return type(value)(cls, *args, **kwargs)
    else:
        return default_exit(path, key, value, new_parent, new_items)


def deepcopy_skip_definition_object(defn):
    def _deepcopy_enter(path, key, value):
        if isinstance(value, Object):
            return value, False
        elif isinstance(value, Definition):
            return value, False
        else:
            return default_enter(path, key, value)

    def _deepcopy_visit(path, key, value):
        if isinstance(value, Object):
            # We have an already realized class instance. We shouldn't deep copy it.
            return key, value
        elif isinstance(value, Definition):
            # unique definitions are supposed to refer to specific objects during 'rendering'
            # We shouldn't copy Definitions
            return key, value
        elif (is_dictlike(value) or is_collection(value)) and not isinstance(value, np.ndarray):
            return key, value
        else:
            return key, deepcopy(value)

    if type(defn) is Definition:
        return remap([defn], enter=_deepcopy_enter, visit=_deepcopy_visit, exit=definition_exit)[0]
    else:
        return remap(defn, enter=_deepcopy_enter, visit=_deepcopy_visit, exit=definition_exit)


class Definition(dict):
    allowed_keys = ['cls', 'args', 'kwargs']
    def __init__(self, *args, **kwargs):
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
        # A true deepcopy
        return deepcopy(self)

    def __eq__(self, rhs):
        if type(self) != type(rhs):
            return False
        # We actually need to check in both directions.
        if not selector_match(self, rhs, strict=True):
            return False
        if not selector_match(rhs, self, strict=True):
            return False
        return True

    def __ne__(self, rhs):
        return not self.__eq__(rhs)

    def concretize(self):
        return concretize_definition(self)

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    @property
    def cls(self):
        return self['cls']

    @property
    def args(self):
        return self['args']

    @property
    def kwargs(self):
        return self['kwargs']


def concretize_definition(defn: Definition):
    # Cache for completed results. All duplicate ConcreteDefinitions should refer to the SAME object and so the same definition
    # Key for this cache will be the ConcreteDefinition hash.
    definition_cache = {}

    def _concretize_definition_enter(path, key, value):
        if id(value) in definition_cache:
            return value, False
        elif type(value) is ConcreteDefinition:
            # The definition is already concrete. don't enter it.
            return value, False
        else:
            return definition_enter(path, key, value)

    def _concretize_definition_visit(path, key, value):
        if id(value) in definition_cache: 
            return key, definition_cache[id(value)]
        elif type(value) is ConcreteDefinition:
            # Value is already Concrete
            return key, value
        elif isinstance(value, Object):
            # We have an already realized class instance. We shouldn't deep copy it.
            return key, value
        elif (is_dictlike(value) or is_collection(value)) and not isinstance(value, np.ndarray):
            return key, value
        else:
            return key, deepcopy(value)

    def _concretize_definition_exit(path, key, values, new_parent, new_items):
        if isinstance(values, Definition):
            for k, v in new_items:
                new_parent[k] = v
            args = new_parent['args']
            kwargs = new_parent['kwargs']
            cls = new_parent['cls']
            # Do argument manipulations
            args, kwargs = cls_super(cls).__arg_manipulation__(*args, **kwargs)
            # Copy args so modifications to this ConcreteDefinition doesn't change the original
            # Values in the original Definitions
            args = deepcopy_skip_definition_object(args)
            kwargs = deepcopy_skip_definition_object(kwargs)
            # Create the now concrete definition
            new_def = ConcreteDefinition(cls, *args, **kwargs) 
            # Check if we've encountered this definition before
            definition_cache[id(values)] = new_def
            return new_def
        else:
            return default_exit(path, key, values, new_parent, new_items)

    if isinstance(defn, Definition):
        return remap(
            [defn],
            enter=_concretize_definition_enter,
            visit=_concretize_definition_visit,
            exit=_concretize_definition_exit)[0]
    else:
        return remap(
            defn,
            enter=_concretize_definition_enter,
            visit=_concretize_definition_visit,
            exit=_concretize_definition_exit)


class ConcreteDefinition(Definition):
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            raise ValueError("ConcreteDefinition must be created with arguments")
        if len(args) > 0:
            if not isclass(args[0]):
                raise TypeError("ConcreteDefinition's first argument must be a class")
        super().__init__(*args, **kwargs)

    def concretize(self):
        return self

    def __hash__(self):
        return digest_to_hashval(hash_function(self))


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

    return remap(structure, enter=definition_enter, visit=hash_visit, exit=hash_exit).hash


# Creating definitions from objects
def build_definition(obj):
    instance_cache = {}

    def _build_definition_enter(path, key, value):
        id_value = id(value)
        if id_value in instance_cache:
            return value, False
        elif isinstance(value, Remember):
            return {}, ItemsView({'cls': type(value), 'args': value.__args__, 'kwargs': value.__kwargs__})
        elif isinstance(value, Definition):
            # We can encounter definitions we have already created
            found_def = False
            for _, v in instance_cache.items():
                if id(v) == id(value):
                    found_def = True
                    break
            if not found_def:
                raise ValueError("We should have built this defintion ourselves")
            else:
                # We don't want to enter already built definitions
                value, False
        else:
            return default_enter(path, key, value)

    def _build_definition_visit(_, key, value):
        id_value = id(value)
        if id_value in instance_cache:
            # First return any instance we have already cached
            return key, instance_cache[id_value]
        elif isinstance(value, Remember):
            raise TypeError("Unexpected type!")
        elif isinstance(value, Definition):
            # We should've stored this definition in the cache already
            found_def = False
            for _, v in instance_cache.items():
                if id(v) == id(value):
                    found_def = True
                    break
            if not found_def:
                raise TypeError("We should have constructed this Definition!")
            else:
                return key, value
        elif is_collection(value) or is_dictlike(value):
            # Don't do anything to the collections
            return key, value
        else:
            # This is a regular value. We need to deepcopy it.
            return key, deepcopy(value)

    def _build_definition_exit(path, key, values, new_parent, new_items):
        if isinstance(values, Remember) and type(new_parent) is dict:
            new_values = {}
            for k, v in new_items:
                new_values[k] = v
            args = new_values['args']
            kwargs = new_values['kwargs']
            # We want to copy the arguments so we don't
            # mutate them unless they're definitions or Objects
            args = deepcopy_skip_definition_object(args)
            kwargs = deepcopy_skip_definition_object(kwargs)

            new_def = Definition(
                type(values),
                *args,
                **kwargs)

            # Cache the instance result
            instance_cache[id(values)] = new_def

            return new_def
        else:
            return default_exit(path, key, values, new_parent, new_items)

    if isinstance(obj, Remember):
        return remap([obj], enter=_build_definition_enter, visit=_build_definition_visit, exit=_build_definition_exit)[0]
    else:
        return remap(obj, enter=_build_definition_enter, visit=_build_definition_visit, exit=_build_definition_exit)



# Creating objects from definitions
def build_from_definition(definition, path=None, repo=None):
    # First, concretize the definition
    concrete_definition = concretize_definition(definition)

    # concrete definitions refer to specific objects

    with manage_repo(path=path, repo=repo) as repo:
        def build_from_definition_visit(_, key, value):
            if type(value) is Definition:
                raise TypeError("Definitions should've been turned into ConcreteDefinitions at this point")
            elif type(value) is ConcreteDefinition:
                # Delegate to a repo to do the loading
                obj = repo.load_object(value)
                return key, obj
            else:
                return key, value

        if isinstance(definition, Definition):
            return remap([concrete_definition], enter=definition_enter, visit=build_from_definition_visit, exit=definition_exit)[0]
        else:
            return remap(concrete_definition, enter=definition_enter, visit=build_from_definition_visit, exit=definition_exit)


def is_nonclass_callable(obj):
    return callable(obj) and not isclass(obj)


## Selecting objects
def selector_match(selector, definition, strict=False):
    # Method for testing if a selector matches a definition
    # if strict is set, it must match exactly, and callables arent' allowed.
    def selector_match_visit(path, key, value):
        # Try to get the value at the right path from the definition
        try:
            def_val = get_path(definition, path)[key]
        except PathAccessError:
            return key, False

        if isclass(def_val):
            # We have a class in the definition.
            # If the selector value is a class, then the definition value must be a subclass.
            # This must also work for objects with metaclasses which aren't type
            if isclass(value):
                return key, issubclass(value, def_val)
            elif callable(value) and not strict:
                # We use the callable to determine if we match
                return key, value(def_val)
            else:
                # we don't have the right type in the selector
                return key, False
        elif (is_collection(def_val) or is_dictlike(def_val)) and not isinstance(def_val, np.ndarray):
            # We do nothing for these collections. Wait for their elements to be matched
            return key, value
        elif isinstance(def_val, np.ndarray):
            if isinstance(value, np.ndarray):
                return key, np.all(def_val == value)
            elif is_nonclass_callable(value) and not strict:
                return key, value(def_val)
            else:
                # type doesn't match.
                return key, False
        else:
            # Plain matching branch
            if is_nonclass_callable(value) and not strict:
                return key, value(def_val)
            elif type(value) != type(value):
                return key, False
            else:
                return key, value == def_val
            

    def selector_match_exit(path, key, old_parent, new_parent, new_items):
        # Type check
        if type(old_parent) != type(new_parent):
            if issubclass(type(old_parent), Definition):
                if type(new_parent) is not dict:
                    # The one case we know about should have new_parent be a dict.
                    return False
            else:
                return False

        if key is None:
            def_values = get_path(definition, path)
        else:
            def_values = get_path(definition, path)[key]

        if strict:
            if len(def_values) != len(new_items):
                return False
        else:
            if is_collection(old_parent) and not is_dictlike(old_parent):
                # For tuples and list arguments, the lengths must match.
                if len(def_values) != len(new_items):
                    return False
        final = True
        for val in map(lambda t: t[1], new_items):
            final = final & val
        return final

    # We reduce across the selector because we are only checking the values supplied
    # In the selector.
    return remap(selector, enter=definition_enter, visit=selector_match_visit, exit=selector_match_exit)


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

    def load_object(self, obj_def: ConcreteDefinition):
        if type(obj_def) is not ConcreteDefinition:
            raise TypeError("Only ConcreteDefinition is supported")
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
