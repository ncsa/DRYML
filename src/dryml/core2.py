import uuid
import time
import hashlib
from inspect import signature, Parameter, isclass
from dryml.utils import is_dictlike
from boltons.iterutils import remap, is_collection, get_path, PathAccessError
from functools import cached_property
import numpy as np


def collide_attributes(obj, attr_list):
    # Check if these attributes are already defined. Throw an error if they are.
    colliding_attrs = []
    for attr in attr_list:
        if hasattr(obj, attr):
            colliding_attrs.append(attr)
    if len(colliding_attrs) > 0:
        raise AttributeError(f"Attributes {colliding_attrs} already exist on object. Cannot create object.")


class CreationControl(type):
    # Support metaclass to allow more complex metaclass behavior
    def __create_instance__(cls):
        return cls.__new__(cls)

    def __call__(cls, *args, **kwargs):
        obj = cls.__create_instance__()
        args, kwargs = obj.__arg_manipulation__(*args, **kwargs)
        obj.__pre_init__(*args, **kwargs)
        obj.__initialize_instance__(*args, **kwargs)
        return obj


class Object(metaclass=CreationControl):
    def __arg_manipulation__(self, *args, **kwargs):
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
    def __arg_manipulation__(self, *args, **kwargs):
        args, kwargs = super().__arg_manipulation__(*args, **kwargs)
        if 'uid' not in kwargs:
            kwargs['uid'] = str(uuid.uuid4())
        return args, kwargs

    def __init__(self, *args, uid=None, **kwargs):
        super().__init__(*args, **kwargs)
        # unique ID
        self.uid = uid


class Metadata(Object):
    def __arg_manipulation__(self, *args, **kwargs):
        args, kwargs = super().__arg_manipulation__(*args, **kwargs)
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
def is_definition(obj):
    if type(obj) is Definition:
        return True
    else:
        return False


def build_from_definition(definition):
    if isinstance(definition, Definition):
        return remap([definition], visit=build_from_definition_visit)[0]
    else:
        return remap(definition, visit=build_from_definition_visit)


def build_from_definition_visit(_, key, value):
    if isinstance(value, Definition):
        args = build_from_definition(value.args)
        kwargs = build_from_definition(value.kwargs)
        obj = value.cls(*args, **kwargs)
        return key, obj
    else:
        return key, value

        raise TypeError(f"Definition must be of type Definition. Got {type(definition)} instead.")


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


