import uuid
import time
import hashlib
from inspect import signature, Parameter, isclass
from dryml.utils import is_dictlike
from boltons.iterutils import remap, is_collection


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
        self.__args__ = args
        # We merge the default kwargs with the kwargs passed in.
        # Defaults are first so they can be overwritten.
        self.__kwargs__ = { **default_kwargs, **kwargs }


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
        init = False
        if len(args) > 0:
            if isclass(args[0]):
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

    @property
    def cls(self):
        return self['cls']

    @property
    def args(self):
        return self['args']

    @property
    def kwargs(self):
        return self['kwargs']


# Definition hash support
class HashHelper(object):
    def __init__(self, the_hash):
        self.hash = the_hash


def hash_value(value):
    # Hashes for supported values here
    if isclass(value):
        return hashlib.sha256(str(value).encode()).hexdigest()
    elif hasattr(value, '__hash__'):
        hash_val = hash(value)
        # Suggestion from gpt-4
        return hex(hash_val & ((1 << 64) - 1))[2:]
    else:
        raise TypeError(f"Value of type {type(value)} not supported for hashing.")


def hash_visit(path, key, value):
    # Skip if it's a hashlib hasher
    if isinstance(value, HashHelper):
        return key, value.hash
    elif is_dictlike(value) or is_collection(value):
        return key, value
    
    return key, hash_value(value)


def hash_exit(path, key, old_parent, new_parent, new_items):
    # At this point, all items should be hashes

    # sort the items. format is [(key, value)]
    new_items = sorted(new_items, key=lambda x: x[0])

    # Combine the hashes
    hasher = hashlib.sha256()
    # Add a string representation of the old parent type
    hasher.update(str(type(old_parent)).encode())
    for k, v in new_items:
        hasher.update(v.encode())
    new_hash = hasher.hexdigest()

    return HashHelper(new_hash)


def hash_function(structure):
    result = remap(structure, visit=hash_visit, exit=hash_exit)
    return result.hash


# Creating definitions from objects
def build_definition(obj):
    # Copy the object's args and kwargs

    # If the object has Remember as a subclass
    if issubclass(type(obj), Remember):
        args = build_definition(obj.__args__)
        kwargs = build_definition(obj.__kwargs__)
        return Definition(
            type(obj),
            *args,
            **kwargs)

    if is_dictlike(obj) or is_collection(obj):
        return remap(obj, visit=build_definition_visit)

    # Do nothing
    return obj


def build_definition_visit(_, key, value):
    return key, build_definition(value)


# Creating objects from definitions
def is_definition(obj):
    if type(obj) is Definition:
        return True
    else:
        return False


def build_from_definition(obj):
    # First, detect a definition
    if is_definition(obj):
        args = build_from_definition(obj['args'])
        kwargs = build_from_definition(obj['kwargs'])
        return obj['cls'](*args, **kwargs)

    if is_dictlike(obj) or is_collection(obj):
        return remap(obj, visit=build_from_definition_visit)

    # Do nothing with value
    return obj


def build_from_definition_visit(_, key, value):
    return key, build_from_definition(value)
