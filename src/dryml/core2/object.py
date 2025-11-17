from functools import cached_property
from pathlib import Path
import uuid
import time
import os
import inspect
import pickle

from dryml.core2.util import collide_attributes, \
    pickle_save, pickle_load, get_kwarg_defaults, is_stream
from dryml.core2.definition import \
    deepcopy_skip_definition_object, build_definition


def _validate_init_sig(cls, *args, **kwargs):
    """
    Raise TypeError *now* if `cls.__init__` cannot accept the args.

    We bind only to the *signature*; we never execute the body, so it is
    safe for Lazy/Heavy objects.
    """
    sig = inspect.signature(cls.__init__)

    # The first parameter is `self`; use `None` as placeholder.
    try:
        sig.bind_partial(None, *args, **kwargs)
    except TypeError as err:
        raise TypeError(
            f"{cls.__name__} cannot be constructed with "
            f"args={args!r}, kwargs={kwargs!r}: {err}"
        ) from None


class Dryml(type):
    # Support metaclass to enable close control of the python
    # object creation process

    def __create_instance__(cls):
        # Base instance creation 
        return cls.__new__(cls)

    def __call__(cls, *args, **kwargs):
        # Lazy initialization
        obj = cls.__create_instance__()
        # Perform class specific argument preparation
        args, kwargs = cls.__prepare_args__(*args, **kwargs)

        _validate_init_sig(cls, *args, **kwargs)

        # Perform object pre-initialization
        obj.__pre_init__(*args, **kwargs)

        return obj


class Lazy(metaclass=Dryml):
    # Base type for using CreationControl metaclass.
    # Provides basic implementations for all methods used
    # In the CreationControl process

    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        # __prepare_args__ should be an idempotent function
        return args, kwargs

    @classmethod
    def __strip_unique_args__(cls, *args, **kwargs):
        # __strip_unique_args__ should be an idempotent function
        return args, kwargs

    # Since methods are part of the class, we only have to remove data from the object. We mark the protected data here. Keep up to date with attributes added 
    def __pre_init__(self, *args, **kwargs):
        # Set up structures for tracking args and store them.
        collide_attributes(self, [
            '__initialized__',
            '__locked__',
            '__args__',
            '__kwargs__',])
        default_kwargs = get_kwarg_defaults(type(self))
        # TODO investigate whether we should include a check to make sure the user isn't passing
        # any Definition objects. I think we should probably disallow that.
        self.__args__ = deepcopy_skip_definition_object(args)
        # We merge the default kwargs with the kwargs passed in.
        # Defaults are first so they can be overwritten.
        self.__kwargs__ = deepcopy_skip_definition_object({ **default_kwargs, **kwargs })
        self.__initialized__ = False
        self.__locked__ = False
        self.__orig_keys__ = None
        self.__orig_keys__ = list(self.__dict__.keys())

    def __init__(self):
        pass

    @cached_property
    def definition(self):
        # Get a `Definition` object for this particular object.
        return build_definition(self)

    @cached_property
    def concrete_definition(self):
        # Get a `ConcreteDefinition` object for this particular object.
        return self.definition.concretize()

    def __hash__(self):
        # Lazys are hashable through through its `ConcreteDefinition`
        return hash(self.concrete_definition)

    def __repr__(self):
        return f"<{self.__class__.__name__} at {hex(id(self))}>(args={self.__args__}, kwargs={self.__kwargs__})"

    def save(self, repo=None):
        from dryml.core2.repo import save_object
        save_object(self, repo=repo)

    def save_to_dir(self, dest_dir: str):
        pickle_save(self.concrete_definition, os.path.join(dest_dir, 'def.pkl'))
        self.save_imp(dest_dir)

    def save_imp(self, dest_dir: str):
        pass

    def load(self, repo=None):
        from dryml.core2.repo import load_object
        load_object(self, repo=repo)
            

    def load_from_dir(self, src_dir: str):
        loaded_def = pickle_load(os.path.join(src_dir, "def.pkl"))
        assert loaded_def == self.concrete_definition, f"Loaded definition {loaded_def} doesn't match expected definition {self.concrete_definition}"
        self.load_imp(src_dir)

    def load_imp(self, src_dir: str):
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
            raise RuntimeError("Cannot initialize object. Lazy is locked.")
        self.__init__(*self.__args__, **self.__kwargs__)
        self.__initialized__ = True

    def __unload__(self):
        if self.__locked__:
            raise RuntimeError("Cannot unload object. Lazy is locked.")
        # Remove all attributes besides self._orig_attrs
        for attr in list(self.__dict__.keys()):
            if attr not in self.__orig_keys__:
                delattr(self, attr)
        self.__initialized__ = False


class Pickleable(Lazy):
    def save_imp(self, dest_dir: str):
        # Grab all heavy-state data
        heavy_state = {}
        for key in self.__dict__:
            if key not in self.__orig_keys__:
                heavy_state[key] = getattr(self, key)

        # Save the entire object as a pickle
        pickle_save(heavy_state, os.path.join(dest_dir, "heavy.pkl"))

    def load_imp(self, src_dir: str):
        # heavy-state data is stored in heavy.pkl
        heavy_state = pickle_load(os.path.join(src_dir, "heavy.pkl"))

        self.__dict__.update(heavy_state)


class UniqueID(Lazy):
    # Mixing in this class adds a `uid` keyword argument which is
    # initialized automatically if not provided.
    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        args, kwargs = super().__prepare_args__(*args, **kwargs)
        kwargs.setdefault("uid", str(uuid.uuid4()))
        return args, kwargs

    @classmethod
    def __strip_unique_args__(cls, *args, **kwargs):
        args, kwargs = super().__strip_unique_args__(*args, **kwargs)
        kwargs = kwargs.copy()
        if 'uid' in kwargs:
            del kwargs['uid']
        return args, kwargs

    def __init__(self, *args, uid=None, **kwargs):
        super().__init__(*args, **kwargs)
        # unique ID
        self.uid = uid


class Metadata(Lazy):
    # Mixing in this class adds a `metadata` keyword argument which is
    # used to store a basic 'description', and 'creation_time' metadata
    # along with any other metadata the user wishes to store.
    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        args, kwargs = super().__prepare_args__(*args, **kwargs)
        if 'metadata' not in kwargs:
            kwargs['metadata'] = {
            }
        if 'description' not in kwargs['metadata']:
            kwargs['metadata']['description'] = ""
        if 'creation_time' not in kwargs['metadata']:
            kwargs['metadata']['creation_time'] = time.time()
        return args, kwargs

    @classmethod
    def __strip_unique_args__(cls, *args, **kwargs):
        args, kwargs = super().__strip_unique_args__(*args, **kwargs)
        kwargs = kwargs.copy()
        if 'metadata' in kwargs:
            del kwargs['metadata']
        return args, kwargs

    def __init__(self, *args, metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata
