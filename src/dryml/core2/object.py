from functools import cached_property
import uuid
import time
import os
import inspect

from dryml.core2.util import collide_attributes, \
    pickle_to_file, unpickler, get_kwarg_defaults
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
        # Object initialization
        obj = cls.__create_instance__()
        # Perform class specific argument preparation
        args, kwargs = cls.__prepare_args__(*args, **kwargs)

        _validate_init_sig(cls, *args, **kwargs)

        # Perform object pre-initialization
        obj.__pre_init__(*args, **kwargs)

        # Run our initialization method
        obj.__initialize_instance__(*args, **kwargs)
        return obj


class Object(metaclass=Dryml):
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

    def __pre_init__(self, *args, **kwargs):
        pass

    def __initialize_instance__(self, *args, **kwargs):
        return self.__init__(*args, **kwargs)

    def __init__(self):
        pass


class Memorizer(Object):
    # Support class which remembers the arguments used when creating it.
    # TODO: Check Invariant: Memorizer Object shouldn't contain arguments to Definitions.
    # TODO: Check Invariant: Memorizer Object should only contain arguments which are other Memorizer objects or plain old data.
    def __pre_init__(self, *args, **kwargs):
        # Set up structures for tracking args and store them.
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
        # Get a `Definition` object for this particular object.
        return build_definition(self)

    @cached_property
    def concrete_definition(self):
        # Get a `ConcreteDefinition` object for this particular object.
        return self.definition.concretize()

    def __hash__(self):
        # Objects are hashable through through its `ConcreteDefinition`
        return hash(self.concrete_definition)

    def __repr__(self):
        return f"<{self.__class__.__name__} at {hex(id(self))}>(args={self.__args__}, kwargs={self.__kwargs__})"


class Lazy(Memorizer):
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


class Metadata(Object):
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
    def __strip_unique_args__(cls_super, *args, **kwargs):
        args, kwargs = super().__prepare_args__(*args, **kwargs)
        kwargs = kwargs.copy()
        if 'metadata' in kwargs:
            del kwargs['metadata']
        return args, kwargs

    def __init__(self, *args, metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata


class Serializable(Memorizer):
    # Enables uniform mechanism for saving/loading to disk
    def save(self, dest, **kwargs):
        from dryml.core2.repo import save_object
        return save_object(self, dest, **kwargs)

    def _save_to_dir(self, dir: str):
        # Directory into which the model should save its 'heavy' content
        # Full save procedure handled elsewhere
        # We expect the directory to exist. Caller should handle this
        if not os.path.exists(dir):
            raise ValueError(f"Path {dir} does not exist. Can't save")

        # Save the definition
        def_file = os.path.join(dir, 'def.pkl')
        pickle_to_file(self.definition, def_file)

        return self._save_to_dir_imp(dir)

    def _save_to_dir_imp(self, dir: str):
        output_file = os.path.join(dir, 'object.pkl')
        pickle_to_file(self, output_file)

        return True

    def _load_from_dir(self, dir: str):
        # Load 'heavy' content from directory
        # Again directory should exist. Caller will handle it.
        if not os.path.exists(dir):
            raise ValueError(f"Path {dir} does not exist. Can't load")

        def_file = os.path.join(dir, 'def.pkl')
        with open(def_file, 'rb') as f:
            definition = unpickler(f.read())

        if definition != self.definition:
            raise ValueError(f"Definition ({definition}) for data in directory {dir} doesn't match this object ({self.definition}). Can't load")

        self._load_from_dir_imp(dir)

    def _load_from_dir_imp(self, dir: str):
        input_file = os.path.join(dir, 'object.pkl')
        with open(input_file, 'rb') as f:
            obj = unpickler(f.read())
        self.__dict__.update(obj.__dict__)

    def __getstate__(self):
        state = self.__dict__.copy()
        # We shouldn't pickle the __args__ and __kwargs__. This is handled by another part of the saving process
        del state['__args__']
        del state['__kwargs__']
        return state
