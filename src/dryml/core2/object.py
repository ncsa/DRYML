from functools import cached_property
import uuid
import time
import os

from dryml.core2.util import cls_super, collide_attributes, \
    pickle_to_file, unpickler, get_kwarg_defaults
from dryml.core2.definition import \
    deepcopy_skip_definition_object, build_definition


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

    @staticmethod
    def __strip_unique_args__(cls_super, *args, **kwargs):
        # __strip_unique_args__ should be an idempotent function
        return args, kwargs

    def __pre_init__(self, *args, **kwargs):
        pass

    def __initialize_instance__(self, *args, **kwargs):
        return self.__init__(*args, **kwargs)

    def __init__(self):
        pass


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

    @staticmethod
    def __strip_unique__(cls_super, *args, **kwargs):
        kwargs = kwargs.copy()
        if 'uid' in kwargs:
            del kwargs['uid']
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

    @staticmethod
    def __strip_unique__(cls_super, *args, **kwargs):
        kwargs = kwargs.copy()
        if 'metadata' in kwargs:
            del kwargs['metadata']
        return args, kwargs

    def __init__(self, *args, metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata


class Serializable(Remember):
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
