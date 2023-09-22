import dill
import os
import zipfile
from collections.abc import Mapping
from inspect import currentframe, getmodule, isclass, \
    Parameter, signature


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

    def __call__(self):
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


def get_class_str(obj):
    if isinstance(obj, type):
        return '.'.join([getmodule(obj).__name__,
                         obj.__name__])
    else:
        return '.'.join([getmodule(obj).__name__,
                         obj.__class__.__name__])


def is_nonclass_callable(obj):
    return callable(obj) and not isclass(obj)


def get_kwarg_defaults(cls):
    kwarg_defaults = {}
    for current_class in reversed(cls.mro()):
        if hasattr(current_class, '__init__'):
            init_signature = signature(current_class.__init__)
            for name, param in init_signature.parameters.items():
                if param.default != Parameter.empty:
                    kwarg_defaults[name] = param.default
    return kwarg_defaults


def is_dictlike(val):
    return isinstance(val, Mapping)


def zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Ignoring dirs for now. May need to edit this in the future.
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)


def hashval_to_digest(val):
    # Suggestion from gpt-4
    return hex(val & ((1 << 64) - 1))[2:]


def digest_to_hashval(digest):
    return int(digest, 16)


def unpickler(stream):
    "Method to ensure all objects are unpickled in the same way"
    return dill.loads(stream)


def pickler(obj):
    "Method to ensure all objects are pickled in the same way"
    return dill.dumps(obj, protocol=5)


def pickle_to_file(obj, path):
    with open(path, 'wb') as f:
        f.write(pickler(obj))
