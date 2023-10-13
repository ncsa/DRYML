import dill
import os
import zipfile
import importlib
import tempfile
from typing import Optional, Callable
from collections.abc import Mapping, ItemsView
from inspect import currentframe, getmodule, isclass, \
    Parameter, signature
from boltons.iterutils import remap, is_collection, PathAccessError, default_enter, default_exit
from icecream import ic


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


def get_class_by_name(module: str, cls: str, reload: bool = False):
    module = importlib.import_module(module)
    # If indicated, reload the module.
    if reload:
        module = importlib.reload(module)
    return getattr(module, cls)


def get_class_from_str(cls_str: str, reload: bool = False):
    cls_split = cls_str.split('.')
    module_string = '.'.join(cls_split[:-1])
    cls_name = cls_split[-1]
    return get_class_by_name(module_string, cls_name, reload=reload)


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


def zip_directory(folder_path, zip_dest):
    with zipfile.ZipFile(zip_dest, 'w', zipfile.ZIP_DEFLATED) as zipf:
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


def get_remember_view(obj):
    return ItemsView({'cls': type(obj), 'args': obj.__args__, 'kwargs': obj.__kwargs__})


def get_unique_objects(obj):
    from dryml.core2.object import Remember

    unique_objs = {}

    def _get_unique_objects_enter(path, key, value):
        if isinstance(value, Remember):
            # Check if we've visited this one already
            def_val = value.definition.concretize()

            if def_val in unique_objs:
                return value, False
            else:
                return {}, get_remember_view(value)
        else:
            return default_enter(path, key, value)

    def _get_unique_objects_visit(path, key, value):
        # We aren't processing anything
        return key, value

    def _get_unique_objects_exit(path, key, value, new_parent, new_items):
        if isinstance(value, Remember):
            # We're exiting a Remember object
            def_val = value.definition.concretize()

            unique_objs[def_val] = value

        return default_exit(path, key, value, new_parent, new_items)

    if isinstance(obj, Remember):
        remap(
            [obj],
            enter=_get_unique_objects_enter,
            visit=_get_unique_objects_visit,
            exit=_get_unique_objects_exit)[0]
        return list(unique_objs.values())
    else:
        remap(
            obj,
            enter=_get_unique_objects_enter,
            visit=_get_unique_objects_visit,
            exit=_get_unique_objects_exit)
        return list(unique_objs.values())


def apply_func(
        obj, func, func_args=None, sel=Optional[Callable],
        func_kwargs=None):
    if func_args is None:
        func_args = ()
    if func_kwargs is None:
        func_kwargs = {}

    obj_list = get_unique_objects(obj)

    for obj in obj_list:
        if sel is None or sel(obj):
            func(obj, *func_args, **func_kwargs)


def get_temp_directory():
    return tempfile.TemporaryDirectory()


def get_relevant_context(frame):
    context = []
    nested_funcs = []
    while frame:
        code = frame.f_code
        func_name = code.co_name
        
        if 'self' in frame.f_locals:
            class_name = frame.f_locals['self'].__class__.__name__
            func_name = f"{class_name}.{func_name}"
            context.insert(0, func_name)
            if nested_funcs:
                context[-1] = f"{context[-1]}{'.' + '.'.join(nested_funcs)}"
            break  # Stop once you get the enclosing method and class
        else:
            nested_funcs.insert(0, func_name)
        
        frame = frame.f_back

    return '.'.join(context)
