import dill
import os
import zipfile
import importlib
import tempfile
import io
from typing import Optional, Callable
from collections.abc import Mapping, ItemsView
from inspect import currentframe, getmodule, isclass, \
    Parameter, signature
from boltons.iterutils import remap, default_enter, default_exit


def collide_attributes(obj, attr_list):
    # Check if these attributes are already defined. Throw an error if they are.
    colliding_attrs = []
    for attr in attr_list:
        if hasattr(obj, attr):
            colliding_attrs.append(attr)
    if len(colliding_attrs) > 0:
        raise AttributeError(f"Attributes {colliding_attrs} already exist on object. Cannot create object.")


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


def is_stream(obj) -> bool:
    return isinstance(obj, io.IOBase)


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


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        f.write(pickler(obj))


def pickle_load(path):
    with open(path, 'rb') as f:
        return unpickler(f.read())


def get_memorizer_view(obj):
    return ItemsView({'cls': type(obj), 'args': obj.__args__, 'kwargs': obj.__kwargs__})


def get_definition_view(defn):
    return ItemsView(defn)


def get_unique_objects(obj):
    from dryml.core2.object import Lazy

    unique_objs = {}

    def _get_unique_objects_enter(path, key, value):
        if isinstance(value, Lazy):
            # Check if we've visited this one already
            def_val = value.definition.concretize()

            if def_val in unique_objs:
                return value, False
            else:
                return {}, get_memorizer_view(value)
        else:
            return default_enter(path, key, value)

    def _get_unique_objects_visit(path, key, value):
        # We aren't processing anything
        return key, value

    def _get_unique_objects_exit(path, key, value, new_parent, new_items):
        if isinstance(value, Lazy):
            # We're exiting a Lazy object
            def_val = value.definition.concretize()

            unique_objs[def_val] = value

        return default_exit(path, key, value, new_parent, new_items)

    if isinstance(obj, Lazy):
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
