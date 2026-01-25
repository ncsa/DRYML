from __future__ import annotations

import dill
import os
import zipfile
import importlib
import tempfile
import io
import glob
from typing import Optional, Callable
from collections.abc import Mapping, ItemsView
from inspect import getmodule, isclass, \
    Parameter, signature


def _validate_init_sig(cls, *args, **kwargs):
    """
    Raise TypeError *now* if `cls.__init__` cannot accept the args.

    We bind only to the *signature*; we never execute the body, so it is
    safe for objects.
    """
    sig = signature(cls.__init__)

    # The first parameter is `self`; use `None` as placeholder.
    try:
        sig.bind_partial(None, *args, **kwargs)
    except TypeError as err:
        raise TypeError(
            f"{cls.__name__} cannot be constructed with "
            f"args={args!r}, kwargs={kwargs!r}: {err}"
        ) from None


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


def get_object_view(obj):
    return ItemsView({'cls': obj.definition.cls, 'args': obj.definition.args, 'kwargs': obj.definition.kwargs})


def get_unique_concrete_definitions(obj_or_def) -> set["ConcreteDefinition"]:
    from ..object import Object
    from ..definition import Definition
    from ..definition import ConcreteDefinition


    if isinstance(obj_or_def, Object):
        return get_unique_concrete_definitions(obj_or_def.definition)
    if isinstance(obj_or_def, ConcreteDefinition):
        args_cdefs = list(get_unique_concrete_definitions(obj_or_def.args))
        kwargs_cdefs = list(get_unique_concrete_definitions(obj_or_def.kwargs))
        return set([obj_or_def] + args_cdefs + kwargs_cdefs)

    if isinstance(obj_or_def, Definition):
        raise ValueError("Unexpected Definition found in object graph!")

    if isinstance(obj_or_def, (list, tuple, set)):
        all_cdefs = []
        for el in obj_or_def:
            all_cdefs.extend(list(get_unique_concrete_definitions(el)))
        return set(all_cdefs)

    if isinstance(obj_or_def, Mapping):
        all_cdefs = []
        for el in obj_or_def.values():
            all_cdefs.extend(list(get_unique_concrete_definitions(el)))
        return set(all_cdefs)

    return set()


def get_unique_objects(obj, repo) -> list["Object"]:
    unique_cdefs = get_unique_concrete_definitions(obj)

    return [repo[cdef] for cdef in unique_cdefs]


def apply_func(
        obj, func, func_args=None, sel=Optional[Callable],
        func_kwargs=None, repo=None):
    if func_args is None:
        func_args = ()
    if func_kwargs is None:
        func_kwargs = {}

    obj_list = get_unique_objects(obj, repo)

    ic(obj_list, func)

    for obj in obj_list:
        ic(obj)
        if sel is None or sel(obj, verbose=True):
            ic("Selected", obj)
            func(obj, *func_args, **func_kwargs)


def get_temp_directory():
    return tempfile.TemporaryDirectory()

def dir_store_inspect(root_path: str):
    files = glob.glob(os.path.join(root_path, '**', 'def.pkl'), recursive=True)
    # Strip root directory
    return list(map(lambda f: f[len(root_path)+1:], files))
