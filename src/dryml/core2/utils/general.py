from __future__ import annotations

import dill
import os
import zipfile
import importlib
import tempfile
import sys
from typing import Optional, Callable
from collections.abc import Mapping, Iterable, ItemsView
from inspect import getmodule, \
    Parameter, signature
import importlib.util


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


def validate_class(cls):
    assert isinstance(cls, type), f"Expected a class, got {cls!r}"
    return cls


def collide_attributes(obj, attr_list):
    # Check if these attributes are already defined. Throw an error if they are.
    colliding_attrs = []
    for attr in attr_list:
        if hasattr(obj, attr):
            colliding_attrs.append(attr)
    if len(colliding_attrs) > 0:
        raise AttributeError(f"Attributes {colliding_attrs} already exist on object. Cannot create object.")


def get_class_str(obj):
    from ..symbol import ImportRef, SourceSpec

    if isinstance(obj, ImportRef):
        return obj.import_path().replace(":", ".")
    if isinstance(obj, SourceSpec):
        return obj.name or "<source>"
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


def get_kwarg_defaults(cls):
    kwarg_defaults = {}
    for current_class in reversed(cls.mro()):
        if hasattr(current_class, '__init__'):
            init_signature = signature(current_class.__init__)
            for name, param in init_signature.parameters.items():
                if param.default != Parameter.empty:
                    kwarg_defaults[name] = param.default
    return kwarg_defaults


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

    for obj in obj_list:
        if sel is None or sel(obj, verbose=True):
            func(obj, *func_args, **func_kwargs)


def get_temp_directory():
    return tempfile.TemporaryDirectory()


def revision_path(file_stem: str, file_ext: str, dir: str, revision: str|None=None):
    filename = [file_stem]
    if revision is not None:
        filename.append(revision)
    filename.append(file_ext)
    return os.path.join(dir, '.'.join(filename))


def get_revision(filepath: str, file_stem: str, file_ext: str):
    filename = os.path.basename(filepath)
    return filename[len(file_stem)+1:-(len(file_ext)+1)]


def is_iterator(obj:Any) -> bool:
    return isinstance(obj, Iterator)


def adjust_class_module(cls):
    # Set module properly
    # From https://stackoverflow.com/questions/1095543/
    #              get-name-of-calling-functions-module-in-python
    # We go up two functions, one to get to the calling function,
    # Another to get to that function's caller. That should be
    # in a module.
    frm = inspect.stack()[2]
    calling_mod = inspect.getmodule(frm[0])
    cls.__module__ = calling_mod.__name__


def _normalize_module_names(names: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(names, str):
        names = (names,)
    else:
        names = tuple(names)

    if not names:
        raise ValueError("At least one module name must be provided.")

    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                "Module names must be str or an iterable of str, "
                f"got element of type {type(name)!r}."
            )
        if not name:
            raise ValueError("Module names must be non-empty strings.")

    return names


def _resolve_module_name(name: str, package: str | None = None) -> str:
    """
    Resolve a possibly-relative module name to its absolute form.

    Examples
    --------
    _resolve_module_name("jax") -> "jax"
    _resolve_module_name(".jax.context", package="dryml.context")
        -> "dryml.context.jax.context"
    """
    if name.startswith("."):
        if package is None:
            raise ValueError(
                f"Relative module name {name!r} requires 'package' to be set."
            )
        return importlib.util.resolve_name(name, package)

    return name


def _resolve_module_names(
    names: str | Iterable[str],
    package: str | None = None,
) -> tuple[str, ...]:
    names = _normalize_module_names(names)
    return tuple(_resolve_module_name(name, package=package) for name in names)


def _module_key_matches(name: str, *, include_children: bool) -> bool:
    if name in sys.modules:
        return True

    if include_children:
        prefix = name + "."
        return any(mod_name.startswith(prefix) for mod_name in sys.modules)

    return False


def module_is_imported(
    names: str | Iterable[str],
    *,
    match_all: bool = True,
    package: str | None = None,
    include_children: bool = False,
) -> bool:
    """
    Check whether one or more modules are already imported in this process.

    Parameters
    ----------
    names
        A module name or iterable of module names. Relative names like
        '.jax.context' are allowed if `package` is provided.
    match_all
        If True, return True only if all names are already imported.
        If False, return True if any name is already imported.
    package
        Package context used to resolve relative module names.
    include_children
        If True, treat a package as imported if any submodule is imported.

        Example:
            module_is_imported("jax", include_children=True)

        will return True if either "jax" or something like "jax.numpy"
        is present in sys.modules.
    """
    names = _resolve_module_names(names, package=package)
    results = (_module_key_matches(name, include_children=include_children)
               for name in names)
    return all(results) if match_all else any(results)


def module_is_available(
    names: str | Iterable[str],
    *,
    match_all: bool = True,
    package: str | None = None,
) -> bool:
    """
    Check whether one or more modules are discoverable without importing them.

    Parameters
    ----------
    names
        A module name or iterable of module names. Relative names like
        '.jax.context' are allowed if `package` is provided.
    match_all
        If True, return True only if all names are discoverable.
        If False, return True if any name is discoverable.
    package
        Package context used to resolve relative module names.

    Notes
    -----
    This does not guarantee that importing the module will succeed. It only
    checks whether an import spec can be found.
    """
    names = _resolve_module_names(names, package=package)
    results = (importlib.util.find_spec(name) is not None for name in names)
    return all(results) if match_all else any(results)
