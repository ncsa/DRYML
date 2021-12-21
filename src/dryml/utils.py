import os
import collections
import inspect
import hashlib
import importlib
import pickle
from typing import Type


def is_nonstring_iterable(val):
    if isinstance(val, collections.abc.Iterable) and type(val) \
             not in [str, bytes]:
        return True
    else:
        return False


def is_dictlike(val):
    return isinstance(val, collections.abc.Mapping)


def init_arg_list_handler(arg_list):
    if arg_list is None:
        return []
    else:
        return arg_list


def init_arg_dict_handler(arg_dict):
    if arg_dict is None:
        return {}
    else:
        return arg_dict


def get_class_str(obj):
    if isinstance(obj, type):
        return '.'.join([inspect.getmodule(obj).__name__,
                         obj.__name__])
    else:
        return '.'.join([inspect.getmodule(obj).__name__,
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


def get_current_cls(cls: Type, reload: bool = False):
    return get_class_by_name(
        inspect.getmodule(cls).__name__,
        cls.__name__, reload=reload)


def get_hashed_id(hashstr: str):
    return hashlib.md5(hashstr.encode('utf-8')).hexdigest()


def path_needs_directory(path):
    """
    Method to determine whether a path is absolute, relative, or just
    a plain filename. If it's a plain filename, it needs a directory
    """
    head, tail = os.path.split(path)
    if head == '':
        return True
    else:
        return False


def pickler(obj):
    "Method to ensure all objects are pickled in the same way"
    # Consider updating to protocol=5 when python 3.7 is deprecated
    return pickle.dumps(obj, protocol=4)
