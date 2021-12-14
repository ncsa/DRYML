import collections
import inspect

def is_nonstring_iterable(val):
    if isinstance(val, collections.abc.Iterable) and type(val) not in [str, bytes]:
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
        return '.'.join([inspect.getmodule(obj).__name__, obj.__name__])
    else:
        return '.'.join([inspect.getmodule(obj).__name__, obj.__class__.__name__])
