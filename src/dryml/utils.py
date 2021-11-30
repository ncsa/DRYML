import collections

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
