import inspect


def validate_class(cls):
    assert isinstance(cls, type), f"Expected a class, got {cls!r}"
    return cls


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



