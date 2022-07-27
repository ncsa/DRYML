import os
import shutil
import pathlib
import collections
import inspect
import hashlib
import importlib
import pickle
from typing import Type, Union, IO
import zipfile


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


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


def show_sig(sig):
    for key in sig.parameters:
        par = sig.parameters[key]
        print(f"{par.name} - {par.kind} - {par.default}")


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


def get_temp_checkpoint_dir(dry_id):
    home_dir = os.environ['HOME']
    # Create checkpoint dir
    temp_checkpoint_dir = os.path.join(home_dir, '.dryml', dry_id)
    # Create directory if needed
    pathlib.Path(temp_checkpoint_dir).mkdir(parents=True, exist_ok=True)
    return temp_checkpoint_dir


def cleanup_checkpoint_dir(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)


def head(get_result):
    from dryml import DryObject
    if issubclass(type(get_result), DryObject):
        return get_result
    return get_result[0]


def tail(get_result):
    from dryml import DryObject
    if issubclass(type(get_result), DryObject):
        return get_result
    return get_result[-1]


def count(get_result):
    from dryml import DryObject
    if issubclass(type(get_result), DryObject):
        return 1
    return len(get_result)


def show_contained_objects(save_file: Union[str, IO[bytes]]):
    if type(save_file) is str or IO[bytes]:
        zf = zipfile.ZipFile(save_file, mode='r')
    else:
        zf = save_file

    print(zf.namelist())


file_blocklist = [
   'meta_data.pkl',
   'cls_def.dill',
   'dry_args.pkl',
   'dry_kwargs.pkl',
   'dry_mut.pkl',
]


def create_zip_branch(input_file, tag):
    with zipfile.ZipFile(input_file, mode='r') as zf:
        content_names = zf.namelist()
        content_nodes = {}
        for name in content_names:
            if name[-4:] == '.zip':
                with zf.file.open(name, mode='r') as f:
                    content_nodes = {
                        **content_nodes,
                        **create_zip_branch(f, name)}
            else:
                content_nodes[name] = {}
    return {tag: content_nodes}


def create_object_tree_from_dryfile(input_file, tag, report_class=True):
    import dryml
    with dryml.DryObjectFile(input_file) as dry_f:
        obj_def = dry_f.definition()
        content_names = dry_f.z_file.namelist()
        content_nodes = {}
        for name in content_names:
            if name[-4:] == '.dry':
                # this is another dry file.
                with dry_f.z_file.open(name, mode='r') as f:
                    content_nodes = {
                        **content_nodes,
                        **create_object_tree_from_dryfile(
                            f,
                            name,
                            report_class=report_class)}

            elif name not in file_blocklist:
                if name[-4:] == '.zip':
                    with dry_f.z_file.open(name, mode='r') as f:
                        content_nodes = {
                            **content_nodes,
                            **create_zip_branch(f, name)}
                else:
                    content_nodes[name] = {}

        if tag is not None:
            label = f"{tag}: {obj_def.dry_id}"
        else:
            label = f"{obj_def.dry_id}"
        if report_class:
            label += f" ({dryml.utils.get_class_str(obj_def.cls)})"

        if len(content_nodes) > 0:
            return {label: content_nodes}
        else:
            return {label: {}}


def show_object_tree_from_dryfile(input_file, report_class=True):
    dry_tree = create_object_tree_from_dryfile(
        input_file,
        None,
        report_class=report_class)
    from asciitree import LeftAligned
    from asciitree.drawing import BoxStyle, BOX_LIGHT
    tr = LeftAligned(
        draw=BoxStyle(
            gfx=BOX_LIGHT,
            horiz_len=1
        )
    )
    print(tr(dry_tree))
