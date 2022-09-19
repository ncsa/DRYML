import os
import shutil
import pathlib
import collections
import inspect
import hashlib
import importlib
import pickle
from typing import Type, Union, IO, Optional, Callable
import zipfile
import io
import numpy as np


def is_in_typelist(val, typelist):
    for t in typelist:
        if isinstance(val, t):
            return True
    return False


supported_scalar_types = (
    str, bytes, int, float, bool, np.float, np.float32,
    np.float64, np.int, np.int64, np.int32, np.int16, np.int8,
    np.bool, np.short, np.ushort, np.uint, np.uint64, np.uint32,
    np.uint16, np.uint8, np.byte, np.ubyte, np.single, np.double,
    np.longdouble, type)


def is_supported_scalar_type(val):
    if val is None:
        return True
    return is_in_typelist(val, supported_scalar_types)


supported_listlike_types = (
    tuple, list)


def is_supported_listlike(val):
    return is_in_typelist(val, supported_listlike_types)


def map_listlike(func, val):
    the_type = type(val)
    return the_type(map(func, val))


supported_dictlike_types = (
    dict,)


def is_supported_dictlike(val):
    return is_in_typelist(val, supported_dictlike_types)


def map_dictlike(func, val):
    the_type = type(val)
    return the_type({
        k: func(val[k]) for k in val})


def equal_listlike(equal_func, list_a, list_b):
    if len(list_a) != len(list_b):
        return False

    for i in range(len(list_a)):
        el_a = list_a[i]
        el_b = list_b[i]
        if not equal_func(el_a, el_b):
            return False

    return True


def equal_dictlike(equal_func, dict_a, dict_b):
    keys_a = set(dict_a.keys())
    keys_b = set(dict_b.keys())
    if keys_a != keys_b:
        return False

    for key in keys_a:
        val_a = dict_a[key]
        val_b = dict_b[key]
        if not equal_func(val_a, val_b):
            return False

    return True


def is_nonstring_iterable(val):
    if isinstance(val, collections.abc.Iterable) and type(val) \
             not in [str, bytes]:
        return True
    else:
        return False


def is_dictlike(val):
    return isinstance(val, collections.abc.Mapping)


def is_iterator(val):
    # Implementation from https://stackoverflow.com/a/36230057
    if (
            hasattr(val, '__iter__') and
            hasattr(val, '__next__') and
            callable(val.__iter__) and
            val.__iter__() is val):
        return True
    else:
        return False


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
    from dryml import Object
    if issubclass(type(get_result), Object):
        return get_result
    return get_result[0]


def tail(get_result):
    from dryml import Object
    if issubclass(type(get_result), Object):
        return get_result
    return get_result[-1]


def count(get_result):
    from dryml import Object
    from dryml.dry_repo import DryRepoContainer
    if isinstance(get_result, Object):
        return 1
    elif isinstance(get_result, DryRepoContainer):
        return 1
    return len(get_result)


def show_contained_objects(save_file: Union[str, IO[bytes]]):
    if type(save_file) is str or IO[bytes]:
        zf = zipfile.ZipFile(save_file, mode='r')
    else:
        zf = save_file

    print(zf.namelist())


def show_contained_objects_md5(save_file: Union[str, IO[bytes]]):
    close_file = False
    if type(save_file) is zipfile.ZipFile:
        zf = save_file
    else:
        zf = zipfile.ZipFile(save_file, mode='r')
        close_file = True

    for name in zf.namelist():
        with zf.open(name, 'r') as f:
            m = hashlib.md5()
            m.update(f.read())
            print(f"{name}: {m.hexdigest()}")

    if close_file:
        zf.close()


file_blocklist = [
   'meta_data.pkl',
   'cls_def.dill',
   'dry_args.pkl',
   'dry_kwargs.pkl',
   'dry_mut.pkl',
]


def create_zip_branch(input_file, tag):
    content_nodes = {}
    try:
        with zipfile.ZipFile(input_file, mode='r') as zf:
            content_names = zf.namelist()
            for name in content_names:
                if name[-4:] == '.zip':
                    with zf.open(name, mode='r') as f:
                        content_nodes = {
                            **content_nodes,
                            **create_zip_branch(f, name)}
                else:
                    content_nodes[name] = {}
    except zipfile.BadZipFile:
        tag = f"{tag} - Bad Zipfile"
    return {tag: content_nodes}


def create_object_tree_from_dryfile(input_file, tag, report_class=True):
    obj_def = None
    content_nodes = {}
    try:
        import dryml
        with dryml.ObjectFile(input_file) as dry_f:
            def_problem = False
            try:
                obj_def = dry_f.definition()
            except KeyError as e:
                def_problem = True
                tag = f"{tag} - Issue reading object definition ({e})"

            if not def_problem:
                content_names = dry_f.z_file.namelist()
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
    except zipfile.BadZipFile:
        tag = f"{tag} - Bad Zip File"

    if tag is not None:
        if obj_def is not None:
            label = f"{tag}: {obj_def.dry_id}"
        else:
            label = f"{tag}"
    else:
        if obj_def is not None:
            label = f"{obj_def.dry_id}"
        else:
            label = "None"

    if report_class and obj_def is not None:
        label += f" ({dryml.utils.get_class_str(obj_def.cls)})"

    if len(content_nodes) > 0:
        return {label: content_nodes}
    else:
        return {label: {}}


def create_file_tree_from_zipfile(input_file, tag):
    content_nodes = {}
    try:
        with zipfile.ZipFile(input_file) as zf:
            content_names = zf.namelist()
            for name in content_names:
                if name[-4:] == '.dry' or name[-4:] == '.zip':
                    # this is another zip file.
                    with zf.open(name, mode='r') as f:
                        content_nodes = {
                            **content_nodes,
                            **create_zip_branch(
                                f,
                                name)}

                else:
                    content_nodes[name] = {}
    except zipfile.BadZipFile:
        tag = f"{tag} - Bad Zip File"

    label = f"{tag}"

    if len(content_nodes) > 0:
        return {label: content_nodes}
    else:
        return {label: {}}


def show_object_tree_from_dryfile(input_file, report_class=True):
    cur_pos = None
    if isinstance(input_file, io.IOBase):
        try:
            # Get current file's position, and rewind to beginning
            cur_pos = input_file.tell()
            input_file.seek(0)
        except io.UnsupportedOperation:
            print(
                "Can't show file content of file of type "
                f"{type(input_file)}. seeking not supported.")
            return
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
    if isinstance(input_file, io.IOBase):
        # Go back to where we were in the file.
        input_file.seek(cur_pos)


def show_files_from_zipfile(input_file):
    cur_pos = None
    if isinstance(input_file, io.IOBase):
        try:
            # Save current file position and go to beginning of file
            cur_pos = input_file.tell()
            input_file.seek(0)
        except io.UnsupportedOperation:
            print(
                f"Can't show content of file {input_file}. "
                "seeking not supported.")
            return
    zip_tree = create_file_tree_from_zipfile(
        input_file,
        None)
    from asciitree import LeftAligned
    from asciitree.drawing import BoxStyle, BOX_LIGHT
    tr = LeftAligned(
        draw=BoxStyle(
            gfx=BOX_LIGHT,
            horiz_len=1
        )
    )
    print(tr(zip_tree))
    if isinstance(input_file, io.IOBase):
        # Return file to original position
        input_file.seek(cur_pos)


def apply_func(
        obj, func, func_args=None, sel=Optional[Callable],
        func_kwargs=None):
    if func_args is None:
        func_args = ()
    if func_kwargs is None:
        func_kwargs = {}

    for sub_obj in obj.__dry_obj_container_list__:
        apply_func(
            sub_obj, func, func_args=func_args, sel=sel,
            func_kwargs=func_kwargs)

    if sel is None or sel(obj):
        func(obj, *func_args, **func_kwargs)
