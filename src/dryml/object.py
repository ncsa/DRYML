# File to define saving/loading convenience functions

from __future__ import annotations

import os
import dill
import pickle
import io
import zipfile
import uuid
import re
import numpy as np
import time

from typing import IO, Union, Optional, Type, Callable
from dryml.config import ObjectDef, Meta, MissingIdError, MissingMetadataError
from dryml.utils import get_current_cls, pickler, static_var, \
    is_supported_scalar_type, is_supported_listlike, is_supported_dictlike, \
    map_dictlike, map_listlike, get_class_from_str, get_class_str, \
    diff_recursive
from dryml.context.context_tracker import combine_requests, context, \
    NoContextError
from dryml.file_intermediary import FileIntermediary
from dryml.save_cache import SaveCache


FileType = Union[str, IO[bytes]]


def file_resolve(file: str, exact_path: bool = False) -> str:
    if os.path.splitext(file)[1] == '' and not exact_path:
        file = f"{file}.dry"
    return file


class ObjectFile(object):
    contained_dry_file_re = re.compile(r"^dry_objects/([a-f0-9-]*)\.dry$")

    # Supports 'save cached' file writing.
    def __init__(self, file: FileType, exact_path: bool = False,
                 mode: str = 'r', must_exist: bool = True,
                 save_cache=None, save_caching=True):

        if type(file) is zipfile.ZipFile:
            raise TypeError(
                "Passing zipfiles directly is currently not supported.")

        self.mode = mode

        # If file is a string, resolve it to a filepath, and save this filepath
        if type(file) is str:
            filepath = file
            filepath = file_resolve(filepath, exact_path=exact_path)
            if must_exist and not os.path.exists(filepath):
                raise ValueError(f"File {filepath} doesn't exist!")
            self.filepath = filepath

        if self.mode == 'w':
            self._z_file = None
            if hasattr(self, 'filepath'):
                self.binary_file = open(self.filepath, 'wb')
            else:
                self.binary_file = file
        elif self.mode == 'r':
            if hasattr(self, 'filepath'):
                self.binary_file = open(self.filepath, 'rb')
                self._z_file = zipfile.ZipFile(self.binary_file, mode=mode)
            else:
                if type(file) is zipfile.ZipFile:
                    self._z_file = file
                    if self._z_file.mode != mode:
                        raise ValueError(
                            "Utilized Zipfile doesn't match requested mode!")
                else:
                    self._z_file = zipfile.ZipFile(file, mode=mode)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    @property
    def z_file(self):
        # We need this property to create the zip file on demand
        # since sometimes We might write directly to the binary file.
        if self.mode == 'w':
            # Check if we've already opened a zip file.
            if self._z_file is not None:
                return self._z_file

            self.int_file = FileIntermediary()
            self.close_int_file = True

            self._z_file = zipfile.ZipFile(self.int_file, mode=self.mode)
            return self._z_file
        elif self.mode == 'r':
            return self._z_file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def int_file_detach(self):
        if not hasattr(self, 'int_file'):
            raise RuntimeError(
                "ObjectFile doesnt' have an intermediate file to detach.")
        self.close_int_file = False
        return self.int_file

    def close(self):
        # Close the zipfile.
        if self._z_file is not None:
            # Avoid accessing z_file property if it doesn't already exist.
            # Accessing the property triggers the int_file to be created,
            # And an empty file to be written out.
            self.z_file.close()

        # If we have a binary open, sync any intermediary and close it.
        if hasattr(self, 'binary_file'):
            if self.mode == 'w':
                if hasattr(self, 'int_file'):
                    # If there's an intermediary file open, write it to the
                    # binary file.
                    self.int_file.write_to_file(self.binary_file)
            if hasattr(self, 'filepath'):
                # We opened the binary file. we should close it.
                self.binary_file.close()

        # Close the intermediary if needed.
        if hasattr(self, 'int_file'):
            if self.close_int_file:
                self.int_file.close()

    # def update_file(self, obj: Object):
    #     self.cache_object_data_obj(obj)

    def save_meta_data(self):
        # Meta_data
        meta_data = {
            'version': 1
        }

        meta_dump = pickler(meta_data)
        with self.z_file.open('meta_data.pkl', mode='w') as f:
            f.write(meta_dump)

    def load_meta_data(self):
        try:
            with self.z_file.open('meta_data.pkl', 'r') as meta_file:
                meta_data = pickle.loads(meta_file.read())
        except KeyError as e:
            print(f"Issue getting MetaData! Zipfile contains the "
                  f"following files: {self.z_file.namelist()}")
            print(f"Zipfile: {self.z_file}")
            raise e
        return meta_data

    def save_class_def_v1(self, obj_def: ObjectDef, update: bool = False):
        # We need to pickle the class definition.
        # By default, error out if class has changed. Check this.
        mod_cls = get_current_cls(obj_def.cls)
        if obj_def.cls != mod_cls and not update:
            raise ValueError("Can't save class definition! It's been changed!")
        try:
            cls_def = dill.dumps(mod_cls)
            with self.z_file.open('cls_def.dill', mode='w') as f:
                f.write(cls_def)
        except TypeError as e:
            if '_abc_data' in str(e):
                # Fallback to pickling the class name.
                # In some cases, the class's _abc_impl
                # attribute seems wrongly constructed?
                # https://github.com/uqfoundation/dill/issues/332
                cls_str = get_class_str(mod_cls)
                with self.z_file.open('cls_str.txt', mode='w') as f:
                    f.write(cls_str.encode('utf-8'))
            else:
                raise e

    def load_class_def_v1(self, update: bool = True, reload: bool = False):
        "Helper function for loading a version 1 class definition"
        namelist = self.z_file.namelist()
        # Get class definition
        if 'cls_def.dill' in namelist:
            with self.z_file.open('cls_def.dill') as cls_def_file:
                if update:
                    # Get original model definition
                    cls_def_init = dill.loads(cls_def_file.read())
                    try:
                        cls_def = get_current_cls(cls_def_init, reload=reload)
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to update module class {e}")
                else:
                    cls_def = dill.loads(cls_def_file.read())
            return cls_def
        elif 'cls_str.txt' in namelist:
            with self.z_file.open('cls_str.txt') as cls_str_file:
                cls_str = cls_str_file.read().decode('utf-8')
                cls_def = get_class_from_str(cls_str, reload=reload)
            return cls_def
        else:
            raise RuntimeError("No stored class data!")

    def save_definition_v1(self, obj_def: ObjectDef, update: bool = False):
        "Save object def"
        # Save obj def
        self.save_class_def_v1(obj_def, update=update)

        # Save args from object def
        with self.z_file.open('dry_args.pkl', mode='w') as args_file:
            args_file.write(pickler(obj_def.args))

        # Save kwargs from object def
        with self.z_file.open('dry_kwargs.pkl', mode='w') as kwargs_file:
            kwargs_file.write(pickler(obj_def.kwargs))

        # Save mutability from object def
        with self.z_file.open('dry_mut.pkl', mode='w') as mut_file:
            mut_file.write(pickler(obj_def.dry_mut))

    def load_definition_v1(self, update: bool = True, reload: bool = False):
        "Load object def"

        # Load obj def
        cls = self.load_class_def_v1(update=update, reload=reload)

        # Load args
        with self.z_file.open('dry_args.pkl', mode='r') as args_file:
            args = pickle.loads(args_file.read())

        # Load kwargs
        with self.z_file.open('dry_kwargs.pkl', mode='r') as kwargs_file:
            kwargs = pickle.loads(kwargs_file.read())

        # Load mutability
        with self.z_file.open('dry_mut.pkl', mode='r') as mut_file:
            mut = pickle.loads(mut_file.read())

        return ObjectDef(cls, *args, dry_mut=mut, **kwargs)

    def definition(self, update: bool = True, reload: bool = False):
        meta_data = self.load_meta_data()
        if meta_data['version'] == 1:
            return self.load_definition_v1(update=update, reload=reload)
        else:
            raise RuntimeError(
                f"File version {meta_data['version']} not supported!")

    def load_object_v1(self, update: bool = True,
                       reload: bool = False,
                       as_cls: Optional[Type] = None) -> Object:
        # Load object
        obj_def = self.load_definition_v1(update=update, reload=reload)
        if as_cls is not None:
            obj_def.cls = as_cls

        # Create object
        obj = obj_def.build(load_zip=self.z_file)

        # Load object content
        if not obj.load_object(self.z_file):
            raise RuntimeError("Error loading object!")

        # Build object instance
        return obj

    def load_object_content(self, obj: Object) -> bool:
        file_def = self.definition()
        obj_def = obj.definition()
        if file_def != obj_def or file_def.dry_id != obj_def.dry_id:
            raise ValueError(
                f"File {self.z_file} doesn't store data for object "
                f"{obj.dry_id} at the top level.")

        if not obj.load_object(self.z_file):
            return False
        return True

    def load_object(self, update: bool = False,
                    reload: bool = False,
                    as_cls: Optional[Type] = None) -> Object:
        meta_data = self.load_meta_data()
        version = meta_data['version']
        if version == 1:
            return self.load_object_v1(
                update=update, reload=reload, as_cls=as_cls)
        else:
            raise RuntimeError(f"DRY version {version} unknown")

    def save_object_v1(self, obj: Object, update: bool = False,
                       as_cls: Optional[Type] = None,
                       save_cache=None) -> bool:

        # First, check the save cache.
        if save_cache is not None:
            # WARNING, the id of an object is only unique for its lifetime.
            # We must assume objects aren't going out of scope
            # or being deleted for this to work.
            if id(obj) in save_cache.obj_cache:
                # We found the object, write the cached file to
                # The passed binary.
                saved_int_file = save_cache.obj_cache[id(obj)]
                saved_int_file.write_to_file(self.binary_file)
                return True

        # Save subordinate objects.
        for sub_obj in obj.__dry_obj_container_list__:
            # Open a file inside the zip to contain the new object.
            obj_id = sub_obj.dry_id
            save_path = f'dry_objects/{obj_id}.dry'
            if save_path not in self.z_file.namelist():
                with self.z_file.open(save_path, 'w') as f:
                    if not sub_obj.save_self(f, save_cache=save_cache):
                        return False

        # Save meta data
        self.save_meta_data()

        # Save config v1
        obj_def = obj.definition()
        if as_cls is not None:
            obj_def.cls = as_cls

        self.save_definition_v1(obj_def, update=update)

        # Save object content
        ret_val = obj.save_object(self.z_file, save_cache=save_cache)

        if save_cache is not None:
            save_cache.obj_cache[id(obj)] = self.int_file_detach()

        return ret_val

    def contained_object_ids(self):
        """
        enumerates the ids of contained subordinate objects
        """

        return list(map(
            lambda m: m.groups(1)[0],
            filter(lambda m: m is not None,
                   map(lambda n: ObjectFile.contained_dry_file_re.match(n),
                       self.z_file.namelist()))))

    def get_contained_object_file(self, dry_id):
        return self.z_file.open(f"dry_objects/{dry_id}.dry")


@static_var('load_repo', None)
def load_object(file: FileType, update: bool = False,
                exact_path: bool = False,
                reload: bool = False,
                as_cls: Optional[Type] = None,
                repo=None) -> Object:
    """
    A method for loading an object from disk.
    """
    reset_repo = False
    load_obj = True

    # Handle repo management variables
    if repo is not None:
        if load_object.load_repo is not None:
            raise RuntimeError(
                "different repos not currently supported")
        else:
            # Set the call_repo
            load_object.load_repo = repo
            reset_repo = True

    # We now need the object definition
    with ObjectFile(file, exact_path=exact_path) as dry_file:
        obj_def = dry_file.definition()
        # Check whether a repo was given in a prior call
        if load_object.load_repo is not None:
            try:
                # Load the object from the repo
                obj = load_object.load_repo.get_obj(obj_def)
                if obj.definition() != obj_def:
                    raise RuntimeError("Found issue!")
                load_obj = False
            except Exception:
                pass

        if load_obj:
            obj = dry_file.load_object(update=update,
                                       reload=reload,
                                       as_cls=as_cls)
            if as_cls is not None or reload:
                if as_cls is not None:
                    cls = as_cls
                elif reload:
                    cls = get_class_from_str(get_class_str(obj_def.cls))
                new_def = ObjectDef(
                    cls,
                    *obj_def.args,
                    dry_mut=obj_def.dry_mut,
                    **obj_def.kwargs)
            else:
                new_def = obj_def
            if not obj.definition() == new_def:
                raise RuntimeError(
                    f"Loaded object doesn't have expected definition!\n"
                    f"expected: {new_def}\ngot: {obj.definition()}")

    # Reset the repo for this function
    if reset_repo:
        load_object.load_repo = None

    return obj


@static_var('load_repo', None)
def load_object_content(
        obj: Object,
        file: FileType) -> bool:
    """
    A method for loading an object from disk.
    """

    # We now need the object definition
    with ObjectFile(file) as dry_file:
        file_def = dry_file.definition()
        obj_def = obj.definition()
        if file_def != obj_def:
            file_contained_obj_ids = dry_file.contained_object_ids()
            diff_recursive(file_def, obj_def)
            raise ValueError(
                f"File {file} doesn't store data for object {obj.dry_id} "
                "at the top level.\n"
                f"file_def: {file_def}\n"
                f"obj_def: {obj_def}\n"
                f"Contained Subordinate Objects: {file_contained_obj_ids}")

        # Load contained objects
        file_contained_obj_ids = dry_file.contained_object_ids()
        for sub_obj in obj.__dry_obj_container_list__:
            if sub_obj.dry_id not in file_contained_obj_ids:
                raise ValueError(
                    f"File {file} doesn't contain subordinate object data for "
                    f"subordinate object {sub_obj.dry_id}! file contains: "
                    f"{file_contained_obj_ids} namelist: "
                    f"{dry_file.z_file.namelist()}")
            sub_obj_f = dry_file.get_contained_object_file(sub_obj.dry_id)
            if not load_object_content(sub_obj, sub_obj_f):
                print(f"Error loading subordinate object {sub_obj.dry_id}")
                return False

        # Load content of this object
        if not dry_file.load_object_content(obj):
            print(f"Error loading self: {obj.dry_id}")
            return False

    return True


def save_object(obj: Object, file: FileType, version: int = 1,
                exact_path: bool = False, update: bool = False,
                as_cls: Optional[Type] = None,
                save_cache=None) -> bool:
    # Initialize a save cache by default.
    close_save_cache = False
    if save_cache is None:
        close_save_cache = True
        save_cache = SaveCache()
    with ObjectFile(file, exact_path=exact_path, mode='w',
                    must_exist=False) as dry_file:
        if version == 1:
            ret_val = dry_file.save_object_v1(
                obj, update=update, as_cls=as_cls, save_cache=save_cache)
        else:
            raise ValueError(f"File version {version} unknown. Can't save!")

    # Close save caches.
    if save_cache is not None and close_save_cache:
        del save_cache

    return ret_val


def change_object_cls(obj: Object, cls: Type, update: bool = False,
                      reload: bool = False) -> Object:
    buffer = io.BytesIO()
    if not save_object(obj, buffer):
        raise RuntimeError("Error saving object!")
    return load_object(buffer, update=update, reload=reload,
                       as_cls=cls)


def obj_to_def(val):
    if is_supported_scalar_type(val):
        return val
    if isinstance(val, Object):
        definition = val.definition()
        return definition
    elif is_supported_listlike(val):
        return map_listlike(obj_to_def, val)
    elif is_supported_dictlike(val):
        return map_dictlike(obj_to_def, val)
    else:
        raise TypeError(
            f"Unsupported value {val} of type "
            f"{type(val)} encountered!")


# Define a base  Object
class Object(metaclass=Meta):
    # Only ever set for this class.
    __dry_meta_base__ = True

    # Define the dry_id
    def __init__(self, *args, dry_id=None, dry_metadata=None, **kwargs):
        # Dry ID
        if dry_id is None:
            self.dry_kwargs['dry_id'] = str(uuid.uuid4())

        # Handle Metadata
        if dry_metadata is None:
            # Create a default empty dry_metadata
            dry_metadata = {}
        if 'description' not in dry_metadata:
            dry_metadata['description'] = ""
        if 'creation_time' not in dry_metadata:
            dry_metadata['creation_time'] = time.time()
        self.dry_kwargs['dry_metadata'] = dry_metadata

        self._definition = None

    def definition(self):
        # Return if we have a definition created already
        if self._definition is not None:
            return self._definition

        # recursively build the definitions
        new_args = obj_to_def(self.dry_args)
        new_kwargs = obj_to_def(self.dry_kwargs)

        self._definition = ObjectDef(
            type(self),
            *new_args,
            **new_kwargs)

        return self._definition

    def save_self(self, file: FileType, version: int = 1, **kwargs) -> bool:
        return save_object(self, file, version=version, **kwargs)

    def __str__(self):
        return str(self.definition())

    def __repr__(self):
        return str(self.definition())

    @property
    def dry_id(self):
        if 'dry_id' not in self.dry_kwargs:
            raise MissingIdError()
        return self.dry_kwargs['dry_id']

    @property
    def dry_metadata(self):
        if 'dry_metadata' not in self.dry_kwargs:
            raise MissingMetadataError()
        return self.dry_kwargs['dry_metadata']

    def __hash__(self):
        return hash(self.dry_id)

    def dry_context_requirements(self):
        context_reqs = {self.__dry_compute_context__: [{}]}
        for obj in self.__dry_obj_container_list__:
            obj_reqs = obj.dry_context_requirements()
            for ctx_name in obj_reqs:
                if ctx_name in context_reqs:
                    context_reqs[ctx_name].append(obj_reqs[ctx_name])
                else:
                    context_reqs[ctx_name] = [obj_reqs[ctx_name]]

        for ctx_name in context_reqs:
            context_reqs[ctx_name] = combine_requests(context_reqs[ctx_name])

        return context_reqs

    def compute_activate(self):
        ctx_mgr = context()
        if ctx_mgr is None:
            raise NoContextError()

        # Activate contained objects first.
        # We can activate subordinate objects and depend
        # on them being loaded.
        if hasattr(self, '__dry_obj_container_list__'):
            for obj in self.__dry_obj_container_list__:
                obj.compute_activate()

        # Prepare this object's compute
        if not ctx_mgr.contains_activated_object(self):
            self.compute_prepare()

            # Load this object's compute if needed
            self.load_compute()

            # Add object to manager tracker.
            ctx_mgr.add_activated_object(self)

    @staticmethod
    def graph_label(obj, report_class=True):
        if report_class:
            return f"{obj.dry_id} ({type(obj).__name__})"
        else:
            return obj.dry_id

    def _dry_obj_graph(self, report_class=True):
        root = {}
        if hasattr(self, '__dry_obj_container_list__') and \
           len(self.__dry_obj_container_list__) > 0:
            for obj in self.__dry_obj_container_list__:
                label = Object.graph_label(
                    obj, report_class=report_class)
                root[label] = obj._dry_obj_graph()
        return root

    def dry_obj_graph(self, report_class=True):
        from asciitree import LeftAligned
        from asciitree.drawing import BoxStyle, BOX_LIGHT
        label = Object.graph_label(
            self, report_class=report_class)
        tree = {label: self._dry_obj_graph()}
        tr = LeftAligned(
            draw=BoxStyle(
                gfx=BOX_LIGHT,
                horiz_len=1
            )
        )
        print(tr(tree))


class ObjectFactory(object):
    def __init__(self, obj_def: ObjectDef, callbacks=[]):
        if 'dry_id' in obj_def:
            raise ValueError(
                "An Object factory can't use a definition with a dry_id")
        self.obj_def = obj_def
        self.callbacks = callbacks

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def __call__(self, repo=None):
        obj = self.obj_def.build(repo=repo)
        for callback in self.callbacks:
            # Call each callback
            callback(obj)
        return obj


class Wrapper(Object):
    """
    An object wrapper for simple python objects
    """
    @Meta.collect_args
    @Meta.collect_kwargs
    def __init__(self, cls: Type, *args, **kwargs):
        self.obj = cls(*args, **kwargs)


class Callable(Object):
    """
    A wrapper for a callable object to cement some arguments
    """

    @Meta.collect_args
    @Meta.collect_kwargs
    def __init__(
            self, obj: Wrapper, *args, **kwargs):
        self.call_args = args
        self.call_kwargs = kwargs
        self.obj = obj

    def __call__(self, *args, **kwargs):
        return self.obj.obj(
            *(self.call_args+args),
            **{**self.call_kwargs, **kwargs})


def get_contained_objects(obj: Object) -> [Object]:
    contained_objs = set()
    for contained_obj in obj.__dry_obj_container_list__:
        sub_set = get_contained_objects(contained_obj)
        # Add contained objects contained objects.
        contained_objs.update(sub_set)
        # Add the contained object.
        contained_objs.add(contained_obj)
    return contained_objs


class DryObjectPlaceholder(object):
    def __init__(self, ID, obj_def):
        self.ID = ID
        self.obj_def = obj_def


class DryObjectPlaceholderData(object):
    def __init__(self, ID: int, data):
        self.ID = ID
        self.data = data


def generate_unique_id():
    ii32 = np.iinfo(np.int32)
    return np.random.randint(ii32.min, ii32.max)


def create_placeholder(obj: Object) -> (DryObjectPlaceholder,
                                        DryObjectPlaceholderData):
    ID = generate_unique_id()
    obj_def = obj.definition()
    ph_def = DryObjectPlaceholder(ID, obj_def)
    temp_buf = io.BytesIO()
    if not obj.save_self(temp_buf):
        raise RuntimeError(
            "Couldn't save object {obj.definition()} to temp buffer!")

    temp_buf.seek(0)
    ph_data = DryObjectPlaceholderData(ID, temp_buf.read())
    return (ph_def, ph_data)


def rebuild_object(ph_def, ph_data, verbose=False) -> Object:
    if verbose:
        print(f"Rebuilding object {ph_def} with data {ph_data}")
    obj = ph_def.obj_def.build()
    buf = io.BytesIO(ph_data.data)
    if not load_object_content(obj, buf):
        raise RuntimeError(f"Failed to rebuild object {ph_def.obj_def}")
    return obj


def prep_args_kwargs(args, kwargs):
    obj_ph_data_map = {}
    obj_ph_map = {}
    new_args = []
    for i in range(len(args)):
        arg = args[i]
        assigned = False
        if issubclass(type(arg), Object):
            if id(arg) in obj_ph_data_map:
                new_args.append(obj_ph_map[id(arg)])
            else:
                ph = create_placeholder(arg)
                new_args.append(ph[0])
                obj_ph_map[id(arg)] = ph[0]
                obj_ph_data_map[id(arg)] = ph[1]
            assigned = True
        if not assigned:
            new_args.append(args[i])

    for key in kwargs:
        arg = kwargs[key]
        if issubclass(type(arg), Object):
            if id(arg) in obj_ph_data_map:
                args[i] = obj_ph_map[id(arg)]
            else:
                ph = create_placeholder(arg)
                args[i] = ph[0]
                obj_ph_map[id(arg)] = ph[0]
                obj_ph_data_map[id(arg)] = ph[1]

    return (new_args, kwargs), list(obj_ph_data_map.values())


def reconstruct_args_kwargs(args, kwargs, ph_data, verbose=False):
    ph_dict = {}
    constructed_objs = {}
    for ph in ph_data:
        ph_dict[ph.ID] = ph

    for i in range(len(args)):
        arg = args[i]
        if type(arg) is DryObjectPlaceholder:
            if arg.ID in constructed_objs:
                args[i] = constructed_objs[arg.ID]
            else:
                obj = rebuild_object(arg, ph_dict[arg.ID], verbose=verbose)
                constructed_objs[arg.ID] = obj
                args[i] = obj

    for key in kwargs:
        arg = kwargs[key]
        if type(arg) is DryObjectPlaceholder:
            if arg.ID in constructed_objs:
                kwargs[key] = constructed_objs[arg.ID]
            else:
                obj = rebuild_object(arg, ph_dict[arg.ID], verbose=verbose)
                constructed_objs[arg.ID] = obj
                kwargs[key] = obj


class ObjectNode(object):
    """
    Helper class to Build tree of objects
    """

    def __init__(self, obj, children):
        # All nodes have a list of children
        self.children = children
        self.obj = obj

    def apply_func(self, func: Callable):
        if self.obj is not None:
            func(self.obj)


def apply_df_imp(func: Callable, node: ObjectNode, seen_set: set[ObjectNode]):
    # seen set will be empty when called on the root node.
    for child_node in node.children:
        if child_node not in seen_set:
            # Apply function to child node.
            apply_df_imp(func, child_node, seen_set)
    # Apply to this node
    node.apply_func(func)
    # Set this node as seen.
    seen_set.add(node)


def apply_bf_imp(
        func: Callable,
        visit_list: list[ObjectNode], seen_set: set[ObjectNode]):
    # the visit list should be populated when called from the root node.
    while len(visit_list) > 0:
        # Pop a node from the front of the list.
        node = visit_list.pop(0)
        # Apply function first to this node.
        node.apply(func)
        # Add this node to the seen set.
        seen_set.add(node)
        # add nodes children that haven't already been seen to the visit list.
        for child_node in node.children:
            if child_node not in seen_set:
                visit_list.append(child_node)


class ObjectTree(ObjectNode):
    """
    Root node object.
    """

    def __init__(self, children):
        super().__init__(None, children)

    def apply_func(self, func: Callable):
        # Don't do anything on the root node.
        pass

    def apply_df(self, func: Callable):
        apply_df_imp(func, self, set())

    def apply_bf(self, func: Callable):
        apply_bf_imp(func, self, self.children, set())


def build_obj_tree_imp(obj: Object, node_dict):
    """
    Returns ObjectNode of passed object.
    """

    if id(obj) in node_dict:
        return node_dict[id(obj)]

    # Get unique children list
    unique_children_map = {}
    for child_obj in obj.__dry_obj_container_list__:
        if id(child_obj) not in unique_children_map:
            unique_children_map[id(child_obj)] = child_obj

    # Build child node list
    child_nodes = []
    for child_id in unique_children_map:
        child_obj = unique_children_map[child_id]
        child_nodes.append(build_obj_tree_imp(child_obj, node_dict))

    # Create node, and record.
    res_node = ObjectNode(obj, child_nodes)
    node_dict[id(obj)] = res_node
    return res_node


def build_obj_tree(obj_or_list: Union[Object, list[Object]]):
    """
    Construct tree
    """
    # We construct depth first.
    node_dict = {}

    if type(obj_or_list) is Object:
        obj_list = [obj_or_list]
    else:
        obj_list = obj_or_list

    # Get unique objects
    obj_dict = {}
    for obj in obj_list:
        if id(obj) not in obj_dict:
            obj_dict[id(obj)] = obj

    obj_list = obj_dict.values()

    root_children = []
    for obj in obj_list:
        root_children.append(build_obj_tree_imp(obj, node_dict))

    # Returned finished tree
    return ObjectTree(root_children)
