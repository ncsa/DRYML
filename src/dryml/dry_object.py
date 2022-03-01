# File to define saving/loading convenience functions

from __future__ import annotations

import os
import dill
import pickle
import io
import zipfile
import uuid
from typing import IO, Union, Optional, Type
from dryml.dry_config import DryObjectDef, DryMeta
from dryml.utils import get_current_cls, pickler, static_var
from dryml.context.context_tracker import consolidate_contexts
import tempfile

FileType = Union[str, IO[bytes]]


def file_resolve(file: str, exact_path: bool = False) -> str:
    if os.path.splitext(file)[1] == '' and not exact_path:
        file = f"{file}.dry"
    return file


class DryObjectFile(object):
    def __init__(self, file: FileType, exact_path: bool = False,
                 mode: str = 'r', must_exist: bool = True):

        if type(file) is str:
            filepath = file
            filepath = file_resolve(filepath, exact_path=exact_path)
            if must_exist and not os.path.exists(filepath):
                raise ValueError(f"File {filepath} doesn't exist!")
            if mode == 'w':
                # Since we're writing a file, we first need to
                # Open a temp file. This is because
                # python zipfile module doesn't support have good support
                # for updating zipfiles.
                self.temp_file = tempfile.NamedTemporaryFile()
                self.filepath = filepath
                self.file = zipfile.ZipFile(self.temp_file.name, mode=mode)
            else:
                # Since we're reading the file, we can
                # Open as zipfile directly
                self.file = zipfile.ZipFile(filepath, mode=mode)
        elif type(file) is not zipfile.ZipFile:
            self.file = zipfile.ZipFile(file, mode=mode)
        else:
            self.file = file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.file.close()
        if hasattr(self, 'temp_file'):
            # We need to overwrite any existing file
            with open(self.filepath, 'wb') as f:
                self.temp_file.seek(0)
                f.write(self.temp_file.read())
            self.temp_file.close()

    # def update_file(self, obj: DryObject):
    #     self.cache_object_data_obj(obj)

    def save_meta_data(self):
        # Meta_data
        meta_data = {
            'version': 1
        }

        meta_dump = pickler(meta_data)
        with self.file.open('meta_data.pkl', mode='w') as f:
            f.write(meta_dump)

    def load_meta_data(self):
        with self.file.open('meta_data.pkl', 'r') as meta_file:
            meta_data = pickle.loads(meta_file.read())
        return meta_data

    def save_class_def_v1(self, obj_def: DryObjectDef, update: bool = False):
        # We need to pickle the class definition.
        # By default, error out if class has changed. Check this.
        mod_cls = get_current_cls(obj_def.cls)
        if obj_def.cls != mod_cls and not update:
            raise ValueError("Can't save class definition! It's been changed!")
        cls_def = dill.dumps(mod_cls)
        with self.file.open('cls_def.dill', mode='w') as f:
            f.write(cls_def)

    def load_class_def_v1(self, update: bool = True, reload: bool = False):
        "Helper function for loading a version 1 class definition"
        # Get class definition
        with self.file.open('cls_def.dill') as cls_def_file:
            if update:
                # Get original model definition
                cls_def_init = dill.loads(cls_def_file.read())
                try:
                    cls_def = get_current_cls(cls_def_init, reload=reload)
                except Exception as e:
                    raise RuntimeError(f"Failed to update module class {e}")
            else:
                cls_def = dill.loads(cls_def_file.read())
        return cls_def

    def save_definition_v1(self, obj_def: DryObjectDef, update: bool = False):
        "Save object def"
        # Save obj def
        self.save_class_def_v1(obj_def, update=update)

        # Save args from object def
        with self.file.open('dry_args.pkl', mode='w') as args_file:
            args_file.write(pickler(obj_def.args.data))

        # Save kwargs from object def
        with self.file.open('dry_kwargs.pkl', mode='w') as kwargs_file:
            kwargs_file.write(pickler(obj_def.kwargs.data))

        # Save mutability from object def
        with self.file.open('dry_mut.pkl', mode='w') as mut_file:
            mut_file.write(pickler(obj_def.dry_mut))

    def load_definition_v1(self, update: bool = True, reload: bool = False):
        "Load object def"

        # Load obj def
        cls = self.load_class_def_v1(update=update, reload=reload)

        # Load args
        with self.file.open('dry_args.pkl', mode='r') as args_file:
            args = pickle.loads(args_file.read())

        # Load kwargs
        with self.file.open('dry_kwargs.pkl', mode='r') as kwargs_file:
            kwargs = pickle.loads(kwargs_file.read())

        # Load mutability
        with self.file.open('dry_mut.pkl', mode='r') as mut_file:
            mut = pickle.loads(mut_file.read())

        return DryObjectDef(cls, *args, dry_mut=mut, **kwargs)

    def definition(self, update: bool = True, reload: bool = False):
        meta_data = self.load_meta_data()
        if meta_data['version'] == 1:
            return self.load_definition_v1(update=update, reload=reload)
        else:
            raise RuntimeError(
                f"File version {meta_data['version']} not supported!")

    def load_object_v1(self, update: bool = True,
                       reload: bool = False,
                       as_cls: Optional[Type] = None) -> DryObject:
        # Load object
        obj_def = self.load_definition_v1(update=update, reload=reload)
        if as_cls is not None:
            obj_def.cls = as_cls

        # Create object
        obj = obj_def.build(load_zip=self.file)

        # Load object content
        if not obj.load_object(self.file):
            raise RuntimeError("Error loading object!")

        # Build object instance
        return obj

    def load_object(self, update: bool = False,
                    reload: bool = False,
                    as_cls: Optional[Type] = None) -> DryObject:
        meta_data = self.load_meta_data()
        version = meta_data['version']
        if version == 1:
            return self.load_object_v1(
                update=update, reload=reload, as_cls=as_cls)
        else:
            raise RuntimeError(f"DRY version {version} unknown")

    def save_object_v1(self, obj: DryObject, update: bool = False,
                       as_cls: Optional[Type] = None) -> bool:
        # Save meta data
        self.save_meta_data()

        # Save config v1
        obj_def = obj.definition()
        if as_cls is not None:
            obj_def.cls = as_cls

        self.save_definition_v1(obj_def, update=update)

        # Save object content
        return obj.save_object(self.file)


@static_var('load_repo', None)
def load_object(file: FileType, update: bool = False,
                exact_path: bool = False,
                reload: bool = False,
                as_cls: Optional[Type] = None,
                repo=None) -> DryObject:
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
    with DryObjectFile(file, exact_path=exact_path) as dry_file:
        obj_def = dry_file.definition()
        # Check whether a repo was given in a prior call
        if load_object.load_repo is not None:
            try:
                # Load the object from the repo
                obj = load_object.load_repo.get_obj(obj_def)
                load_obj = False
            except Exception:
                pass

        if load_obj:
            obj = dry_file.load_object(update=update,
                                       reload=reload,
                                       as_cls=as_cls)

    # Reset the repo for this function
    if reset_repo:
        load_object.load_repo = None

    return obj


def save_object(obj: DryObject, file: FileType, version: int = 1,
                exact_path: bool = False, update: bool = False,
                as_cls: Optional[Type] = None) -> bool:
    with DryObjectFile(file, exact_path=exact_path, mode='w',
                       must_exist=False) as dry_file:
        if version == 1:
            return dry_file.save_object_v1(obj, update=update, as_cls=as_cls)
        else:
            raise ValueError(f"File version {version} unknown. Can't save!")


def change_object_cls(obj: DryObject, cls: Type, update: bool = False,
                      reload: bool = False) -> DryObject:
    buffer = io.BytesIO()
    if not save_object(obj, buffer):
        raise RuntimeError("Error saving object!")
    return load_object(buffer, update=update, reload=reload,
                       as_cls=cls)


# Define a base Dry Object
class DryObject(metaclass=DryMeta):
    # Only ever set for this class.
    __dry_meta_base__ = True

    # Define the dry_id
    def __init__(self, *args, dry_id=None, **kwargs):
        if dry_id is None:
            self.dry_kwargs['dry_id'] = str(uuid.uuid4())

    def definition(self):
        return DryObjectDef(
            type(self),
            *self.dry_args,
            **self.dry_kwargs)

    def save_self(self, file: FileType, version: int = 1, **kwargs) -> bool:
        return save_object(self, file, version=version, **kwargs)

    def __str__(self):
        return str(self.definition())

    def __repr__(self):
        return str(self.definition())

    @property
    def dry_id(self):
        return self.dry_kwargs.get('dry_id', None)

    def dry_compute_context(self) -> str:
        contexts = [self.__dry_compute_context__]
        for obj in self.__dry_obj_container_list__:
            contexts.append(obj.dry_compute_context())

        return consolidate_contexts(contexts)


class DryObjectFactory(object):
    def __init__(self, obj_def: DryObjectDef, callbacks=[]):
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


class ObjectWrapper(DryObject):
    """
    An object wrapper for simple python objects
    """
    def __init__(self, cls: Type, obj_args=None, obj_kwargs=None):
        if obj_args is None:
            obj_args = []
            self.dry_kwargs['obj_args'] = obj_args
        if obj_kwargs is None:
            obj_kwargs = {}
            self.dry_kwargs['obj_kwargs'] = obj_kwargs

        self.obj = cls(*obj_args, **obj_kwargs)


class CallableWrapper(DryObject):
    """
    A wrapper for a callable object to cement some arguments
    """

    def __init__(
            self, obj: ObjectWrapper, obj_args=None, obj_kwargs=None,
            call_args=None, call_kwargs=None):
        if obj_args is None:
            obj_args = []
            self.dry_kwargs['obj_args'] = obj_args
        if obj_kwargs is None:
            obj_kwargs = {}
            self.dry_kwargs['obj_kwargs'] = obj_kwargs
        if call_args is None:
            call_args = []
            self.dry_kwargs['call_args'] = call_args
        self.call_args = call_args
        if call_kwargs is None:
            call_kwargs = {}
            self.dry_kwargs['call_kwargs'] = call_kwargs
        self.call_kwargs = call_kwargs
        self.obj = obj

    def __call__(self, *args, **kwargs):
        return self.obj.obj(
            *(self.call_args+args),
            **{**self.call_kwargs, **kwargs})
