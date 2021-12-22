# File to define saving/loading convenience functions

from __future__ import annotations

import collections
import os
import dill
import pickle
import io
import zipfile
import uuid
import copy
from typing import IO, Union, Optional, Type, Mapping
from dryml.dry_config import DryKwargs, DryArgs
from dryml.utils import init_arg_list_handler, init_arg_dict_handler, \
    get_hashed_id, get_class_str, get_current_cls, get_class_from_str, \
    pickler

FileType = Union[str, IO[bytes]]


def file_resolve(file: str, exact_path: bool = False) -> str:
    if os.path.splitext(file)[1] == '' and not exact_path:
        file = f"{file}.dry"
    return file


def load_zipfile(file: FileType, exact_path: bool = False,
                 mode='r', must_exist: bool = True) -> zipfile.ZipFile:
    if type(file) is str:
        filepath = file
        filepath = file_resolve(filepath, exact_path=exact_path)
        if must_exist and not os.path.exists(filepath):
            raise ValueError(f"File {filepath} doesn't exist!")
        file = zipfile.ZipFile(filepath, mode=mode)
    if type(file) is not zipfile.ZipFile:
        file = zipfile.ZipFile(file, mode=mode)
    return file


def compute_obj_hash_str(cls: type, args: DryArgs, kwargs: DryKwargs,
                         no_id: bool = True):
    class_hash_str = get_class_str(cls)
    args_hash_str = args.get_hash_str()
    # Remove dry_id so we can test for object 'class'
    if no_id:
        kwargs_copy = copy.copy(kwargs)
        if 'dry_id' in kwargs_copy:
            kwargs_copy.pop('dry_id')
        kwargs_hash_str = kwargs_copy.get_hash_str()
    else:
        kwargs_hash_str = kwargs.get_hash_str()
    return class_hash_str+args_hash_str+kwargs_hash_str


class DryObjectDefinition(collections.UserDict):
    # We may have to adjust this behavior in the future.
    # For now, this is the only way I can think to have nested
    # collections of dry objects all find objects from the same
    # repository.
    call_repo = None

    @staticmethod
    def from_dict(def_dict: Mapping):
        if isinstance(def_dict['cls'], str):
            cls = get_class_from_str(def_dict['cls'])
        return DryObjectDefinition(
            cls,
            *def_dict.get('dry_args', DryArgs()),
            **def_dict.get('dry_kwargs', DryKwargs()))

    def __init__(self, cls: Type,
                 *args, **kwargs):
        super().__init__()
        self['cls'] = cls
        self['dry_args'] = DryArgs(args)
        self['dry_kwargs'] = DryKwargs(kwargs)

    def to_dict(self):
        return {
            'cls': get_class_str(self['cls']),
            'dry_args': self['dry_args'].data,
            'dry_kwargs': self['dry_kwargs'].data,
        }

    def __call__(self, repo=None):
        "Construct an object"
        reset_repo = False
        construct_object = True

        if repo is not None:
            from dryml.dry_repo import DryRepo
            if not isinstance(repo, DryRepo):
                raise TypeError(
                    "Only objects of type DryRepo are supported for"
                    " repo argument")
            if DryObjectDefinition.call_repo is not None:
                raise RuntimeError(
                    "different call repos not currently supported")
            else:
                # Set the call_repo
                DryObjectDefinition.call_repo = repo
                reset_repo = True

        # Check whether a repo was given in a prior call
        if DryObjectDefinition.call_repo is not None:
            cat_hash = self.get_category_hash()
            try:
                ind_hash = self.get_individual_hash()
            except Exception as e:
                print("Can't use a repo with a non-specific object"
                      " definition")
                raise e
            try:
                obj = DryObjectDefinition.call_repo.get_obj_by_hash(
                    cat_hash, ind_hash)
                construct_object = False
            except Exception:
                pass

        if construct_object:
            obj = self['cls'](*self['dry_args'], **self['dry_kwargs'])

        # Reset the repo for this function
        if reset_repo:
            DryObjectDefinition.call_repo = None

        # Return the result
        return obj

    def __hash__(self):
        return self.get_hash(no_id=False)

    def __eq__(self, rhs: DryObjectDefinition):
        return hash(self) == hash(rhs)

    def get_hash_str(self, no_id=False):
        if not no_id:
            if 'dry_id' not in self['dry_kwargs']:
                raise RuntimeError(
                    "Tried to make an individual hash on a "
                    "DryObjectDefinition without a dry_id!")
        return compute_obj_hash_str(self['cls'], self['dry_args'],
                                    self['dry_kwargs'], no_id=no_id)

    def get_hash(self, no_id=False):
        return hash(self.get_hash_str(no_id=no_id))

    def get_individual_hash(self):
        return get_hashed_id(self.get_hash_str(no_id=False))

    def get_category_hash(self):
        return get_hashed_id(self.get_hash_str(no_id=True))


class DryObjectFile(object):
    def __init__(self, file: FileType, exact_path: bool = False,
                 mode: str = 'r', must_exist: bool = True,
                 obj: Optional[DryObject] = None,
                 reload: bool = False, as_cls: Optional[Type] = None):
        if type(file) is str:
            # Save the filename
            self.filename = file
        else:
            self.filename = None
        self.file = load_zipfile(file, exact_path=exact_path,
                                 mode=mode, must_exist=must_exist)

        self.cls = None
        self.args = None
        self.kwargs = None

        if obj is not None:
            # Cache object data, can be used for updating
            self.update_file(obj)
        else:
            # If we're reading the file, we probably want to compute a
            # hash or create an object. We can safely read and cache
            # hash data now.
            if mode == 'r':
                self.cache_object_data_file(self.file, reload=reload)

        if as_cls is not None:
            # Set the type we want to save as
            self.cls = as_cls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def close(self):
        self.file.close()

    def update_file(self, obj: DryObject):
        self.cache_object_data_obj(obj)

    def load_config_v1(self):
        "Helper function for loading a version 1 config file"
        with self.file.open('config.pkl', 'r') as config_file:
            config_data = pickle.loads(config_file.read())
        return config_data

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

    def load_meta_data(self):
        with self.file.open('meta_data.pkl', 'r') as meta_file:
            meta_data = pickle.loads(meta_file.read())
        return meta_data

    def cache_object_data_v1_file(self, update: bool = True,
                                  reload: bool = False):
        # Cache object data from a file
        config_data = self.load_config_v1()

        # Get arguments
        self.kwargs = DryKwargs(config_data['kwargs'])
        self.args = DryArgs(config_data['args'])

        # Get class definition
        self.cls = self.load_class_def_v1(update=update, reload=reload)

    def cache_object_data_file(self, update: bool = True,
                               reload: bool = False):
        meta_data = self.load_meta_data()
        version = meta_data['version']
        if version == 1:
            self.cache_object_data_v1_file(update=update, reload=reload)
        else:
            raise RuntimeError(f"DRY version {version} unknown")

    def cache_object_data_obj(self, obj: DryObject):
        # Cache object data from an object
        self.kwargs = obj.dry_kwargs
        self.args = obj.dry_args
        self.cls = type(obj)

    def load_object_v1(self, update: bool = True,
                       reload: bool = False) -> DryObject:
        # Load object
        self.cache_object_data_v1_file(update=update, reload=reload)

        # Create object
        obj = self.cls(*self.args, **self.kwargs)

        # Load object content
        obj.load_object_imp(self.file)

        # Build object instance
        return obj

    def load_object(self, update: bool = False,
                    reload: bool = False) -> DryObject:
        meta_data = self.load_meta_data()
        version = meta_data['version']
        if version == 1:
            return self.load_object_v1(update=update, reload=reload)
        else:
            raise RuntimeError(f"DRY version {version} unknown")

    def save_meta_data(self):
        # Meta_data
        meta_data = DryKwargs({
            'version': 1
        })

        meta_dump = pickler(meta_data)
        with self.file.open('meta_data.pkl', mode='w') as f:
            f.write(meta_dump)

    def save_config_v1(self):
        config_data = DryKwargs({
            'kwargs': self.kwargs.data,
            'args': self.args.data
        })

        config_dump = pickler(config_data)
        with self.file.open('config.pkl', mode='w') as f:
            f.write(config_dump)

    def save_class_def_v1(self, update: bool = False):
        # We need to pickle the class definition.
        # By default, error out if class has changed. Check this.
        mod_cls = get_current_cls(self.cls)
        if self.cls != mod_cls and not update:
            raise ValueError("Can't save class definition! It's been changed!")
        cls_def = dill.dumps(mod_cls)
        with self.file.open('cls_def.dill', mode='w') as f:
            f.write(cls_def)

    def save_object_v1(self, obj: DryObject, update: bool = False,
                       as_cls: Optional[Type] = None) -> bool:
        self.cache_object_data_obj(obj)
        if as_cls is not None:
            self.cls = as_cls

        # Save meta data
        self.save_meta_data()

        # Save config v1
        self.save_config_v1()

        # Save class def
        self.save_class_def_v1(update=update)

        # Save object content
        obj.save_object_imp(self.file)

        return True

    def get_hash_str(self, no_id=True):
        return compute_obj_hash_str(self.cls, self.args,
                                    self.kwargs, no_id=no_id)

    def get_hash(self, no_id=True):
        return hash(self.get_hash_str(no_id=no_id))

    def get_category_hash(self):
        return get_hashed_id(self.get_hash_str(no_id=True))

    def get_individual_hash(self):
        return get_hashed_id(self.get_hash_str(no_id=False))


def load_object(file: Union[FileType, DryObjectFile],
                update: bool = False, exact_path: bool = False,
                reload: bool = False) -> DryObject:
    """
    A method for loading an object from disk.
    """
    if not isinstance(file, DryObjectFile):
        with DryObjectFile(file, exact_path=exact_path) as dry_file:
            result_object = dry_file.load_object(update=update,
                                                 reload=reload)
    else:
        result_object = dry_file.load_object(update=update, reload=reload)
    return result_object


def save_object(obj: DryObject, file: FileType, version: int = 1,
                exact_path: bool = False, update: bool = False,
                as_cls: Optional[Type] = None) -> bool:
    with DryObjectFile(file, exact_path=exact_path, mode='w',
                       must_exist=False, as_cls=as_cls) as dry_file:
        if version == 1:
            return dry_file.save_object_v1(obj, update=update, as_cls=as_cls)
        else:
            raise ValueError(f"File version {version} unknown. Can't save!")


def change_object_cls(obj: DryObject, cls: Type, update: bool = False,
                      reload: bool = False) -> DryObject:
    buffer = io.BytesIO()
    save_object(obj, buffer, as_cls=cls)
    return load_object(buffer, update=update, reload=reload)


# Define a base Dry Object
class DryObject(object):
    def __init__(self, *args, dry_args=None, dry_kwargs=None,
                 dry_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Use DryKwargs/DryArgs object to coerse args/kwargs to proper
        # json serializable form.
        self.dry_args = DryArgs(init_arg_list_handler(dry_args))
        self.dry_kwargs = DryKwargs(init_arg_dict_handler(dry_kwargs))
        # Generate unique id for this object. (Meant to separate between
        # multiple instances of same object)
        if dry_id is None:
            self.dry_kwargs['dry_id'] = str(uuid.uuid4())
        else:
            self.dry_kwargs['dry_id'] = dry_id

    def get_definition(self):
        return DryObjectDefinition(
            type(self),
            *self.dry_args,
            **self.dry_kwargs)

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Should be the last object inherited
        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Should be the last object inherited
        return True

    def save_self(self, file: FileType, version: int = 1, **kwargs) -> bool:
        return save_object(self, file, version=version, **kwargs)

    def __hash__(self):
        return self.get_hash(no_id=False)

    def get_hash_str(self, no_id=True):
        return compute_obj_hash_str(type(self), self.dry_args,
                                    self.dry_kwargs, no_id=no_id)

    def get_hash(self, no_id=True):
        return hash(self.get_hash_str(no_id=no_id))

    def get_category_hash(self):
        return get_hashed_id(self.get_hash_str(no_id=True))

    def get_individual_hash(self):
        return get_hashed_id(self.get_hash_str(no_id=False))

    def is_same_category(self, rhs):
        # Not an exact type test to support dynamic object reloading
        if get_class_str(type(self)) != get_class_str(type(rhs)):
            return False
        if self.dry_args != rhs.dry_args:
            return False
        kwargs_copy = copy.copy(self.dry_kwargs)
        kwargs_copy.pop('dry_id')
        rhs_kwargs_copy = copy.copy(rhs.dry_kwargs)
        rhs_kwargs_copy.pop('dry_id')
        if kwargs_copy != rhs_kwargs_copy:
            return False
        return True

    def is_identical(self, rhs):
        if not self.is_same_category(rhs):
            return False
        if self.dry_kwargs['dry_id'] != rhs.dry_kwargs['dry_id']:
            return False
        return True


class DryObjectFactory(object):
    def __init__(self, obj_def: DryObjectDefinition, callbacks=[]):
        if 'dry_id' in obj_def:
            raise ValueError(
                "An Object factory can't use a definition with a dry_id")
        self.obj_def = obj_def
        self.callbacks = callbacks

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def __call__(self):
        obj = self.obj_def()
        for callback in self.callbacks:
            # Call each callback
            callback(obj)
        return obj
