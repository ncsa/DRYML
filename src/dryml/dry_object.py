# File to define saving/loading convenience functions

from __future__ import annotations

import os
import dill
import json
import pickle
import io
import zipfile
import uuid
import copy
import inspect
import importlib
from typing import IO, Union, Optional
from dryml.dry_config import DryConfig, DryList
from dryml.utils import init_arg_list_handler, init_arg_dict_handler, get_hashed_id, get_class_str

FileType = Union[str, IO[bytes]]

def file_resolve(file: str, exact_path:bool = False) -> str:
    if os.path.splitext(file)[1] == '' and not exact_path:
        file = f"{file}.dry"
    return file


def load_zipfile(file: FileType, exact_path:bool=False, mode='r', must_exist:bool=True) -> zipfile.ZipFile:
    if type(file) is str:
        filepath = file
        filepath = file_resolve(filepath, exact_path=exact_path)
        if must_exist and not os.path.exists(filepath):
            raise ValueError(f"File {filepath} doesn't exist!")
        file = zipfile.ZipFile(filepath, mode=mode)
    if type(file) is not zipfile.ZipFile:
        file = zipfile.ZipFile(file, mode=mode)
    return file

def compute_obj_hash_str(cls:type, args:DryList, kwargs:DryConfig, no_id=True):
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


class DryObjectFile(object):
    def __init__(self, file: FileType, exact_path:bool=False, mode='r', must_exist=True, obj:Optional[DryObject]=None, reload:bool=False):
        if type(file) is str:
            # Save the filename
            self.filename = file
        else:
            self.filename = None
        self.file = load_zipfile(file, exact_path=exact_path, mode=mode, must_exist=must_exist)

        self.cls = None
        self.args = None
        self.kwargs = None

        if obj is not None:
            # Cache object data, can be used for updating
            self.update_file(obj)
        else:
            # If we're reading the file, we probably want to compute a 
            # hash or create an object. We can safely read and cache hash data now.
            if mode == 'r':
                 self.cache_object_data_file(self.file, reload=reload)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def close(self):
        self.file.close()

    def update_file(self, obj:DryObject):
        self.cache_object_data_obj(obj)

    def load_config_v1(self):
        "Helper function for loading a version 1 config file"
        with self.file.open('config.pkl', 'r') as config_file:
            config_data = pickle.loads(config_file.read())
        return config_data

    def load_class_def_v1(self, update:bool=True, reload:bool=False):
        "Helper function for loading a version 1 class definition"
        # Get class definition
        with self.file.open('cls_def.dill') as cls_def_file:
            if update:
                # Get original model definition
                cls_def_init = dill.loads(cls_def_file.read())
                try:
                    module = importlib.import_module(inspect.getmodule(cls_def_init).__name__)
                    # If indicated, reload the module.
                    if reload:
                        module = importlib.reload(module)
                    cls_def = getattr(module, cls_def_init.__name__)
                except:
                    raise RuntimeError("Failed to update module class")
            else:
                cls_def = dill.loads(cls_def_file.read())
        return cls_def

    def load_meta_data(self):
        with self.file.open('meta_data.pkl', 'r') as meta_file:
            meta_data = pickle.loads(meta_file.read())
        return meta_data

    def cache_object_data_v1_file(self, update:bool=True, reload:bool=False):
        # Cache object data from a file
        config_data = self.load_config_v1()

        # Get arguments
        self.kwargs = DryConfig(config_data['kwargs'])
        self.args = DryList(config_data['args'])

        # Get class definition
        self.cls = self.load_class_def_v1(update=update, reload=reload)

    def cache_object_data_file(self, update:bool=True, reload:bool=False):
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

    def load_object_v1(self, update:bool=True, reload:bool=False) -> DryObject:
        # Load object
        self.cache_object_data_v1_file(update=update, reload=reload)

        # Create object
        obj = self.cls(*self.args, **self.kwargs)

        # Load object content
        obj.load_object_imp(self.file)

        # Build object instance
        return obj

    def load_object(self, update:bool=False, reload:bool=False):
    	meta_data = self.load_meta_data()
    	version = meta_data['version']
    	if version == 1:
        	return self.load_object_v1(update=update, reload=reload)
    	else:
        	raise RuntimeError(f"DRY version {version} unknown")

    def save_meta_data(self):
        # Meta_data
        meta_data = DryConfig({
            'version': 1
        })

        meta_dump = pickle.dumps(meta_data, protocol=5)
        self.file.writestr('meta_data.pkl', meta_dump)

    def save_config_v1(self):
        config_data = DryConfig({
            'kwargs': self.kwargs.data,
            'args': self.args.data
        })

        config_dump = pickle.dumps(config_data, protocol=5)
        self.file.writestr('config.pkl', config_dump)

    def save_class_def_v1(self, update:bool=False):
        # We need to pickle the class definition.
        # By default, error out if class has changed. Check this.
        mod = inspect.getmodule(self.cls)
        mod_cls = getattr(mod, self.cls.__name__)
        if self.cls != mod_cls and not update:
            raise ValueError("Can't save class definition! It's been changed!")
        cls_def = dill.dumps(mod_cls)
        self.file.writestr('cls_def.dill', cls_def)

    def save_object_v1(self, obj: DryObject, update:bool=False) -> bool:
        self.cache_object_data_obj(obj)

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
        return compute_obj_hash_str(self.cls, self.args, self.kwargs, no_id=no_id)

    def get_hash(self, no_id=True):
        return hash(self.get_hash_str(no_id=no_id))

    def get_category_hash(self):
        return get_hashed_id(self.get_hash_str(no_id=True))

    def get_individual_hash(self):
        return get_hashed_id(self.get_hash_str(no_id=False))

def load_object(file: Union[FileType,DryObjectFile], update:bool=False, exact_path:bool=False, reload:bool=False) -> Type[DryObject]:
    if not isinstance(file, DryObjectFile):
    	with DryObjectFile(file, exact_path=exact_path) as dry_file:
        	result_object = dry_file.load_object(update=update, reload=reload)
    else:
        result_object = dry_file.load_object(update=update, reload=reload)
    return result_object

def save_object(obj: DryObject, file: FileType, version: int=1, exact_path:bool=False, update:bool=False) -> bool:
    with DryObjectFile(file, exact_path=exact_path, mode='w', must_exist=False) as dry_file:
        if version == 1:
            return dry_file.save_object_v1(obj, update=update)
        else:
            raise ValueError(f"File version {version} unknown. Can't save!")

# Define a base Dry Object 
class DryObject(object):
    def __init__(self, *args, dry_args=None, dry_kwargs=None, dry_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Use DryConfig/DryList object to coerse args/kwargs to proper json serializable form.
        self.dry_args = DryList(init_arg_list_handler(dry_args))
        self.dry_kwargs = DryConfig(init_arg_dict_handler(dry_kwargs))
        # Generate unique id for this object. (Meant to separate between multiple instances of same object)
        if dry_id is None:
            self.dry_kwargs['dry_id'] = str(uuid.uuid4())
        else:
            self.dry_kwargs['dry_id'] = dry_id

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Should be the last object inherited
        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Should be the last object inherited
        return True

    def save_self(self, file: FileType, version: int=1, **kwargs) -> bool:
        return save_object(self, file, version=version, **kwargs)

    def get_hash_str(self, no_id=True):
        return compute_obj_hash_str(type(self), self.dry_args, self.dry_kwargs, no_id=no_id)

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
    def __init__(self, cls, *args, callbacks=[], **kwargs):
        self.cls = cls
        self.args = DryList(args)
        self.kwargs = DryConfig(kwargs)
        self.callbacks = callbacks

    def add_callback(callback):
        self.callbacks.append(callback)

    def __call__(self):
        obj = self.cls(*self.args, **self.kwargs) 
        for callback in self.callbacks:
            # Call each callback
            callback(obj)
        return obj

    def get_hash_str(self):
        # For DryObjectFactory, we can't use id.
        return compute_obj_hash_str(self.cls, self.args, self.kwargs, no_id=True)

    def get_hash(self):
        return hash(self.get_hash_str())

    def get_category_hash(self):
        return get_hashed_id(self.get_hash_str())
