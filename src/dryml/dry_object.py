# File to define saving/loading convenience functions

from __future__ import annotations

import os
import dill
import json
import io
import zipfile
import uuid
import copy
from typing import Type, IO, Union
from dryml.dry_config import DryConfig

def file_resolve(file: str, exact_path:bool = False) -> str:
    if os.path.splitext(file)[1] == '' and not exact_path:
        file = f"{file}.dry"
    return file

def load_zipfile(file: Union[str, IO[bytes]], exact_path:bool=False) -> zipfile.ZipFile:
    if type(file) is str:
        file = file_resolve(file, exact_path=exact_path)
        if not os.path.exists(file):
            raise ValueError(f"File {filepath} doesn't exist!")
        file = zipfile.ZipFile(file, mode='r')
    if type(file) is not zipfile.ZipFile:
        file = zipfile.ZipFile(file, mode='r')
    return file

def load_config_v1(dry_file: DryObjectFile):
    "Helper function for loading a version 1 config file"
    with dry_file.file.open('config.json', 'r') as config_file:
        config_data = json.loads(config_file.read())
    return config_data

def load_class_def_v1(dry_file: DryObjectFile, update:bool=True):
    "Helper function for loading a version 1 class definition"
    # Get class definition
    with dry_file.file.open('cls_def.dill') as cls_def_file:
        if update:
            cls_def = dill.loads(cls_def_file.read(), ignore=False)
        else:
            cls_def = dill.loads(cls_def_file.read())
    return cls_def

def load_meta_data(dry_file: DryObjectFile):
    with dry_file.file.open('meta_data.json', 'r') as meta_file:
        meta_data = json.loads(meta_file.read())
    return meta_data

class DryObjectFile(object):
    def __init__(self, file: Union[str, IO[bytes]], exact_path:bool=False):
        self.file = load_zipfile(file, exact_path=exact_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def close(self):
        self.file.close()

def load_object_v1(file: DryObjectFile, update:bool=True) -> Type[DryObject]:
    config_data = load_config_v1(file)

    # Get arguments
    kwargs = config_data['kwargs']
    args = config_data['args']

    # Get class definition
    cls_def = load_class_def_v1(file, update=update)

    # Create object
    obj = cls_def(*args, **kwargs)

    # Load object content
    obj.load_object_imp(file)

    # Build object instance
    return obj

def save_object_v1(obj: Type[DryObject], file: IO[bytes]) -> bool:
    # Create zipfile
    output_file = zipfile.ZipFile(file, mode='w')

    # Meta_data
    meta_data = {
        'version': 1
    }

    meta_dump = json.dumps(meta_data)
    output_file.writestr('meta_data.json', meta_dump)

    config_data = {
        'kwargs': obj.dry_kwargs,
        'args': obj.dry_args
    }

    config_dump = json.dumps(config_data)
    output_file.writestr('config.json', config_dump)

    # Now, we need to pickle the class definition
    cls_def = dill.dumps(type(obj))
    output_file.writestr('cls_def.dill', cls_def)

    # Close the output file
    output_file.close()

    return True

def load_object(file: Union[str, IO[bytes]], update:bool=False, exact_path:bool=False) -> Type[DryObject]:
    with DryObjectFile(file) as file:
        meta_data = load_meta_data(file)
        version = meta_data['version']
        if version == 1:
            result_object = load_object_v1(file, update=update)
        else:
            raise RuntimeError(f"DRY version {version} unknown")
    return result_object

def save_object(obj: Type[DryObject], file: Union[str, IO[bytes]], version: int=1, exact_path:bool=False) -> bool:
    # Handle creation of bytes-like
    if type(file) is str:
        file = file_resolve(file)
        file = open(file, 'wb')
    if version == 1:
        return save_object_v1(obj, file)
    else:
        raise ValueError(f"File version {version} unknown. Can't save!")

# Define a base Dry Object 
class DryObject(object):
    def __init__(self, *args, dry_args={}, dry_kwargs={}, dry_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Use DryConfig object to coerse args/kwargs to proper json serializable form.
        self.dry_kwargs = DryConfig(dry_kwargs)
        self.dry_args = DryConfig(dry_args)
        # Generate unique id for this object. (Meant to separate between multiple instances of same object)
        if dry_id is None:
            self.dry_kwargs['dry_id'] = str(uuid.uuid4())
        else:
            self.dry_kwargs['dry_id'] = dry_id

    def load_object_imp(self, file: IO[bytes]) -> bool:
        # Helper function to load object specific data should return a boolean indicating if loading was successful
        return True

    def save_self(self, file: Union[str, IO[bytes]], version: int=1, **kwargs) -> bool:
        return save_object(self, file, version=version, **kwargs)

    def get_hash_str(self, no_id=True):
        class_hash_str = str(type(self))
        args_hash_str = self.dry_args.get_hash_str()
        # Remove dry_id so we can test for object 'class'
        if no_id:
            kwargs_copy = self.dry_kwargs.copy()
            kwargs_copy.pop('dry_id')
            kwargs_hash_str = kwargs_copy.get_hash_str()
        else:
            kwargs_hash_str = self.dry_kwargs.get_hash_str()
        return class_hash_str+args_hash_str+kwargs_hash_str

    def get_hash(self):
        return hash(self.get_hash_str())

class DryObjectFactory(object):
    def __init__(self, cls, *args, callbacks=[], **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.callbacks = callbacks

    def add_callback(callback):
        self.callbacks.append(callback)

    def __call__(self):
        obj = self.cls(*self.args, **self.kwargs) 
        for callback in self.callbacks:
            # Call each callback
            callback(obj)
        return obj
