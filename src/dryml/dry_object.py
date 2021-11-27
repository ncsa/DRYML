# File to define saving/loading convenience functions

from __future__ import annotations

import os
import dill
import json
import io
import zipfile
import uuid
from typing import Type, IO, Union
from dryml.dry_config import DryConfig

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

    def load_object_imp(self, file: IO[bytes]):
        # Helper function to load object specific data
        return True

    @staticmethod
    def load_object_v1(file: IO[bytes], update:bool=False) -> Type[DryObject]:
        with file.open('config.json', 'r') as config_file:
            config_data = json.loads(config_file.read())

        # Get arguments
        kwargs = config_data['kwargs']
        args = config_data['args']

        # Get class definition
        with file.open('cls_def.dill') as cls_def_file:
            if update:
                cls_def = dill.loads(cls_def_file.read(), ignore=False)
            else:
                cls_def = dill.loads(cls_def_file.read())

        print(args)
        print(kwargs)

        # Create object
        obj = cls_def(*args, **kwargs)

        # Load object content
        obj.load_object_imp(file)

        # Build object instance
        return obj

    @staticmethod
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

    @staticmethod
    def load_object(file: Union[str, IO[bytes]], update:bool=False, exact_path:bool=False) -> Type[DryObject]:
        if type(file) is str:
            if os.path.splitext(file)[1] == '' and not exact_path:
                file = f"{file}.dry"
            if not os.path.exists(file):
                raise ValueError(f"File {filepath} doesn't exist!")
            file = zipfile.ZipFile(file, mode='r')
        if type(file) is not zipfile.ZipFile:
            file = zipfile.ZipFile(file, mode='r')
        with file.open('meta_data.json', 'r') as meta_file:
            meta_data = json.loads(meta_file.read())
        version = meta_data['version']
        if version == 1:
            result_object = DryObject.load_object_v1(file, update=update)
        else:
            raise RuntimeError(f"DRY version {version} unknown")
        file.close()
        return result_object

    @staticmethod
    def save_object(obj: Type[DryObject], file: Union[str, IO[bytes]], version: int=1, exact_path:bool=False) -> bool:
        # Handle creation of bytes-like
        if type(file) is str:
            if os.path.splitext(file)[1] == '' and not exact_path:
                # No extension specified, add default.
                file = f"{file}.dry"
            file = open(file, 'wb')
        if version == 1:
            return DryObject.save_object_v1(obj, file)
        else:
            raise ValueError(f"File version {version} unknown. Can't save!")

    def save_self(self, file: Union[str, IO[bytes]], version: int=1, **kwargs) -> bool:
        return DryObject.save_object(self, file, version=version, **kwargs)

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
