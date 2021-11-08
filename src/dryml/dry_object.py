# File to define saving/loading convenience functions

import os
import dill
import dbm
import configparser
import io
import zipfile
from typing import Type, IO, Union

# Define a base Dry Object 
class DryObject(object):
    def __init__(self, *args, dry_args={}, dry_kwargs={}, **kwargs)
        super().__init__(*args, **kwargs)
        self.dry_kwargs = dry_kwargs
        self.dry_args = dry_args

    @classmethod
    def load_object_v1(file: Union[str, IO[bytes]]) -> Type[DryObject]:
        return

    @classmethod
    def save_object_v1(obj: Type[DryObject], file: IO[bytes]) -> bool:
        # Meta_data
        meta_data = configparser.ConfigParser()
        meta_data['DRY'] = {
            'dry_version': 1
        }

        config_dbm = io.BytesIO(b"")
        dbm.open(config_dbm, 'w')

        

        return True

    @classmethod
    def load_object(filepath: str) -> Type[DryObject]:
        if not os.path.exists(filepath):
            raise ValueError(f"File {filepath} doesn't exist!")

    @classmethod
    def save_object(obj: Type[DryObject], file: Union[str, IO[bytes]], version: int=1) -> bool:
        # Handle creation of bytes-like
        if type(file) is str:
            file = open(file, 'wb')
        if version == 1:
            return save_object_v1(obj, file)
        else:
            raise ValueError(f"File version {version} unknown. Can't save!")

    def save_self(self, file: Union[str, IO[bytes]], version: int=1) -> bool:
        return DryObject.save_object(self, file, version=version)
