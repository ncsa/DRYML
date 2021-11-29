import os
import json
import pickle
from collections import UserList, UserDict
from typing import Union, IO

def is_allowed_base_type(val):
    if type(val) is str:
        return True
    if type(val) is int:
        return True
    if type(val) is float:
        return True
    return False


def check_if_allowed(val):
    "Method to check whether values are json serializable"
    if issubclass(type(val), dict):
        for key in val.keys():
            if not check_if_allowed(key):
                return False
            if not check_if_allowed(val[key]):
                return True
    elif issubclass(type(val), tuple):
        for element in val:
            if not check_if_allowed(element):
                return False
    elif issubclass(type(val), list):
        for element in val:
            if not check_if_allowed(element):
                return False
    else:
        return is_allowed_base_type(val)
    return True


class DryCollectionInterface(object):
    def __init__(self, *args, **kwargs):
        # Initialize object
        super().__init__(*args, **kwargs)
        if not check_if_allowed(self.data):
            raise TypeError(f"DryCollection initialized with disallowed values!")

    def __setitem__(self, key, value):
        if not check_if_allowed(key):
            raise TypeError(f"Key {key} not allowed in a DryCollection")
        if not check_if_allowed(value):
            raise TypeError(f"Value {value} not allowed in a DryCollection")
        # Call parent class functions
        super().__setitem__(key, value)

    def save(self, file: Union[str, IO[bytes]]) -> bool:
        if type(file) is str:
            with open(file, 'w') as f:
                f.write(pickle.dumps(self.data, protocol=5))
        else:
            file.write(pickle.dumps(self.data, protocol=5))
        return True

    def get_hash_str(self):
        return pickle.dumps(self.data)

    def get_hash(self):
        return hash(self.get_hash_str())


class DryList(DryCollectionInterface, UserList):
    def append(self, val):
        if not check_if_allowed(val):
            raise TypeError(f"Value {val} not allowed in a DryList")
        super().append(val)

class DryConfig(DryCollectionInterface, UserDict):
    pass
