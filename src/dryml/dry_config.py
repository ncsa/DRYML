import collections
from typing import Union, IO
from dryml.utils import is_nonstring_iterable, is_dictlike, pickler


def is_allowed_base_type(val):
    if type(val) in (str, bytes, int, float):
        return True
    return False


def check_if_allowed(val):
    "Method to check whether values are json serializable"
    if is_dictlike(val):
        for key in val.keys():
            if not check_if_allowed(key):
                return False
            if not check_if_allowed(val[key]):
                return True
    elif is_nonstring_iterable(val):
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
            raise TypeError(
                "DryCollection initialized with disallowed values!")

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
                f.write(pickler(self.data))
        else:
            file.write(pickler(self.data))
        return True

    def get_hash_str(self):
        return str(self.data)

    def get_hash(self):
        return hash(self.get_hash_str())


class DryArgs(DryCollectionInterface, collections.UserList):
    def append(self, val):
        if not check_if_allowed(val):
            raise TypeError(f"Value {val} not allowed in a DryList")
        super().append(val)


class DryKwargs(DryCollectionInterface, collections.UserDict):
    pass
