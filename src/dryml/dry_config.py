import collections
import copy
import abc
from typing import Union, IO, Type, Mapping
from dryml.utils import is_nonstring_iterable, is_dictlike, pickler, \
    get_class_from_str, get_class_str, get_hashed_id


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


class DryConfigInterface(object):
    def __init__(self, *args, **kwargs):
        # Initialize object
        super().__init__(*args, **kwargs)
        if not check_if_allowed(self.data):
            raise TypeError(
                "DryConfig initialized with disallowed values!")

    def __setitem__(self, key, value):
        if not check_if_allowed(key):
            raise TypeError(f"Key {key} not allowed in a DryConfig object")
        if not check_if_allowed(value):
            raise TypeError(f"Value {value} not allowed in a DryConfig object")
        # Call parent class functions
        super().__setitem__(key, value)

    def save(self, file: Union[str, IO[bytes]]) -> bool:
        if type(file) is str:
            with open(file, 'w') as f:
                f.write(pickler(self.data))
        else:
            file.write(pickler(self.data))
        return True


class DryArgs(DryConfigInterface, collections.UserList):
    def append(self, val):
        if not check_if_allowed(val):
            raise TypeError(f"Value {val} not allowed in a DryList")
        super().append(val)


class DryKwargs(DryConfigInterface, collections.UserDict):
    pass


class DryObjectDef(collections.UserDict):
    build_repo = None

    @staticmethod
    def from_dict(def_dict: Mapping):
        return DryObjectDef(
            def_dict['cls'],
            *def_dict.get('dry_args', DryArgs()),
            dry_mut=def_dict.get('dry_mut', False),
            **def_dict.get('dry_kwargs', DryKwargs()))

    def __init__(self, cls: Union[Type, str],
                 *args, dry_mut: bool = False, **kwargs):
        super().__init__()
        self.cls = cls
        self.data['dry_mut'] = dry_mut
        self.args = args
        self.kwargs = kwargs

    @property
    def cls(self):
        return self['cls']

    @cls.setter
    def cls(self, val: Union[Type, str]):
        self['cls'] = val

    @property
    def dry_mut(self):
        return self['dry_mut']

    @property
    def args(self):
        return self['dry_args']

    @args.setter
    def args(self, value):
        self['dry_args'] = value

    @property
    def kwargs(self):
        return self['dry_kwargs']

    @kwargs.setter
    def kwargs(self, value):
        self['dry_kwargs'] = value

    def __setitem__(self, key, value):
        if key not in ['cls', 'dry_args', 'dry_kwargs']:
            raise ValueError(
                f"Setting Key {key} not supported by DryObjectDef")

        if key == 'cls':
            if type(value) is type or type(value) is abc.ABCMeta:
                self.data['cls'] = value
            elif type(value) is str:
                self.data['cls'] = get_class_from_str(value)
            else:
                raise TypeError(
                    f"Value of type {type(value)} not supported "
                    "for class assignment!")

        if key == 'dry_args':
            self.data['dry_args'] = DryArgs(value)

        if key == 'dry_kwargs':
            self.data['dry_kwargs'] = DryKwargs(value)

    def to_dict(self, cls_str: bool = False):
        return {
            'cls': self.cls if not cls_str else get_class_str(self.cls),
            'dry_mut': self.dry_mut,
            'dry_args': self.args.data,
            'dry_kwargs': self.kwargs.data,
        }

    def build(self, repo=None):
        "Construct an object"
        reset_repo = False
        construct_object = True

        if repo is not None:
            if DryObjectDef.build_repo is not None:
                raise RuntimeError(
                    "different repos not currently supported")
            else:
                # Set the call_repo
                DryObjectDef.build_repo = repo
                reset_repo = True

        # Check whether a repo was given in a prior call
        if DryObjectDef.build_repo is not None:
            try:
                obj = DryObjectDef.build_repo.get_obj(self)
                construct_object = False
            except Exception:
                pass

        if construct_object:
            obj = self.cls(*self.args, **self.kwargs)

        # Reset the repo for this function
        if reset_repo:
            DryObjectDef.build_repo = None

        # Return the result
        return obj

    def get_cat_def(self):
        def_dict = self.to_dict()
        kwargs_copy = copy.copy(def_dict['dry_kwargs'].data)
        if 'dry_id' in kwargs_copy:
            kwargs_copy.pop('dry_id')
        def_dict['dry_kwargs'] = kwargs_copy
        return DryObjectDef.from_dict(def_dict)

    def get_hash_str(self, no_id: bool = False):
        class_hash_str = get_class_str(self.cls)
        args_hash_str = str(self.args.data)
        # Remove dry_id so we can test for object 'class'
        if no_id:
            kwargs_copy = copy.copy(self.kwargs.data)
            if 'dry_id' in kwargs_copy:
                kwargs_copy.pop('dry_id')
            kwargs_hash_str = str(kwargs_copy)
        else:
            kwargs_hash_str = str(self.kwargs.data)
        return class_hash_str+args_hash_str+kwargs_hash_str

    def __hash__(self):
        if 'dry_id' not in self.kwargs:
            raise RuntimeError(
                "Tried to make an individual hash on a "
                "DryObjectDef without a dry_id!")
        return hash(self.get_individual_id())

    def get_individual_id(self):
        if 'dry_id' not in self.kwargs:
            raise RuntimeError(
                "Tried to make an individual id on a "
                "DryObjectDef without a dry_id!")
        return get_hashed_id(self.get_hash_str(no_id=False))

    def get_category_id(self):
        return get_hashed_id(self.get_hash_str(no_id=True))
