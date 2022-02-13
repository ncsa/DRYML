import collections
import copy
import abc
import inspect
import functools
import zipfile
from typing import Union, IO, Type, Mapping
from dryml.utils import is_nonstring_iterable, is_dictlike, pickler, \
    get_class_from_str, get_class_str, get_hashed_id, init_arg_list_handler, \
    init_arg_dict_handler


class DryMeta(abc.ABCMeta):
    def __new__(cls, clsname, bases, attrs):
        # Create class
        new_cls = super().__new__(cls, clsname, bases, attrs)

        # Get init function
        init_func = new_cls.__init__

        # Run check for self
        DryMeta.check_for_self(init_func, clsname)

        # Detect if we have the base class.
        base = False
        if '__dry_meta_base__' in attrs:
            base = True

        new_cls.__init__ = DryMeta.make_dry_init(new_cls, init_func, base=base)
        new_cls.load_object = DryMeta.make_load_object(new_cls)
        new_cls.save_object = DryMeta.make_save_object(new_cls)

        return new_cls

    @staticmethod
    def check_for_self(f, clsname):
        sig = inspect.signature(f)
        keys = list(sig.parameters.keys())
        first_par = sig.parameters[keys[0]]
        if first_par.name != 'self':
            raise RuntimeError(
                "__init__ signature of DryMeta Objects must have a first "
                f"argument named self. Found when initializing type {clsname}")

    @staticmethod
    def track_args(f):
        """
        Build a list of explicitly mentioned arguments which will be
        consumed by dry args systems
        """
        # Get function signature
        sig = inspect.signature(f)

        # Build list of positional and keyword arguments for dry_init
        track_args = getattr(f, '__dry_args__', [])
        track_kwargs = getattr(f, '__dry_kwargs__', [])
        for key in sig.parameters:
            par = sig.parameters[key]
            if par.name == 'self':
                continue
            if par.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            if par.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            if par.default == inspect.Parameter.empty and \
               par.kind != inspect.Parameter.KEYWORD_ONLY:
                track_args.append(par.name)
            else:
                track_kwargs.append((par.name, par.default))

        f.__dry_args__ = track_args
        f.__dry_kwargs__ = track_kwargs

        return f

    @staticmethod
    def skip_args(f):
        f.__dry_skip_args__ = True
        return f

    @staticmethod
    def collect_args(f):
        f.__dry_collect_args__ = True
        return f

    @staticmethod
    def collect_kwargs(f):
        f.__dry_collect_kwargs__ = True
        return f

    # Create scope with __class__ defined so super can find a cell
    # with the right name.
    # Python issue: https://bugs.python.org/issue29944
    @staticmethod
    def make_dry_init(__class__, init_func, base=False):
        # Track arguments
        init_func = DryMeta.track_args(init_func)

        @functools.wraps(init_func)
        def dry_init(self, *args, dry_args=None, dry_kwargs=None, **kwargs):
            # First, preprocess the arguments
            if hasattr(__class__, 'args_preprocess'):
                (pargs, pkwargs) = __class__.args_preprocess(
                    self, *args, dry_args=None, dry_kwargs=None, **kwargs)
            else:
                pargs = args
                pkwargs = kwargs

            # Initialize dry arguments
            dry_args = init_arg_list_handler(dry_args)
            dry_kwargs = init_arg_dict_handler(dry_kwargs)

            # grab args/kwargs into dry_args/dry_kwargs
            skip_args = False
            if hasattr(init_func, '__dry_skip_args__'):
                if init_func.__dry_skip_args__:
                    skip_args = True
            collect_args = False
            if hasattr(init_func, '__dry_collect_args__'):
                if init_func.__dry_collect_args__:
                    collect_args = True
            if skip_args and collect_args:
                raise ValueError(
                    "Cannot set both __dry_skip_args__ and "
                    "__dry_collect_args__ on an __init__ function.")
            if not skip_args:
                if collect_args:
                    # Collect all the arguments
                    num_args = len(pargs)
                else:
                    num_args = len(init_func.__dry_args__)
                for i in range(num_args):
                    dry_args.append(pargs[i])
            collect_kwargs = False
            used_kwargs = []
            exception_kwargs = ['dry_args', 'dry_kwargs', 'dry_id']
            if hasattr(init_func, '__dry_collect_kwargs__'):
                if init_func.__dry_collect_kwargs__:
                    collect_kwargs = True
            if collect_kwargs:
                for k, v in pkwargs.items():
                    if k not in exception_kwargs:
                        # Get all the arguments
                        dry_kwargs[k] = v
                        used_kwargs.append(k)
            else:
                for k, v in init_func.__dry_kwargs__:
                    dry_kwargs[k] = pkwargs.get(k, v)
                    used_kwargs.append(k)

            # Grab unaltered arguments to pass to super
            if not skip_args:
                super_args = args[num_args:]
            else:
                super_args = []
            used_kwargs += ['dry_args', 'dry_kwargs']
            super_kwargs = {
                k: v for k, v in kwargs.items() if k not in used_kwargs
            }

            if base:
                # At the base, we need to set dry_args/dry_kwargs finally
                # Use DryKwargs/DryArgs object to coerse args/kwargs to proper
                # json serializable form.
                self.dry_args = DryArgs(dry_args)
                self.dry_kwargs = DryKwargs(dry_kwargs)

                # Construct parents
                super().__init__(
                    *super_args,
                    **super_kwargs,
                )
            else:
                # Construct parents
                super().__init__(
                    *super_args,
                    dry_args=dry_args,
                    dry_kwargs=dry_kwargs,
                    **super_kwargs,
                )

            # Execute user init
            # Here we make sure to remove special arguments
            used_kwargs = ['dry_args', 'dry_kwargs', 'dry_id']
            sub_kwargs = { k: v for k, v in kwargs.items() if k not in used_kwargs }
            init_func(self, *args, **sub_kwargs)

        return dry_init

    @staticmethod
    def make_load_object(__class__):
        """
        Method for making a load_object function
        """
        def load_object(self, file: zipfile.ZipFile) -> bool:
            if not hasattr(__class__, '__dry_meta_base__'):
                # If we're not the base, call the super class's load.
                super().load_object(self, file)
            if hasattr(__class__, 'load_object_imp'):
                if not __class__.load_object_imp(self, file):
                    return False
            return True
        return load_object

    @staticmethod
    def make_save_object(__class__):
        """
        Method for making a save_object function
        """
        def save_object(self, file: zipfile.ZipFile) -> bool:
            if hasattr(__class__, 'save_object_imp'):
                if not __class__.save_object_imp(self, file):
                    return False
            if not hasattr(__class__, '__dry_meta_base__'):
                # If we're not the base, call the super class's load.
                super().save_object(self, file)
            return True
        return save_object


def is_allowed_value_type(val):
    if type(val) in (str, bytes, int, float):
        return True
    if type(val) is type:
        return True
    if val is None:
        return True
    if issubclass(type(val), DryMeta):
        return True
    print(f"Value {val} is not an allowed value type.")
    return False


def is_allowed_key_type(val):
    if type(val) in (str, bytes, int, float):
        return True
    if type(val) is tuple:
        for el in val:
            if not is_allowed_key_type(el):
                print(f"element {el} within {val} is not an allowed key type.")
                return False
        return True
    print(f"Key {val} is not an allowed key type.")
    return False


def check_if_allowed(val):
    "Method to check whether values are json serializable"
    if is_dictlike(val):
        for key in val.keys():
            if not is_allowed_key_type(key):
                return False
            if not check_if_allowed(val[key]):
                return True
    elif is_nonstring_iterable(val):
        for element in val:
            if not check_if_allowed(element):
                return False
    else:
        return is_allowed_value_type(val)
    return True


def adapt_val(val):
    # we need to turn DryMeta types into definitions
    if issubclass(type(val), DryMeta):
        return val.definition().to_dict()
    return val


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
        super().__setitem__(key, adapt_val(value))

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
            raise TypeError(f"Value {val} not allowed in DryArgs")
        super().append(adapt_val(val))


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
            if type(value) is type or issubclass(type(value), abc.ABCMeta):
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
            'dry_def': True,
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
        kwargs_copy = copy.copy(def_dict['dry_kwargs'])
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
