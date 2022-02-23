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


class IncompleteDefinitionError(Exception):
    pass


# Create a couple global variables
build_repo = None
build_cache = None


# Detect and Construct dryobjects as we encounter them.
def detect_and_construct(val, load_zip=None):
    from dryml.dry_object import DryObject
    if isinstance(val, DryObject):
        # If we already have a DryObject, do nothing
        return val
    elif isinstance(val, DryObjectDef):
        # We have an object definition. Build the object.
        return val.build(load_zip=load_zip)
    if is_dictlike(val):
        # We might have an object definition
        if 'dry_def' in val:
            if val['dry_def']:
                # We have a dry object definition.
                val_def = DryObjectDef.from_dict(val)
                return val_def.build(load_zip=load_zip)
        # Otherwise, we have a mapping, and need to build its content
        dict_val = {k: detect_and_construct(v, load_zip=load_zip)
                    for k, v in val.items()}
        return dict_val
    elif is_nonstring_iterable(val):
        if type(val) is tuple:
            # We have a list-like item, and must build its contents
            return tuple(map(
                lambda v: detect_and_construct(v, load_zip=load_zip), val))
        elif type(val) is list or type(val) is DryArgs:
            # We have a list-like item, and must build its contents
            return list(map(
                lambda v: detect_and_construct(v, load_zip=load_zip), val))
        else:
            raise ValueError(
                f"Unsupported iterable type {type(val)}!")
    else:
        # We have some other value, and assume it can be passed directly.
        # Such as 'int, float, etc...'
        return val


def is_concrete_val(input_object):
    from dryml import DryObject
    # Is this object a dry definition?
    if isinstance(input_object, DryObject):
        # A DryObject itself is a concrete value
        return True
    elif isinstance(input_object, DryObjectDef) or \
            is_dictlike(input_object) and 'dry_def' in input_object:
        # Check that there's a Dry ID here.
        if 'dry_id' not in input_object['dry_kwargs']:
            return False
        # Check the args
        for arg in input_object['dry_args']:
            if not is_concrete_val(arg):
                return False
        # Check the kwargs
        for key in input_object['dry_kwargs']:
            kwarg = input_object['dry_kwargs'][key]
            if not is_concrete_val(kwarg):
                return False
    elif is_nonstring_iterable(input_object):
        for obj in input_object:
            if not is_concrete_val(obj):
                return False
    elif is_dictlike(input_object):
        for key in input_object:
            val = input_object[key]
            if not is_concrete_val(val):
                return False
    elif callable(input_object):
        # Callables aren't concrete. They don't have a definite value
        return False
    # Assuming all other types are concrete for now.
    return True


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
    def collect_args(f):
        f.__dry_collect_args__ = True
        return f

    # Create scope with __class__ defined so super can find a cell
    # with the right name.
    # Python issue: https://bugs.python.org/issue29944
    @staticmethod
    def make_dry_init(__class__, init_func, base=False):
        # Track arguments
        init_func = DryMeta.track_args(init_func)
        sig = inspect.signature(init_func)

        # Detect var signature components
        no_var_pars = False
        no_final_kwargs = False
        last_par_kind = None
        for key in sig.parameters:
            par = sig.parameters[key]
            last_par_kind = par.kind
            if par.kind == inspect.Parameter.VAR_POSITIONAL:
                no_var_pars = True
        if last_par_kind != inspect.Parameter.VAR_KEYWORD:
            no_final_kwargs = True

        @functools.wraps(init_func)
        def dry_init(self, *args, dry_args=None, dry_kwargs=None, **kwargs):
            # Initialize dry arguments
            dry_args = init_arg_list_handler(dry_args)
            dry_kwargs = init_arg_dict_handler(dry_kwargs)

            # grab all args into dry_args
            collect_args = False
            if hasattr(init_func, '__dry_collect_args__'):
                if init_func.__dry_collect_args__:
                    collect_args = True
            if collect_args:
                # Collect all the arguments
                num_args = len(args)
            else:
                num_args = len(init_func.__dry_args__)
            for i in range(num_args):
                dry_args.append(args[i])

            used_kwargs = []
            for k, v in init_func.__dry_kwargs__:
                dry_kwargs[k] = kwargs.get(k, v)
                used_kwargs.append(k)

            # Grab unaltered arguments to pass to super
            super_args = args[num_args:]
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
            used_kwargs = ['dry_args', 'dry_kwargs']
            if no_final_kwargs:
                known_kwargs = list(map(
                    lambda t: t[0],
                    init_func.__dry_kwargs__))
                sub_kwargs = {
                    k: detect_and_construct(v) for k, v in kwargs.items()
                    if k in known_kwargs}
            else:
                sub_kwargs = {
                    k: detect_and_construct(v) for k, v in kwargs.items()
                    if k not in used_kwargs}
            if no_var_pars:
                args = args[:num_args]
            # Make sure we process args.
            args = list(map(detect_and_construct, args))

            # Store list of DryObjects we need to later save.
            self.__dry_obj_container_list__ = []
            from dryml import DryObject
            for arg in args:
                if isinstance(arg, DryObject):
                    self.__dry_obj_container_list__.append(arg)
            for name in sub_kwargs:
                obj = sub_kwargs[name]
                if isinstance(obj, DryObject):
                    self.__dry_obj_container_list__.append(obj)

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
            # Save contained dry objects passed as arguments to construct
            for obj in self.__dry_obj_container_list__:
                obj_id = obj.definition().get_individual_id()
                save_path = f'dry_objects/{obj_id}.dry'
                if save_path not in file.namelist():
                    with file.open(save_path, 'w') as f:
                        if not obj.save_self(f):
                            return False
            if not hasattr(__class__, '__dry_meta_base__'):
                # If we're not the base, call the super class's load.
                super().save_object(self, file)
            return True
        return save_object


def adapt_key(val):
    if type(val) in (str, bytes, int, float):
        return val
    if type(val) is tuple:
        return tuple(map(adapt_key, val))
    raise ValueError(f"Key {val} not supported by Dry Configuration")


def adapt_val(val):
    if type(val) in (str, bytes, int, float, bool):
        return val
    if type(val) is type:
        return val
    if val is None:
        return val
    if type(val) is DryMeta:
        return val
    from dryml import DryObject
    if issubclass(type(val), DryObject):
        return val.definition().to_dict()
    if issubclass(type(val), DryObjectDef):
        # Handle DryObjectDef, otherwise it'll get mangled
        return val.to_dict()
    if type(val) is tuple:
        adjusted_value = list(map(adapt_val, val))
        return tuple(adjusted_value)
    if is_dictlike(val):
        adjusted_value = {adapt_key(k): adapt_val(v) for k, v in val.items()}
        return adjusted_value
    if is_nonstring_iterable(val):
        adjusted_value = list(map(adapt_val, val))
        return adjusted_value
    raise ValueError(f"value {val} not supported by Dry Configuration")


class DryConfigInterface(object):
    def __init__(self, *args, **kwargs):
        # Initialize object
        adapted_args = adapt_val(args)
        adapted_kwargs = adapt_val(kwargs)
        super().__init__(*adapted_args, **adapted_kwargs)

    def __setitem__(self, key, value):
        adapted_key = adapt_key(key)
        adapted_val = adapt_val(value)
        # Call parent class functions
        super().__setitem__(adapted_key, adapted_val)

    def save(self, file: Union[str, IO[bytes]]) -> bool:
        if type(file) is str:
            with open(file, 'w') as f:
                f.write(pickler(self.data))
        else:
            file.write(pickler(self.data))
        return True


class DryArgs(DryConfigInterface, collections.UserList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.data)):
            self.data[i] = adapt_val(self.data[i])

    def append(self, val):
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

    def build(self, repo=None, load_zip=None):
        "Construct an object"
        reset_repo = False
        reset_cache = False
        construct_object = True

        global build_repo
        global build_cache

        if repo is not None:
            if build_repo is not None:
                raise RuntimeError(
                    "different repos not currently supported")
            else:
                # Set the call_repo
                build_repo = repo
                reset_repo = True

        if build_cache is None:
            build_cache = {}
            reset_cache = True

        construction_required = True
        try:
            obj_id = self.get_individual_id()
            construction_required = False
        except IncompleteDefinitionError:
            pass

        # Check the cache
        if not construction_required and build_cache is not None:
            if obj_id in build_cache:
                obj = build_cache[obj_id]
                construct_object = False

        # Check the repo
        if not construction_required and build_repo is not None:
            try:
                obj = build_repo.get_obj(self)
                build_cache[obj_id] = obj
                construct_object = False
            except KeyError:
                # Didn't find the object in the repo
                pass

        # Check the zipfile
        if not construction_required and \
                construct_object and load_zip is not None:
            target_filename = f"dry_objects/{obj_id}.dry"
            from dryml import load_object
            if target_filename in load_zip.namelist():
                with load_zip.open(target_filename) as f:
                    obj = load_object(f)
                    build_cache[obj_id] = obj
                    construct_object = False

        # Finally, actually construct the object
        if construct_object:
            args = detect_and_construct(self.args, load_zip=load_zip)
            kwargs = detect_and_construct(self.kwargs, load_zip=load_zip)

            obj = self.cls(*args, **kwargs)

        # Reset the repo for this function
        if reset_repo:
            build_repo = None

        # Reset the build cache
        if reset_cache:
            build_cache = None

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

    def is_concrete(self):
        return is_concrete_val(self)

    def __hash__(self):
        if not self.is_concrete():
            raise IncompleteDefinitionError(
                "Definition {self} has no dry_id!")
        return hash(self.get_individual_id())

    def get_individual_id(self):
        if not self.is_concrete():
            raise IncompleteDefinitionError(
                "Definition {self} has no dry_id!")
        return get_hashed_id(self.get_hash_str(no_id=False))

    def get_category_id(self):
        return get_hashed_id(self.get_hash_str(no_id=True))
