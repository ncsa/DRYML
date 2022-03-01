import collections
import copy
import abc
import inspect
import functools
import zipfile
import io
from typing import Union, IO, Type, Mapping
from dryml.utils import is_nonstring_iterable, is_dictlike, pickler, \
    get_class_from_str, get_class_str, get_hashed_id, init_arg_list_handler, \
    init_arg_dict_handler
from dryml.context.context_tracker import get_context_class, \
    WrongContextError, context, NoContextError


class IncompleteDefinitionError(Exception):
    pass


class ComputeModeAlreadyActiveError(Exception):
    pass


class ComputeModeLoadError(Exception):
    pass


class ComputeModeNotActiveError(Exception):
    pass


class ComputeModeSaveError(Exception):
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
        # Set default init function as python's default
        # init tries to use super which shouldn't be done
        # here.
        if '__init__' not in attrs:
            attrs['__init__'] = DryMeta.default_init

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

        # Set new methods which need the class object
        # to be set properly
        new_cls.__init__ = DryMeta.make_dry_init(new_cls, init_func, base=base)
        new_cls.load_object = DryMeta.make_load_object(new_cls)
        new_cls.save_object = DryMeta.make_save_object(new_cls)
        new_cls.load_compute = DryMeta.make_load_compute(new_cls)
        new_cls.save_compute = DryMeta.make_save_compute(new_cls)
        new_cls.compute_prepare = DryMeta.make_compute_prepare(new_cls)
        new_cls.compute_cleanup = DryMeta.make_compute_cleanup(new_cls)

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
        existing_kwargs = list(map(lambda t: t[0], track_kwargs))
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
                if par.name not in existing_kwargs:
                    track_kwargs.append((par.name, par.default))

        f.__dry_args__ = track_args
        f.__dry_kwargs__ = track_kwargs

        return f

    @staticmethod
    def collect_args(f):
        f.__dry_collect_args__ = True
        return f

    @staticmethod
    def default_init(self, *args, **kwargs):
        pass

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

            # Initialize compute data holder
            if not hasattr(self, '__dry_compute_data__'):
                self.__dry_compute_data__ = None

            # Initialize compute mode indicator
            if not hasattr(self, '__dry_compute_mode__'):
                self.__dry_compute_mode__ = False

            # manage whether to collect all arguments
            collect_args = False
            if hasattr(init_func, '__dry_collect_args__'):
                if init_func.__dry_collect_args__:
                    collect_args = True

            # Determine how many arguments to collect
            if collect_args:
                # Collect all the arguments
                num_args = len(args)
            else:
                num_args = len(init_func.__dry_args__)

            # Collect arguments
            for i in range(num_args):
                dry_args.append(args[i])

            # Collect keyword arguments
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

            # Call user defined init
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
                super().load_object(file)
            if hasattr(__class__, 'load_imp'):
                if not __class__.load_imp(self, file):
                    return False
            # Load compute data
            compute_data_path = 'compute_data.zip'
            if compute_data_path in file.namelist():
                with file.open(compute_data_path, 'r') as f:
                    self.__dry_compute_data__ = io.BytesIO(f.read())
            else:
                self.__dry_compute_data__ = None
            return True
        return load_object

    @staticmethod
    def make_save_object(__class__):
        """
        Method for making a save_object function
        """
        def save_object(self, file: zipfile.ZipFile) -> bool:
            if hasattr(__class__, 'save_imp'):
                if not __class__.save_imp(self, file):
                    return False
            # Save any compute data from compute components.
            # If compute mode is active
            if self.__dry_compute_mode__:
                self.save_compute()
            # Save compute data if it's there
            if self.__dry_compute_data__ is not None:
                compute_data_path = 'compute_data.zip'
                data_buff = self.__dry_compute_data__
                # Seek to beginning of io stream before read
                data_buff.seek(0)
                with file.open(compute_data_path, 'w') as f:
                    f.write(data_buff.read())

            # Save contained dry objects passed as arguments to construct
            for obj in self.__dry_obj_container_list__:
                obj_id = obj.definition().get_individual_id()
                save_path = f'dry_objects/{obj_id}.dry'
                if save_path not in file.namelist():
                    with file.open(save_path, 'w') as f:
                        if not obj.save_self(f):
                            return False
            if not hasattr(__class__, '__dry_meta_base__'):
                # If we're not the base, call the super class's save.
                super().save_object(file)
            return True
        return save_object

    @staticmethod
    def make_compute_prepare(__class__):
        """
        Method for making compute_prepare function
        """

        def compute_prepare(self):
            if self.__dry_compute_mode__:
                raise ComputeModeAlreadyActiveError()

            # Prepare contained objects
            for obj in self.__dry_obj_container_list__:
                obj.compute_prepare()

            # Prepare self
            if not hasattr(__class__, '__dry_meta_base__'):
                # Call the super class's compute_prepare
                super().compute_prepare()

            # Check required context for this class
            required_context_name = 'default'
            if hasattr(self, '__dry_compute_context__'):
                required_context_name = self.__dry_compute_context__
                required_ctx_cls = get_context_class(required_context_name)
                current_ctx = context()

                if current_ctx is None:
                    raise NoContextError("There is no context active.")
                else:
                    if not issubclass(type(current_ctx), required_ctx_cls):
                        raise WrongContextError(
                            f"{current_ctx} doesn't satisfy "
                            f"requirement {required_ctx_cls}")

            # Execute user compute prepare method
            if hasattr(self, 'compute_prepare_imp'):
                self.compute_prepare_imp()

            self.__dry_compute_mode__ = True

        return compute_prepare

    @staticmethod
    def make_compute_cleanup(__class__):
        """
        Method for making compute_cleanup function
        """

        def compute_cleanup(self):
            if not self.__dry_compute_mode__:
                raise ComputeModeNotActiveError()

            # Cleanup this object
            if hasattr(self, 'compute_cleanup_imp'):
                self.compute_cleanup_imp()

            # Cleanup contained objects
            for obj in self.__dry_obj_container_list__:
                obj.compute_cleanup()

            # Cleanup parent classes
            if not hasattr(__class__, '__dry_meta_base__'):
                # If we're not the base, call the super class's save.
                super().compute_cleanup()

            self.__dry_compute_mode__ = False

        return compute_cleanup

    @staticmethod
    def make_load_compute(__class__):
        """
        Method for making a load_compute function
        """

        def load_compute(self) -> bool:
            if not hasattr(__class__, '__dry_meta_base__'):
                # If we're not the base, call the super class's load.
                super().load_compute()
            # Load our contained objects
            for obj in self.__dry_obj_container_list__:
                if not obj.load_compute():
                    return False
            # Load this object's compute
            if hasattr(__class__, 'load_compute_imp'):
                if self.__dry_compute_data__ is not None:
                    # Need to seek the io stream to the beginning
                    self.__dry_compute_data__.seek(0)
                    with zipfile.ZipFile(
                            self.__dry_compute_data__, mode='r') as zf:
                        if not __class__.load_compute_imp(
                                self, zf):
                            return False
            return True
        return load_compute

    @staticmethod
    def make_save_compute(__class__):
        """
        Method for making a save_compute function
        """
        def save_compute(self) -> bool:
            if hasattr(__class__, 'save_compute_imp'):
                self.__dry_compute_data__ = io.BytesIO()
                with zipfile.ZipFile(
                        self.__dry_compute_data__, mode='w') as zf:
                    if not __class__.save_compute_imp(
                            self, zf):
                        return False
            # Save contained dry objects passed as arguments to construct
            for obj in self.__dry_obj_container_list__:
                if not obj.save_compute():
                    return False
            if not hasattr(__class__, '__dry_meta_base__'):
                # If we're not the base, call the super class's load.
                super().save_compute()
            return True
        return save_compute


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
