import collections
import copy
import abc
import inspect
import functools
import zipfile
from typing import Union, Type, Mapping
from dryml.core.utils import is_nonstring_iterable, is_dictlike, \
    get_class_from_str, get_class_str, get_hashed_id, init_arg_list_handler, \
    init_arg_dict_handler, is_supported_scalar_type, is_supported_listlike, \
    is_supported_dictlike, map_dictlike, map_listlike, equal_recursive
from dryml.context.context_tracker import WrongContextError, \
    context, NoContextError
from dryml.context.process import compute_context
from dryml.core.save_cache import SaveCache
from dryml.core.file_intermediary import FileIntermediary
import uuid


class MissingIdError(Exception):
    pass


class MissingMetadataError(Exception):
    pass


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


class ExpectedArgumentError(Exception):
    pass


class BuildStratTracker(object):
    def __init__(self):
        self.tracker = {}

    def __getitem__(self, dry_id):
        if dry_id not in self.tracker:
            self.tracker[dry_id] = set()
        return self.tracker[dry_id]

    def __repr__(self):
        return f"{self.tracker}"


# Create a couple global variables
build_repo = None
build_cache = None
build_verbose = None
def_cache = None
build_strat = None


def is_concrete_val(input_object):
    from dryml import Object
    # Is this object a dry definition?
    if isinstance(input_object, Object):
        # A Object itself is a concrete value
        return True
    elif isinstance(input_object, ObjectDef) or \
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
    elif issubclass(type(input_object), type):
        # We have a type which is 'callable'
        return True
    elif callable(input_object):
        # Callables aren't concrete. They don't have a definite value
        return False
    # Assuming all other types are concrete for now.
    return True


class Meta(abc.ABCMeta):
    def __new__(cls, clsname, bases, attrs):
        # Set default init function as python's default
        # init tries to use super which shouldn't be done
        # here.
        if '__init__' not in attrs:
            attrs['__init__'] = Meta.default_init

        # Create class
        new_cls = super().__new__(cls, clsname, bases, attrs)

        # Get init function
        init_func = new_cls.__init__

        # Run check for self
        Meta.check_for_self(init_func, clsname)

        # Detect if we have the base class.
        base = False
        if '__dry_meta_base__' in attrs:
            base = True

        # Set new methods which need the class object
        # to be set properly
        new_cls.__init__ = Meta.make_dry_init(new_cls, init_func, base=base)
        new_cls.load_object = Meta.make_load_object(new_cls)
        new_cls.save_object = Meta.make_save_object(new_cls)
        new_cls.load_compute = Meta.make_load_compute(new_cls)
        new_cls.save_compute = Meta.make_save_compute(new_cls)
        new_cls.compute_prepare = Meta.make_compute_prepare(new_cls)
        new_cls.compute_cleanup = Meta.make_compute_cleanup(new_cls)

        # Wrap marked compute methods
        if hasattr(new_cls, '__dry_compute_methods__'):
            for (attr, ctx_kwargs) in new_cls.__dry_compute_methods__:
                if hasattr(new_cls, attr):
                    method = getattr(new_cls, attr)
                    if 'ctx_dont_create_context' not in ctx_kwargs:
                        ctx_kwargs['ctx_dont_create_context'] = True
                    setattr(
                        new_cls, attr,
                        compute_context(**ctx_kwargs)(method))

        return new_cls

    @staticmethod
    def check_for_self(f, clsname):
        sig = inspect.signature(f)
        keys = list(sig.parameters.keys())
        first_par = sig.parameters[keys[0]]
        if first_par.name != 'self':
            raise RuntimeError(
                "__init__ signature of Meta Objects must have a first "
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
    def collect_kwargs(f):
        f.__dry_collect_kwargs__ = True
        return f

    @staticmethod
    def mark_compute(f):
        f.__dry_compute_mark__ = True
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
        init_func = Meta.track_args(init_func)
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

            # Initialize compute context indicator
            if not hasattr(self, '__dry_compute_context__'):
                self.__dry_compute_context__ = 'default'

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

            if num_args > len(args):
                raise ExpectedArgumentError(
                    f"Object class {__class__} init expects {num_args} "
                    f"positional arguments, got {len(args)}. Did you forget "
                    f"to specify a required positional argument?")

            for i in range(num_args):
                dry_args.append(args[i])

            collect_kwargs = False
            if hasattr(init_func, '__dry_collect_kwargs__'):
                if init_func.__dry_collect_kwargs__:
                    collect_kwargs = True

            # Collect keyword arguments and save into dry_kwargs
            used_kwargs = []
            for k, v in init_func.__dry_kwargs__:
                # Need to use .get since we are passing a default (v).
                dry_kwargs[k] = kwargs.get(k, v)
                used_kwargs.append(k)

            # Collect remaining kwargs if needed.
            dont_collect_kwargs = ['dry_id', 'dry_metadata']
            if collect_kwargs:
                for k in kwargs:
                    if k not in dont_collect_kwargs:
                        if k not in used_kwargs:
                            dry_kwargs[k] = kwargs[k]
                            used_kwargs.append(k)

            # Grab unaltered arguments to pass to super
            super_args = args[num_args:]
            used_kwargs += ['dry_args', 'dry_kwargs']
            super_kwargs = {
                k: v for k, v in kwargs.items() if k not in used_kwargs
            }

            if base:
                # At the base, we need to validate the dry args
                # and dry kwargs
                validate_val_obj(dry_args)
                validate_val_obj(dry_kwargs)

                self.dry_args = tuple(dry_args)
                self.dry_kwargs = dry_kwargs

                # Construct parents
                try:
                    super().__init__(
                        *super_args,
                        **super_kwargs,
                    )
                except TypeError as e:
                    print(f"error constructing possibly 'object' parent. args:"
                          f" {super_args} kwargs: {super_kwargs}")
                    raise e
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
                    k: v for k, v in kwargs.items() if k in known_kwargs}
            else:
                sub_kwargs = {
                    k: v for k, v in kwargs.items() if k not in used_kwargs}

            # Remove dry_id from being used in non-base constructors.
            # Kludge solution.
            if not base:
                if 'dry_id' in sub_kwargs:
                    del sub_kwargs['dry_id']
                if 'dry_metadata' in sub_kwargs:
                    del sub_kwargs['dry_metadata']

            if no_var_pars:
                args = args[:num_args]

            # Store list of Objects we need to later save.
            if not hasattr(self, '__dry_obj_container_list__'):
                self.__dry_obj_container_list__ = []

            def _add_dry_objs(el):
                from dryml import Object
                if isinstance(el, Object):
                    self.__dry_obj_container_list__.append(el)
                if is_nonstring_iterable(el):
                    for elm in el:
                        _add_dry_objs(elm)
                if is_dictlike(el):
                    for key in el:
                        elm = el[key]
                        _add_dry_objs(elm)

            _add_dry_objs(args)
            _add_dry_objs(sub_kwargs)

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
                # We're not the base, call the super class's load.
                if not super().load_object(file):
                    print("issue with super class load")
                    return False
            else:
                # Load compute data at the base.
                compute_data_path = 'compute_data.zip'
                if self.__dry_compute_data__ is not None:
                    del self.__dry_compute_data__
                if compute_data_path in file.namelist():
                    with file.open(compute_data_path, 'r') as f:
                        new_compute_data = FileIntermediary()
                        new_compute_data.write(f.read())
                        self.__dry_compute_data__ = new_compute_data
                else:
                    self.__dry_compute_data__ = None

            # Call this class's load object.
            if hasattr(__class__, 'load_object_imp'):
                retval = __class__.load_object_imp(self, file)
                if type(retval) is not bool:
                    raise TypeError("load_object_imp must return a bool.")
                if not retval:
                    print("issue with object load implementation")
                    return False
            return True
        return load_object

    @staticmethod
    def make_save_object(__class__):
        """
        Method for making a save_object function
        """
        def save_object(self, file: zipfile.ZipFile, save_cache=None) -> bool:
            if hasattr(__class__, '__dry_meta_base__'):
                # We're at the base, so load the compute data.
                # Save any compute data from compute components.
                # If compute mode is active
                if self.__dry_compute_mode__:
                    self.save_compute(save_cache=save_cache)

                # Save compute data if it's there
                if self.__dry_compute_data__ is not None:
                    compute_data_path = 'compute_data.zip'
                    data_buff = self.__dry_compute_data__
                    with file.open(compute_data_path, 'w') as f:
                        data_buff.write_to_file(f)

            # Call this class's save object.
            if hasattr(__class__, 'save_object_imp'):
                retval = __class__.save_object_imp(self, file)
                if type(retval) is not bool:
                    raise TypeError(
                        "save_object_imp must always return a bool.")
                if not retval:
                    return False

            if not hasattr(__class__, '__dry_meta_base__'):
                # If we're not the base, call the super class's save.
                super().save_object(file)

            # Save contained dry objects passed as arguments to construct
            # for obj in self.__dry_obj_container_list__:
            #     obj_id = obj.dry_id
            #     save_path = f'dry_objects/{obj_id}.dry'
            #     if save_path not in file.namelist():
            #         with file.open(save_path, 'w') as f:
            #             if not obj.save_self(f):
            #                 return False

            return True
        return save_object

    @staticmethod
    def make_compute_prepare(__class__):
        """
        Method for making compute_prepare function
        """

        def compute_prepare(self, _top=True):
            is_top_call = False
            if _top:
                # We know that we're the first call.
                _top = False
                is_top_call = True

                # throw error
                if self.__dry_compute_mode__:
                    raise ComputeModeAlreadyActiveError()

                # Check for context ONLY ONCE.
                # Check required context for this class
                required_context_name = \
                    getattr(self, '__dry_compute_context__', 'default')

                # Get context manager
                ctx_manager = context()

                # Check requirements against context
                if ctx_manager is None:
                    raise NoContextError("There is no context active.")
                else:
                    ctx_reqs = {required_context_name: {}}
                    if not ctx_manager.satisfies(ctx_reqs):
                        raise WrongContextError(
                            f"{ctx_manager} currently doesn't satisfy "
                            f"requirements {ctx_reqs}")

                # Prepare contained objects I would do this only at top.
                # for obj in self.__dry_obj_container_list__:
                #     obj.compute_prepare()

            # Prepare self super classes
            if not hasattr(__class__, '__dry_meta_base__'):
                # Call the super class's compute_prepare
                super().compute_prepare(_top=_top)

            # Execute user compute prepare method
            if hasattr(self, 'compute_prepare_imp'):
                self.compute_prepare_imp()

            if is_top_call:
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

            # Cleanup contained objects
            # for obj in self.__dry_obj_container_list__:
            #     obj.compute_cleanup()

            # Cleanup this object
            if hasattr(self, 'compute_cleanup_imp'):
                self.compute_cleanup_imp()

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

        def load_compute(self, f=None) -> bool:
            if f is None:
                # Load contained objects
                # if f is None, we're at the top, so load here.
                # for obj in self.__dry_obj_container_list__:
                #     # We call load_compute without f since those objects
                #     # will have individual saved compute states.
                #     if not obj.load_compute():
                #         return False

                if self.__dry_compute_data__ is None:
                    # There is no data to load.
                    return True
                else:
                    # Set file pointer.
                    f = self.__dry_compute_data__

            # self.__dry_compute_data__ exists, so we need to load it.

            if not hasattr(__class__, '__dry_meta_base__'):
                # If we're not the base, call the super class's load.
                # So, we're loading from the ground up.
                super().load_compute(f=f)

            # Load this object's compute
            if hasattr(__class__, 'load_compute_imp'):
                if f is not None:
                    # Need to seek the io stream to the beginning
                    f.seek(0)
                    with zipfile.ZipFile(
                            f, mode='r') as zf:
                        imp_res = __class__.load_compute_imp(
                            self, zf)
                        if type(imp_res) is not bool:
                            raise TypeError(
                                "load_compute_imp must return a bool.")
                        if not imp_res:
                            return False
            # Finally report success
            return True
        return load_compute

    @staticmethod
    def make_save_compute(__class__):
        """
        Method for making a save_compute function
        """
        def save_compute(self, f=None, save_cache=None) -> bool:
            if save_cache is None:
                save_cache = SaveCache()
            top_call = False
            if f is None:
                top_call = True
                # We're at the top of the call stack.
                # Create io bytes stream
                f = FileIntermediary()

                # Save contained dry objects passed as arguments to construct
                for obj in self.__dry_obj_container_list__:
                    if not obj.save_compute(save_cache=save_cache):
                        return False

            # Now we check if we're in the save cache.
            if id(self) in save_cache.compute_cache:
                # We don't have to go any further.
                return True

            # Call class save implementation
            if hasattr(__class__, 'save_compute_imp'):
                with zipfile.ZipFile(
                        f, mode='w') as zf:
                    compute_imp_res = __class__.save_compute_imp(
                        self, zf)
                    if type(compute_imp_res) is not bool:
                        raise TypeError("save_compute_imp must return a bool.")
                    if not compute_imp_res:
                        return False

            # Call super class save
            if not hasattr(__class__, '__dry_meta_base__'):
                # If we're not the base, call the super class's load.
                super().save_compute(f=f, save_cache=save_cache)

            if top_call:
                # Only set the save file if there's data.
                if not f.is_empty():
                    self.__dry_compute_data__ = f
                else:
                    # delete the io stream created.
                    del f

                # Add self to the save cache, so we don't repeat.
                save_cache.compute_cache.add(id(self))

            return True
        return save_compute


def validate_key(key):
    if type(key) in (str, bytes, int, float):
        return
    if type(key) is tuple:
        return
    raise TypeError(f"Unsupported key ({key}) of type {type(key)}")


# Assumption, for Objects, definitions are not valid values.
# All definitions should be resolved into objects.
def validate_val_obj(val):
    if is_supported_scalar_type(val):
        return
    from dryml import Object
    if isinstance(val, Object):
        # Assumption: Objects have already validated their values
        return
    if isinstance(val, ObjectDef):
        raise TypeError(
            "Object Definitions not valid for use within Object")
    if is_supported_listlike(val):
        for el in val:
            validate_val_obj(el)
        return
    if is_supported_dictlike(val):
        for k in val:
            validate_val_obj(val[k])
        return
    raise ValueError(
        f"value ({val}) of type {type(val)} not supported")


def validate_val_def(val):
    if is_supported_scalar_type(val):
        return
    from dryml import Object
    if isinstance(val, Object):
        # Assumption: Objects have already validated their values
        return
    if isinstance(val, ObjectDef):
        return
    if is_supported_listlike(val):
        for el in val:
            validate_val_def(el)
        return
    if is_supported_dictlike(val):
        for k in val:
            validate_val_def(val[k])
        return
    raise ValueError(
        f"value ({val}) of type {type(val)} not supported")


def strip_dry_id(obj, render_cache=None):
    # Create render cache
    if render_cache is None:
        render_cache = {}

    # Check render cache if this obj has been seen before
    obj_id = id(obj)
    if obj_id in render_cache:
        return render_cache[obj_id]

    # Construct new object
    if is_dictlike(obj):
        new_dict = {}
        for k in obj:
            if k != 'dry_id':
                new_dict[k] = strip_dry_id(obj[k], render_cache=render_cache)
        render_cache[obj_id] = new_dict
        return new_dict
    elif is_nonstring_iterable(obj):
        new_list = [strip_dry_id(o) for o in obj]
        if type(obj) is tuple:
            return tuple(new_list)
        else:
            return new_list
    else:
        return obj


class RenderCache(object):
    def __init__(self):
        self.unique_cache = {}
        self.nonunique_cache = {}


def def_to_obj(val, repo=None, load_zip=None):
    def applier(val):
        return def_to_obj(val, repo=repo, load_zip=load_zip)
    from dryml import Object
    if is_supported_scalar_type(val):
        return val
    elif isinstance(val, ObjectDef):
        return val.build(repo=repo, load_zip=load_zip)
    elif isinstance(val, Object):
        return val
    elif is_supported_listlike(val):
        return map_listlike(applier, val)
    elif is_supported_dictlike(val):
        return map_dictlike(applier, val)
    else:
        raise TypeError(
            f"Unsupported value {val} of type {type(val)} encountered!")


def def_to_cat_def(val, cache=None):
    def applier(val):
        return def_to_cat_def(val, cache=cache)
    from dryml import Object
    if is_supported_scalar_type(val):
        return val
    elif isinstance(val, ObjectDef):
        return val.get_cat_def(recursive=True, cache=cache)
    elif isinstance(val, Object):
        return val.definition().get_cat_def(recursive=True, cache=cache)
    elif is_supported_listlike(val):
        return map_listlike(applier, val)
    elif is_supported_dictlike(val):
        return map_dictlike(applier, val)
    else:
        raise TypeError(
            f"Unsupported value {val} of type {type(val)} encountered!")


class ObjectDef(collections.UserDict):
    @staticmethod
    def from_dict(def_dict: Mapping, render_cache=None):
        raise RuntimeError("Functionality Questionable")
        # Construct cache object if needed
        if render_cache is None:
            render_cache = RenderCache()

        # Check render cache
        if 'dry_id' in def_dict['dry_kwargs']:
            # Check for unique
            dry_id = def_dict['dry_kwargs']['dry_id']
            if dry_id in render_cache.unique_cache:
                return render_cache.unique_cache[dry_id]
        else:
            # Check for nonunique
            dict_id = id(def_dict)
            if dict_id in render_cache.nonunique_cache:
                return render_cache.nonunique_cache[dict_id]

        # We have to construct a new definition.
        def transform_el(el):
            if type(el) is dict:
                if 'dry_def' in el:
                    # Detect dry def.
                    return ObjectDef.from_dict(
                        el, render_cache=render_cache)
            if is_dictlike(el):
                # We have just a normal dictionary.
                return {k: transform_el(el[k]) for k in el}
            elif is_nonstring_iterable(el):
                new_list = [transform_el(v) for v in el]
                if type(el) is tuple:
                    return tuple(new_list)
                else:
                    return new_list
            else:
                return el

        cls = def_dict['cls']
        args = transform_el(def_dict.get('dry_args', ()))
        kwargs = transform_el(def_dict.get('dry_kwargs', {}))

        # construct new definition
        new_def = ObjectDef(
            cls,
            *args,
            dry_mut=def_dict.get('dry_mut', False),
            **kwargs)

        # Save our result in the cache.
        if 'dry_id' in def_dict['dry_kwargs']:
            dry_id = def_dict['dry_kwargs']['dry_id']
            render_cache.unique_cache[dry_id] = new_def
        else:
            def_id = id(def_dict)
            render_cache.nonunique_cache[def_id] = new_def

        # Return the result.
        return new_def

    def __init__(self, cls: Union[Type, str],
                 *args, dry_mut: bool = False, **kwargs):
        self._tracking_id = uuid.uuid4()

        super().__init__()
        if cls is None:
            raise ValueError(
                "Can't construct an object Definition with None type.")
        self.cls = cls
        self.data['dry_mut'] = dry_mut
        # Validate arguments
        validate_val_def(args)
        self.data['dry_args'] = args
        validate_val_def(kwargs)
        self.data['dry_kwargs'] = kwargs

    def __eq__(self, other):
        return equal_recursive(self, other)

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

    @property
    def kwargs(self):
        return self['dry_kwargs']

    @property
    def dry_id(self):
        if 'dry_id' not in self['dry_kwargs']:
            raise MissingIdError()
        return self['dry_kwargs']['dry_id']

    @property
    def tracking_id(self):
        return self._tracking_id

    def __setitem__(self, key, value):
        if key not in ['cls', 'dry_mut', 'dry_args', 'dry_kwargs']:
            raise ValueError(
                f"Setting Key {key} not supported by ObjectDef")

        if key == 'cls':
            if type(value) is type or issubclass(type(value), abc.ABCMeta):
                self.data['cls'] = value
            elif type(value) is str:
                self.data['cls'] = get_class_from_str(value)
            else:
                raise TypeError(
                    f"Value of type {type(value)} not supported "
                    "for class assignment!")
        else:
            self.data[key] = value

    def to_dict(self, cls_str: bool = False, render_cache=None):
        raise RuntimeError("Functionality Questionable")
        from dryml import Object

        # Create cache if it doesn't yet exist.
        if render_cache is None:
            render_cache = RenderCache()

        # Search for a rendered version of this definition first.
        if self.is_concrete():
            dry_id = self.dry_id
            if dry_id in render_cache.unique_cache:
                return render_cache.unique_cache[dry_id]
        else:
            def_id = id(self)
            if def_id in render_cache.nonunique_cache:
                return render_cache.nonunique_cache[def_id]

        # Create recursive helper function
        def transform_el(el):
            if type(el) is ObjectDef:
                return el.to_dict(
                    cls_str=cls_str, render_cache=render_cache)
            elif type(el) is Object:
                the_def = el.definition()
                return the_def.to_dict(
                    cls_str=cls_str, render_cache=render_cache)
            elif is_dictlike(el):
                # May cause a problem if we have a dict version
                # of a ObjectDef within another ObjectDef
                return {k: transform_el(el[k]) for k in el}
            elif is_nonstring_iterable(el):
                new_list = [transform_el(e) for e in el]
                if type(el) is tuple:
                    return tuple(new_list)
                else:
                    return new_list
            else:
                return el

        # Create the new dictionary
        new_dict = {
            'cls': self.cls if not cls_str else get_class_str(self.cls),
            'dry_mut': self.dry_mut,
            'dry_args': transform_el(self.args),
            'dry_kwargs': transform_el(self.kwargs),
            'dry_def': True,
        }

        # Save result to cache
        if self.is_concrete():
            render_cache.unique_cache[self.dry_id] = new_dict
        else:
            def_id = id(self)
            render_cache.nonunique_cache[def_id] = new_dict

        # Return result
        return new_dict

    def build(self, repo=None, load_zip=None, verbose=None):
        "Construct an object"
        reset_repo = False
        reset_cache = False
        reset_strat = False
        reset_verbose = False
        construct_object = True

        global build_repo
        global build_cache
        global def_cache
        global build_strat
        global build_verbose

        # Handle setting of build_verbose
        if build_verbose is None:
            if verbose is None:
                build_verbose = False
            else:
                if type(verbose) is not bool:
                    raise TypeError("verbose must be a bool!")
                build_verbose = verbose
            reset_verbose = True
        else:
            if verbose is not None:
                if type(verbose) is not bool:
                    raise TypeError("verbose must be a bool!")
                if verbose != build_verbose:
                    raise ValueError(
                        "Can't change verbose once set by a superior call.")

        # Handle creation of build repo
        if repo is not None:
            if build_repo is not None:
                if repo is not build_repo:
                    raise RuntimeError(
                        "different repos not currently supported")
            else:
                # Set the call_repo
                build_repo = repo
                reset_repo = True

        # Handle creation of build cache
        if build_cache is None:
            build_cache = {}
            def_cache = {}
            reset_cache = True

        # Handle creation of build strat cache
        if build_strat is None:
            build_strat = BuildStratTracker()
            reset_strat = True

        # Define a cleanup function to call in the event of error
        # and at the end of the function.
        def cleanup():
            global build_repo
            global build_cache
            global def_cache
            global build_strat
            global build_verbose

            # Reset the repo for this function
            if reset_repo:
                build_repo = None

            # Reset the build cache
            if reset_cache:
                build_cache = None
                def_cache = None

            # Reset the build strat cache
            if reset_strat:
                build_strat = None

            # Reset the verbose indicator
            if reset_verbose:
                build_verbose = None

        # Create some book-keeping variables
        obj = None
        construction_required = True
        construct_object = True

        # Check whether this SPECIFIC definition has been built yet.
        if self.tracking_id in def_cache:
            obj = def_cache[self.tracking_id]
            if build_verbose:
                print(
                    "Object with id {obj.dry_id} built for definition "
                    f"with tracking id {self.tracking_id} was found "
                    "in the definition cache.")
            construction_required = False
            construct_object = False

        # Indicates whether we need to construct the object because this
        # definition isn't concrete
        if obj is None and self.is_concrete():
            # This is a concrete object and we can get an id.
            obj_id = self.dry_id
            construction_required = False

        # Check the cache
        if obj is None and not construction_required and \
                build_cache is not None:
            try:
                obj = build_cache[obj_id]
                if build_verbose:
                    print(f"Found object with id {obj_id} in "
                          "the build cache.")
                construct_object = False
            except KeyError:
                pass

        # Check the repo
        if obj is None and (not construction_required) and \
                (build_repo is not None) and \
                ('repo' not in build_strat[obj_id]):
            try:
                build_strat[obj_id].add('repo')
                obj = build_repo.get_obj(self, load=True)
                build_cache[obj_id] = obj
                def_cache[self.tracking_id] = obj
                construct_object = False
                build_strat[obj_id].remove('repo')
                if build_verbose:
                    print(f"Found object with id {obj_id} in the "
                          "repository.")
            except KeyError:
                # Didn't find the object in the repo
                pass

        try:
            # Check the zipfile
            if obj is None and (not construction_required) \
                    and construct_object and (load_zip is not None) \
                    and ('zip' not in build_strat[obj_id]):
                target_filename = f"dry_objects/{obj_id}.dry"
                from dryml import load_object
                if target_filename in load_zip.namelist():
                    build_strat[obj_id].add('zip')
                    with load_zip.open(target_filename) as f:
                        obj = load_object(f)
                        build_cache[obj_id] = obj
                        def_cache[self.tracking_id] = obj
                        construct_object = False
                    build_strat[obj_id].remove('zip')
                    if build_verbose:
                        print(f"Found object with id {obj_id} in the "
                              "zip file.")

            # Finally, actually construct the object
            if obj is None and construct_object:
                new_args = def_to_obj(self.args, repo=repo, load_zip=load_zip)
                new_kwargs = def_to_obj(
                    self.kwargs,
                    repo=repo,
                    load_zip=load_zip)

                obj = self.cls(*new_args, **new_kwargs)

                # Save object in the build cache.
                obj_id = obj.dry_id
                build_cache[obj_id] = obj
                def_cache[self.tracking_id] = obj
                if build_verbose:
                    print(f"Explicitly constructed new object. id: {obj_id}")

            elif obj is None and not construct_object:
                raise RuntimeError(
                    "Unexpected condition encountered when "
                    "building from definition")

        except Exception as e:
            cleanup()
            raise e

        cleanup()

        # Return the result
        return obj

    def get_cat_def(self, recursive=True, cache=None):
        if recursive:
            # Create cache if needed
            if cache is None:
                cache = {}

            # check the cache for whether this definition has been rendered yet
            if id(self) in cache:
                return cache[id(self)]

            # Recursively apply get_cat_def to args, kwargs
            new_args = def_to_cat_def(self.args, cache=cache)
            new_kwargs = def_to_cat_def(self.kwargs, cache=cache)
            if 'dry_id' in new_kwargs:
                del new_kwargs['dry_id']
            if 'dry_metadata' in new_kwargs:
                del new_kwargs['dry_metadata']

            # Create new definition
            cat_def = ObjectDef(
                self.cls, *new_args, dry_mut=self.dry_mut, **new_kwargs)

            # Store the result
            cache[id(self)] = cat_def

            # return the result
            return cat_def

        else:
            # Remove only the top level dry_id.
            new_def = ObjectDef(
                self.cls, *self.args, dry_mut=self.dry_mut, **self.kwargs)
            del new_def.kwargs['dry_id']
            return new_def

    def get_hash_str(self, no_id: bool = False, no_metadata: bool = False):
        class_hash_str = get_class_str(self.cls)
        args_hash_str = str(self.args)
        # Remove dry_id so we can test for object 'class'
        if no_id or no_metadata:
            kwargs_copy = copy.copy(self.kwargs)
            if no_id:
                if 'dry_id' in kwargs_copy:
                    kwargs_copy.pop('dry_id')
            if no_metadata:
                if 'dry_metadata' in kwargs_copy:
                    kwargs_copy.pop('dry_metadata')
            kwargs_hash_str = str(kwargs_copy)
        else:
            kwargs_hash_str = str(self.kwargs)
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
        return get_hashed_id(self.get_hash_str())

    def get_category_id(self):
        return get_hashed_id(self.get_hash_str(no_id=True, no_metadata=True))
