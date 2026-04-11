from __future__ import annotations

from copy import deepcopy
from inspect import isclass
import numpy as np
import sys
#import weakref
from collections.abc import Mapping
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .utils.stable_hash import stable_int_hash, stable_hash_function
from .utils.types import is_dictlike, is_nonclass_callable
from .utils.general import get_class_str, hashval_to_digest, \
    get_object_view
from .utils.recurse import cycle_detect
from .types import is_pod, compatible_containers, container_types
from .freeze import FrozenDict, FrozenList, FrozenTuple, FrozenSet, FrozenNDArray, frozen_container_types
from .errors import PathAccessError
from .policies import InstancePolicy, CachePolicy

# Special value to skip args
SKIP_ARGS = object()


class DefInterface(ABC):
    @property
    @abstractmethod
    def cls(self):
        ...

    @property
    @abstractmethod
    def args(self):
        ...

    @property
    @abstractmethod
    def kwargs(self):
        ...

    @abstractmethod
    def concretize(self, repo: "Repo | None"=None) -> Any:
        ...

    def categorical(self, recursive=False):
        return categorical_definition(self, recursive=recursive)

    def build(
            self, *,
            repo: "Repo | None" = None,
            instance: "InstancePolicy" = "reuse",
            restore_state: bool = True,
            build_missing: bool =True,
            cache: "CachePolicy" = "weak",
            revision: "str | None" = None) -> Object:
        from .repo import manage_repo
        with manage_repo(repo=repo) as sub_repo:
            concrete_def = self.concretize(repo=sub_repo)
            return sub_repo.load_object(
                concrete_def,
                instance=instance,
                restore_state=restore_state,
                build_missing=build_missing,
                cache=cache,
                revision=revision)

    def match(self, other_def, *, strict: bool=False, verbose: bool=False, **sel_kwargs) -> bool:
        from .definition import selector_match  # or wherever you keep it
        return selector_match(self, other_def, strict=strict, verbose=verbose, **sel_kwargs)

    def __call__(self, other_def, *, strict: bool=False, verbose: bool=False, **sel_kwargs):
        return self.match(other_def, strict=strict, verbose=verbose, **sel_kwargs)

    def stable_hash(self):
        return stable_hash_function(self)


@dataclass(slots=True, init=False, eq=False)
class Definition(DefInterface, Mapping):
    """
    Mutable, selector-capable spec.
    - Unhashable by contract.
    - Behaves like a mapping with keys: cls, (args?), kwargs
    """

    _cls: Callable[..., Any] | type | None
    _args: tuple[Any, ...] | None
    kwargs: dict[str, Any]

    __hash__ = None  # critical: prevent dict/set usage

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            if not callable(args[0]) and not isclass(args[0]):
                if args[0] is SKIP_ARGS and len(args) == 1:
                    self._cls = None
                    self._args = None
                    self.kwargs = kwargs
                else:
                    raise ValueError("First positional argument must be a class or callable.")
            if len(args) > 1 and args[1] is SKIP_ARGS:
                if len(args) > 2:
                    raise ValueError("SKIP_ARGS must be the only positional argument besides the class.")

                self._cls = args[0]
                self._args = None
                self.kwargs = kwargs

            else:
                self._cls = args[0]
                self._args = args[1:]
                self.kwargs = kwargs
        else:
            self._cls = None
            self._args = args
            self.kwargs = kwargs

    @property
    def cls(self):
        return self._cls

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self.kwargs

    @property
    def skip_args(self) -> bool:
        if self._args is None:
            return True
        else:
            return False

    # --- mapping interface (for view compatibility) ---
    def __getitem__(self, k: str) -> Any:
        if k == "cls":
            if self._cls is None:
                raise KeyError("cls")
            return self._cls
        if k == "args":
            if self._args is None:
                raise KeyError("args")
            return self._args
        if k == "kwargs":
            return self.kwargs
        raise KeyError(k)

    def __iter__(self) -> Iterator[str]:
        if self._cls is not None:
            yield "cls"
        if self._args is not None:
            yield "args"
        yield "kwargs"

    def __len__(self) -> int:
        return 1 + (1 if self._args is not None else 0) + (1 if self.cls is not None else 0)

    def __eq__(self, rhs):
        if type(self) != type(rhs):
            return False
        # We actually need to check in both directions.
        if not selector_match(self, rhs, strict=True):
            return False
        if not selector_match(rhs, self, strict=True):
            return False
        return True

    def __ne__(self, rhs):
        return not self.__eq__(rhs)

    def __repr__(self):
        arg_elements = []
        arg_str = ""
        if self._cls is not None:
            arg_elements.append(f"{self._cls}")
        else:
            arg_elements.append("_")
        if self._args is not None:
            for arg in self._args:
                arg_elements.append(arg)
        for key, v in self.kwargs.items():
            arg_elements.append(f"{key}={v}")
        arg_str = ", ".join(map(str, arg_elements))
        return f"{type(self).__name__}({arg_str})"

    def __getstate__(self):
        # Only structural data; drop ephemeral links
        return {
            'cls': self._cls,
            'args': self._args,
            'kwargs': self.kwargs
        }

    def __setstate__(self, state):
        self._cls = state['cls']
        self._args = state['args']
        self.kwargs = state['kwargs']

    def __deepcopy__(self, memo):
        """
        Snapshot cls/args/kwargs, but re-use _obj and any repo reference.
        """
        cp_args = []
        if self._cls is not None:
            cp_args.append(self._cls)
        if self._args is None:
            cp_args.append(SKIP_ARGS)
        else:
            cp_args.extend(deepcopy(self._args, memo))
        cp_kwargs = deepcopy(self.kwargs, memo)

        new = type(self)(*cp_args, **cp_kwargs)

        ## Reuse identity-ish bits
        #new._repo = self._repo

        return new

    def concretize(self, repo: "Repo | None"=None) -> Any:
        return concretize_func(self, repo=repo)


@dataclass(frozen=True, slots=True)
class ConcreteDefinition(DefInterface, Mapping):
    """
    Mutable, selector-capable spec.
    - Unhashable by contract.
    - Behaves like a mapping with keys: cls, (args?), kwargs
    """

    cls: type
    args: FrozenTuple[Any, ...]
    kwargs: FrozenDict[str, Any]

    def __hash__(self) -> int:
        return stable_int_hash(self.stable_hash())

    def __getitem__(self, k: str) -> Any:
        if k == "cls": return self.cls
        if k == "args": return self.args
        if k == "kwargs": return self.kwargs
        raise KeyError(k)

    def __iter__(self) -> Iterator[str]:
        yield from ("cls", "args", "kwargs")

    def __len__(self) -> int:
        return 3

    def __repr__(self):
        arg_elements = []
        arg_str = ""
        arg_elements.append(f"{self.cls}")
        for arg in self.args:
            arg_elements.append(arg)
        for key, v in self.kwargs.items():
            arg_elements.append(f"{key}={v}")
        arg_str = ", ".join(map(str, arg_elements))
        return f"{type(self).__name__}({arg_str})"

    def __eq__(self, rhs):
        if type(self) != type(rhs):
            return False
        # We actually need to check in both directions.
        if not selector_match(self, rhs, strict=True):
            return False
        if not selector_match(rhs, self, strict=True):
            return False
        return True

    def __ne__(self, rhs):
        return not self.__eq__(rhs)


    def __deepcopy__(self, memo):
        """
        Snapshot cls/args/kwargs, but re-use _obj and any repo reference.
        """
        return type(self)(
            self.cls,
            deepcopy(self.args, memo),
            deepcopy(self.kwargs, memo))

    def copy(self):
        return deepcopy(self)

    def to_definition(self):
        return thaw_concrete(self)

    def concretize(self, repo: "Repo | None"=None) -> Any:
        return self


def get_path(obj_or_def, path):
    from .object import Object
    if len(path) == 0:
        return obj_or_def

    key = path[0]
    if key is None:
        return obj_or_def

    new_path = path[1:]
    if isinstance(obj_or_def, Object):
        if key == 'cls':
            value = obj_or_def.definition.cls
        elif key == 'args':
            value = obj_or_def.definition.args
        elif key == 'kwargs':
            value = obj_or_def.definition.kwargs
        else:
            raise PathAccessError(path)
    else:
        try:
            value = obj_or_def[key]
        except (KeyError, IndexError, ValueError, TypeError, PathAccessError) as e:
            if type(e) is PathAccessError:
                raise PathAccessError(path)
            else:
                raise PathAccessError(path)


    try:
        return get_path(value, new_path)
    except (KeyError, IndexError, ValueError, PathAccessError) as e:
        if type(e) is PathAccessError:
            raise PathAccessError(path)
        else:
            raise PathAccessError(path)


def render_path(path, key):
    path = ["root",] + path
    if key is not None:
        path = path + [key,]

    return "/".join(map(str, path))


@cycle_detect()
def concretize_func(obj: Any, path: list[str|int]|None=None, repo: "Repo | None"=None) -> Any:
    if path is None:
        path = [''] # Treat this as the root node
    from .repo import manage_repo
    from .object import Object
    with manage_repo(repo=repo) as sub_repo:
        if is_pod(obj) or isinstance(obj, FrozenNDArray):
            # We have a plain old data type
            return obj
        elif isinstance(obj, frozen_container_types):
            # We have a frozen container
            return obj
        elif isinstance(obj, ConcreteDefinition):
            # ConcreteDefinitions are already concrete
            return obj
        elif isinstance(obj, tuple):
            # Freeze the tuple
            return FrozenTuple([concretize_func(v, path + [i], repo=sub_repo) for i, v in enumerate(obj)])
        elif isinstance(obj, list):
            # Freeze the list
            return FrozenList([concretize_func(v, path + [i], repo=sub_repo) for i, v in enumerate(obj)])
        elif isinstance(obj, set):
            # Freeze the set
            return FrozenSet([concretize_func(v, path + [i], repo=sub_repo) for i, v in enumerate(obj)])

        elif isinstance(obj, dict):
            # Freeze the dict
            new_dict = {}
            for k, v in obj.items():
                new_dict[k] = concretize_func(v, path + [k], repo=sub_repo)
            return FrozenDict(new_dict)

        elif isinstance(obj, np.ndarray):
            return FrozenNDArray.from_array(obj)

        elif isinstance(obj, Object):
            sub_repo.cache_weak(obj)
            return obj.__cdef__

        elif isinstance(obj, Definition):
            # Check the repo object

            # Normalize args
            if obj.args is None:
                raise ValueError(f"Cannot concretize Definition with missing args at path {'/'.join(map(str, path))}")
            if obj.cls is None:
                raise ValueError(f"Cannot concretize Definition with missing cls at path {'/'.join(map(str, path))}")
            # Normalize args
            c_args, c_kwargs = obj.cls.__prepare_args__(*obj.args, **obj.kwargs)
            c_args = concretize_func(c_args, path + ['args'], repo=sub_repo)
            c_kwargs = concretize_func(c_kwargs, path + ['kwargs'], repo=sub_repo)
            
            return ConcreteDefinition(obj.cls, c_args, c_kwargs)
        elif isinstance(obj, type):
            # We allow plain types
            return obj
        else:
            raise TypeError(f"Cannot concretize object of type {type(obj)} at path {'/'.join(map(str, path))}")


def categorical_definition(defn: DefInterface, recursive=True, cache=None):
    from .object import Object
    # Copy the Definition
    new_def = deepcopy(defn)

    level = 0

    if cache is None:
        cache = {}

    if isinstance(new_def, (Object, ConcreteDefinition)):
        # we need to thaw first.
        new_def = new_def.to_definition()

    @cycle_detect()
    def _categorical(obj):
        nonlocal level
        level += 1
        if level > 1 and (not recursive):
            # Don't recurse further
            return obj
        try:
            if is_pod(obj):
                # We have a plain old data type
                return obj
            # At this point we know we're recursive, so proceed
            if isinstance(obj, (tuple, list, set)):
                # We have an iterable
                return type(obj)([_categorical(v) for v in obj])
            if isinstance(obj, dict):
                # We have a dict
                return {k: _categorical(v) for k, v in obj.items()}

            if isinstance(obj, Definition):
                # descend into Definition args/kwargs
                defn_args = _categorical(obj.args) if obj.args is not None else None
                defn_kwargs = _categorical(obj.kwargs)
                temp_args = defn_args if defn.args is not None else tuple()
                new_args, new_kwargs = defn.cls.__strip_unique_args__(
                    *temp_args,
                    **defn_kwargs)
                if obj.args is None:
                    new_args = None

                new_defn_args = []
                if obj.cls is not None:
                    new_defn_args.append(obj.cls)
                if obj.args is not None:
                    new_defn_args.extend(new_args)

                return Definition(
                    *new_defn_args,
                    **new_kwargs)

            else:
                raise TypeError(f"Cannot categorical-ify object of type {type(obj)}")
        finally:
            level -= 1

    return _categorical(new_def)


@cycle_detect()
def thaw_concrete(cdef_or_obj: Any, cache = None) -> Any:
    """
    Thaw a ConcreteDefinition or frozen container into a mutable Definition or container.
    All nested elements are thawed recursively.
    ConcreteDefinition -> Definition
    Object -> Definition
    """
    from .object import Object
    if cache is None:
        cache = {}
    if id(cdef_or_obj) in cache:
        return cache[id(cdef_or_obj)]
    if is_pod(cdef_or_obj):
        # We have a plain old data type
        return cdef_or_obj
    if isinstance(cdef_or_obj, FrozenNDArray):
        # We have a frozen array
        new_val = cdef_or_obj.thaw()
        cache[id(cdef_or_obj)] = new_val
        return new_val
    # Thaw containers
    if isinstance(cdef_or_obj, FrozenList):
        new_val = [thaw_concrete(v, cache=cache) for v in cdef_or_obj]
        cache[id(cdef_or_obj)] = new_val
        return new_val

    if isinstance(cdef_or_obj, FrozenSet):
        new_val = set([thaw_concrete(v, cache=cache) for v in cdef_or_obj])
        cache[id(cdef_or_obj)] = new_val
        return new_val

    if isinstance(cdef_or_obj, FrozenTuple):
        new_val = tuple([thaw_concrete(v, cache=cache) for v in cdef_or_obj])
        cache[id(cdef_or_obj)] = new_val
        return new_val

    if isinstance(cdef_or_obj, FrozenDict):
        # Freeze the dict
        new_dict = {}
        for k, v in cdef_or_obj.items():
            new_dict[k] = thaw_concrete(v, cache=cache)
        cache[id(cdef_or_obj)] = new_dict
        return new_dict

    if isinstance(cdef_or_obj, ConcreteDefinition):
        thaw_args = thaw_concrete(cdef_or_obj.args, cache=cache)
        thaw_kwargs = thaw_concrete(cdef_or_obj.kwargs, cache=cache)
        new_def = Definition(cdef_or_obj.cls, *thaw_args, **thaw_kwargs)
        cache[id(cdef_or_obj)] = new_def
        return new_def

    if isinstance(cdef_or_obj, Definition):
        # We already have a Definition, just return it
        return cdef_or_obj
    if isinstance(cdef_or_obj, Object):
        new_def = thaw_concrete(cdef_or_obj.definition, cache=cache)
        cache[id(cdef_or_obj)] = new_def
        return new_def
    else:
        raise TypeError(f"Cannot thaw object of type {type(cdef_or_obj)}")


## Selecting objects
def selector_match(
        selector,
        target,
        strict=True,
        cls_str_compare=False,
        verbose=False,
        full_diagnostic=False,
        output_stream=sys.stderr):
    """
    full_diagnostic - If true, doesn't stop on the first match failure. Continues to compare to produce a full report of what differs.
    """

    from .object import Object

    dryml_obj_types = (Object, Definition, ConcreteDefinition)

    def _selector_match_func(
        path: list[str|int]|None=None):

        # Method for testing if a selector matches a definition
        # if strict is set, it must match exactly, and callables arent' allowed.
        # cls_str_compare forces a string based name comparison between classes.
        # Additionally, Definitions which skip args also aren't allowed
        from .object import Object
        if path is None:
            path = []

        def _selector_print(msg: str):
            if verbose:
                output_stream.write(f"[{render_path(path, None)}]: {msg}\n")

        # Check we can access the current path in the definition
        try:
            target_val = get_path(target, path)
        except PathAccessError:
            _selector_print("Path doesn't exist in target")
            return False

        try:
            sel_val = get_path(selector, path)
        except PathAccessError:
            _selector_print("Path doesn't exist in selector")
            return False

        if isinstance(target_val, Definition):
            if strict and target_val.skip_args:
                raise TypeError(f"Definitions which skip args aren't allowed in strict mode {render_path(path, None)}")

        # class specific conditionals
        if isclass(sel_val) and isclass(target_val):
            if strict:
                condition = sel_val is target_val
                if not condition:
                    _selector_print("Classes differ")
                return condition
            else:
                condition = issubclass(target_val, sel_val)
                if not condition:
                    _selector_print(f"Classes not subclass: {get_class_str(target_val)} is not a subclass of {get_class_str(sel_val)}")
                return condition
        elif isinstance(sel_val, str) and isclass(target_val):
            # We can do a class string comparison
            condition = (sel_val == get_class_str(target_val))
            if not condition:
                _selector_print(f"Class string comparison failed: {sel_val} != {get_class_str(target_val)}")
            return condition
        # Double container comparison
        elif isinstance(sel_val, container_types) and isinstance(target_val, container_types):
            # Check that containers are compatible.
            val_conditions = map(lambda t: isinstance(sel_val, t), compatible_containers.values())
            def_conditions = map(lambda t: isinstance(target_val, t), compatible_containers.values())
            containers_match = any(list(map(lambda t: t[0] and t[1], zip(val_conditions, def_conditions))))
            if not containers_match:
                _selector_print(f"Container types don't match. {type(sel_val)} in the selector {type(target_val)} in the target")
                return False

            # tuple/set/list check
            if isinstance(target_val, compatible_containers['tuple']) or isinstance(target_val, compatible_containers['set']) or isinstance(target_val, compatible_containers['list']):
                # tuples must match length
                if len(sel_val) != len(target_val):
                    _selector_print(f"Container lengths don't match. {len(sel_val)} in the selector {len(target_val)} in the target")
                    return False

                # Descend into each element
                compare_failed = False
                for i in range(len(sel_val)):
                    res = _selector_match_func(path + [i,])
                    if not res:
                        compare_failed = True
                        if not full_diagnostic:
                            break
                return not compare_failed

            # dict check
            if isinstance(target_val, compatible_containers['dict']):
                # Check each element in order.
                # Descend into each element
                # Only check the mentioned keys in the value dict.
                compare_failed = False
                for k in sel_val.keys():
                    if k not in target_val:
                        compare_failed = True
                        if not full_diagnostic:
                            break
                    res = _selector_match_func(path + [k,])
                    if not res:
                        compare_failed = True
                        if not full_diagnostic:
                            break
                return not compare_failed
            raise TypeError(f"Unhandled container type ({target_val}) in selector_match")

        elif isinstance(sel_val, np.ndarray) and isinstance(target_val, np.ndarray):
            condition = (sel_val.shape == target_val.shape)
            if not condition:
                _selector_print(
                    f" Mismatched array shapes {sel_val.shape} != {target_val.shape}")
                return False
            condition = np.all(target_val == sel_val)
            if not condition:
                _selector_print(
                    "Unequal Arrays")
                return False
            return True

        elif isinstance(sel_val, dryml_obj_types) and isinstance(target_val, dryml_obj_types):
            if isinstance(sel_val, Object):
                sel_def = sel_val.definition
            else:
                sel_def = sel_val
            compare_failed = False
            # Descent into dryml objects
            if sel_def.cls is not None:
                if not _selector_match_func(path + ['cls',]):
                    if full_diagnostic:
                        compare_failed = True
                    else:
                        return False
            # args selection
            if sel_def.args is not None:
                if not _selector_match_func(path + ['args',]):
                    if full_diagnostic:
                        compare_failed = True
                    else:
                        return False
            # kwargs selection
            compare_failed = not _selector_match_func(path + ['kwargs',])

            return not compare_failed
        
        elif is_nonclass_callable(sel_val):
            if strict:
                raise TypeError(f"Callable selectors are not allowed in strict mode {render_path(path, None)}")
            condition = sel_val(target_val)
            if not condition:
                _selector_print(
                    f"Callable test failed")
            return condition

        else:
            # Plain matching branch
            if type(sel_val) is not type(target_val):
                _selector_print(
                        "Type mismatch")
                return False
            else:
                condition = (sel_val == target_val)
                if not condition:
                    _selector_print(
                        "Values differ")
                return condition

    return _selector_match_func()
