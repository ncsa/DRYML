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
from .utils.graph import GraphCtx, GraphTransformer
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


# ----------------------------------------------------------------------
# Concretization
# ----------------------------------------------------------------------

class ConcretizeTransformer(GraphTransformer):
    """
    Convert runtime Python / Object / Definition structures into the
    canonical 'concrete' representation used by ConcreteDefinition args/kwargs.

    Policy lives here in core, not in utils.graph.
    """

    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        return (
            is_pod(obj)
            or isinstance(obj, FrozenNDArray)
            or isinstance(obj, frozen_container_types)
            or isinstance(obj, ConcreteDefinition)
            or isinstance(obj, type)
        )

    def transform_atomic(self, obj: Any, ctx: GraphCtx) -> Any:
        return obj

    def should_track_cycle(self, obj: Any, ctx: GraphCtx) -> bool:
        from .definition import Definition

        return (
            super().should_track_cycle(obj, ctx)
            or isinstance(obj, Definition)
        )

    def transform_tuple(self, obj: tuple[Any, ...], ctx: GraphCtx) -> Any:
        return FrozenTuple(self.transform(v, ctx.child(i)) for i, v in enumerate(obj))

    def transform_list(self, obj: list[Any], ctx: GraphCtx) -> Any:
        return FrozenList(self.transform(v, ctx.child(i)) for i, v in enumerate(obj))

    def transform_set(self, obj: set[Any], ctx: GraphCtx) -> Any:
        return FrozenSet(self.transform(v, ctx.child(i)) for i, v in enumerate(obj))

    def transform_dict(self, obj: dict[Any, Any], ctx: GraphCtx) -> Any:
        # Match current behavior: recurse into values, preserve keys as-is.
        out = {}
        for k, v in obj.items():
            child_path = k if isinstance(k, (str, int)) else str(k)
            out[k] = self.transform(v, ctx.child(child_path))
        return FrozenDict(out)

    def transform_other(self, obj: Any, ctx: GraphCtx) -> Any:
        from .definition import Definition, ConcreteDefinition
        from .object import Object

        repo = ctx.state["repo"]

        if isinstance(obj, np.ndarray):
            return FrozenNDArray.from_array(obj)

        if isinstance(obj, Object):
            repo.cache_weak(obj)
            return obj.__cdef__

        if isinstance(obj, Definition):
            if obj.cls is None:
                raise ValueError(
                    f"Cannot concretize Definition with missing cls at {ctx.path_str()}"
                )
            if obj.args is None:
                raise ValueError(
                    f"Cannot concretize Definition with missing args at {ctx.path_str()}"
                )

            prep_args, prep_kwargs = obj.cls.__prepare_args__(*obj.args, **obj.kwargs)

            c_args = self.transform(prep_args, ctx.child("args"))
            c_kwargs = self.transform(prep_kwargs, ctx.child("kwargs"))

            if not isinstance(c_args, FrozenTuple):
                raise TypeError(
                    f"Prepared args did not concretize to FrozenTuple at {ctx.path_str()}"
                )
            if not isinstance(c_kwargs, FrozenDict):
                raise TypeError(
                    f"Prepared kwargs did not concretize to FrozenDict at {ctx.path_str()}"
                )

            return ConcreteDefinition(obj.cls, c_args, c_kwargs)

        raise TypeError(
            f"Cannot concretize object of type {type(obj).__name__} at {ctx.path_str()}"
        )


def concretize_func(
    obj: Any,
    path: list[str | int] | None = None,
    repo: "Repo | None" = None,
) -> Any:
    from .repo import manage_repo

    with manage_repo(repo=repo) as sub_repo:
        ctx = GraphCtx(
            path=tuple(path) if path is not None else (),
            state={"repo": sub_repo},
        )
        return ConcretizeTransformer().transform(obj, ctx)


# ----------------------------------------------------------------------
# Categorical definition
# ----------------------------------------------------------------------

class CategoricalDefinitionTransformer(GraphTransformer):
    """
    Strip unique args recursively (or only at the root when recursive=False).
    """

    def __init__(self, recursive: bool = True):
        super().__init__()
        self.recursive = recursive

    def transform(self, obj: Any, ctx: GraphCtx | None = None) -> Any:
        if ctx is not None and (not self.recursive) and ctx.path:
            # Match the intended current behavior: only operate at the root.
            return obj
        return super().transform(obj, ctx)

    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        return is_pod(obj) or isinstance(obj, type)

    def transform_atomic(self, obj: Any, ctx: GraphCtx) -> Any:
        return obj

    def should_track_cycle(self, obj: Any, ctx: GraphCtx) -> bool:
        from .definition import Definition

        return super().should_track_cycle(obj, ctx) or isinstance(obj, Definition)

    def transform_other(self, obj: Any, ctx: GraphCtx) -> Any:
        from .definition import Definition

        if isinstance(obj, Definition):
            if obj.cls is None:
                raise ValueError(
                    f"Cannot categorical-ify Definition with missing cls at {ctx.path_str()}"
                )

            defn_args = (
                self.transform(obj.args, ctx.child("args"))
                if obj.args is not None
                else None
            )
            defn_kwargs = self.transform(obj.kwargs, ctx.child("kwargs"))

            temp_args = defn_args if obj.args is not None else tuple()
            new_args, new_kwargs = obj.cls.__strip_unique_args__(
                *temp_args,
                **defn_kwargs,
            )

            new_defn_args = [obj.cls]
            if obj.args is not None:
                new_defn_args.extend(new_args)
            else:
                from .definition import SKIP_ARGS
                new_defn_args.append(SKIP_ARGS)

            return Definition(*new_defn_args, **new_kwargs)

        raise TypeError(
            f"Cannot categorical-ify object of type {type(obj).__name__} at {ctx.path_str()}"
        )


def categorical_definition(defn, recursive=True, cache=None):
    from .definition import ConcreteDefinition
    from .object import Object

    if cache is None:
        cache = {}

    if isinstance(defn, Object):
        root = thaw_concrete(defn.definition, cache=cache)
    elif isinstance(defn, ConcreteDefinition):
        root = thaw_concrete(defn, cache=cache)
    else:
        root = deepcopy(defn)

    return CategoricalDefinitionTransformer(recursive=recursive).transform(root)


# ----------------------------------------------------------------------
# Thawing
# ----------------------------------------------------------------------

class ThawConcreteTransformer(GraphTransformer):
    """
    Convert ConcreteDefinition / frozen containers back into mutable runtime
    Definition / Python containers.
    """

    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        return is_pod(obj)

    def transform_atomic(self, obj: Any, ctx: GraphCtx) -> Any:
        return obj

    def memo_key(self, obj: Any, ctx: GraphCtx):
        from .definition import ConcreteDefinition, Definition
        from .object import Object

        if isinstance(
            obj,
            (
                FrozenNDArray,
                FrozenList,
                FrozenSet,
                FrozenTuple,
                FrozenDict,
                ConcreteDefinition,
                Definition,
                Object,
            ),
        ):
            return id(obj)
        return None

    def should_track_cycle(self, obj: Any, ctx: GraphCtx) -> bool:
        from .definition import ConcreteDefinition, Definition
        from .object import Object

        return (
            super().should_track_cycle(obj, ctx)
            or isinstance(
                obj,
                (
                    FrozenList,
                    FrozenSet,
                    FrozenTuple,
                    FrozenDict,
                    ConcreteDefinition,
                    Definition,
                    Object,
                ),
            )
        )

    def dispatch(self, obj: Any, ctx: GraphCtx) -> Any:
        """
        Important: handle frozen/core-specific node types before the generic
        container dispatch in GraphTransformer, since some frozen container
        types may be tuple-/set-/mapping-tagged.
        """
        from .definition import ConcreteDefinition, Definition
        from .object import Object

        if isinstance(obj, FrozenNDArray):
            return self.transform_frozen_ndarray(obj, ctx)

        if isinstance(obj, FrozenList):
            return self.transform_frozen_list(obj, ctx)

        if isinstance(obj, FrozenSet):
            return self.transform_frozen_set(obj, ctx)

        if isinstance(obj, FrozenTuple):
            return self.transform_frozen_tuple(obj, ctx)

        if isinstance(obj, FrozenDict):
            return self.transform_frozen_dict(obj, ctx)

        if isinstance(obj, ConcreteDefinition):
            return self.transform_concrete_definition(obj, ctx)

        if isinstance(obj, Definition):
            return self.transform_definition(obj, ctx)

        if isinstance(obj, Object):
            return self.transform_object(obj, ctx)

        return super().dispatch(obj, ctx)

    def transform_frozen_ndarray(self, obj: FrozenNDArray, ctx: GraphCtx) -> Any:
        return obj.thaw()

    def transform_frozen_list(self, obj: FrozenList, ctx: GraphCtx) -> list[Any]:
        return [self.transform(v, ctx.child(i)) for i, v in enumerate(obj)]

    def transform_frozen_set(self, obj: FrozenSet, ctx: GraphCtx) -> set[Any]:
        return {self.transform(v, ctx.child(i)) for i, v in enumerate(obj)}

    def transform_frozen_tuple(self, obj: FrozenTuple, ctx: GraphCtx) -> tuple[Any, ...]:
        return tuple(self.transform(v, ctx.child(i)) for i, v in enumerate(obj))

    def transform_frozen_dict(self, obj: FrozenDict, ctx: GraphCtx) -> dict[Any, Any]:
        out = {}
        for k, v in obj.items():
            child_path = k if isinstance(k, (str, int)) else str(k)
            out[k] = self.transform(v, ctx.child(child_path))
        return out

    def transform_concrete_definition(self, obj, ctx: GraphCtx) -> Any:
        from .definition import Definition

        thaw_args = self.transform(obj.args, ctx.child("args"))
        thaw_kwargs = self.transform(obj.kwargs, ctx.child("kwargs"))
        return Definition(obj.cls, *thaw_args, **thaw_kwargs)

    def transform_definition(self, obj, ctx: GraphCtx) -> Any:
        return obj

    def transform_object(self, obj, ctx: GraphCtx) -> Any:
        return self.transform(obj.definition, ctx)

    def transform_other(self, obj: Any, ctx: GraphCtx) -> Any:
        raise TypeError(
            f"Cannot thaw object of type {type(obj).__name__} at {ctx.path_str()}"
        )


def thaw_concrete(cdef_or_obj: Any, cache=None) -> Any:
    if cache is None:
        cache = {}
    ctx = GraphCtx(memo=cache)
    return ThawConcreteTransformer().transform(cdef_or_obj, ctx)


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
