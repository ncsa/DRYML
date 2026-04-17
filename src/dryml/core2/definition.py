from __future__ import annotations

from copy import deepcopy
from inspect import isclass
import numpy as np
import sys
#import weakref
from collections.abc import Mapping
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .utils.stable_hash import stable_int_hash, stable_hash_function
from .utils.types import is_nonclass_callable
from .utils.general import get_class_str
from .utils.graph import GraphCtx, GraphTransformer, GraphMatcher
from .types import is_pod
from .freeze import FrozenDict, FrozenTuple
from .errors import PathAccessError
from .policies import InstancePolicy, CachePolicy
from .canonical import (
    to_canonical,
    thaw_value,
    NodeKind,
    matching_container_family,
    node_kind)

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
    _kwargs: dict[str, Any]

    __hash__ = None  # critical: prevent dict/set usage

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            if not callable(args[0]) and not isclass(args[0]):
                if args[0] is SKIP_ARGS and len(args) == 1:
                    self._cls = None
                    self._args = None
                    self._kwargs = kwargs
                else:
                    raise ValueError("First positional argument must be a class or callable.")
            if len(args) > 1 and args[1] is SKIP_ARGS:
                if len(args) > 2:
                    raise ValueError("SKIP_ARGS must be the only positional argument besides the class.")

                self._cls = args[0]
                self._args = None
                self._kwargs = kwargs

            else:
                self._cls = args[0]
                self._args = args[1:]
                self._kwargs = kwargs
        else:
            self._cls = None
            self._args = args
            self._kwargs = kwargs

    @property
    def cls(self):
        return self._cls

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

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
            'kwargs': self._kwargs
        }

    def __setstate__(self, state):
        self._cls = state['cls']
        self._args = state['args']
        self._kwargs = state['kwargs']

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
        cp_kwargs = deepcopy(self._kwargs, memo)

        new = type(self)(*cp_args, **cp_kwargs)

        ## Reuse identity-ish bits
        #new._repo = self._repo

        return new

    def thaw(self, memo: dict|None=None) -> Any:
        return self

    def concretize(self, repo: "Repo | None"=None) -> Any:
        return to_canonical(self, repo=repo)


@dataclass(frozen=True, slots=True)
class ConcreteDefinition(DefInterface, Mapping):
    """
    Mutable, selector-capable spec.
    - Unhashable by contract.
    - Behaves like a mapping with keys: cls, (args?), kwargs
    """

    cls: type
    args: FrozenTuple[Any, ...] = field(default_factory = lambda: FrozenTuple())
    kwargs: FrozenDict[str, Any] = field(default_factory = lambda: FrozenDict({}))

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

    def thaw(self, memo: dict | None = None) -> Any:
        return thaw_value(self, memo=memo)

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


def categorical_definition(defn, recursive=True, memo=None):
    from .definition import ConcreteDefinition
    from .object import Object

    if memo is None:
        memo = {}

    if isinstance(defn, Object):
        root = thaw_concrete(defn.definition, memo=memo)
    elif isinstance(defn, ConcreteDefinition):
        root = thaw_concrete(defn, memo=memo)
    else:
        root = deepcopy(defn)

    return CategoricalDefinitionTransformer(recursive=recursive).transform(root)


class SelectorMatcher(GraphMatcher):
    def __init__(
        self,
        *,
        strict: bool = True,
        cls_str_compare: bool = False,
        verbose: bool = False,
        full_diagnostic: bool = False,
        output_stream=sys.stderr,
    ):
        self.strict = strict
        self.cls_str_compare = cls_str_compare
        self.verbose = verbose
        self.full_diagnostic = full_diagnostic
        self.output_stream = output_stream

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------

    def _render(self, ctx: GraphCtx) -> str:
        return render_path(list(ctx.path), None)

    def _print(self, ctx: GraphCtx, msg: str) -> None:
        if self.verbose:
            self.output_stream.write(f"[{self._render(ctx)}]: {msg}\n")

    def _normalize_dryml(self, x: Any):
        from .object import Object
        if isinstance(x, Object):
            return x.definition
        return x

    def _is_structural(self, x: Any) -> bool:
        return node_kind(x) not in {
            NodeKind.POD,
            NodeKind.TYPE,
            NodeKind.IDENTITY_VALUE,
            NodeKind.UNSUPPORTED,
        }

    # ------------------------------------------------------------------
    # graph matcher hooks
    # ------------------------------------------------------------------

    def memo_key(self, selector: Any, target: Any, ctx: GraphCtx):
        if self._is_structural(selector) or self._is_structural(target):
            return (id(selector), id(target))
        return None

    def should_track_cycle(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        return self._is_structural(selector) or self._is_structural(target)

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    def dispatch(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        from .object import Object

        dryml_obj_types = (Object, Definition, ConcreteDefinition)

        if isinstance(target, Definition):
            if self.strict and target.skip_args:
                raise TypeError(
                    f"Definitions which skip args aren't allowed in strict mode {self._render(ctx)}"
                )

        # class comparisons
        if isclass(selector) and isclass(target):
            if self.strict:
                condition = selector is target
                if not condition:
                    self._print(ctx, "Classes differ")
                return condition
            else:
                condition = issubclass(target, selector)
                if not condition:
                    self._print(
                        ctx,
                        f"Classes not subclass: {get_class_str(target)} "
                        f"is not a subclass of {get_class_str(selector)}",
                    )
                return condition

        if self.cls_str_compare and isinstance(selector, str) and isclass(target):
            condition = selector == get_class_str(target)
            if not condition:
                self._print(
                    ctx,
                    f"Class string comparison failed: {selector} != {get_class_str(target)}",
                )
            return condition

        # centralized container compatibility
        family = matching_container_family(selector, target)
        if family is not None:
            if family == "tuple":
                return self.match_tuple(selector, target, ctx)
            if family == "list":
                return self.match_list(selector, target, ctx)
            if family == "set":
                return self.match_set(selector, target, ctx)
            if family == "dict":
                return self.match_dict(selector, target, ctx)

            raise TypeError(f"Unhandled container family {family!r} in selector_match")

        # ndarray
        if isinstance(selector, np.ndarray) and isinstance(target, np.ndarray):
            if selector.shape != target.shape:
                self._print(
                    ctx,
                    f"Mismatched array shapes {selector.shape} != {target.shape}",
                )
                return False
            condition = np.all(target == selector)
            if not condition:
                self._print(ctx, "Unequal arrays")
            return condition

        # dryml object / definition
        if isinstance(selector, dryml_obj_types) and isinstance(target, dryml_obj_types):
            return self.match_dryml(selector, target, ctx)

        # callable selector
        if is_nonclass_callable(selector):
            if self.strict:
                raise TypeError(
                    f"Callable selectors are not allowed in strict mode {self._render(ctx)}"
                )
            condition = selector(target)
            if not condition:
                self._print(ctx, "Callable test failed")
            return condition

        return self.match_other(selector, target, ctx)

    # ------------------------------------------------------------------
    # container overrides with diagnostics / selector semantics
    # ------------------------------------------------------------------

    def match_tuple(self, selector, target, ctx: GraphCtx) -> bool:
        if len(selector) != len(target):
            self._print(
                ctx,
                f"Container lengths don't match. {len(selector)} in selector, "
                f"{len(target)} in target",
            )
            return False

        compare_failed = False
        for i, (sel_v, tgt_v) in enumerate(zip(selector, target)):
            res = self.match(sel_v, tgt_v, ctx.child(i))
            if not res:
                compare_failed = True
                if not self.full_diagnostic:
                    return False
        return not compare_failed

    def match_list(self, selector, target, ctx: GraphCtx) -> bool:
        if len(selector) != len(target):
            self._print(
                ctx,
                f"Container lengths don't match. {len(selector)} in selector, "
                f"{len(target)} in target",
            )
            return False

        compare_failed = False
        for i, (sel_v, tgt_v) in enumerate(zip(selector, target)):
            res = self.match(sel_v, tgt_v, ctx.child(i))
            if not res:
                compare_failed = True
                if not self.full_diagnostic:
                    return False
        return not compare_failed

    def match_set(self, selector, target, ctx: GraphCtx) -> bool:
        if len(selector) != len(target):
            self._print(
                ctx,
                f"Set lengths don't match. {len(selector)} in selector, "
                f"{len(target)} in target",
            )
            return False

        unmatched = list(target)
        compare_failed = False

        for i, sel_v in enumerate(selector):
            found = None
            for j, tgt_v in enumerate(unmatched):
                if self.match(sel_v, tgt_v, ctx.child(i)):
                    found = j
                    break
            if found is None:
                compare_failed = True
                self._print(ctx.child(i), "No matching set element found")
                if not self.full_diagnostic:
                    return False
            else:
                unmatched.pop(found)

        return not compare_failed

    def match_dict(self, selector, target, ctx: GraphCtx) -> bool:
        # Selector semantics: only keys mentioned in selector must match.
        compare_failed = False

        for k in selector.keys():
            child_ctx = ctx.child(k if isinstance(k, (str, int)) else str(k))
            if k not in target:
                compare_failed = True
                self._print(child_ctx, "Key missing in target")
                if not self.full_diagnostic:
                    return False
                continue

            res = self.match(selector[k], target[k], child_ctx)
            if not res:
                compare_failed = True
                if not self.full_diagnostic:
                    return False

        return not compare_failed

    # ------------------------------------------------------------------
    # dryml-specific matching
    # ------------------------------------------------------------------

    def match_dryml(self, selector, target, ctx: GraphCtx) -> bool:
        sel_def = self._normalize_dryml(selector)
        tgt_def = self._normalize_dryml(target)

        compare_failed = False

        if sel_def.cls is not None:
            if not self.match(sel_def.cls, tgt_def.cls, ctx.child("cls")):
                compare_failed = True
                if not self.full_diagnostic:
                    return False

        if sel_def.args is not None:
            if not self.match(sel_def.args, tgt_def.args, ctx.child("args")):
                compare_failed = True
                if not self.full_diagnostic:
                    return False

        if not self.match(sel_def.kwargs, tgt_def.kwargs, ctx.child("kwargs")):
            compare_failed = True
            if not self.full_diagnostic:
                return False

        return not compare_failed

    # ------------------------------------------------------------------
    # plain values
    # ------------------------------------------------------------------

    def match_other(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        if type(selector) is not type(target):
            self._print(ctx, "Type mismatch")
            return False

        condition = selector == target
        if not condition:
            self._print(ctx, "Values differ")
        return condition


def selector_match(
    selector,
    target,
    strict=True,
    cls_str_compare=False,
    verbose=False,
    full_diagnostic=False,
    output_stream=sys.stderr,
):
    return SelectorMatcher(
        strict=strict,
        cls_str_compare=cls_str_compare,
        verbose=verbose,
        full_diagnostic=full_diagnostic,
        output_stream=output_stream,
    ).match(selector, target)


def concretize_func(
    obj: Any,
    path: list[str | int] | None = None,
    repo: "Repo | None" = None,
) -> Any:
    return to_canonical(obj, repo=repo, path=path)


def thaw_concrete(cdef_or_obj: Any, memo: dict|None =None) -> Any:
    return thaw_value(cdef_or_obj, memo=memo)
