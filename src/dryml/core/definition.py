from __future__ import annotations

from copy import deepcopy
from inspect import isclass
import numpy as np
import sys
#import weakref
from collections.abc import Mapping
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

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
from .symbol import ImportRef, SourceSpec, maybe_symbol_ref, resolve_symbol

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
            reuse_weak: bool = True,
            cache: "CachePolicy" = "weak",
            revision: "str | None" = None) -> Object:
        """Materialize this definition through a repository lifecycle.

        Args:
            repo: Optional repository to use for concretization and loading.
            instance: Instance reuse policy.
            restore_state: Whether an existing saved state may be restored.
            build_missing: Whether a missing definition may be constructed.
            reuse_weak: Whether weak cached instances may be reused.
            cache: Runtime cache policy.
            revision: Optional saved-state revision.

        Returns:
            The materialized live Object.

        Raises:
            RuntimeTransitionError: Before any concretization, cache access,
                restoration, or construction in strict orchestration.

        Side Effects:
            Holds the runtime publication lease through the repository and
            construction lifecycle. WARN/OFF advanced runtime scopes can admit
            the private fresh scope without changing public object-mode status.
        """

        from dryml.runtime import assert_object_materialization_allowed
        from .repo import manage_repo
        from .session import _construction_config
        # The guard must precede concretization and the private fresh scope so
        # strict orchestration cannot reach cache, restore, or user code.
        with assert_object_materialization_allowed(operation="definition_build"):
            with manage_repo(repo=repo) as sub_repo:
                concrete_def = self.concretize(repo=sub_repo)
                loader = sub_repo.load_or_build if build_missing else sub_repo.load
                with _construction_config():
                    return loader(
                        concrete_def,
                        instance=instance,
                        restore_state=restore_state,
                        reuse_weak=reuse_weak,
                        cache=cache,
                        revision=revision)

    def match(self, other_def, *, strict: bool=False, verbose: bool=False, **sel_kwargs) -> bool:
        from .selector import Selector

        sel_kwargs.pop("full_diagnostic", None)
        sel_kwargs.pop("output_stream", None)
        sel_kwargs.pop("cls_str_compare", None)
        if "class_match" in sel_kwargs and "cls_policy" not in sel_kwargs:
            sel_kwargs["cls_policy"] = sel_kwargs.pop("class_match")
        if isinstance(self, Definition):
            return Selector(self, strict=strict, **sel_kwargs).matches(other_def, verbose=verbose)
        from .query.query import _query_match

        class_match = sel_kwargs.pop("cls_policy", sel_kwargs.pop("class_match", "selector"))
        return _query_match(self, other_def, strict=strict, class_match=class_match)

    def __call__(self, other_def, *, strict: bool=False, verbose: bool=False, **sel_kwargs):
        return self.match(other_def, strict=strict, verbose=verbose, **sel_kwargs)

    def stable_hash(self):
        return stable_hash_function(self)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Definition(DefInterface, Mapping):
    """
    Immutable structural graph expression.

    Values supplied at construction and update boundaries are deeply frozen so
    user-owned containers cannot mutate the Definition after creation.
    """

    _cls: Callable[..., Any] | type | ImportRef | SourceSpec | None
    _args: FrozenTuple | None
    _kwargs: FrozenDict[str, Any]
    _stable_hash_cache: str | None = field(default=None, init=False, repr=False, compare=False, hash=False)

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_stable_hash_cache", None)
        if len(args) > 0:
            if not callable(args[0]) and not isclass(args[0]) and not isinstance(args[0], (ImportRef, SourceSpec)):
                if args[0] is SKIP_ARGS and len(args) == 1:
                    object.__setattr__(self, "_cls", None)
                    object.__setattr__(self, "_args", None)
                    object.__setattr__(self, "_kwargs", self._freeze_kwargs(kwargs))
                else:
                    raise ValueError("First positional argument must be a class, callable, or symbol reference.")
            if len(args) > 1 and args[1] is SKIP_ARGS:
                if len(args) > 2:
                    raise ValueError("SKIP_ARGS must be the only positional argument besides the class.")

                object.__setattr__(self, "_cls", self._freeze_value(args[0]))
                object.__setattr__(self, "_args", None)
                object.__setattr__(self, "_kwargs", self._freeze_kwargs(kwargs))

            else:
                object.__setattr__(self, "_cls", self._freeze_value(args[0]))
                object.__setattr__(self, "_args", FrozenTuple(self._freeze_value(v) for v in args[1:]))
                object.__setattr__(self, "_kwargs", self._freeze_kwargs(kwargs))
        else:
            object.__setattr__(self, "_cls", None)
            object.__setattr__(self, "_args", FrozenTuple())
            object.__setattr__(self, "_kwargs", self._freeze_kwargs(kwargs))

    @staticmethod
    def _freeze_value(value):
        from .canonical import freeze_def_value

        return freeze_def_value(value)

    @classmethod
    def _freeze_kwargs(cls, kwargs):
        return FrozenDict({k: cls._freeze_value(v) for k, v in kwargs.items()})

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
        if type(self) is not type(rhs):
            return False
        return _structural_value_equal(self.cls, rhs.cls) and _structural_value_equal(self.args, rhs.args) and _structural_value_equal(self.kwargs, rhs.kwargs)

    def __hash__(self) -> int:
        return stable_int_hash(self.stable_hash())

    def stable_hash(self) -> str:
        cached = self._stable_hash_cache
        if cached is not None:
            return cached
        value = stable_hash_function(self)
        object.__setattr__(self, "_stable_hash_cache", value)
        return value

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
        object.__setattr__(self, "_cls", self._freeze_value(state['cls']))
        object.__setattr__(self, "_args", None if state['args'] is None else FrozenTuple(self._freeze_value(v) for v in state['args']))
        object.__setattr__(self, "_kwargs", self._freeze_kwargs(dict(state['kwargs'])))
        object.__setattr__(self, "_stable_hash_cache", None)

    def __deepcopy__(self, memo):
        """
        Return self because Definition is immutable.
        """
        return self

    def thaw(self, memo: dict|None=None) -> Any:
        return self

    def freeze(self):
        """Return a quoted Definition snapshot for expression-as-data use."""

        return self.quote()

    def concretize(self, repo: "Repo | None"=None) -> Any:
        return to_canonical(self, repo=repo)

    def with_cls(self, cls) -> "Definition":
        args = (SKIP_ARGS,) if self.args is None else tuple(self.args)
        return Definition(cls, *args, **dict(self.kwargs))

    def with_args(self, *args) -> "Definition":
        if self.cls is None:
            return Definition(*args, **dict(self.kwargs))
        return Definition(self.cls, *args, **dict(self.kwargs))

    def with_arg(self, index, value) -> "Definition":
        if self.args is None:
            raise IndexError("Cannot update positional args on a skip-args Definition.")
        args = list(self.args)
        args[index] = value
        return self.with_args(*args)

    def with_kwargs(self, **kwargs) -> "Definition":
        merged = dict(self.kwargs)
        merged.update(kwargs)
        args = (SKIP_ARGS,) if self.args is None else tuple(self.args)
        if self.cls is None:
            return Definition(*args, **merged)
        return Definition(self.cls, *args, **merged)

    def with_kwarg(self, key, value) -> "Definition":
        return self.with_kwargs(**{key: value})

    def without_kwarg(self, key) -> "Definition":
        kwargs = dict(self.kwargs)
        kwargs.pop(key)
        args = (SKIP_ARGS,) if self.args is None else tuple(self.args)
        if self.cls is None:
            return Definition(*args, **kwargs)
        return Definition(self.cls, *args, **kwargs)

    def at(self, path) -> "DefinitionLens":
        return DefinitionLens(self, path)

    def ref(self):
        from .links import Ref
        return Ref(self)

    def mat(self):
        from .links import Mat
        return Mat(self)

    def quote(self):
        from .quoted import QuotedDef
        return QuotedDef(self)

    def as_selector(self, **policy):
        from .selector import Selector
        return Selector(self, **policy)

    def as_space(self):
        from .search_space import SearchSpace
        return SearchSpace.from_def(self)


@dataclass(frozen=True, slots=True)
class DefinitionLens:
    definition: Definition
    path: Any

    def set(self, value: Any) -> Definition:
        from .canonical import freeze_def_value
        from .utils.graph.value import replace_subtree

        return replace_subtree(self.definition, self.path, freeze_def_value(value))


@dataclass(frozen=True, slots=True)
class ConcreteDefinition(DefInterface, Mapping):
    """
    Exact canonical materializable object identity.

    Values are deeply frozen and validated at construction. Equality is exact
    structural equality, and hashes are stable across processes.
    """

    cls: type | ImportRef | SourceSpec
    args: FrozenTuple[Any, ...] = field(default_factory = lambda: FrozenTuple())
    kwargs: FrozenDict[str, Any] = field(default_factory = lambda: FrozenDict({}))
    _stable_hash_cache: str | None = field(default=None, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        from .canonical import freeze_concrete_value

        args = self.args if isinstance(self.args, FrozenTuple) else FrozenTuple(self.args)
        kwargs = self.kwargs if isinstance(self.kwargs, FrozenDict) else FrozenDict(self.kwargs)
        object.__setattr__(
            self,
            "args",
            FrozenTuple(freeze_concrete_value(v, path=("args", i)) for i, v in enumerate(args)),
        )
        object.__setattr__(
            self,
            "kwargs",
            FrozenDict({k: freeze_concrete_value(v, path=("kwargs", k)) for k, v in kwargs.items()}),
        )

    def __hash__(self) -> int:
        return stable_int_hash(self.stable_hash())

    def stable_hash(self) -> str:
        cached = self._stable_hash_cache
        if cached is not None:
            return cached
        value = stable_hash_function(self)
        object.__setattr__(self, "_stable_hash_cache", value)
        return value

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
        if type(self) is not type(rhs):
            return False
        return _structural_value_equal(self.cls, rhs.cls) and _structural_value_equal(self.args, rhs.args) and _structural_value_equal(self.kwargs, rhs.kwargs)

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

    def freeze(self):
        """Return a non-materializing canonical reference to this CDef."""

        return self.ref()

    def ref(self):
        from .links import Ref
        return Ref(self)

    def mat(self):
        from .links import Mat
        return Mat(self)

    def thaw(self, memo: dict | None = None) -> Any:
        return thaw_value(self, memo=memo)

    def concretize(self, repo: "Repo | None"=None) -> Any:
        return self


def freeze(value: Any) -> Any:
    """Return the new immutable graph wrapper for a DRYML definition value."""

    from .object import Object

    if isinstance(value, Object):
        return value.definition.freeze()
    if isinstance(value, ConcreteDefinition):
        return value.freeze()
    if isinstance(value, Definition):
        return value.freeze()
    raise TypeError(
        "dryml.freeze expects Object, ConcreteDefinition, or Definition; "
        f"got {type(value).__name__}."
    )


def _structural_value_equal(left: Any, right: Any) -> bool:
    from .freeze import FrozenDict, FrozenList, FrozenNDArray, FrozenSet, FrozenTuple
    from .links import DefLink
    from .quoted import QuotedDef, SelectorSpec
    from .selector import Selector

    if left is right:
        return True
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        return left.shape == right.shape and left.dtype == right.dtype and bool(np.array_equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, (Definition, ConcreteDefinition)):
        return left == right
    if isinstance(left, DefLink):
        return left.kind is right.kind and _structural_value_equal(left.target, right.target)
    if isinstance(left, QuotedDef):
        return _structural_value_equal(left.value, right.value)
    if isinstance(left, SelectorSpec):
        return _structural_value_equal(left.selector, right.selector)
    if isinstance(left, Selector):
        return left.strict == right.strict and left.cls_policy == right.cls_policy and _structural_value_equal(left.root, right.root)
    if isinstance(left, (FrozenList, FrozenTuple, tuple, list)):
        return len(left) == len(right) and all(_structural_value_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (FrozenDict, dict)):
        if len(left) != len(right) or tuple(left.keys()) != tuple(right.keys()):
            return False
        return all(_structural_value_equal(left[k], right[k]) for k in left.keys())
    if isinstance(left, (FrozenSet, frozenset, set)):
        if len(left) != len(right):
            return False
        unmatched = list(right)
        for item in left:
            for idx, other in enumerate(unmatched):
                if _structural_value_equal(item, other):
                    unmatched.pop(idx)
                    break
            else:
                return False
        return True
    result = left == right
    if isinstance(result, np.ndarray):
        return bool(np.all(result))
    return bool(result)


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

    def dispatch(self, obj: Any, ctx: GraphCtx) -> Any:
        from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple

        if isinstance(obj, FrozenDict):
            return {k: self.transform(v, ctx.child(k if isinstance(k, (str, int)) else str(k))) for k, v in obj.items()}
        if isinstance(obj, FrozenList):
            return [self.transform(v, ctx.child(i)) for i, v in enumerate(obj)]
        if isinstance(obj, FrozenTuple):
            return tuple(self.transform(v, ctx.child(i)) for i, v in enumerate(obj))
        if isinstance(obj, FrozenSet):
            return {self.transform(v, ctx.child(f"<set:{i}>")) for i, v in enumerate(obj)}
        return super().dispatch(obj, ctx)

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
            live_cls = resolve_symbol(obj.cls)
            new_args, new_kwargs = live_cls.__strip_unique_args__(
                *temp_args,
                **defn_kwargs,
            )

            new_defn_args = [live_cls]
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
        if hasattr(ctx.path, "legacy_tuple"):
            return render_path(list(ctx.path.legacy_tuple()), None)
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
            NodeKind.IMPORT_REF,
            NodeKind.SOURCE_SPEC,
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

        selector_ref = maybe_symbol_ref(selector, functions=False)
        target_ref = maybe_symbol_ref(target, functions=False)
        if selector_ref is not None and target_ref is not None:
            if not self.strict:
                try:
                    selector_obj = resolve_symbol(selector_ref)
                    target_obj = resolve_symbol(target_ref)
                except Exception:
                    selector_obj = None
                    target_obj = None
                if isclass(selector_obj) and isclass(target_obj):
                    condition = issubclass(target_obj, selector_obj)
                    if not condition:
                        self._print(
                            ctx,
                            f"Classes not subclass: {get_class_str(target_obj)} "
                            f"is not a subclass of {get_class_str(selector_obj)}",
                        )
                    return condition

            condition = selector_ref == target_ref
            if not condition:
                self._print(ctx, "Symbol refs differ")
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

        selector_items = list(selector)
        target_items = list(target)
        edges = [
            [j for j, tgt_v in enumerate(target_items) if self.match(sel_v, tgt_v, ctx.child(i))]
            for i, sel_v in enumerate(selector_items)
        ]
        order = sorted(range(len(selector_items)), key=lambda idx: len(edges[idx]))
        matched_to_selector: dict[int, int] = {}

        def augment(sel_idx: int, seen: set[int]) -> bool:
            for tgt_idx in edges[sel_idx]:
                if tgt_idx in seen:
                    continue
                seen.add(tgt_idx)
                if tgt_idx not in matched_to_selector or augment(matched_to_selector[tgt_idx], seen):
                    matched_to_selector[tgt_idx] = sel_idx
                    return True
            return False

        for sel_idx in order:
            if not augment(sel_idx, set()):
                self._print(ctx.child(sel_idx), "No matching set element found")
                return False

        return True

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
        if isinstance(sel_def, Definition):
            from .arg_roles import apply_definition_arg_roles

            sel_def = apply_definition_arg_roles(sel_def)

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
