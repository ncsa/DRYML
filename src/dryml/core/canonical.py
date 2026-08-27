from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypeAlias, Callable

import numpy as np

from .freeze import (
    FrozenDict,
    FrozenList,
    FrozenNDArray,
    FrozenSet,
    FrozenTuple,
)
from .types import is_pod
from .utils.graph import GraphCtx, GraphTransformer
from .policies import RepoLoadOptions
from .errors import CannotConcretizeParameterizedDefinition, CannotConcretizeSelectorReference, CycleError
from .cdef_identity import V2_IDENTITY_VERSION

if TYPE_CHECKING:
    from .definition import ConcreteDefinition
    from .repo import Repo

# Identity semantic values:
# runtime == canonical == thawed
from .dtype import DType
from .tensor_spec import TensorSpec
from .cardinality import Cardinality
from .config import ConfigRef
from .factory import FactorySpec
# If Backend is a real runtime type/class, include it too.
# from .backend import Backend

from .symbol import ImportRef, SourceSpec, symbol_ref, resolve_symbol


# ----------------------------------------------------------------------
# Identity semantic values
# ----------------------------------------------------------------------

# These are values that should pass through unchanged across all three
# surfaces: runtime, canonical, and thawed.
IDENTITY_VALUE_TYPES = (
    DType,
    TensorSpec,
    Cardinality,
    ConfigRef,
    FactorySpec,
)


def is_identity_value(x: Any) -> bool:
    """
    Semantic value objects that do not need representation changes across
    runtime/canonical/thawed surfaces.

    Important:
    - DType / TensorSpec belong here.
    - Future represented specs like SourceSpec should *not* go here,
      because they will decode back to a callable at runtime/thaw time.
    """
    if isinstance(x, IDENTITY_VALUE_TYPES):
        return True

    # Enums like Dim / Layout / Dynamic are also fine as identity values.
    if isinstance(x, Enum):
        return True

    return False

# ----------------------------------------------------------------------
# Canonical type aliases
# ----------------------------------------------------------------------

CanonicalAtom: TypeAlias = (
    None
    | bool
    | int
    | float
    | complex
    | bytes
    | str
    | type
)


CanonicalKey: TypeAlias = str | int


def is_canonical_key(x: Any) -> bool:
    """
    Canonical mapping keys are restricted to exactly str and int.

    Note that bool is intentionally *not* accepted, even though bool is a
    subclass of int.
    """
    return (type(x) is str) or (type(x) is int)


def validate_canonical_key(
    k: Any,
    *,
    where: str = "mapping key",
    path: tuple[str | int, ...] | None = None,
) -> None:
    if is_canonical_key(k):
        return

    loc = "<root>" if not path else "/".join(map(str, path))
    raise TypeError(
        f"Invalid {where} of type {type(k).__name__} at {loc}. "
        "Only str and int keys are allowed."
    )


if TYPE_CHECKING:
    CanonicalValue: TypeAlias = (
        CanonicalAtom
        | FrozenNDArray
        | FrozenList
        | FrozenTuple
        | FrozenSet
        | FrozenDict
        | "ConcreteDefinition"
    )
else:
    CanonicalValue: TypeAlias = Any


# Canonical mappings are FrozenDict[CanonicalKey, CanonicalValue]

# ----------------------------------------------------------------------
# Container compatibility
# ----------------------------------------------------------------------

COMPATIBLE_CONTAINERS = {
    "dict": (dict, FrozenDict),
    "list": (list, FrozenList),
    "tuple": (tuple, FrozenTuple),
    "set": (set, FrozenSet),
}

CONTAINER_FAMILY_ORDER = (
    "dict",
    "list",
    "tuple",
    "set",
)


def container_family(x: Any) -> str | None:
    """
    Return the DRYML container family name for x, or None if x is not a
    supported container.

    Order matters because some frozen container types may satisfy multiple
    isinstance checks (for example tuple-tagged frozen lists).
    """
    for name in CONTAINER_FAMILY_ORDER:
        typ = COMPATIBLE_CONTAINERS[name]
        if isinstance(x, typ):
            return name
    return None


def matching_container_family(a: Any, b: Any) -> str | None:
    """
    Return the compatible container family for a pair of values, or None if
    they are not container-compatible.

    This should be used instead of classifying each side independently and
    comparing the family names, because some types can satisfy multiple
    container checks.
    """
    for name in CONTAINER_FAMILY_ORDER:
        typ = COMPATIBLE_CONTAINERS[name]
        if isinstance(a, typ) and isinstance(b, typ):
            return name
    return None


def is_container_value(x: Any) -> bool:
    return container_family(x) is not None


# ----------------------------------------------------------------------
# Node classification
# ----------------------------------------------------------------------

class NodeKind(Enum):
    POD = auto()
    TYPE = auto()
    IDENTITY_VALUE = auto()

    NDARRAY = auto()
    FROZEN_NDARRAY = auto()

    FUNCTION = auto()
    IMPORT_REF = auto()
    SOURCE_SPEC = auto()

    LIST = auto()
    TUPLE = auto()
    SET = auto()
    DICT = auto()

    FROZEN_LIST = auto()
    FROZEN_TUPLE = auto()
    FROZEN_SET = auto()
    FROZEN_DICT = auto()

    DEFINITION = auto()
    CONCRETE_DEFINITION = auto()
    DEFLINK = auto()
    QUOTED_DEF = auto()
    SELECTOR_SPEC = auto()
    SELECTOR = auto()
    PAR = auto()
    OBJECT = auto()

    UNSUPPORTED = auto()


def node_kind(x: Any) -> NodeKind:
    from .definition import ConcreteDefinition, Definition
    from .links import DefLink
    from .object import Object
    from .params import Par
    from .quoted import QuotedDef, SelectorSpec
    from .selector import Selector

    if isinstance(x, type):
        return NodeKind.TYPE

    if is_pod(x):
        return NodeKind.POD

    if is_identity_value(x):
        return NodeKind.IDENTITY_VALUE

    if isinstance(x, FrozenNDArray):
        return NodeKind.FROZEN_NDARRAY

    if isinstance(x, np.ndarray):
        return NodeKind.NDARRAY

    # order matters for tagged frozen containers
    if isinstance(x, FrozenList):
        return NodeKind.FROZEN_LIST
    if isinstance(x, FrozenTuple):
        return NodeKind.FROZEN_TUPLE
    if isinstance(x, FrozenSet):
        return NodeKind.FROZEN_SET
    if isinstance(x, FrozenDict):
        return NodeKind.FROZEN_DICT

    if isinstance(x, list):
        return NodeKind.LIST
    if isinstance(x, tuple):
        return NodeKind.TUPLE
    if isinstance(x, set):
        return NodeKind.SET
    if isinstance(x, dict):
        return NodeKind.DICT

    if isinstance(x, DefLink):
        return NodeKind.DEFLINK

    if isinstance(x, QuotedDef):
        return NodeKind.QUOTED_DEF

    if isinstance(x, SelectorSpec):
        return NodeKind.SELECTOR_SPEC

    if isinstance(x, Selector):
        return NodeKind.SELECTOR

    if isinstance(x, Par):
        return NodeKind.PAR

    if isinstance(x, ConcreteDefinition):
        return NodeKind.CONCRETE_DEFINITION

    if isinstance(x, Definition):
        return NodeKind.DEFINITION

    if isinstance(x, Object):
        return NodeKind.OBJECT

    if isinstance(x, ImportRef):
        return NodeKind.IMPORT_REF

    if isinstance(x, SourceSpec):
        return NodeKind.SOURCE_SPEC

    if isinstance(x, Callable):
        return NodeKind.FUNCTION

    return NodeKind.UNSUPPORTED


def is_canonical_value(x: Any) -> bool:
    return node_kind(x) in {
        NodeKind.POD,
        NodeKind.IDENTITY_VALUE,
        NodeKind.FROZEN_NDARRAY,
        NodeKind.FROZEN_LIST,
        NodeKind.FROZEN_TUPLE,
        NodeKind.FROZEN_SET,
        NodeKind.FROZEN_DICT,
        NodeKind.CONCRETE_DEFINITION,
        NodeKind.QUOTED_DEF,
        NodeKind.SELECTOR_SPEC,
        NodeKind.IMPORT_REF,
        NodeKind.SOURCE_SPEC,
    } or _is_naked_core_type(x)


def _is_naked_core_type(x: Any) -> bool:
    module = getattr(x, "__module__", "")
    return isinstance(x, type) and (module == "dryml.core" or module.startswith("dryml.core."))


def is_runtime_leaf(x: Any) -> bool:
    """
    Values that should usually be treated as terminal by runtime-oriented
    graph traversals.
    """
    return node_kind(x) in {
        NodeKind.POD,
        NodeKind.TYPE,
        NodeKind.IDENTITY_VALUE,
        NodeKind.NDARRAY,
        NodeKind.FROZEN_NDARRAY,
        NodeKind.FUNCTION,
        NodeKind.IMPORT_REF,
        NodeKind.SOURCE_SPEC,
    }


# ----------------------------------------------------------------------
# Child traversal / rebuilding helpers
# ----------------------------------------------------------------------

def _path_part_from_key(k: Any) -> str | int:
    validate_canonical_key(k, where="mapping key")
    return k


def iter_value_children(x: Any):
    """
    Yield (path_part, child) for the value-children of a supported container.

    For dict-like containers, keys are preserved and only values are yielded.
    Keys must be canonical keys (str or int).
    """
    kind = node_kind(x)

    if kind in {
        NodeKind.LIST,
        NodeKind.TUPLE,
        NodeKind.SET,
        NodeKind.FROZEN_LIST,
        NodeKind.FROZEN_TUPLE,
        NodeKind.FROZEN_SET,
    }:
        for i, v in enumerate(x):
            yield i, v
        return

    if kind in {NodeKind.DICT, NodeKind.FROZEN_DICT}:
        for k, v in x.items():
            yield _path_part_from_key(k), v
        return

    raise TypeError(f"{type(x).__name__} does not have value-children")


def map_value_children(x: Any, fn):
    """
    Map fn(path_part, child) over the value-children of a supported container,
    rebuilding a container of the same core family/type.
    """
    kind = node_kind(x)

    if kind is NodeKind.LIST:
        return [fn(i, v) for i, v in enumerate(x)]

    if kind is NodeKind.TUPLE:
        return tuple(fn(i, v) for i, v in enumerate(x))

    if kind is NodeKind.SET:
        return {fn(i, v) for i, v in enumerate(x)}

    if kind is NodeKind.DICT:
        return {k: fn(_path_part_from_key(k), v) for k, v in x.items()}

    if kind is NodeKind.FROZEN_LIST:
        return FrozenList(fn(i, v) for i, v in enumerate(x))

    if kind is NodeKind.FROZEN_TUPLE:
        return FrozenTuple(fn(i, v) for i, v in enumerate(x))

    if kind is NodeKind.FROZEN_SET:
        return FrozenSet(fn(i, v) for i, v in enumerate(x))

    if kind is NodeKind.FROZEN_DICT:
        return FrozenDict({k: fn(_path_part_from_key(k), v) for k, v in x.items()})

    raise TypeError(f"{type(x).__name__} does not support map_value_children")


def map_dict_items(x: Any, key_fn, value_fn):
    """
    Map key_fn/value_fn over dict-like containers, rebuilding the same family.
    Only valid for dict / FrozenDict.

    Rebuilt keys must be canonical keys (str or int).
    """
    kind = node_kind(x)

    if kind is NodeKind.DICT:
        out = {}
        for k, v in x.items():
            new_k = key_fn(k)
            validate_canonical_key(new_k, where="mapping key")
            out[new_k] = value_fn(k, v)
        return out

    if kind is NodeKind.FROZEN_DICT:
        out = {}
        for k, v in x.items():
            new_k = key_fn(k)
            validate_canonical_key(new_k, where="mapping key")
            out[new_k] = value_fn(k, v)
        return FrozenDict(out)

    raise TypeError(f"{type(x).__name__} is not dict-like")


# ----------------------------------------------------------------------
# Container transform helpers
# ----------------------------------------------------------------------

RUNTIME_SEQ_KINDS = {
    NodeKind.LIST,
    NodeKind.TUPLE,
    NodeKind.SET,
}

CANONICAL_SEQ_KINDS = {
    NodeKind.FROZEN_LIST,
    NodeKind.FROZEN_TUPLE,
    NodeKind.FROZEN_SET,
}

RUNTIME_DICT_KINDS = {
    NodeKind.DICT,
}

CANONICAL_DICT_KINDS = {
    NodeKind.FROZEN_DICT,
}


def is_value_container_kind(kind: NodeKind) -> bool:
    return kind in (
        RUNTIME_SEQ_KINDS
        | CANONICAL_SEQ_KINDS
        | RUNTIME_DICT_KINDS
        | CANONICAL_DICT_KINDS
    )


def rebuild_sequence_kind(kind: NodeKind, values):
    if kind is NodeKind.LIST:
        return list(values)
    if kind is NodeKind.TUPLE:
        return tuple(values)
    if kind is NodeKind.SET:
        return set(values)

    if kind is NodeKind.FROZEN_LIST:
        return FrozenList(values)
    if kind is NodeKind.FROZEN_TUPLE:
        return FrozenTuple(values)
    if kind is NodeKind.FROZEN_SET:
        return FrozenSet(values)

    raise TypeError(f"{kind} is not a sequence/set container kind")


def rebuild_dict_kind(kind: NodeKind, items: dict[Any, Any]):
    if kind is NodeKind.DICT:
        return dict(items)
    if kind is NodeKind.FROZEN_DICT:
        return FrozenDict(items)
    raise TypeError(f"{kind} is not a dict container kind")


def target_container_kind(kind: NodeKind, *, target: str) -> NodeKind:
    """
    target:
      - 'same'       -> preserve current kind
      - 'canonical'  -> runtime containers become frozen/canonical
      - 'runtime'    -> frozen/canonical containers become plain runtime
    """
    if target == "same":
        return kind

    if target == "canonical":
        mapping = {
            NodeKind.LIST: NodeKind.FROZEN_LIST,
            NodeKind.TUPLE: NodeKind.FROZEN_TUPLE,
            NodeKind.SET: NodeKind.FROZEN_SET,
            NodeKind.DICT: NodeKind.FROZEN_DICT,

            NodeKind.FROZEN_LIST: NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_TUPLE: NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_SET: NodeKind.FROZEN_SET,
            NodeKind.FROZEN_DICT: NodeKind.FROZEN_DICT,
        }
        return mapping[kind]

    if target == "runtime":
        mapping = {
            NodeKind.LIST: NodeKind.LIST,
            NodeKind.TUPLE: NodeKind.TUPLE,
            NodeKind.SET: NodeKind.SET,
            NodeKind.DICT: NodeKind.DICT,

            NodeKind.FROZEN_LIST: NodeKind.LIST,
            NodeKind.FROZEN_TUPLE: NodeKind.TUPLE,
            NodeKind.FROZEN_SET: NodeKind.SET,
            NodeKind.FROZEN_DICT: NodeKind.DICT,
        }
        return mapping[kind]

    raise ValueError(f"Unknown target container mode: {target!r}")


def transform_container(
    x: Any,
    value_fn,
    *,
    target: str = "same",
    transform_keys: bool = False,
):
    """
    Generic container entry/exit helper.

    - `value_fn(path_part, value)` maps child values
    - `target` controls whether the rebuilt container is:
        * same kind
        * canonical kind
        * runtime kind

    Dict key policy:
      - keys must be canonical keys (str or int)
      - keys are preserved, not transformed
      - `transform_keys=True` is not supported
    """
    kind = node_kind(x)
    if not is_value_container_kind(kind):
        raise TypeError(f"{type(x).__name__} is not a supported container")

    if transform_keys:
        raise ValueError(
            "Dict key transformation is disabled. "
            "Canonical dict keys must be preserved."
        )

    out_kind = target_container_kind(kind, target=target)

    # sequence / set families
    if kind in (RUNTIME_SEQ_KINDS | CANONICAL_SEQ_KINDS):
        mapped = [value_fn(p, v) for p, v in iter_value_children(x)]
        return rebuild_sequence_kind(out_kind, mapped)

    # dict families
    if kind in (RUNTIME_DICT_KINDS | CANONICAL_DICT_KINDS):
        items = {}
        for k, v in x.items():
            key = _path_part_from_key(k)   # validates key and preserves it
            items[key] = value_fn(key, v)
        return rebuild_dict_kind(out_kind, items)

    raise TypeError(f"Unhandled container kind {kind}")


# ----------------------------------------------------------------------
# Internal transformers
# ----------------------------------------------------------------------

class _ToCanonicalTransformer(GraphTransformer):
    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        if node_kind(obj) in (CANONICAL_SEQ_KINDS | CANONICAL_DICT_KINDS):
            return False
        return is_canonical_value(obj)

    def transform_atomic(self, obj: Any, ctx: GraphCtx) -> Any:
        return obj

    def should_track_cycle(self, obj: Any, ctx: GraphCtx) -> bool:
        kind = node_kind(obj)
        return kind in {
            NodeKind.LIST,
            NodeKind.TUPLE,
            NodeKind.SET,
            NodeKind.DICT,
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_DICT,
            NodeKind.DEFINITION,
        }

    def dispatch(self, obj: Any, ctx: GraphCtx) -> Any:
        from .definition import ConcreteDefinition
        from .object import Object

        kind = node_kind(obj)
        repo = ctx.state["repo"]

        if kind is NodeKind.NDARRAY:
            return FrozenNDArray.from_array(obj)

        if kind is NodeKind.TYPE:
            if _is_naked_core_type(obj):
                return obj
            return symbol_ref(obj)

        if kind in {
            NodeKind.LIST,
            NodeKind.TUPLE,
            NodeKind.SET,
            NodeKind.DICT,
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_DICT,
        }:
            return transform_container(
                obj,
                lambda p, v: self.transform(v, ctx.child(p)),
                target="canonical",
                transform_keys=False,
            )

        if kind is NodeKind.OBJECT:
            repo.cache_weak(obj)
            return obj.__cdef__

        if kind is NodeKind.DEFLINK:
            from .cdef_graph import EdgeKind
            from .links import Ref
            from .selector import Selector

            if isinstance(obj.target, Selector):
                raise CannotConcretizeSelectorReference(tuple(ctx.path), obj.target)

            target = self.transform(obj.target, ctx.child("target"))
            if obj.kind is EdgeKind.MATERIALIZE:
                return target
            if obj.kind is EdgeKind.REF:
                return Ref(target)
            raise TypeError(f"Unknown DefLink kind {obj.kind!r} at {ctx.path_str()}")

        if kind in {NodeKind.QUOTED_DEF, NodeKind.SELECTOR_SPEC}:
            return obj

        if kind is NodeKind.SELECTOR:
            from .quoted import SelectorSpec
            return SelectorSpec(obj)

        if kind is NodeKind.PAR:
            raise CannotConcretizeParameterizedDefinition(tuple(ctx.path), obj)

        if kind is NodeKind.DEFINITION:
            if obj.cls is None:
                raise ValueError(
                    f"Cannot concretize Definition with missing cls at {ctx.path_str()}"
                )
            if obj.args is None:
                raise ValueError(
                    f"Cannot concretize Definition with missing args at {ctx.path_str()}"
                )

            live_cls = resolve_symbol(obj.cls)
            prep_args, prep_kwargs = live_cls.__prepare_args__(*obj.args, **obj.kwargs)
            from .arg_roles import apply_bound_arg_roles
            from .bound_args import BoundArguments, bind_complete_arguments

            bound_args = bind_complete_arguments(live_cls, tuple(prep_args), dict(prep_kwargs))
            bound_args = apply_bound_arg_roles(live_cls, bound_args)
            canonical_bound_args = BoundArguments(
                (name, self.transform(value, ctx.child(name)))
                for name, value in bound_args.items()
            )
            c_cls = self.transform(live_cls, ctx.child("cls"))
            return ConcreteDefinition._from_bound_record(c_cls, canonical_bound_args)

        if kind is NodeKind.FUNCTION:
            return symbol_ref(obj)

        raise TypeError(
            f"Cannot canonicalize object of type {type(obj).__name__} at {ctx.path_str()}"
        )


class _ThawValueTransformer(GraphTransformer):
    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        return node_kind(obj) in {
            NodeKind.POD,
            NodeKind.TYPE,
            NodeKind.IDENTITY_VALUE,
        }

    def transform_atomic(self, obj: Any, ctx: GraphCtx) -> Any:
        return obj

    def memo_key(self, obj: Any, ctx: GraphCtx):
        kind = node_kind(obj)
        if kind in {
            NodeKind.FROZEN_NDARRAY,
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_DICT,
            NodeKind.CONCRETE_DEFINITION,
            NodeKind.DEFINITION,
            NodeKind.OBJECT,
        }:
            return id(obj)
        return None

    def should_track_cycle(self, obj: Any, ctx: GraphCtx) -> bool:
        kind = node_kind(obj)
        return kind in {
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_DICT,
            NodeKind.CONCRETE_DEFINITION,
            NodeKind.DEFINITION,
            NodeKind.OBJECT,
        }

    def dispatch(self, obj: Any, ctx: GraphCtx) -> Any:
        kind = node_kind(obj)

        if kind is NodeKind.FROZEN_NDARRAY:
            return obj.thaw()

        if kind in {
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_DICT,
        }:
            return transform_container(
                obj,
                lambda p, v: self.transform(v, ctx.child(p)),
                target="runtime",
                transform_keys=False,
            )

        if kind is NodeKind.CONCRETE_DEFINITION:
            return thaw_definition_surface_value(obj)

        if kind is NodeKind.DEFLINK:
            return obj

        if kind in {NodeKind.QUOTED_DEF, NodeKind.SELECTOR_SPEC, NodeKind.PAR, NodeKind.SELECTOR}:
            return obj

        if kind is NodeKind.DEFINITION:
            return obj

        if kind is NodeKind.OBJECT:
            return self.transform(obj.definition, ctx)

        if kind in {NodeKind.IMPORT_REF, NodeKind.SOURCE_SPEC}:
            return resolve_symbol(obj)

        if kind is NodeKind.FUNCTION:
            return obj

        raise TypeError(
            f"Cannot thaw value of type {type(obj).__name__} at {ctx.path_str()}"
        )


class _FromCanonicalTransformer(GraphTransformer):
    def __init__(self, repo: "Repo", config: RepoLoadOptions, *, resolve_cdef=None):
        self.repo = repo
        self.config = config
        self.resolve_cdef = resolve_cdef

    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        return node_kind(obj) in {
            NodeKind.POD,
            NodeKind.TYPE,
            NodeKind.IDENTITY_VALUE,
        }

    def transform_atomic(self, obj: Any, ctx: GraphCtx) -> Any:
        return obj

    def memo_key(self, obj: Any, ctx: GraphCtx):
        if node_kind(obj) is NodeKind.CONCRETE_DEFINITION:
            return obj
        return None

    def should_track_cycle(self, obj: Any, ctx: GraphCtx) -> bool:
        return node_kind(obj) in {
            NodeKind.LIST,
            NodeKind.TUPLE,
            NodeKind.SET,
            NodeKind.DICT,
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_DICT,
            NodeKind.DEFINITION,
            NodeKind.OBJECT,
        }

    def dispatch(self, obj: Any, ctx: GraphCtx) -> Any:
        kind = node_kind(obj)

        if kind is NodeKind.FROZEN_NDARRAY:
            return obj.thaw() if hasattr(obj, "thaw") else np.array(obj, copy=True)

        if kind is NodeKind.NDARRAY:
            return np.array(obj, copy=True)

        if kind is NodeKind.CONCRETE_DEFINITION:
            if self.resolve_cdef is not None:
                return self.resolve_cdef(obj)
            return self.repo._materialize_cdef(
                obj,
                options=self.config,
                memo=ctx.memo,
                path=list(ctx.path),
            )

        if kind is NodeKind.DEFLINK:
            from .cdef_graph import EdgeKind
            if obj.kind is EdgeKind.REF:
                return obj.target
            return self.transform(obj.target, ctx.child("target"))

        if kind in {NodeKind.QUOTED_DEF, NodeKind.SELECTOR_SPEC, NodeKind.PAR, NodeKind.SELECTOR}:
            return obj

        if kind is NodeKind.DEFINITION:
            cdef = to_canonical(obj, repo=self.repo)
            if self.resolve_cdef is not None:
                return self.resolve_cdef(cdef)
            return self.repo._materialize_cdef(
                cdef,
                options=self.config,
                memo=ctx.memo,
                path=list(ctx.path),
            )

        if kind is NodeKind.OBJECT:
            if self.resolve_cdef is not None:
                return self.resolve_cdef(obj.definition)
            if self.config.instance == "new":
                return self.repo._materialize_cdef(
                    obj.definition,
                    options=self.config,
                    memo=ctx.memo,
                    path=list(ctx.path),
                )

            if self.repo.get_cached(obj.definition, reuse_weak=self.config.reuse_weak) is None:
                self.repo.cache_weak(obj)

            return obj

        if kind in {NodeKind.IMPORT_REF, NodeKind.SOURCE_SPEC}:
            return resolve_symbol(obj)

        if kind in {
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_DICT,
            NodeKind.LIST,
            NodeKind.TUPLE,
            NodeKind.SET,
            NodeKind.DICT,
        }:
            return transform_container(
                obj,
                lambda p, v: self.transform(v, ctx.child(p)),
                target="runtime",
            )

        raise TypeError(
            f"Cannot de-canonicalize value of type {type(obj).__name__} at {ctx.path_str()}"
        )

# ----------------------------------------------------------------------
# Thin public wrappers for now
# ----------------------------------------------------------------------

def to_canonical(
    x: Any,
    *,
    repo: "Repo | None" = None,
    path: list[str | int] | tuple[str | int, ...] | None = None,
):
    from .repo import manage_repo

    with manage_repo(repo=repo) as sub_repo:
        ctx = GraphCtx(
            path=tuple(path) if path is not None else (),
            state={"repo": sub_repo},
        )
        return _ToCanonicalTransformer().transform(x, ctx)


def thaw_value(
    x: Any,
    *,
    memo: dict | None = None,
    path: list[str | int] | tuple[str | int, ...] | None = None,
):
    if memo is None:
        memo = {}

    ctx = GraphCtx(
        path=tuple(path) if path is not None else (),
        memo=memo,
    )
    return _ThawValueTransformer().transform(x, ctx)


def from_canonical(
    x: Any,
    *,
    repo: "Repo",
    options: RepoLoadOptions | None = None,
    instance: str = "reuse",
    restore_state: bool = True,
    build_missing: bool = False,
    reuse_weak: bool = True,
    cache: str = "weak",
    revision=None,
    memo: dict | None = None,
    path: list[str | int] | tuple[str | int, ...] | None = None,
    resolve_cdef=None,
):
    if memo is None:
        memo = {}

    cfg = options or RepoLoadOptions(
        instance=instance,
        restore_state=restore_state,
        build_missing=build_missing,
        reuse_weak=reuse_weak,
        cache=cache,
        revision=revision,
    )

    ctx = GraphCtx(
        path=tuple(path) if path is not None else (),
        memo=memo,
    )
    return _FromCanonicalTransformer(repo, cfg, resolve_cdef=resolve_cdef).transform(x, ctx)


def thaw_definition_surface_value(value: Any) -> Any:
    """Thaw canonical containers for Definition.thaw while preserving symbols."""

    from .definition import Definition

    kind = node_kind(value)
    if kind is NodeKind.FROZEN_NDARRAY:
        return value.thaw()
    if kind in {
        NodeKind.LIST,
        NodeKind.TUPLE,
        NodeKind.SET,
        NodeKind.DICT,
        NodeKind.FROZEN_LIST,
        NodeKind.FROZEN_TUPLE,
        NodeKind.FROZEN_SET,
        NodeKind.FROZEN_DICT,
    }:
        return transform_container(
            value,
            lambda p, v: thaw_definition_surface_value(v),
            target="runtime",
        )
    if kind is NodeKind.CONCRETE_DEFINITION:
        if value.identity_version == V2_IDENTITY_VERSION:
            from .materialization import project_cdef_call

            args, kwargs = project_cdef_call(value)
            return Definition(
                value.cls,
                *thaw_definition_surface_value(args),
                **thaw_definition_surface_value(kwargs),
            )
        args = thaw_definition_surface_value(value.args)
        kwargs = thaw_definition_surface_value(value.kwargs)
        return Definition(value.cls, *args, **kwargs)
    if kind is NodeKind.DEFLINK:
        from .links import DefLink

        return DefLink(value.kind, thaw_definition_surface_value(value.target))
    if kind in {NodeKind.QUOTED_DEF, NodeKind.SELECTOR_SPEC, NodeKind.SELECTOR, NodeKind.PAR}:
        return value
    return value


def freeze_def_value(value: Any) -> Any:
    """Deep-freeze values admitted into Definition expressions."""

    return _freeze_def_value(value, stack=set())


def _freeze_def_value(value: Any, *, stack: set[int]) -> Any:
    from .definition import ConcreteDefinition, Definition
    from .links import DefLink
    from .object import Object
    from .params import Par
    from .quoted import QuotedDef, SelectorSpec
    from .selector import Selector

    if isinstance(value, Object):
        return value.definition
    if isinstance(value, (Definition, ConcreteDefinition, DefLink, QuotedDef, SelectorSpec, Selector, Par)):
        return value
    kind = node_kind(value)
    if kind is NodeKind.NDARRAY:
        return FrozenNDArray.from_array(value)
    if kind in {
        NodeKind.LIST,
        NodeKind.TUPLE,
        NodeKind.SET,
        NodeKind.DICT,
        NodeKind.FROZEN_LIST,
        NodeKind.FROZEN_TUPLE,
        NodeKind.FROZEN_SET,
        NodeKind.FROZEN_DICT,
    }:
        oid = id(value)
        if oid in stack:
            raise CycleError(f"at {type(value).__name__}")
        stack.add(oid)
        try:
            return transform_container(value, lambda p, v: _freeze_def_value(v, stack=stack), target="canonical")
        finally:
            stack.remove(oid)
    if kind in {
        NodeKind.POD,
        NodeKind.TYPE,
        NodeKind.IDENTITY_VALUE,
        NodeKind.FROZEN_NDARRAY,
        NodeKind.IMPORT_REF,
        NodeKind.SOURCE_SPEC,
    }:
        return value
    if kind is NodeKind.FUNCTION:
        try:
            return symbol_ref(value)
        except Exception as e:
            raise TypeError(
                "Anonymous function values are not allowed in Definition values; "
                "use Satisfies(predicate, name=...) for selector predicates."
            ) from e
    raise TypeError(f"Cannot freeze Definition value of type {type(value).__name__}.")


def freeze_concrete_value(
        value: Any,
        *,
        path: tuple[str | int, ...] | None = None) -> Any:
    """Deep-freeze values admitted into ConcreteDefinition values."""

    return _freeze_concrete_value(value, stack=set(), path=() if path is None else tuple(path))


def _path_label(path: tuple[str | int, ...]) -> str:
    return "/".join(map(str, path)) if path else "<root>"


def _freeze_concrete_value(value: Any, *, stack: set[int], path: tuple[str | int, ...]) -> Any:
    kind = node_kind(value)
    if kind in {
        NodeKind.LIST,
        NodeKind.TUPLE,
        NodeKind.SET,
        NodeKind.DICT,
        NodeKind.FROZEN_LIST,
        NodeKind.FROZEN_TUPLE,
        NodeKind.FROZEN_SET,
        NodeKind.FROZEN_DICT,
    }:
        oid = id(value)
        if oid in stack:
            raise CycleError(f"at {_path_label(path)}: {type(value).__name__}")
        stack.add(oid)
        try:
            return transform_container(
                value,
                lambda p, v: _freeze_concrete_value(v, stack=stack, path=path + (p,)),
                target="canonical",
                transform_keys=False,
            )
        finally:
            stack.remove(oid)
    if kind is NodeKind.PAR:
        raise CannotConcretizeParameterizedDefinition(path, value)
    if kind is NodeKind.DEFINITION:
        raise TypeError(
            f"ConcreteDefinition values cannot contain unresolved Definition values at {_path_label(path)}; concretize first."
        )
    if kind is NodeKind.DEFLINK:
        from .definition import ConcreteDefinition
        from .cdef_graph import EdgeKind
        from .selector import Selector
        if isinstance(value.target, Selector):
            raise CannotConcretizeSelectorReference(path, value.target)
        if not isinstance(value.target, ConcreteDefinition):
            raise TypeError(
                f"ConcreteDefinition link values at {_path_label(path)} must target ConcreteDefinition values."
            )
        if value.kind is EdgeKind.MATERIALIZE:
            return value.target
        return value
    if kind in {NodeKind.QUOTED_DEF, NodeKind.SELECTOR_SPEC}:
        from .utils.stable_hash import stable_hash_function

        try:
            stable_hash_function(value)
        except Exception as e:
            raise TypeError(
                f"ConcreteDefinition selector-as-data value at {_path_label(path)} must be stable-hashable."
            ) from e
        return value
    if kind is NodeKind.SELECTOR:
        from .quoted import SelectorSpec
        from .utils.stable_hash import stable_hash_function

        spec = SelectorSpec(value)
        try:
            stable_hash_function(spec)
        except Exception as e:
            raise TypeError(
                f"ConcreteDefinition selector-as-data value at {_path_label(path)} must be stable-hashable."
            ) from e
        return spec
    return freeze_def_value(value)


def freeze_selector_value(value: Any) -> Any:
    """Deep-freeze values stored inside quoted selector/expression wrappers."""

    return freeze_def_value(value)


def freeze_link_target(value: Any) -> Any:
    """Freeze and validate a link target."""

    from .definition import ConcreteDefinition, Definition
    from .object import Object
    from .selector import Selector

    if isinstance(value, Object):
        return value.definition
    if isinstance(value, (Definition, ConcreteDefinition, Selector)):
        return value
    raise TypeError(f"DefLink target must be Definition, ConcreteDefinition, Selector, or Object; got {type(value).__name__}.")
