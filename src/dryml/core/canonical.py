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
from .policies import CachePolicy
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
from .reference_values import ObjectId, ObjectRef, StateRef, StateSelectorRef
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
    ObjectId,
    ObjectRef,
    StateRef,
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
    REFERENCE_VALUE = auto()
    STATE_SELECTOR_REF = auto()

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

    if isinstance(x, (ObjectRef, StateRef)):
        return NodeKind.REFERENCE_VALUE

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

    if isinstance(x, StateSelectorRef):
        return NodeKind.STATE_SELECTOR_REF

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
        NodeKind.REFERENCE_VALUE,
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
        NodeKind.REFERENCE_VALUE,
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
    def memo_key(self, obj: Any, ctx: GraphCtx):
        """Memoize graph-bearing sources by Python identity within one scope."""

        kind = node_kind(obj)
        if kind in {NodeKind.DEFINITION, NodeKind.OBJECT, NodeKind.STATE_SELECTOR_REF}:
            return id(obj)
        return None

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

        if kind is NodeKind.STATE_SELECTOR_REF:
            resolver = getattr(repo, "resolve_state_selector", None)
            if not callable(resolver):
                raise TypeError(
                    "StateSelectorRef requires a managing Repo with resolve_state_selector()."
                )
            resolved = resolver(obj)
            if not isinstance(resolved, StateRef):
                raise TypeError("StateSelectorRef resolution must return a StateRef.")
            if resolved.object != obj.object:
                raise ValueError("StateSelectorRef resolution returned a StateRef outside its ObjectRef scope.")
            return resolved

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
            if not isinstance(live_cls, type):
                raise TypeError(
                    f"ConcreteDefinition class target at {ctx.path_str()} must resolve to a class, "
                    f"got {type(live_cls).__name__}."
                )
            prep_args, prep_kwargs = live_cls.__prepare_args__(*obj.args, **obj.kwargs)
            from .arg_roles import apply_bound_arg_roles
            from .bound_args import BoundArguments, bind_complete_arguments

            bound_args = bind_complete_arguments(live_cls, tuple(prep_args), dict(prep_kwargs))
            bound_args = apply_bound_arg_roles(live_cls, bound_args)
            canonical_bound_args = BoundArguments(
                (name, self.transform(value, ctx.child(name)))
                for name, value in bound_args.items()
            )
            # A source-backed class is already its canonical class authority.
            # Re-symbolizing the transient class produced by ``exec`` loses
            # its source provenance and can fail because it has no module file.
            c_cls = obj.cls if isinstance(obj.cls, SourceSpec) else self.transform(live_cls, ctx.child("cls"))
            from .object import Serializable

            return ConcreteDefinition._from_bound_record(
                c_cls,
                canonical_bound_args,
                stateful_role=issubclass(live_cls, Serializable),
            )

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
            NodeKind.STATE_SELECTOR_REF,
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
            return thaw_definition_surface_value(obj, memo=ctx.memo)

        if kind is NodeKind.REFERENCE_VALUE:
            return obj

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
    def __init__(self, repo: "Repo", cache: CachePolicy, *, resolve_cdef=None, resolve_reference=None):
        self.repo = repo
        self.cache = cache
        self.resolve_cdef = resolve_cdef
        self.resolve_reference = resolve_reference

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
            from .cdef_identity import cdef_node_key

            return cdef_node_key(obj)
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
                cache=self.cache,
                memo=ctx.memo,
                path=list(ctx.path),
            )

        if kind is NodeKind.REFERENCE_VALUE:
            from .reference_values import StateRef
            from .repo_plan import apply_exact_reference_identity

            if isinstance(obj, StateRef) and self.resolve_reference is not None:
                return self.resolve_reference(obj)
            if isinstance(obj, StateRef):
                return self.repo.load_state_ref(obj)

            reference = obj.object if isinstance(obj, StateRef) else obj
            realized = self.repo._materialize_cdef(
                reference.definition,
                cache=self.cache,
                memo=ctx.memo,
                path=list(ctx.path),
            )
            apply_exact_reference_identity(realized, reference)
            return realized

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
                cache=self.cache,
                memo=ctx.memo,
                path=list(ctx.path),
            )

        if kind is NodeKind.OBJECT:
            if self.resolve_cdef is not None:
                return self.resolve_cdef(obj.definition)
            if self.repo.get_cached(obj.definition) is None:
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

    if repo is not None and callable(getattr(repo, "resolve_state_selector", None)):
        ctx = GraphCtx(
            path=tuple(path) if path is not None else (),
            state={"repo": repo},
        )
        return _ToCanonicalTransformer().transform(x, ctx)

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
    cache: CachePolicy = "weak",
    memo: dict | None = None,
    path: list[str | int] | tuple[str | int, ...] | None = None,
    resolve_cdef=None,
    resolve_reference=None,
):
    from dryml.runtime import materialization_admission

    with materialization_admission(operation="canonical_reconstruction"):
        if memo is None:
            memo = {}

        ctx = GraphCtx(
            path=tuple(path) if path is not None else (),
            memo=memo,
        )
        return _FromCanonicalTransformer(
            repo, cache, resolve_cdef=resolve_cdef,
            resolve_reference=resolve_reference,
        ).transform(x, ctx)


def thaw_definition_surface_value(value: Any, *, memo: dict | None = None) -> Any:
    """Thaw canonical values while retaining private CDef graph topology.

    Args:
        value: Canonical value to project onto a Definition construction surface.
        memo: Optional operation-local graph memo shared by recursive calls.

    Returns:
        A mutable Definition surface whose shared CDef nodes remain shared
        Definition instances.
    """

    from .definition import Definition

    if memo is None:
        memo = {}

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
            lambda p, v: thaw_definition_surface_value(v, memo=memo),
            target="runtime",
        )
    if kind is NodeKind.CONCRETE_DEFINITION:
        from .cdef_identity import cdef_node_key

        key = cdef_node_key(value)
        cached = memo.get(key)
        if cached is not None:
            return cached
        from .materialization import project_cdef_call

        args, kwargs = project_cdef_call(value)
        result = Definition(
            value.cls,
            *thaw_definition_surface_value(args, memo=memo),
            **thaw_definition_surface_value(kwargs, memo=memo),
        )
        memo[key] = result
        return result
    if kind is NodeKind.DEFLINK:
        from .links import DefLink

        return DefLink(value.kind, thaw_definition_surface_value(value.target, memo=memo))
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
    if isinstance(value, (Definition, ConcreteDefinition, DefLink, QuotedDef, SelectorSpec, Selector, Par, ObjectRef, StateRef, StateSelectorRef)):
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
        NodeKind.REFERENCE_VALUE,
        NodeKind.STATE_SELECTOR_REF,
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
    if isinstance(value, (Definition, ConcreteDefinition, Selector, ObjectRef, StateRef, StateSelectorRef)):
        return value
    raise TypeError(f"DefLink target must be Definition, ConcreteDefinition, Selector, ObjectRef, StateRef, StateSelectorRef, or Object; got {type(value).__name__}.")
