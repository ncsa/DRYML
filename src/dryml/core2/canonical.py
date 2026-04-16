from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .freeze import (
    FrozenDict,
    FrozenList,
    FrozenNDArray,
    FrozenSet,
    FrozenTuple,
)
from .types import is_pod

if TYPE_CHECKING:
    from .definition import ConcreteDefinition
    from .repo import Repo


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

    NDARRAY = auto()
    FROZEN_NDARRAY = auto()

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
    OBJECT = auto()

    UNSUPPORTED = auto()


def node_kind(x: Any) -> NodeKind:
    from .definition import ConcreteDefinition, Definition
    from .object import Object

    if is_pod(x):
        return NodeKind.POD

    if isinstance(x, type):
        return NodeKind.TYPE

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

    if isinstance(x, ConcreteDefinition):
        return NodeKind.CONCRETE_DEFINITION

    if isinstance(x, Definition):
        return NodeKind.DEFINITION

    if isinstance(x, Object):
        return NodeKind.OBJECT

    return NodeKind.UNSUPPORTED


def is_canonical_value(x: Any) -> bool:
    return node_kind(x) in {
        NodeKind.POD,
        NodeKind.TYPE,
        NodeKind.FROZEN_NDARRAY,
        NodeKind.FROZEN_LIST,
        NodeKind.FROZEN_TUPLE,
        NodeKind.FROZEN_SET,
        NodeKind.FROZEN_DICT,
        NodeKind.CONCRETE_DEFINITION,
    }


def is_runtime_leaf(x: Any) -> bool:
    """
    Values that should usually be treated as terminal by runtime-oriented
    graph traversals.
    """
    return node_kind(x) in {
        NodeKind.POD,
        NodeKind.TYPE,
        NodeKind.NDARRAY,
        NodeKind.FROZEN_NDARRAY,
    }


# ----------------------------------------------------------------------
# Child traversal / rebuilding helpers
# ----------------------------------------------------------------------

def _path_part_from_key(k: Any) -> str | int:
    return k if isinstance(k, (str, int)) else str(k)


def iter_value_children(x: Any):
    """
    Yield (path_part, child) for the value-children of a supported container.

    For dict-like containers, keys are preserved and only values are yielded.
    """
    kind = node_kind(x)

    if kind in {NodeKind.LIST, NodeKind.TUPLE, NodeKind.SET,
                NodeKind.FROZEN_LIST, NodeKind.FROZEN_TUPLE, NodeKind.FROZEN_SET}:
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
    """
    kind = node_kind(x)

    if kind is NodeKind.DICT:
        out = {}
        for k, v in x.items():
            out[key_fn(k)] = value_fn(k, v)
        return out

    if kind is NodeKind.FROZEN_DICT:
        out = {}
        for k, v in x.items():
            out[key_fn(k)] = value_fn(k, v)
        return FrozenDict(out)

    raise TypeError(f"{type(x).__name__} is not dict-like")


# ----------------------------------------------------------------------
# Thin public wrappers for now
# ----------------------------------------------------------------------

def to_canonical(x: Any, *, repo: "Repo | None" = None) -> CanonicalValue:
    """
    Thin wrapper for now. The actual implementation still lives in the
    concretization transformer.
    """
    from .definition import concretize_func
    return concretize_func(x, repo=repo)


def thaw_value(x: Any) -> Any:
    """
    Thin wrapper for now. The actual implementation still lives in the
    thaw transformer.
    """
    from .definition import thaw_concrete
    return thaw_concrete(x)


def from_canonical(
    x: Any,
    *,
    repo: "Repo",
    instance="reuse",
    restore_state: bool = True,
    build_missing: bool = False,
    reuse_weak: bool = True,
    cache="weak",
    revision=None,
):
    """
    Thin wrapper for now. Later this can become the direct de-canonicalization
    entry point if you want to move more logic out of Repo.
    """
    return repo.load_object(
        x,
        instance=instance,
        restore_state=restore_state,
        build_missing=build_missing,
        reuse_weak=reuse_weak,
        cache=cache,
        revision=revision,
    )
