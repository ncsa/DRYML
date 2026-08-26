from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from ...cdef_identity import V2_IDENTITY_VERSION
from ...definition import ConcreteDefinition, Definition, SKIP_ARGS
from ...freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from ...utils.stable_hash import stable_hash_function
from .path import (
    Arg,
    DefinitionPathLike,
    GraphPath,
    Index,
    Key,
    Kwarg,
    Parameter,
    PathSegment,
    QueryPathError,
    SetMember,
    normalize_path,
)


@dataclass(frozen=True, slots=True)
class ValueEdge:
    segment: PathSegment
    value: Any


def set_member_segment(value: Any, *, ordinal: int = 0) -> SetMember:
    return SetMember(stable_hash_function(value), ordinal)


def iter_set_members(values: Iterable[Any]) -> tuple[tuple[SetMember, Any], ...]:
    buckets: dict[str, list[Any]] = defaultdict(list)
    for value in values:
        try:
            fp = stable_hash_function(value)
        except TypeError as e:
            raise QueryPathError(
                "Cannot build stable set path for a non-stably-hashable set member."
            ) from e
        buckets[fp].append(value)

    out: list[tuple[SetMember, Any]] = []
    for fp in sorted(buckets):
        repr_groups: dict[str, list[Any]] = defaultdict(list)
        for value in buckets[fp]:
            repr_groups[repr(value)].append(value)
        ambiguous = [values for values in repr_groups.values() if len(values) > 1 and any(value != values[0] for value in values[1:])]
        if ambiguous:
            raise QueryPathError(
                "Cannot build deterministic set paths for unequal members with the same stable hash and repr."
            )
        bucket = sorted(buckets[fp], key=repr)
        for ordinal, value in enumerate(bucket):
            out.append((SetMember(fp, ordinal), value))
    return tuple(out)


def resolve_set_member(values: Iterable[Any], segment: SetMember) -> Any:
    matches = [value for seg, value in iter_set_members(values) if seg == segment]
    if not matches:
        raise QueryPathError(f"Set member {segment!s} was not found.")
    if len(matches) > 1:
        raise QueryPathError(f"Set member {segment!s} is ambiguous.")
    return matches[0]


def iter_value_edges(value: Any) -> tuple[ValueEdge, ...]:
    if isinstance(value, ConcreteDefinition) and value.identity_version == V2_IDENTITY_VERSION:
        return tuple(ValueEdge(Parameter(name), child) for name, child in value.parameters.items())

    if isinstance(value, (Definition, ConcreteDefinition)):
        edges: list[ValueEdge] = []
        if value.args is not None:
            for idx, child in enumerate(value.args):
                edges.append(ValueEdge(Arg(idx), child))
        for key, child in value.kwargs.items():
            edges.append(ValueEdge(Kwarg(key), child))
        return tuple(edges)

    if isinstance(value, (FrozenDict, dict)):
        return tuple(ValueEdge(Key(key), child) for key, child in value.items())

    if isinstance(value, (FrozenList, FrozenTuple, list, tuple)):
        return tuple(ValueEdge(Index(idx), child) for idx, child in enumerate(value))

    if isinstance(value, (FrozenSet, set, frozenset)):
        return tuple(ValueEdge(seg, child) for seg, child in iter_set_members(value))

    return ()


def get_subtree(obj: Any, path: DefinitionPathLike = "$") -> Any:
    norm = normalize_path(path)
    cur = obj
    for idx, seg in enumerate(norm):
        try:
            cur = _get_child(cur, seg)
        except Exception as e:
            failing = GraphPath(norm.segments[:idx + 1])
            raise QueryPathError(f"Failed to resolve segment {seg!s} at {failing!s}.") from e
    return cur


def replace_subtree(obj: Any, path: DefinitionPathLike, replacement: Any) -> Any:
    norm = normalize_path(path)
    if len(norm) == 0:
        return replacement
    seg = norm[0]
    rest = GraphPath(norm.segments[1:])
    child = get_subtree(obj, GraphPath((seg,)))
    new_child = replace_subtree(child, rest, replacement)
    return _replace_child(obj, seg, new_child)


def _get_child(obj: Any, seg: PathSegment) -> Any:
    if isinstance(obj, ConcreteDefinition) and obj.identity_version == V2_IDENTITY_VERSION:
        if isinstance(seg, Parameter):
            return obj.parameters[seg.name]
        raise TypeError(f"{seg!s} is not valid on a V2 concrete definition.")

    if isinstance(obj, (Definition, ConcreteDefinition)):
        if isinstance(seg, Kwarg):
            return obj.kwargs[seg.name]
        if isinstance(seg, Arg):
            if obj.args is None:
                raise KeyError(seg.index)
            return obj.args[seg.index]
        raise TypeError(f"{seg!s} is not valid on a definition.")

    if isinstance(obj, (dict, FrozenDict)):
        if isinstance(seg, Key):
            return obj[seg.key]
        if isinstance(seg, Kwarg):
            return obj[seg.name]
        raise TypeError(f"{seg!s} is not valid on a mapping.")

    if isinstance(obj, (list, tuple, FrozenList, FrozenTuple)):
        if isinstance(seg, Index):
            return obj[seg.index]
        raise TypeError(f"{seg!s} is not valid on a sequence.")

    if isinstance(obj, (set, frozenset, FrozenSet)):
        if isinstance(seg, SetMember):
            return resolve_set_member(obj, seg)
        if isinstance(seg, Index):
            # Compatibility for old user-authored paths. Graph-generated paths
            # use SetMember because numeric set positions are not semantic.
            return iter_set_members(obj)[seg.index][1]
        raise TypeError(f"{seg!s} is not valid on a set.")

    raise TypeError(f"Cannot traverse into {type(obj).__name__}.")


def _replace_child(obj: Any, seg: PathSegment, child: Any) -> Any:
    if isinstance(obj, (Definition, ConcreteDefinition)):
        args = None if obj.args is None else list(obj.args)
        kwargs = dict(obj.kwargs)
        if isinstance(seg, Kwarg):
            if seg.name not in kwargs:
                raise QueryPathError(f"Missing kwarg {seg.name!r} while replacing {seg!s}.")
            kwargs[seg.name] = child
        elif isinstance(seg, Arg):
            if args is None:
                raise QueryPathError(f"Cannot replace arg {seg.index}; definition skips args.")
            args[seg.index] = child
        else:
            raise QueryPathError(f"{seg!s} is not valid on a definition.")

        if args is None:
            return Definition(obj.cls, SKIP_ARGS, **kwargs)
        return Definition(obj.cls, *args, **kwargs)

    if isinstance(obj, list):
        if not isinstance(seg, Index):
            raise QueryPathError(f"{seg!s} is not valid on a list.")
        out = list(obj)
        out[seg.index] = child
        return out

    if isinstance(obj, tuple) and not isinstance(obj, (FrozenList, FrozenTuple)):
        if not isinstance(seg, Index):
            raise QueryPathError(f"{seg!s} is not valid on a tuple.")
        out = list(obj)
        out[seg.index] = child
        return tuple(out)

    if isinstance(obj, FrozenList):
        if not isinstance(seg, Index):
            raise QueryPathError(f"{seg!s} is not valid on a FrozenList.")
        out = list(obj)
        out[seg.index] = child
        return FrozenList(out)

    if isinstance(obj, FrozenTuple):
        if not isinstance(seg, Index):
            raise QueryPathError(f"{seg!s} is not valid on a FrozenTuple.")
        out = list(obj)
        out[seg.index] = child
        return FrozenTuple(out)

    if isinstance(obj, dict):
        key = _mapping_key_from_segment(seg)
        if key not in obj:
            raise QueryPathError(f"Missing mapping key {key!r} while replacing {seg!s}.")
        out = dict(obj)
        out[key] = child
        return out

    if isinstance(obj, FrozenDict):
        key = _mapping_key_from_segment(seg)
        if key not in obj:
            raise QueryPathError(f"Missing mapping key {key!r} while replacing {seg!s}.")
        out = dict(obj.items())
        out[key] = child
        return FrozenDict(out)

    if isinstance(obj, set):
        return _replace_set_member(obj, seg, child, set)

    if isinstance(obj, frozenset) and not isinstance(obj, FrozenSet):
        return _replace_set_member(obj, seg, child, frozenset)

    if isinstance(obj, FrozenSet):
        return _replace_set_member(obj, seg, child, FrozenSet)

    raise QueryPathError(f"Cannot replace a child on {type(obj).__name__}.")


def _mapping_key_from_segment(seg: PathSegment) -> Any:
    if isinstance(seg, Key):
        return seg.key
    if isinstance(seg, Kwarg):
        return seg.name
    raise QueryPathError(f"{seg!s} is not valid on a mapping.")


def _replace_set_member(obj: Iterable[Any], seg: PathSegment, child: Any, factory):
    if not isinstance(seg, SetMember):
        raise QueryPathError("Replacing set members requires a stable SetMember path segment.")
    old = resolve_set_member(obj, seg)
    out = set(obj)
    out.remove(old)
    try:
        out.add(child)
    except TypeError as e:
        raise QueryPathError("Replacement set member must be hashable.") from e
    if len(out) != len(set(obj)):
        raise QueryPathError("Replacement collapsed two distinct set members.")
    return factory(out)
