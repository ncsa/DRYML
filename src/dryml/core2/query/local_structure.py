from __future__ import annotations

from typing import Any, Literal, Protocol

from ..cdef_graph import EdgeKind
from ..definition import ConcreteDefinition, Definition
from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from ..links import DefLink
from ..object import Object
from ..params import Par
from ..quoted import QuotedDef, SelectorSpec
from ..selector import Selector
from ..symbol import maybe_symbol_ref
from ..utils.stable_hash import stable_hash_function
from ..utils.types import is_nonclass_callable
from .model import ClassMatchPolicy
from .path import DefinitionPath, Key, Kwarg, SetMember, iter_value_edges


LocalStructureMode = Literal["target-full", "target-local", "selector-full", "selector-local"]


class LocalStructureCycleError(Exception):
    pass


class LocalStructureConsumer(Protocol):
    def feature(self, kind: str, path: DefinitionPath | None, payload: Any = None) -> None:
        ...

    def definition_boundary(
            self,
            path: DefinitionPath,
            definition: Definition | ConcreteDefinition,
            *,
            edge_kind: EdgeKind = EdgeKind.MATERIALIZE,
            unordered: bool = False) -> None:
        ...


def canonical_class_key(cls: Any) -> str:
    ref = maybe_symbol_ref(cls, functions=False)
    if ref is not None:
        return stable_hash_function(ref)
    return stable_hash_function(cls)


def scalar_key(value: Any) -> str | None:
    try:
        return stable_hash_function(value)
    except TypeError:
        return None


def walk_local_structure(
        value: Any,
        consumer: LocalStructureConsumer,
        *,
        mode: LocalStructureMode,
        class_match: ClassMatchPolicy = "selector",
        path: DefinitionPath | None = None,
        unordered_set_boundaries: bool = False) -> None:
    path = DefinitionPath() if path is None else path
    _walk(
        value,
        path,
        consumer,
        active={},
        mode=mode,
        class_match=class_match,
        unordered_set_boundaries=unordered_set_boundaries,
    )


def _walk(
        value: Any,
        path: DefinitionPath,
        consumer: LocalStructureConsumer,
        *,
        active: dict[int, DefinitionPath],
        mode: LocalStructureMode,
        class_match: ClassMatchPolicy,
        unordered_set_boundaries: bool) -> None:
    if isinstance(value, Object):
        value = value.definition

    is_target = mode.startswith("target")
    is_local = mode.endswith("local")
    if isinstance(value, DefLink):
        if is_local and path:
            target = value.target.root if isinstance(value.target, Selector) else value.target
            consumer.definition_boundary(path, target, edge_kind=value.kind)
        return

    if isinstance(value, Selector):
        value = SelectorSpec(value)

    if isinstance(value, (QuotedDef, SelectorSpec)):
        quoted = value.value if isinstance(value, QuotedDef) else value.selector
        consumer.feature("QUOTED_SELECTOR_AT", path, None)
        root = quoted.root if isinstance(quoted, Selector) else quoted
        if getattr(root, "cls", None) is not None:
            try:
                consumer.feature("QUOTED_SELECTOR_ROOT_CLASS", path, canonical_class_key(root.cls))
            except TypeError:
                pass
        return

    if isinstance(value, Par):
        return

    active_id = id(value) if _tracks_cycles(value) else None
    if active_id is not None:
        first_path = active.get(active_id)
        if first_path is not None:
            raise LocalStructureCycleError(
                f"Selector structure cycle at {path!s}; container first became active at {first_path!s}."
            )
        active[active_id] = path

    try:
        _walk_checked(
            value,
            path,
            consumer,
            active=active,
            mode=mode,
            class_match=class_match,
            unordered_set_boundaries=unordered_set_boundaries,
        )
    finally:
        if active_id is not None:
            active.pop(active_id, None)


def _walk_checked(
        value: Any,
        path: DefinitionPath,
        consumer: LocalStructureConsumer,
        *,
        active: dict[int, DefinitionPath],
        mode: LocalStructureMode,
        class_match: ClassMatchPolicy,
        unordered_set_boundaries: bool) -> None:

    is_target = mode.startswith("target")
    is_selector = mode.startswith("selector")
    is_local = mode.endswith("local")

    if isinstance(value, ConcreteDefinition):
        if is_target:
            if is_local and path:
                consumer.definition_boundary(path, value, edge_kind=EdgeKind.MATERIALIZE)
                return
            if is_local:
                consumer.feature("EXACT_NODE", path, value.stable_hash())
            consumer.feature("EXACT_SUBTREE", path, value.stable_hash())
            consumer.feature("CLASS_KEY", path, canonical_class_key(value.cls))
            _walk_definition_children(
                value,
                path,
                consumer,
                active=active,
                mode=mode,
                class_match=class_match,
                unordered_set_boundaries=unordered_set_boundaries,
            )
            return

        if is_selector and is_local and path:
            consumer.definition_boundary(path, value, edge_kind=EdgeKind.MATERIALIZE)
            return
        consumer.feature("EXACT_SUBTREE", path, value.stable_hash())
        return

    if is_selector and isinstance(value, Definition):
        from ..arg_roles import apply_definition_arg_roles

        value = apply_definition_arg_roles(value)
        if is_local and path:
            consumer.definition_boundary(path, value, edge_kind=EdgeKind.MATERIALIZE)
            return
        if value.cls is not None and class_match == "exact" and not is_nonclass_callable(value.cls):
            try:
                consumer.feature("CLASS_KEY", path, canonical_class_key(value.cls))
            except TypeError:
                pass
        _walk_definition_children(
            value,
            path,
            consumer,
            active=active,
            mode=mode,
            class_match=class_match,
            unordered_set_boundaries=unordered_set_boundaries,
        )
        return

    families = _container_families(value, mode=mode)
    if families or (is_selector and isinstance(value, FrozenList)):
        primary_family = families[0] if families else "list"
        for family in families:
            consumer.feature("CONTAINER_KIND", path, family)
        if primary_family in {"list", "tuple", "set"}:
            consumer.feature("SEQUENCE_LENGTH", path, len(value))
        if primary_family == "dict":
            _walk_mapping(
                value,
                path,
                consumer,
                active=active,
                mode=mode,
                class_match=class_match,
                unordered_set_boundaries=unordered_set_boundaries,
            )
        elif primary_family == "set":
            _walk_set(
                value,
                path,
                consumer,
                active=active,
                mode=mode,
                class_match=class_match,
                unordered_set_boundaries=unordered_set_boundaries,
            )
        else:
            _walk_children(
                value,
                path,
                consumer,
                active=active,
                mode=mode,
                class_match=class_match,
                unordered_set_boundaries=unordered_set_boundaries,
            )
        return

    if is_selector and is_nonclass_callable(value):
        return

    key = scalar_key(value)
    if key is not None:
        consumer.feature("SCALAR_VALUE", path, key)


def _walk_definition_children(
        definition: Definition | ConcreteDefinition,
        path: DefinitionPath,
        consumer: LocalStructureConsumer,
        *,
        active: dict[int, DefinitionPath],
        mode: LocalStructureMode,
        class_match: ClassMatchPolicy,
        unordered_set_boundaries: bool) -> None:
    for edge in iter_value_edges(definition):
        if isinstance(edge.segment, Kwarg):
            if not _can_match_absent(edge.value):
                consumer.feature("HAS_KWARG", path, edge.segment.name)
        _walk(
            edge.value,
            path.child(edge.segment),
            consumer,
            active=active,
            mode=mode,
            class_match=class_match,
            unordered_set_boundaries=unordered_set_boundaries,
        )


def _walk_mapping(
        value: Any,
        path: DefinitionPath,
        consumer: LocalStructureConsumer,
        *,
        active: dict[int, DefinitionPath],
        mode: LocalStructureMode,
        class_match: ClassMatchPolicy,
        unordered_set_boundaries: bool) -> None:
    for edge in iter_value_edges(value):
        if isinstance(edge.segment, Key):
            key_hash = scalar_key(edge.segment.key)
            if key_hash is not None and not _can_match_absent(edge.value):
                consumer.feature("HAS_MAPPING_KEY", path, key_hash)
        _walk(
            edge.value,
            path.child(edge.segment),
            consumer,
            active=active,
            mode=mode,
            class_match=class_match,
            unordered_set_boundaries=unordered_set_boundaries,
        )


def _walk_children(
        value: Any,
        path: DefinitionPath,
        consumer: LocalStructureConsumer,
        *,
        active: dict[int, DefinitionPath],
        mode: LocalStructureMode,
        class_match: ClassMatchPolicy,
        unordered_set_boundaries: bool) -> None:
    for edge in iter_value_edges(value):
        _walk(
            edge.value,
            path.child(edge.segment),
            consumer,
            active=active,
            mode=mode,
            class_match=class_match,
            unordered_set_boundaries=unordered_set_boundaries,
        )


def _walk_set(
        value: Any,
        path: DefinitionPath,
        consumer: LocalStructureConsumer,
        *,
        active: dict[int, DefinitionPath],
        mode: LocalStructureMode,
        class_match: ClassMatchPolicy,
        unordered_set_boundaries: bool) -> None:
    if mode in {"target-full", "selector-full"}:
        return
    if mode == "target-local":
        _walk_children(
            value,
            path,
            consumer,
            active=active,
            mode=mode,
            class_match=class_match,
            unordered_set_boundaries=unordered_set_boundaries,
        )
        return

    for edge in iter_value_edges(value):
        child = edge.value.definition if isinstance(edge.value, Object) else edge.value
        if not isinstance(child, (Definition, ConcreteDefinition)):
            continue
        if unordered_set_boundaries and isinstance(child, Definition):
            consumer.definition_boundary(path, child, edge_kind=EdgeKind.MATERIALIZE, unordered=True)
            continue
        member_path = path.child(edge.segment)
        if not isinstance(edge.segment, SetMember):
            raise TypeError(f"Expected SetMember path segment, got {type(edge.segment).__name__}.")
        consumer.definition_boundary(member_path, child, edge_kind=EdgeKind.MATERIALIZE)


def _container_families(value: Any, *, mode: LocalStructureMode) -> tuple[str, ...]:
    if mode.startswith("target") and isinstance(value, FrozenList):
        return ("list", "tuple")
    if mode.startswith("selector") and isinstance(value, FrozenList):
        return ()
    if isinstance(value, list):
        return ("list",)
    if isinstance(value, (tuple, FrozenTuple)):
        return ("tuple",)
    if isinstance(value, (set, frozenset, FrozenSet)):
        return ("set",)
    if isinstance(value, (dict, FrozenDict)):
        return ("dict",)
    return ()


def _tracks_cycles(value: Any) -> bool:
    return isinstance(value, (
        Definition,
        ConcreteDefinition,
        list,
        tuple,
        set,
        frozenset,
        dict,
        FrozenList,
        FrozenTuple,
        FrozenSet,
        FrozenDict,
    ))


def _can_match_absent(value: Any) -> bool:
    return isinstance(value, Par) and value.matches(None, present=False)
