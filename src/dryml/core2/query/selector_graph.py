from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..definition import ConcreteDefinition, Definition
from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from ..object import Object
from ..utils.types import is_nonclass_callable
from .fingerprint import canonical_class_key, scalar_key
from .model import ClassMatchPolicy, FeatureRequirement, FeatureToken
from .path import Arg, DefinitionPath, Index, Key, Kwarg, iter_set_members


SelectorNodeId = int


class SelectorGraphError(Exception):
    pass


class SelectorGraphCycleError(SelectorGraphError):
    pass


@dataclass(frozen=True, slots=True)
class SelectorGraphNode:
    node_id: SelectorNodeId
    source_path: DefinitionPath
    selector: Definition | ConcreteDefinition
    local_requirements: tuple[FeatureRequirement, ...]
    exact_definition: ConcreteDefinition | None = None


@dataclass(frozen=True, slots=True)
class SelectorGraphEdge:
    parent: SelectorNodeId
    path: DefinitionPath
    child: SelectorNodeId
    unordered: bool = False


@dataclass(frozen=True, slots=True)
class SelectorGraph:
    root: SelectorNodeId
    nodes: tuple[SelectorGraphNode, ...]
    edges: tuple[SelectorGraphEdge, ...]
    _outgoing: dict[SelectorNodeId, tuple[SelectorGraphEdge, ...]] = field(init=False, repr=False, compare=False)
    _incoming: dict[SelectorNodeId, tuple[SelectorGraphEdge, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        node_ids = [node.node_id for node in self.nodes]
        expected_ids = set(range(len(self.nodes)))
        if set(node_ids) != expected_ids or len(node_ids) != len(set(node_ids)):
            raise SelectorGraphError("SelectorGraph node IDs must be unique contiguous tuple indexes.")
        if self.root not in expected_ids:
            raise SelectorGraphError(f"SelectorGraph root {self.root!r} is not a graph node.")

        outgoing: dict[SelectorNodeId, list[SelectorGraphEdge]] = defaultdict(list)
        incoming: dict[SelectorNodeId, list[SelectorGraphEdge]] = defaultdict(list)
        for edge in self.edges:
            if edge.parent not in expected_ids:
                raise SelectorGraphError(f"SelectorGraph edge parent {edge.parent!r} is not a graph node.")
            if edge.child not in expected_ids:
                raise SelectorGraphError(f"SelectorGraph edge child {edge.child!r} is not a graph node.")
            outgoing[edge.parent].append(edge)
            incoming[edge.child].append(edge)

        outgoing_t = {node_id: tuple(edges) for node_id, edges in outgoing.items()}
        incoming_t = {node_id: tuple(edges) for node_id, edges in incoming.items()}
        _validate_acyclic(self.root, expected_ids, outgoing_t)
        object.__setattr__(self, "_outgoing", outgoing_t)
        object.__setattr__(self, "_incoming", incoming_t)

    def node(self, node_id: SelectorNodeId) -> SelectorGraphNode:
        return self.nodes[node_id]

    def outgoing(self, node_id: SelectorNodeId) -> tuple[SelectorGraphEdge, ...]:
        return self._outgoing.get(node_id, ())

    def incoming(self, node_id: SelectorNodeId) -> tuple[SelectorGraphEdge, ...]:
        return self._incoming.get(node_id, ())


def compile_selector_graph(
        selector: Definition | ConcreteDefinition | None,
        *,
        class_match: ClassMatchPolicy = "selector") -> SelectorGraph | None:
    if selector is None:
        return None
    if isinstance(selector, Object):
        selector = selector.definition
    if not isinstance(selector, (Definition, ConcreteDefinition)):
        return None
    compiler = _SelectorGraphCompiler(class_match=class_match)
    root = compiler.add_node(selector, DefinitionPath())
    return SelectorGraph(root=root, nodes=tuple(compiler.nodes), edges=tuple(compiler.edges))


class _SelectorGraphCompiler:
    def __init__(self, *, class_match: ClassMatchPolicy):
        self.class_match = class_match
        self.nodes: list[SelectorGraphNode] = []
        self.edges: list[SelectorGraphEdge] = []
        self.active: dict[int, DefinitionPath] = {}

    def add_node(self, selector: Definition | ConcreteDefinition, source_path: DefinitionPath) -> SelectorNodeId:
        active_id = id(selector) if isinstance(selector, Definition) else None
        if active_id is not None:
            first_path = self.active.get(active_id)
            if first_path is not None:
                raise SelectorGraphCycleError(
                    f"Selector Definition cycle detected at {source_path!s}; first active path was {first_path!s}."
                )
            self.active[active_id] = source_path

        node_id = len(self.nodes)
        self.nodes.append(SelectorGraphNode(node_id, source_path, selector, (), None))
        try:
            counts: Counter[FeatureToken] = Counter()
            exact = selector if isinstance(selector, ConcreteDefinition) else None
            if exact is not None:
                counts[FeatureToken("EXACT_SUBTREE", DefinitionPath(), exact.stable_hash())] += 1
            else:
                self._local_requirements(selector, DefinitionPath(), counts, parent_id=node_id, source_path=source_path)
            requirements = tuple(
                FeatureRequirement(token, count)
                for token, count in sorted(counts.items(), key=lambda item: repr(item[0]))
            )
            self.nodes[node_id] = SelectorGraphNode(node_id, source_path, selector, requirements, exact)
            return node_id
        finally:
            if active_id is not None:
                self.active.pop(active_id, None)

    def _add(self, counts: Counter[FeatureToken], kind: str, path: DefinitionPath | None, payload: Any = None) -> None:
        counts[FeatureToken(kind, path, payload)] += 1

    def _local_requirements(
            self,
            value: Any,
            path: DefinitionPath,
            counts: Counter[FeatureToken],
            *,
            parent_id: SelectorNodeId,
            source_path: DefinitionPath) -> None:
        if isinstance(value, Object):
            value = value.definition

        if isinstance(value, ConcreteDefinition):
            child_id = self.add_node(value, source_path.join(path))
            self._add(counts, "CDEF_EDGE_AT_PATH", path, None)
            self._add(counts, "CDEF_EDGE_EXACT", path, value.stable_hash())
            self.edges.append(SelectorGraphEdge(parent_id, path, child_id))
            return

        if isinstance(value, Definition):
            if path:
                child_id = self.add_node(value, source_path.join(path))
                self._add(counts, "CDEF_EDGE_AT_PATH", path, None)
                self.edges.append(SelectorGraphEdge(parent_id, path, child_id))
                return
            if value.cls is not None and self.class_match == "exact" and not is_nonclass_callable(value.cls):
                try:
                    self._add(counts, "CLASS_KEY", path, canonical_class_key(value.cls))
                except TypeError:
                    pass
            if value.args is not None:
                for idx, child in enumerate(value.args):
                    self._local_requirements(child, path.child(Arg(idx)), counts, parent_id=parent_id, source_path=source_path)
            for key, child in value.kwargs.items():
                self._add(counts, "HAS_KWARG", path, key)
                self._local_requirements(child, path.child(Kwarg(key)), counts, parent_id=parent_id, source_path=source_path)
            return

        families = _container_families(value)
        if families or isinstance(value, FrozenList):
            primary_family = families[0] if families else "list"
            for family in families:
                self._add(counts, "CONTAINER_KIND", path, family)
            if primary_family in {"list", "tuple", "set"}:
                self._add(counts, "SEQUENCE_LENGTH", path, len(value))
            if primary_family == "dict":
                for key, child in value.items():
                    key_hash = scalar_key(key)
                    if key_hash is not None:
                        self._add(counts, "HAS_MAPPING_KEY", path, key_hash)
                    self._local_requirements(child, path.child(Key(key)), counts, parent_id=parent_id, source_path=source_path)
            elif primary_family == "set":
                for member_path, child in iter_set_members(value):
                    member_abs = path.child(member_path)
                    if isinstance(child, Object):
                        child = child.definition
                    if isinstance(child, ConcreteDefinition):
                        child_id = self.add_node(child, source_path.join(member_abs))
                        self._add(counts, "CDEF_EDGE_AT_PATH", member_abs, None)
                        self._add(counts, "CDEF_EDGE_EXACT", member_abs, child.stable_hash())
                        self.edges.append(SelectorGraphEdge(parent_id, member_abs, child_id))
                    elif isinstance(child, Definition):
                        child_id = self.add_node(child, source_path.join(path))
                        self._add(counts, "CDEF_EDGE_AT_PATH", path, None)
                        self.edges.append(SelectorGraphEdge(parent_id, path, child_id, unordered=True))
            else:
                for idx, child in enumerate(value):
                    self._local_requirements(child, path.child(Index(idx)), counts, parent_id=parent_id, source_path=source_path)
            return

        if is_nonclass_callable(value):
            return

        key = scalar_key(value)
        if key is not None:
            self._add(counts, "SCALAR_VALUE", path, key)


def _container_families(value: Any) -> tuple[str, ...]:
    if isinstance(value, FrozenList):
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


def _validate_acyclic(
        root: SelectorNodeId,
        node_ids: set[SelectorNodeId],
        outgoing: dict[SelectorNodeId, tuple[SelectorGraphEdge, ...]]) -> None:
    visited: set[SelectorNodeId] = set()
    active: set[SelectorNodeId] = set()

    def visit(node_id: SelectorNodeId) -> None:
        if node_id in visited:
            return
        if node_id in active:
            raise SelectorGraphCycleError(f"SelectorGraph cycle detected at node {node_id}.")
        active.add(node_id)
        try:
            for edge in outgoing.get(node_id, ()):
                visit(edge.child)
        finally:
            active.remove(node_id)
        visited.add(node_id)

    visit(root)
    for node_id in node_ids:
        visit(node_id)
