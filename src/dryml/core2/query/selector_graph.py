from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..cdef_graph import EdgeKind
from ..definition import ConcreteDefinition, Definition
from ..object import Object
from ..selector import Selector
from .local_structure import LocalStructureCycleError, walk_local_structure
from .model import ClassMatchPolicy, FeatureRequirement, FeatureToken
from .path import DefinitionPath


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
    edge_kind: EdgeKind = EdgeKind.MATERIALIZE


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
        selector: Definition | ConcreteDefinition | Selector | None,
        *,
        class_match: ClassMatchPolicy | None = None) -> SelectorGraph | None:
    if selector is None:
        return None
    if isinstance(selector, Object):
        selector = selector.definition
    if isinstance(selector, Selector):
        if class_match is None:
            class_match = selector.cls_policy
        selector = selector.root
    if not isinstance(selector, (Definition, ConcreteDefinition)):
        return None
    if class_match is None:
        class_match = "selector"
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
                try:
                    walk_local_structure(
                        selector,
                        _GraphLocalConsumer(self, counts, parent_id=node_id, source_path=source_path),
                        mode="selector-local",
                        class_match=self.class_match,
                        unordered_set_boundaries=True,
                    )
                except LocalStructureCycleError as e:
                    raise SelectorGraphCycleError(str(e)) from e
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


class _GraphLocalConsumer:
    def __init__(
            self,
            compiler: _SelectorGraphCompiler,
            counts: Counter[FeatureToken],
            *,
            parent_id: SelectorNodeId,
            source_path: DefinitionPath):
        self.compiler = compiler
        self.counts = counts
        self.parent_id = parent_id
        self.source_path = source_path

    def feature(self, kind: str, path: DefinitionPath | None, payload: Any = None) -> None:
        self.compiler._add(self.counts, kind, path, payload)

    def definition_boundary(
            self,
            path: DefinitionPath,
            definition: Definition | ConcreteDefinition,
            *,
            edge_kind: EdgeKind = EdgeKind.MATERIALIZE,
            unordered: bool = False) -> None:
        child_id = self.compiler.add_node(definition, self.source_path.join(path))
        self.feature(f"CDEF_EDGE_AT_PATH:{edge_kind.value}", path, None)
        if edge_kind is EdgeKind.MATERIALIZE:
            self.feature("CDEF_EDGE_AT_PATH", path, None)
        if isinstance(definition, ConcreteDefinition):
            self.feature(f"CDEF_EDGE_EXACT:{edge_kind.value}", path, definition.stable_hash())
            if edge_kind is EdgeKind.MATERIALIZE:
                self.feature("CDEF_EDGE_EXACT", path, definition.stable_hash())
        self.compiler.edges.append(SelectorGraphEdge(self.parent_id, path, child_id, unordered=unordered, edge_kind=edge_kind))


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
