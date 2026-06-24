from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from .definition import ConcreteDefinition, Definition
from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from .object import Object
from .utils.graph.path import Arg, GraphPath, Index, Key, Kwarg, SetMember
from .utils.graph.value import get_subtree, iter_set_members


class ConcreteDefinitionGraphError(Exception):
    pass


class ConcreteDefinitionGraphCycleError(ConcreteDefinitionGraphError):
    pass


class ConcreteDefinitionGraphOccurrenceLimitError(ConcreteDefinitionGraphError):
    pass


@dataclass(frozen=True, slots=True)
class CDefNode:
    definition: ConcreteDefinition
    stable_hash: str


@dataclass(frozen=True, slots=True)
class CDefEdge:
    parent: ConcreteDefinition
    path: GraphPath
    child: ConcreteDefinition


@dataclass(frozen=True, slots=True)
class CDefOccurrence:
    root: ConcreteDefinition
    path: GraphPath
    definition: ConcreteDefinition


def iter_direct_cdef_edges(cdef: ConcreteDefinition) -> Iterator[tuple[GraphPath, ConcreteDefinition]]:
    if not isinstance(cdef, ConcreteDefinition):
        raise TypeError(f"Expected ConcreteDefinition, got {type(cdef).__name__}.")

    for idx, child in enumerate(cdef.args):
        yield from _iter_direct_edges_from_value(child, GraphPath((Arg(idx),)))
    for key, child in cdef.kwargs.items():
        yield from _iter_direct_edges_from_value(child, GraphPath((Kwarg(key),)))


def _iter_direct_edges_from_value(value: Any, path: GraphPath) -> Iterator[tuple[GraphPath, ConcreteDefinition]]:
    if isinstance(value, Object):
        raise ConcreteDefinitionGraphError(f"Runtime Object found inside ConcreteDefinition graph at {path!s}.")
    if isinstance(value, Definition):
        raise ConcreteDefinitionGraphError(f"Plain Definition found inside ConcreteDefinition graph at {path!s}.")
    if isinstance(value, ConcreteDefinition):
        yield path, value
        return

    if isinstance(value, (FrozenDict, dict)):
        for key, child in value.items():
            yield from _iter_direct_edges_from_value(child, path.child(Key(key)))
        return

    if isinstance(value, (FrozenList, FrozenTuple, list, tuple)):
        for idx, child in enumerate(value):
            yield from _iter_direct_edges_from_value(child, path.child(Index(idx)))
        return

    if isinstance(value, (FrozenSet, set, frozenset)):
        for seg, child in iter_set_members(value):
            yield from _iter_direct_edges_from_value(child, path.child(seg))
        return


class ConcreteDefinitionGraph:
    def __init__(
            self,
            roots: Iterable[ConcreteDefinition],
            nodes: Iterable[CDefNode],
            edges: Iterable[CDefEdge]):
        self._roots = tuple(dict.fromkeys(roots))
        self._nodes = tuple(nodes)
        self._edges = tuple(dict.fromkeys(edges))
        _validate_graph_parts(self._roots, self._nodes, self._edges)
        self._node_by_cdef = {node.definition: node for node in self._nodes}
        outgoing: dict[ConcreteDefinition, list[CDefEdge]] = defaultdict(list)
        incoming: dict[ConcreteDefinition, list[CDefEdge]] = defaultdict(list)
        for edge in self._edges:
            outgoing[edge.parent].append(edge)
            incoming[edge.child].append(edge)
        self._outgoing = {cdef: tuple(edges) for cdef, edges in outgoing.items()}
        self._incoming = {cdef: tuple(edges) for cdef, edges in incoming.items()}

    @classmethod
    def from_root(cls, cdef: ConcreteDefinition) -> "ConcreteDefinitionGraph":
        return cls.from_roots((cdef,))

    @classmethod
    def from_roots(cls, cdefs: Iterable[ConcreteDefinition]) -> "ConcreteDefinitionGraph":
        builder = ConcreteDefinitionGraphBuilder()
        builder.add_roots(cdefs)
        return builder.build()

    @property
    def roots(self) -> tuple[ConcreteDefinition, ...]:
        return self._roots

    def node(self, cdef: ConcreteDefinition) -> CDefNode:
        return self._node_by_cdef[cdef]

    def nodes(self) -> tuple[CDefNode, ...]:
        return self._nodes

    def edges(self) -> tuple[CDefEdge, ...]:
        return self._edges

    def outgoing(self, cdef: ConcreteDefinition) -> tuple[CDefEdge, ...]:
        return self._outgoing.get(cdef, ())

    def incoming(self, cdef: ConcreteDefinition) -> tuple[CDefEdge, ...]:
        return self._incoming.get(cdef, ())

    def walk_nodes(self, *, order: str = "pre") -> tuple[CDefNode, ...]:
        if order not in {"pre", "post"}:
            raise ValueError("order must be 'pre' or 'post'.")
        seen: set[ConcreteDefinition] = set()
        out: list[CDefNode] = []

        def visit(cdef: ConcreteDefinition) -> None:
            if cdef in seen:
                return
            seen.add(cdef)
            if order == "pre":
                out.append(self.node(cdef))
            for edge in self.outgoing(cdef):
                visit(edge.child)
            if order == "post":
                out.append(self.node(cdef))

        for root in self.roots:
            visit(root)
        return tuple(out)

    def topological_order(self, *, dependencies_first: bool = True) -> tuple[ConcreteDefinition, ...]:
        nodes = tuple(node.definition for node in self.walk_nodes(order="post"))
        return nodes if dependencies_first else tuple(reversed(nodes))

    def iter_occurrences(
            self,
            *,
            roots: Iterable[ConcreteDefinition] | None = None,
            include_roots: bool = False,
            target: ConcreteDefinition | None = None,
            max_occurrences: int | None = None,
            limit: int | None = None) -> Iterator[CDefOccurrence]:
        count = 0
        selected_roots = tuple(roots) if roots is not None else self.roots
        for root in selected_roots:
            if include_roots and (target is None or _same_cdef(root, target)):
                count += 1
                if max_occurrences is not None and count > max_occurrences:
                    raise ConcreteDefinitionGraphOccurrenceLimitError(
                        f"Occurrence limit {max_occurrences} exceeded while walking {root.stable_hash()}."
                    )
                yield CDefOccurrence(root, GraphPath(), root)
                if limit is not None and count >= limit:
                    return
            stack = [(edge, edge.path) for edge in reversed(self.outgoing(root))]
            while stack:
                edge, path = stack.pop()
                if target is None or _same_cdef(edge.child, target):
                    count += 1
                    if max_occurrences is not None and count > max_occurrences:
                        raise ConcreteDefinitionGraphOccurrenceLimitError(
                            f"Occurrence limit {max_occurrences} exceeded while walking {root.stable_hash()}."
                        )
                    yield CDefOccurrence(root, path, edge.child)
                    if limit is not None and count >= limit:
                        return
                for child_edge in reversed(self.outgoing(edge.child)):
                    stack.append((child_edge, path.join(child_edge.path)))

    def resolve(self, root: ConcreteDefinition, path: GraphPath) -> ConcreteDefinition:
        if not path:
            return root
        value = get_subtree(root, path)
        if not isinstance(value, ConcreteDefinition):
            raise ConcreteDefinitionGraphError(f"Path {path!s} does not resolve to a ConcreteDefinition boundary.")
        return value

    def paths_to(
            self,
            root: ConcreteDefinition,
            target: ConcreteDefinition,
            *,
            max_paths: int | None = None) -> tuple[GraphPath, ...]:
        return tuple(occ.path for occ in self.iter_occurrences(
            roots=(root,),
            target=target,
            limit=max_paths,
        ))

    def contains(self, root: ConcreteDefinition, target: ConcreteDefinition) -> bool:
        if _same_cdef(root, target):
            return True
        return any(True for _ in self.iter_occurrences(roots=(root,), target=target, limit=1))

    def primary_path(self, root: ConcreteDefinition, target: ConcreteDefinition) -> GraphPath | None:
        paths = self.paths_to(root, target, max_paths=1)
        return paths[0] if paths else None

    def ancestors(self, cdef: ConcreteDefinition) -> tuple[ConcreteDefinition, ...]:
        seen: set[ConcreteDefinition] = set()
        out: list[ConcreteDefinition] = []
        queue = deque(edge.parent for edge in self.incoming(cdef))
        while queue:
            cur = queue.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            queue.extend(edge.parent for edge in self.incoming(cur))
        return tuple(out)

    def descendants(self, cdef: ConcreteDefinition) -> tuple[ConcreteDefinition, ...]:
        seen: set[ConcreteDefinition] = set()
        out: list[ConcreteDefinition] = []
        stack = [edge.child for edge in reversed(self.outgoing(cdef))]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            stack.extend(edge.child for edge in reversed(self.outgoing(cur)))
        return tuple(out)

    def roots_containing(self, cdef: ConcreteDefinition) -> tuple[ConcreteDefinition, ...]:
        return tuple(root for root in self.roots if not _same_cdef(root, cdef) and self.contains(root, cdef))

    def explain(self) -> str:
        return "\n".join((
            f"roots: {len(self.roots)}",
            f"nodes: {len(self.nodes())}",
            f"edges: {len(self.edges())}",
        ))


class ConcreteDefinitionGraphBuilder:
    def __init__(self):
        self._roots: list[ConcreteDefinition] = []
        self._nodes: dict[ConcreteDefinition, CDefNode] = {}
        self._edges: dict[tuple[ConcreteDefinition, GraphPath, ConcreteDefinition], CDefEdge] = {}
        self._edges_by_parent: dict[ConcreteDefinition, list[CDefEdge]] = defaultdict(list)
        self._completed: set[ConcreteDefinition] = set()
        self._active: dict[ConcreteDefinition, GraphPath] = {}

    def add_root(self, cdef: ConcreteDefinition) -> None:
        if not isinstance(cdef, ConcreteDefinition):
            raise TypeError(f"Graph roots must be ConcreteDefinitions, got {type(cdef).__name__}.")
        if not any(_same_cdef(cdef, root) for root in self._roots):
            self._roots.append(cdef)
        self._visit(cdef, GraphPath())

    def add_roots(self, cdefs: Iterable[ConcreteDefinition]) -> None:
        for cdef in cdefs:
            self.add_root(cdef)

    def build(self) -> ConcreteDefinitionGraph:
        node_order = self._node_order()
        order_index = {cdef: idx for idx, cdef in enumerate(node_order)}
        ordered_nodes = [self._nodes[cdef] for cdef in node_order]
        ordered_edges = tuple(sorted(
            self._edges.values(),
            key=lambda edge: (
                order_index.get(edge.parent, len(order_index)),
                str(edge.path),
                edge.child.stable_hash(),
                repr(edge.child),
            ),
        ))
        return ConcreteDefinitionGraph(tuple(self._roots), ordered_nodes, ordered_edges)

    def _visit(self, cdef: ConcreteDefinition, path: GraphPath) -> None:
        if cdef in self._completed:
            return
        if cdef in self._active:
            first = self._active[cdef]
            raise ConcreteDefinitionGraphCycleError(
                f"ConcreteDefinition cycle detected at {path!s}; first active path was {first!s}; hash={cdef.stable_hash()}."
            )

        self._active[cdef] = path
        self._nodes.setdefault(cdef, CDefNode(cdef, cdef.stable_hash()))
        try:
            for edge_path, child in iter_direct_cdef_edges(cdef):
                edge = CDefEdge(cdef, edge_path, child)
                edge_key = (edge.parent, edge.path, edge.child)
                if edge_key not in self._edges:
                    self._edges[edge_key] = edge
                    self._edges_by_parent.setdefault(edge.parent, []).append(edge)
                self._visit(child, path.join(edge_path))
        finally:
            self._active.pop(cdef, None)
            self._completed.add(cdef)

    def _node_order(self) -> tuple[ConcreteDefinition, ...]:
        seen: set[ConcreteDefinition] = set()
        out: list[ConcreteDefinition] = []

        def visit(cdef: ConcreteDefinition) -> None:
            if cdef in seen:
                return
            seen.add(cdef)
            out.append(cdef)
            child_edges = list(self._edges_by_parent.get(cdef, ()))
            child_edges.sort(key=lambda edge: (str(edge.path), edge.child.stable_hash(), repr(edge.child)))
            for edge in child_edges:
                visit(edge.child)

        for root in self._roots:
            visit(root)
        return tuple(out)


def _validate_graph_parts(
        roots: tuple[ConcreteDefinition, ...],
        nodes: tuple[CDefNode, ...],
        edges: tuple[CDefEdge, ...]) -> None:
    node_defs: set[ConcreteDefinition] = set()
    for node in nodes:
        if not isinstance(node, CDefNode):
            raise TypeError(f"Graph nodes must be CDefNode instances, got {type(node).__name__}.")
        if not isinstance(node.definition, ConcreteDefinition):
            raise TypeError(f"Graph node definitions must be ConcreteDefinitions, got {type(node.definition).__name__}.")
        if node.definition in node_defs:
            raise ConcreteDefinitionGraphError(f"Duplicate graph node for {node.definition}.")
        if node.stable_hash != node.definition.stable_hash():
            raise ConcreteDefinitionGraphError(
                f"Graph node stable_hash {node.stable_hash!r} does not match its definition."
            )
        node_defs.add(node.definition)

    for root in roots:
        if not isinstance(root, ConcreteDefinition):
            raise TypeError(f"Graph roots must be ConcreteDefinitions, got {type(root).__name__}.")
        if root not in node_defs:
            raise ConcreteDefinitionGraphError(f"Graph root {root} is missing from nodes.")

    for edge in edges:
        if not isinstance(edge, CDefEdge):
            raise TypeError(f"Graph edges must be CDefEdge instances, got {type(edge).__name__}.")
        if not isinstance(edge.parent, ConcreteDefinition):
            raise TypeError(f"Graph edge parents must be ConcreteDefinitions, got {type(edge.parent).__name__}.")
        if not isinstance(edge.child, ConcreteDefinition):
            raise TypeError(f"Graph edge children must be ConcreteDefinitions, got {type(edge.child).__name__}.")
        if not isinstance(edge.path, GraphPath):
            raise TypeError(f"Graph edge paths must be GraphPath instances, got {type(edge.path).__name__}.")
        if edge.parent not in node_defs:
            raise ConcreteDefinitionGraphError(f"Graph edge parent {edge.parent} is missing from nodes.")
        if edge.child not in node_defs:
            raise ConcreteDefinitionGraphError(f"Graph edge child {edge.child} is missing from nodes.")

    outgoing: dict[ConcreteDefinition, list[ConcreteDefinition]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.parent].append(edge.child)

    visited: set[ConcreteDefinition] = set()
    active: set[ConcreteDefinition] = set()

    def visit(cdef: ConcreteDefinition) -> None:
        if cdef in visited:
            return
        if cdef in active:
            raise ConcreteDefinitionGraphCycleError(
                f"ConcreteDefinition graph cycle detected at hash={cdef.stable_hash()}."
            )
        active.add(cdef)
        try:
            for child in outgoing.get(cdef, ()):
                visit(child)
        finally:
            active.remove(cdef)
        visited.add(cdef)

    for root in roots:
        visit(root)

    for node in nodes:
        visit(node.definition)


def _same_cdef(left: ConcreteDefinition, right: ConcreteDefinition) -> bool:
    if left is right:
        return True
    if left.stable_hash() != right.stable_hash():
        return False
    try:
        return left == right
    except TypeError:
        return False
