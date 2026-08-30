"""Private-node-aware traversal for canonical CDef graphs.

Ordinary CDef equality is intentionally structural. This module is the narrow
authority boundary that instead keys graph traversal by each CDef's private,
process-local node token so aliases and independent equal nodes remain distinct.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Iterator

from .cdef_identity import cdef_node_key, same_cdef, same_cdef_node
from .definition import ConcreteDefinition, Definition
from .links import DefLink
from .params import Par
from .quoted import QuotedDef, SelectorSpec
from .object import Object
from .utils.graph.path import GraphPath, graph_path_sort_key
from .utils.graph.value import iter_value_edges

CDEF_GRAPH_SCHEMA_VERSION = 6


class EdgeKind(Enum):
    """The construction meaning of a direct CDef edge."""

    MATERIALIZE = "materialize"
    REF = "ref"


class GraphClosure(Enum):
    """Traversal closure represented by a ConcreteDefinitionGraph."""

    MATERIALIZATION = "materialization"
    QUERY_INDEX = "query_index"


class ConcreteDefinitionGraphError(Exception):
    """Raised when a CDef graph is malformed or internally inconsistent."""


class ConcreteDefinitionGraphCycleError(ConcreteDefinitionGraphError):
    """Raised when a materializing CDef edge forms a cycle."""


class ConcreteDefinitionGraphOccurrenceLimitError(
    ConcreteDefinitionGraphError
):
    """Raised when occurrence traversal exceeds its configured bound."""


@dataclass(frozen=True, slots=True)
class CDefNode:
    """One graph node paired with its structural CDef digest."""

    definition: ConcreteDefinition
    stable_hash: str


@dataclass(frozen=True, slots=True)
class CDefEdge:
    """One typed direct edge between CDef graph nodes."""

    parent: ConcreteDefinition
    path: GraphPath
    child: ConcreteDefinition
    kind: EdgeKind = EdgeKind.MATERIALIZE


@dataclass(frozen=True, slots=True)
class CDefOccurrence:
    """One materializing root-to-node occurrence path."""

    root: ConcreteDefinition
    path: GraphPath
    definition: ConcreteDefinition
    kind: EdgeKind = EdgeKind.MATERIALIZE


def iter_direct_cdef_edges(
    cdef: ConcreteDefinition,
) -> Iterator[tuple[GraphPath, ConcreteDefinition, EdgeKind]]:
    """Yield the typed direct CDef edges contained by one exact definition."""

    if not isinstance(cdef, ConcreteDefinition):
        raise TypeError(
            f"Expected ConcreteDefinition, got {type(cdef).__name__}."
        )
    for edge in iter_value_edges(cdef):
        yield from _iter_direct_edges_from_value(
            edge.value, GraphPath((edge.segment,))
        )


def _iter_direct_edges_from_value(
    value: Any, path: GraphPath
) -> Iterator[tuple[GraphPath, ConcreteDefinition, EdgeKind]]:
    if isinstance(value, Object):
        raise ConcreteDefinitionGraphError(
            f"Runtime Object found inside ConcreteDefinition graph at {path!s}."
        )
    if isinstance(value, Definition):
        raise ConcreteDefinitionGraphError(
            f"Plain Definition found inside ConcreteDefinition graph at {path!s}."
        )
    if isinstance(value, ConcreteDefinition):
        yield path, value, EdgeKind.MATERIALIZE
        return
    if isinstance(value, DefLink):
        if not isinstance(value.target, ConcreteDefinition):
            raise ConcreteDefinitionGraphError(
                f"DefLink at {path!s} does not resolve to a ConcreteDefinition boundary."
            )
        yield path, value.target, value.kind
        return
    if isinstance(value, (QuotedDef, SelectorSpec)):
        return
    if isinstance(value, Par):
        raise ConcreteDefinitionGraphError(
            f"Unresolved Par found inside ConcreteDefinition graph at {path!s}."
        )
    for edge in iter_value_edges(value):
        yield from _iter_direct_edges_from_value(
            edge.value, path.child(edge.segment)
        )


class ConcreteDefinitionGraph:
    """An immutable CDef graph keyed internally by private node identity."""

    def __init__(
        self,
        roots: Iterable[ConcreteDefinition],
        nodes: Iterable[CDefNode],
        edges: Iterable[CDefEdge],
        *,
        closure: GraphClosure = GraphClosure.MATERIALIZATION,
    ):
        if not isinstance(closure, GraphClosure):
            raise TypeError(
                f"Graph closure must be GraphClosure, got {type(closure).__name__}."
            )
        self._roots = _unique_nodes(roots)
        self._nodes = tuple(nodes)
        self._edges = _unique_edges(edges)
        self._closure = closure
        _validate_graph_parts(self._roots, self._nodes, self._edges, closure)
        self._node_by_key = {
            cdef_node_key(node.definition): node for node in self._nodes
        }
        outgoing: dict[object, list[CDefEdge]] = defaultdict(list)
        incoming: dict[object, list[CDefEdge]] = defaultdict(list)
        for edge in self._edges:
            outgoing[cdef_node_key(edge.parent)].append(edge)
            incoming[cdef_node_key(edge.child)].append(edge)
        self._outgoing = {
            key: tuple(sorted(value, key=_edge_sort_key))
            for key, value in outgoing.items()
        }
        self._incoming = {key: tuple(value) for key, value in incoming.items()}

    @classmethod
    def from_root(
        cls, cdef: ConcreteDefinition, *, expand_ref_targets: bool = False
    ) -> "ConcreteDefinitionGraph":
        """Build a private-node-aware graph rooted at one CDef."""

        return cls.from_roots((cdef,), expand_ref_targets=expand_ref_targets)

    @classmethod
    def from_roots(
        cls,
        cdefs: Iterable[ConcreteDefinition],
        *,
        expand_ref_targets: bool = False,
    ) -> "ConcreteDefinitionGraph":
        """Build a private-node-aware graph for roots in supplied order."""

        closure = (
            GraphClosure.QUERY_INDEX
            if expand_ref_targets
            else GraphClosure.MATERIALIZATION
        )
        builder = ConcreteDefinitionGraphBuilder(
            expand_ref_targets=expand_ref_targets, closure=closure
        )
        builder.add_roots(cdefs)
        return builder.build()

    @classmethod
    def for_query_index(
        cls, cdef: ConcreteDefinition
    ) -> "ConcreteDefinitionGraph":
        """Return the explicit structurally deduplicated query projection."""

        return cls.for_query_index_roots((cdef,))

    @classmethod
    def for_query_index_roots(
        cls, cdefs: Iterable[ConcreteDefinition]
    ) -> "ConcreteDefinitionGraph":
        """Return a structural query projection after node-aware traversal."""

        graph = cls.from_roots(cdefs, expand_ref_targets=True)
        return graph.structural_projection()

    @property
    def closure(self) -> GraphClosure:
        """Return the traversal closure used to build this graph."""

        return self._closure

    @property
    def roots(self) -> tuple[ConcreteDefinition, ...]:
        """Return graph roots in deterministic insertion order."""

        return self._roots

    def node(self, cdef: ConcreteDefinition) -> CDefNode:
        """Return the graph node for this exact private CDef node."""

        return self._node_by_key[cdef_node_key(cdef)]

    def nodes(self) -> tuple[CDefNode, ...]:
        """Return graph nodes in deterministic traversal order."""

        return self._nodes

    def edges(self) -> tuple[CDefEdge, ...]:
        """Return direct graph edges in deterministic traversal order."""

        return self._edges

    def outgoing(self, cdef: ConcreteDefinition) -> tuple[CDefEdge, ...]:
        """Return direct outgoing edges for this exact private node."""

        return self._outgoing.get(cdef_node_key(cdef), ())

    def incoming(self, cdef: ConcreteDefinition) -> tuple[CDefEdge, ...]:
        """Return direct incoming edges for this exact private node."""

        return self._incoming.get(cdef_node_key(cdef), ())

    def structural_projection(self) -> "ConcreteDefinitionGraph":
        """Collapse graph nodes only at the explicit structural-query boundary."""

        representatives: list[ConcreteDefinition] = []
        by_node: dict[object, ConcreteDefinition] = {}
        for node in self._nodes:
            representative = next(
                (
                    item
                    for item in representatives
                    if same_cdef(item, node.definition)
                ),
                None,
            )
            if representative is None:
                representative = node.definition
                representatives.append(representative)
            by_node[cdef_node_key(node.definition)] = representative
        roots = _unique_nodes(
            by_node[cdef_node_key(root)] for root in self._roots
        )
        edges = [
            CDefEdge(
                by_node[cdef_node_key(edge.parent)],
                edge.path,
                by_node[cdef_node_key(edge.child)],
                edge.kind,
            )
            for edge in self._edges
        ]
        return ConcreteDefinitionGraph(
            roots,
            (CDefNode(node, node.stable_hash()) for node in representatives),
            edges,
            closure=GraphClosure.QUERY_INDEX,
        )

    def walk_nodes(self, *, order: str = "pre") -> tuple[CDefNode, ...]:
        """Walk each private node once in pre- or post-order."""

        if order not in {"pre", "post"}:
            raise ValueError("order must be 'pre' or 'post'.")
        seen: set[object] = set()
        out: list[CDefNode] = []

        def visit(cdef: ConcreteDefinition) -> None:
            key = cdef_node_key(cdef)
            if key in seen:
                return
            seen.add(key)
            if order == "pre":
                out.append(self.node(cdef))
            for edge in self.outgoing(cdef):
                visit(edge.child)
            if order == "post":
                out.append(self.node(cdef))

        for root in self.roots:
            visit(root)
        return tuple(out)

    def topological_order(
        self, *, dependencies_first: bool = True
    ) -> tuple[ConcreteDefinition, ...]:
        """Return a private-node topological order for this acyclic graph."""

        nodes = tuple(
            node.definition for node in self.walk_nodes(order="post")
        )
        return nodes if dependencies_first else tuple(reversed(nodes))

    def iter_occurrences(
        self,
        *,
        roots: Iterable[ConcreteDefinition] | None = None,
        include_roots: bool = False,
        target: ConcreteDefinition | None = None,
        max_occurrences: int | None = None,
        limit: int | None = None,
    ) -> Iterator[CDefOccurrence]:
        """Yield materializing occurrence paths without structural node collapse."""

        count = 0
        selected_roots = tuple(roots) if roots is not None else self.roots
        target_key = None if target is None else cdef_node_key(target)
        for root in selected_roots:
            if include_roots and (
                target_key is None or cdef_node_key(root) is target_key
            ):
                count += 1
                if max_occurrences is not None and count > max_occurrences:
                    raise ConcreteDefinitionGraphOccurrenceLimitError(
                        f"Occurrence limit {max_occurrences} exceeded while walking {root.stable_hash()}."
                    )
                yield CDefOccurrence(root, GraphPath(), root)
                if limit is not None and count >= limit:
                    return
            stack = [
                (edge, edge.path)
                for edge in reversed(self.outgoing(root))
                if edge.kind is EdgeKind.MATERIALIZE
            ]
            while stack:
                edge, path = stack.pop()
                if (
                    target_key is None
                    or cdef_node_key(edge.child) is target_key
                ):
                    count += 1
                    if max_occurrences is not None and count > max_occurrences:
                        raise ConcreteDefinitionGraphOccurrenceLimitError(
                            f"Occurrence limit {max_occurrences} exceeded while walking {root.stable_hash()}."
                        )
                    yield CDefOccurrence(root, path, edge.child, edge.kind)
                    if limit is not None and count >= limit:
                        return
                for child_edge in reversed(self.outgoing(edge.child)):
                    if child_edge.kind is EdgeKind.MATERIALIZE:
                        stack.append((child_edge, path.join(child_edge.path)))

    def resolve(
        self, root: ConcreteDefinition, path: GraphPath
    ) -> ConcreteDefinition:
        """Resolve a direct typed graph path from a root without construction."""

        if not path:
            return root
        value = root.graph_path(path)
        if isinstance(value, DefLink):
            return value.target
        if not isinstance(value, ConcreteDefinition):
            raise ConcreteDefinitionGraphError(
                f"Path {path!s} does not resolve to a ConcreteDefinition boundary."
            )
        return value

    def paths_to(
        self,
        root: ConcreteDefinition,
        target: ConcreteDefinition,
        *,
        max_paths: int | None = None,
    ) -> tuple[GraphPath, ...]:
        """Return materializing occurrence paths to one exact private node."""

        return tuple(
            occ.path
            for occ in self.iter_occurrences(
                roots=(root,), target=target, limit=max_paths
            )
        )

    def contains(
        self, root: ConcreteDefinition, target: ConcreteDefinition
    ) -> bool:
        """Report materializing containment using private node identity."""

        return same_cdef_node(root, target) or any(
            True
            for _ in self.iter_occurrences(
                roots=(root,), target=target, limit=1
            )
        )

    def primary_path(
        self, root: ConcreteDefinition, target: ConcreteDefinition
    ) -> GraphPath | None:
        """Return the minimum canonical materializing occurrence path."""

        paths = self.paths_to(root, target)
        return min(paths, key=graph_path_sort_key) if paths else None

    def ancestors(
        self, cdef: ConcreteDefinition
    ) -> tuple[ConcreteDefinition, ...]:
        """Return unique private-node ancestors in breadth-first order."""

        seen: set[object] = set()
        out: list[ConcreteDefinition] = []
        queue = deque(edge.parent for edge in self.incoming(cdef))
        while queue:
            cur = queue.popleft()
            key = cdef_node_key(cur)
            if key in seen:
                continue
            seen.add(key)
            out.append(cur)
            queue.extend(edge.parent for edge in self.incoming(cur))
        return tuple(out)

    def descendants(
        self, cdef: ConcreteDefinition
    ) -> tuple[ConcreteDefinition, ...]:
        """Return unique private-node descendants in deterministic DFS order."""

        seen: set[object] = set()
        out: list[ConcreteDefinition] = []
        stack = [edge.child for edge in reversed(self.outgoing(cdef))]
        while stack:
            cur = stack.pop()
            key = cdef_node_key(cur)
            if key in seen:
                continue
            seen.add(key)
            out.append(cur)
            stack.extend(edge.child for edge in reversed(self.outgoing(cur)))
        return tuple(out)

    def roots_containing(
        self, cdef: ConcreteDefinition
    ) -> tuple[ConcreteDefinition, ...]:
        """Return roots that materially contain this exact private node."""

        return tuple(
            root
            for root in self.roots
            if not same_cdef_node(root, cdef) and self.contains(root, cdef)
        )

    def explain(self) -> str:
        """Return a small deterministic graph summary."""

        return "\n".join(
            (
                f"roots: {len(self.roots)}",
                f"nodes: {len(self.nodes())}",
                f"edges: {len(self.edges())}",
            )
        )


class ConcreteDefinitionGraphBuilder:
    """Incrementally build a CDef graph using only private node keys."""

    def __init__(
        self,
        *,
        expand_ref_targets: bool = False,
        closure: GraphClosure = GraphClosure.MATERIALIZATION,
    ):
        self._expand_ref_targets = expand_ref_targets
        self._closure = closure
        self._roots: list[ConcreteDefinition] = []
        self._nodes: dict[object, CDefNode] = {}
        self._edges: dict[
            tuple[object, GraphPath, object, EdgeKind], CDefEdge
        ] = {}
        self._edges_by_parent: dict[object, list[CDefEdge]] = defaultdict(list)
        self._completed: set[object] = set()
        self._active: dict[object, GraphPath] = {}

    def add_root(self, cdef: ConcreteDefinition) -> None:
        """Add one root and recursively collect its selected closure."""

        if not isinstance(cdef, ConcreteDefinition):
            raise TypeError(
                f"Graph roots must be ConcreteDefinitions, got {type(cdef).__name__}."
            )
        if cdef_node_key(cdef) not in {
            cdef_node_key(root) for root in self._roots
        }:
            self._roots.append(cdef)
        self._visit(cdef, GraphPath())

    def add_roots(self, cdefs: Iterable[ConcreteDefinition]) -> None:
        """Add all supplied roots in order."""

        for cdef in cdefs:
            self.add_root(cdef)

    def build(self) -> ConcreteDefinitionGraph:
        """Freeze the collected graph into deterministic node and edge order."""

        node_order = self._node_order()
        order_index = {
            cdef_node_key(cdef): idx for idx, cdef in enumerate(node_order)
        }
        edges = tuple(
            sorted(
                self._edges.values(),
                key=lambda edge: (
                    order_index[cdef_node_key(edge.parent)],
                    graph_path_sort_key(edge.path),
                    order_index[cdef_node_key(edge.child)],
                    edge.kind.value,
                ),
            )
        )
        return ConcreteDefinitionGraph(
            self._roots,
            (self._nodes[cdef_node_key(item)] for item in node_order),
            edges,
            closure=self._closure,
        )

    def _visit(self, cdef: ConcreteDefinition, path: GraphPath) -> None:
        key = cdef_node_key(cdef)
        if key in self._completed:
            return
        if key in self._active:
            raise ConcreteDefinitionGraphCycleError(
                f"ConcreteDefinition cycle detected at {path!s}; first active path was {self._active[key]!s}; hash={cdef.stable_hash()}."
            )
        self._active[key] = path
        self._nodes.setdefault(key, CDefNode(cdef, cdef.stable_hash()))
        try:
            for edge_path, child, kind in iter_direct_cdef_edges(cdef):
                child_key = cdef_node_key(child)
                edge = CDefEdge(cdef, edge_path, child, kind)
                edge_key = (key, edge.path, child_key, edge.kind)
                if edge_key not in self._edges:
                    self._nodes.setdefault(
                        child_key, CDefNode(child, child.stable_hash())
                    )
                    self._edges[edge_key] = edge
                    self._edges_by_parent[key].append(edge)
                if kind is EdgeKind.MATERIALIZE or self._expand_ref_targets:
                    self._visit(child, path.join(edge_path))
        finally:
            self._active.pop(key, None)
            self._completed.add(key)

    def _node_order(self) -> tuple[ConcreteDefinition, ...]:
        seen: set[object] = set()
        out: list[ConcreteDefinition] = []

        def visit(cdef: ConcreteDefinition) -> None:
            key = cdef_node_key(cdef)
            if key in seen:
                return
            seen.add(key)
            out.append(cdef)
            for edge in sorted(
                self._edges_by_parent.get(key, ()), key=_edge_sort_key
            ):
                visit(edge.child)

        for root in self._roots:
            visit(root)
        return tuple(out)


def _unique_nodes(
    cdefs: Iterable[ConcreteDefinition],
) -> tuple[ConcreteDefinition, ...]:
    seen: set[object] = set()
    out: list[ConcreteDefinition] = []
    for cdef in cdefs:
        key = cdef_node_key(cdef)
        if key not in seen:
            seen.add(key)
            out.append(cdef)
    return tuple(out)


def _unique_edges(edges: Iterable[CDefEdge]) -> tuple[CDefEdge, ...]:
    seen: set[tuple[object, GraphPath, object, EdgeKind]] = set()
    out: list[CDefEdge] = []
    for edge in edges:
        key = (
            cdef_node_key(edge.parent),
            edge.path,
            cdef_node_key(edge.child),
            edge.kind,
        )
        if key not in seen:
            seen.add(key)
            out.append(edge)
    return tuple(out)


def _edge_sort_key(edge: CDefEdge) -> tuple[bytes, str, str]:
    return (
        graph_path_sort_key(edge.path),
        edge.kind.value,
        edge.child.stable_hash(),
    )


def _validate_graph_parts(
    roots: tuple[ConcreteDefinition, ...],
    nodes: tuple[CDefNode, ...],
    edges: tuple[CDefEdge, ...],
    closure: GraphClosure,
) -> None:
    node_defs: dict[object, ConcreteDefinition] = {}
    for node in nodes:
        if not isinstance(node, CDefNode) or not isinstance(
            node.definition, ConcreteDefinition
        ):
            raise TypeError(
                "Graph nodes must contain ConcreteDefinition values."
            )
        key = cdef_node_key(node.definition)
        if key in node_defs:
            raise ConcreteDefinitionGraphError(
                f"Duplicate graph node for {node.definition}."
            )
        if node.stable_hash != node.definition.stable_hash():
            raise ConcreteDefinitionGraphError(
                f"Graph node stable_hash {node.stable_hash!r} does not match its definition."
            )
        node_defs[key] = node.definition
    for root in roots:
        if not isinstance(root, ConcreteDefinition):
            raise TypeError(
                f"Graph roots must be ConcreteDefinitions, got {type(root).__name__}."
            )
        if cdef_node_key(root) not in node_defs:
            raise ConcreteDefinitionGraphError(
                f"Graph root {root} is missing from nodes."
            )
    outgoing: dict[object, list[ConcreteDefinition]] = defaultdict(list)
    reachable_outgoing: dict[object, list[ConcreteDefinition]] = defaultdict(
        list
    )
    for edge in edges:
        if (
            not isinstance(edge, CDefEdge)
            or not isinstance(edge.kind, EdgeKind)
            or not isinstance(edge.path, GraphPath)
        ):
            raise TypeError("Graph edges must contain typed CDefEdge values.")
        parent_key, child_key = cdef_node_key(edge.parent), cdef_node_key(
            edge.child
        )
        if parent_key not in node_defs:
            raise ConcreteDefinitionGraphError(
                f"Graph edge parent {edge.parent} is missing from nodes."
            )
        if child_key not in node_defs:
            raise ConcreteDefinitionGraphError(
                f"Graph edge child {edge.child} is missing from nodes."
            )
        try:
            resolved = edge.parent.graph_path(edge.path)
        except Exception as error:
            raise ConcreteDefinitionGraphError(
                f"Graph edge path {edge.path!s} cannot be resolved on parent {edge.parent}."
            ) from error
        if isinstance(resolved, DefLink):
            if edge.kind is not resolved.kind:
                raise ConcreteDefinitionGraphError(
                    f"Graph edge path {edge.path!s} resolves to a different edge kind."
                )
            resolved = resolved.target
        elif edge.kind is EdgeKind.REF:
            raise ConcreteDefinitionGraphError(
                f"Graph ref edge path {edge.path!s} does not resolve to a Ref boundary."
            )
        if not isinstance(resolved, ConcreteDefinition):
            raise ConcreteDefinitionGraphError(
                f"Graph edge path {edge.path!s} does not resolve to a ConcreteDefinition boundary."
            )
        if closure is GraphClosure.MATERIALIZATION and not same_cdef_node(
            resolved, edge.child
        ):
            raise ConcreteDefinitionGraphError(
                f"Graph edge path {edge.path!s} resolves to a different child definition."
            )
        if closure is GraphClosure.QUERY_INDEX and not same_cdef(
            resolved, edge.child
        ):
            raise ConcreteDefinitionGraphError(
                f"Graph edge path {edge.path!s} resolves to a different child definition."
            )
        reachable_outgoing[parent_key].append(edge.child)
        if edge.kind is EdgeKind.MATERIALIZE:
            outgoing[parent_key].append(edge.child)
    visited: set[object] = set()
    active: set[object] = set()

    def visit(cdef: ConcreteDefinition) -> None:
        key = cdef_node_key(cdef)
        if key in visited:
            return
        if key in active:
            raise ConcreteDefinitionGraphCycleError(
                f"ConcreteDefinition graph cycle detected at hash={cdef.stable_hash()}."
            )
        active.add(key)
        try:
            for child in outgoing.get(key, ()):
                visit(child)
        finally:
            active.remove(key)
        visited.add(key)

    for root in roots:
        visit(root)
    reachable: set[object] = set()

    def visit_reachable(cdef: ConcreteDefinition) -> None:
        key = cdef_node_key(cdef)
        if key in reachable:
            return
        reachable.add(key)
        for child in reachable_outgoing.get(key, ()):
            visit_reachable(child)

    for root in roots:
        visit_reachable(root)
    for key, node in node_defs.items():
        if key not in reachable:
            raise ConcreteDefinitionGraphError(
                f"Graph node {node} is not reachable from any root."
            )


def as_query_index_graph(
    graph: ConcreteDefinitionGraph,
    roots: Iterable[ConcreteDefinition] | None = None,
) -> ConcreteDefinitionGraph:
    """Return the structural query projection for requested graph roots."""

    wanted_roots = tuple(graph.roots if roots is None else roots)
    if (
        graph.closure is GraphClosure.QUERY_INDEX
        and graph.roots == wanted_roots
    ):
        return graph
    return ConcreteDefinitionGraph.for_query_index_roots(wanted_roots)
