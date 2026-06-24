from __future__ import annotations

from collections import deque

from .model import DefinitionId, QueryStats
from .selector_graph import SelectorGraph, SelectorGraphEdge, SelectorGraphNode


def graph_candidate_ids(
        catalog,
        selector_graph: SelectorGraph | None,
        universe_ids: set[DefinitionId],
        *,
        stats: QueryStats | None = None) -> set[DefinitionId]:
    if selector_graph is None:
        if stats is not None:
            stats.candidate_count = len(universe_ids)
        return set(universe_ids)

    with catalog.lock:
        exact_anchor = _choose_exact_anchor(catalog, selector_graph)
        if exact_anchor is not None:
            return _graph_candidate_ids_from_exact_anchor(
                catalog,
                selector_graph,
                universe_ids,
                exact_anchor,
                stats=stats,
            )
        return _graph_candidate_ids_full(catalog, selector_graph, universe_ids, stats=stats)


def _graph_candidate_ids_full(
        catalog,
        selector_graph: SelectorGraph,
        universe_ids: set[DefinitionId],
        *,
        stats: QueryStats | None = None) -> set[DefinitionId]:
    all_ids = set(catalog.definitions_by_id.keys())
    candidates: dict[int, set[DefinitionId]] = {}
    initial_sizes: dict[int, int] = {}

    for node in selector_graph.nodes:
        if node.exact_definition is not None:
            digest = node.exact_definition.stable_hash()
            ids = {
                did
                for did in catalog.ids_by_stable_hash.get(digest, set())
                if catalog.definitions_by_id[did].cdef == node.exact_definition
            }
        else:
            ids = catalog.local_candidate_ids(all_ids, node.local_requirements)
        candidates[node.node_id] = ids
        initial_sizes[node.node_id] = len(ids)

    if stats is not None:
        stats.graph_node_count = len(selector_graph.nodes)
        stats.graph_edge_count = len(selector_graph.edges)
        if selector_graph.nodes:
            anchor_id = min(initial_sizes, key=lambda node_id: (initial_sizes[node_id], str(selector_graph.node(node_id).source_path)))
            stats.graph_anchor_path = selector_graph.node(anchor_id).source_path
            stats.graph_anchor_mode = "exact" if selector_graph.node(anchor_id).exact_definition is not None else "local-posting"

    changed = True
    while changed:
        changed = False
        for edge in selector_graph.edges:
            parent_before = candidates[edge.parent]
            child_before = candidates[edge.child]
            parent_after = _parents_with_matching_child(catalog, edge, parent_before, child_before)
            child_after = _children_with_matching_parent(catalog, edge, parent_after, child_before)
            if parent_after != parent_before:
                candidates[edge.parent] = parent_after
                changed = True
            if child_after != child_before:
                candidates[edge.child] = child_after
                changed = True

    root_candidates = candidates[selector_graph.root] & universe_ids
    if stats is not None:
        stats.graph_candidate_count = len(root_candidates)
        stats.candidate_count = len(root_candidates)
    return root_candidates


def _graph_candidate_ids_from_exact_anchor(
        catalog,
        selector_graph: SelectorGraph,
        universe_ids: set[DefinitionId],
        anchor_id: int,
        *,
        stats: QueryStats | None = None) -> set[DefinitionId]:
    if stats is not None:
        stats.graph_node_count = len(selector_graph.nodes)
        stats.graph_edge_count = len(selector_graph.edges)
        stats.graph_anchor_path = selector_graph.node(anchor_id).source_path
        stats.graph_anchor_mode = "exact"

    anchor_node = selector_graph.node(anchor_id)
    anchor_candidates = _exact_candidate_ids(catalog, anchor_node)
    if not anchor_candidates:
        if stats is not None:
            stats.graph_candidate_count = 0
            stats.candidate_count = 0
        return set()

    candidates: dict[int, set[DefinitionId]] = {anchor_id: anchor_candidates}
    queue = deque((anchor_id,))
    while queue:
        node_id = queue.popleft()
        node_candidates = candidates.get(node_id, set())

        for edge in selector_graph.incoming(node_id):
            parent_ids = _parents_for_children(catalog, edge, node_candidates)
            parent_ids = _filter_node_ids(catalog, selector_graph.node(edge.parent), parent_ids)
            if _merge_candidates(candidates, edge.parent, parent_ids):
                queue.append(edge.parent)

        for edge in selector_graph.outgoing(node_id):
            child_ids = _children_for_parents(catalog, edge, node_candidates)
            child_ids = _filter_node_ids(catalog, selector_graph.node(edge.child), child_ids)
            if _merge_candidates(candidates, edge.child, child_ids):
                queue.append(edge.child)

    root_candidates = candidates.get(selector_graph.root, set()) & universe_ids
    if stats is not None:
        stats.graph_candidate_count = len(root_candidates)
        stats.candidate_count = len(root_candidates)
    return root_candidates


def _choose_exact_anchor(catalog, selector_graph: SelectorGraph) -> int | None:
    exact_nodes = [node for node in selector_graph.nodes if node.exact_definition is not None]
    if not exact_nodes:
        return None
    return min(
        exact_nodes,
        key=lambda node: (
            len(_exact_candidate_ids(catalog, node)),
            str(node.source_path),
        ),
    ).node_id


def _filter_node_ids(catalog, node: SelectorGraphNode, ids: set[DefinitionId]) -> set[DefinitionId]:
    if not ids:
        return set()
    if node.exact_definition is not None:
        return ids & _exact_candidate_ids(catalog, node)
    return catalog.local_candidate_ids(ids, node.local_requirements)


def _exact_candidate_ids(catalog, node: SelectorGraphNode) -> set[DefinitionId]:
    if node.exact_definition is None:
        return set()
    digest = node.exact_definition.stable_hash()
    return {
        did
        for did in catalog.ids_by_stable_hash.get(digest, set())
        if catalog.definitions_by_id[did].cdef == node.exact_definition
    }


def _merge_candidates(candidates: dict[int, set[DefinitionId]], node_id: int, ids: set[DefinitionId]) -> bool:
    old = candidates.get(node_id)
    if old is None:
        candidates[node_id] = set(ids)
        return True
    new = old & ids
    if new == old:
        return False
    candidates[node_id] = new
    return True


def _parents_with_matching_child(
        catalog,
        edge: SelectorGraphEdge,
        parent_ids: set[DefinitionId],
        child_ids: set[DefinitionId]) -> set[DefinitionId]:
    if not parent_ids or not child_ids:
        return set()
    out = set()
    if edge.unordered:
        for parent_id in parent_ids:
            for edge_key in catalog.outgoing_edges.get(parent_id, ()):
                record = catalog.edge_by_key[edge_key]
                if record.child_id in child_ids and record.path.startswith(edge.path):
                    out.add(parent_id)
                    break
        return out

    for parent_id in parent_ids:
        if catalog.child_by_parent_path.get((parent_id, edge.path), set()) & child_ids:
            out.add(parent_id)
    return out


def _parents_for_children(
        catalog,
        edge: SelectorGraphEdge,
        child_ids: set[DefinitionId]) -> set[DefinitionId]:
    if not child_ids:
        return set()
    out = set()
    if edge.unordered:
        for child_id in child_ids:
            for edge_key in catalog.incoming_edges.get(child_id, ()):
                record = catalog.edge_by_key[edge_key]
                if record.path.startswith(edge.path):
                    out.add(record.parent_id)
        return out

    for child_id in child_ids:
        out.update(catalog.parents_by_child_path.get((child_id, edge.path), set()))
    return out


def _children_with_matching_parent(
        catalog,
        edge: SelectorGraphEdge,
        parent_ids: set[DefinitionId],
        child_ids: set[DefinitionId]) -> set[DefinitionId]:
    if not parent_ids or not child_ids:
        return set()
    out = set()
    if edge.unordered:
        for child_id in child_ids:
            for edge_key in catalog.incoming_edges.get(child_id, ()):
                record = catalog.edge_by_key[edge_key]
                if record.parent_id in parent_ids and record.path.startswith(edge.path):
                    out.add(child_id)
                    break
        return out

    for child_id in child_ids:
        if catalog.parents_by_child_path.get((child_id, edge.path), set()) & parent_ids:
            out.add(child_id)
    return out


def _children_for_parents(
        catalog,
        edge: SelectorGraphEdge,
        parent_ids: set[DefinitionId]) -> set[DefinitionId]:
    if not parent_ids:
        return set()
    out = set()
    if edge.unordered:
        for parent_id in parent_ids:
            for edge_key in catalog.outgoing_edges.get(parent_id, ()):
                record = catalog.edge_by_key[edge_key]
                if record.path.startswith(edge.path):
                    out.add(record.child_id)
        return out

    for parent_id in parent_ids:
        out.update(catalog.child_by_parent_path.get((parent_id, edge.path), set()))
    return out
