from __future__ import annotations

from .model import DefinitionId, QueryStats
from .selector_graph import SelectorGraph, SelectorGraphEdge


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
