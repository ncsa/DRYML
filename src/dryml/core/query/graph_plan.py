from __future__ import annotations

from collections import deque

from .domain import DefinitionDomain
from .model import DefinitionId, QueryStats
from .protocols import DefinitionGraphIndex
from .selector_graph import SelectorGraph, SelectorGraphEdge, SelectorGraphNode


def graph_candidate_ids(
        catalog: DefinitionGraphIndex,
        selector_graph: SelectorGraph | None,
        domain: DefinitionDomain | None,
        *,
        stats: QueryStats | None = None) -> set[DefinitionId]:
    if selector_graph is None:
        universe_ids = domain.all_ids() if domain is not None else catalog.all_definition_ids()
        if stats is not None:
            stats.candidate_count = len(universe_ids)
        return set(universe_ids)

    if selector_graph.requires_scan:
        universe_ids = domain.all_ids() if domain is not None else catalog.all_definition_ids()
        if stats is not None:
            stats.candidate_count = len(universe_ids)
        return set(universe_ids)

    anchor = _choose_anchor(catalog, selector_graph)
    if anchor is not None:
        anchor_id, anchor_mode = anchor
        anchor_candidates = _anchor_candidate_ids(catalog, selector_graph.node(anchor_id), anchor_mode)
        return _graph_candidate_ids_from_anchor(
            catalog,
            selector_graph,
            domain,
            anchor_id,
            anchor_candidates,
            anchor_mode,
            stats=stats,
        )
    return _graph_candidate_ids_full(catalog, selector_graph, domain, stats=stats)


def _graph_candidate_ids_full(
        catalog,
        selector_graph: SelectorGraph,
        domain: DefinitionDomain | None,
        *,
        stats: QueryStats | None = None) -> set[DefinitionId]:
    all_ids = catalog.all_definition_ids()
    candidates: dict[int, set[DefinitionId]] = {}
    initial_sizes: dict[int, int] = {}

    for node in selector_graph.nodes:
        if node.exact_definition is not None:
            ids = catalog.exact_ids(node.exact_definition)
        else:
            ids = _local_candidate_ids(catalog, node, within=all_ids)
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

    root_candidates = _apply_domain(candidates[selector_graph.root], domain)
    if stats is not None:
        if stats.universe_size is None:
            stats.universe_size = len(root_candidates)
        stats.graph_candidate_count = len(root_candidates)
        stats.candidate_count = len(root_candidates)
    return root_candidates


def _graph_candidate_ids_from_anchor(
        catalog,
        selector_graph: SelectorGraph,
        domain: DefinitionDomain | None,
        anchor_id: int,
        anchor_candidates: set[DefinitionId],
        anchor_mode: str,
        *,
        stats: QueryStats | None = None) -> set[DefinitionId]:
    if stats is not None:
        stats.graph_node_count = len(selector_graph.nodes)
        stats.graph_edge_count = len(selector_graph.edges)
        stats.graph_anchor_path = selector_graph.node(anchor_id).source_path
        stats.graph_anchor_mode = anchor_mode

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

    root_candidates = _apply_domain(candidates.get(selector_graph.root, set()), domain)
    if stats is not None:
        stats.graph_candidate_count = len(root_candidates)
        stats.candidate_count = len(root_candidates)
    return root_candidates


def _apply_domain(ids: set[DefinitionId], domain: DefinitionDomain | None) -> set[DefinitionId]:
    return set(ids) if domain is None else domain.filter(ids)


def _choose_anchor(catalog, selector_graph: SelectorGraph) -> tuple[int, str] | None:
    anchor_data: list[tuple[int, int, str]] = []
    for node in selector_graph.nodes:
        if node.exact_definition is not None:
            anchor_data.append((node.node_id, catalog.estimate_exact_ids(node.exact_definition), "exact"))
        elif any(node.requirement_branches):
            anchor_data.append((
                node.node_id,
                sum(catalog.estimate_local_candidates(branch) for branch in node.requirement_branches),
                "local-posting",
            ))
    if not anchor_data:
        return None
    node_id, _, mode = min(
        anchor_data,
        key=lambda item: (
            item[1],
            str(selector_graph.node(item[0]).source_path),
        ),
    )
    return node_id, mode


def _anchor_candidate_ids(catalog, node: SelectorGraphNode, mode: str) -> set[DefinitionId]:
    if mode == "exact":
        return _exact_candidate_ids(catalog, node)
    return _local_candidate_ids(catalog, node)


def _filter_node_ids(catalog, node: SelectorGraphNode, ids: set[DefinitionId]) -> set[DefinitionId]:
    if not ids:
        return set()
    if node.exact_definition is not None:
        return ids & _exact_candidate_ids(catalog, node)
    return _local_candidate_ids(catalog, node, within=ids)


def _local_candidate_ids(catalog, node: SelectorGraphNode, *, within=None) -> set[DefinitionId]:
    return set().union(*(
        catalog.local_candidates(branch, within=within)
        for branch in node.requirement_branches
    ))


def _exact_candidate_ids(catalog, node: SelectorGraphNode) -> set[DefinitionId]:
    if node.exact_definition is None:
        return set()
    return catalog.exact_ids(node.exact_definition)


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
    return _parents(catalog, child_ids, edge, within=parent_ids)


def _parents_for_children(
        catalog,
        edge: SelectorGraphEdge,
        child_ids: set[DefinitionId]) -> set[DefinitionId]:
    if not child_ids:
        return set()
    return _parents(catalog, child_ids, edge)


def _children_with_matching_parent(
        catalog,
        edge: SelectorGraphEdge,
        parent_ids: set[DefinitionId],
        child_ids: set[DefinitionId]) -> set[DefinitionId]:
    if not parent_ids or not child_ids:
        return set()
    return _children(catalog, parent_ids, edge, within=child_ids)


def _children_for_parents(
        catalog,
        edge: SelectorGraphEdge,
        parent_ids: set[DefinitionId]) -> set[DefinitionId]:
    if not parent_ids:
        return set()
    return _children(catalog, parent_ids, edge)


def _parents(catalog, child_ids, edge: SelectorGraphEdge, *, within=None):
    return set().union(*(_parents_at_path(catalog, child_ids, edge, path, within=within) for path in (edge.path, *edge.alternate_paths)))


def _parents_at_path(catalog, child_ids, edge, path, *, within=None):
    try:
        return catalog.parents(child_ids, path, unordered=edge.unordered, edge_kind=edge.edge_kind, within=within)
    except TypeError:
        if edge.edge_kind.value != "materialize":
            raise
        return catalog.parents(child_ids, path, unordered=edge.unordered, within=within)


def _children(catalog, parent_ids, edge: SelectorGraphEdge, *, within=None):
    return set().union(*(_children_at_path(catalog, parent_ids, edge, path, within=within) for path in (edge.path, *edge.alternate_paths)))


def _children_at_path(catalog, parent_ids, edge, path, *, within=None):
    try:
        return catalog.children(parent_ids, path, unordered=edge.unordered, edge_kind=edge.edge_kind, within=within)
    except TypeError:
        if edge.edge_kind.value != "materialize":
            raise
        return catalog.children(parent_ids, path, unordered=edge.unordered, within=within)
