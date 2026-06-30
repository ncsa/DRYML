from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from ..codecs import QueryIndexCodec, digest_blob
from ..domain import DefinitionDomain
from ..lowering import CandidateBatch, LoweredEdgeStep, LoweredGraphPlan, LoweredQueryPlan, LoweringDiagnostics, LogicalRelationPlan, PagedResultCursor, PhysicalRelationPlan, QueryTerminal, ScanPolicy
from ..model import QueryWouldScanError
from ..selector_graph import SelectorGraph, SelectorGraphEdge, SelectorGraphNode
from ..utils import feature_token_equal, stable_hash_to_blob


@dataclass(frozen=True, slots=True)
class SQLiteOptimizerPolicy:
    """Rule-based SQLite relation strategy thresholds.

    Args:
        materialize_if_reused: Materialize a relation that one operation will scan repeatedly.
        materialize_if_estimate_gt: Materialize relations whose estimated row count exceeds this threshold.
        materialize_if_sql_length_gt: Materialize relations whose SQL text exceeds this many characters.
        materialize_recursive_owner_inputs: Materialize recursive owner-projection inputs.
        materialize_page_relations: Materialize page-terminal relations. This is disabled by default because
            query-backed paging opens a fresh read view for each page, so temp relations cannot be reused.
    """

    materialize_if_reused: bool = True
    materialize_if_estimate_gt: int = 10_000
    materialize_if_sql_length_gt: int = 20_000
    materialize_recursive_owner_inputs: bool = True
    materialize_page_relations: bool = False


class SQLiteRelationCompiler:
    """Compile DRYML selector graphs into SQLite candidate relations.

    SQLite lowering is conservative: the generated SQL may return false
    positives, but Python verification remains authoritative before a result is
    exposed to callers.
    """

    def __init__(self, con, *, source_key: str, generation: int, codec: QueryIndexCodec, cdef_loader=None):
        self.con = con
        self.source_key = source_key
        self.generation = generation
        self.codec = codec
        self.cdef_loader = cdef_loader

    def lower_selector_graph(
            self,
            selector_graph: SelectorGraph | None,
            domain: DefinitionDomain,
            *,
            terminal: QueryTerminal,
            scan_policy: ScanPolicy,
            diagnostics: LoweringDiagnostics | None = None,
            within_relation: str | None = None) -> LoweredQueryPlan:
        if diagnostics is None:
            diagnostics = LoweringDiagnostics()
        diagnostics.strategy = "sqlite-lowered"
        diagnostics.relation_strategy = "cte"
        diagnostics.inline_relations = ("candidates",)
        diagnostics.scan_policy = scan_policy.mode
        diagnostics.verify_budget = scan_policy.max_verify

        params: list[Any] = []
        if selector_graph is None:
            body_sql = "SELECT d.def_id FROM definitions d"
            scan_reason = "selector has no indexable requirements"
            diagnostics.anchor_relation_kind = "scan"
            self._apply_scan_policy(scan_policy, diagnostics, scan_reason)
        else:
            if not _has_indexable_requirement(selector_graph):
                scan_reason = "selector graph has no indexable requirements"
                diagnostics.anchor_relation_kind = "scan"
                self._apply_scan_policy(scan_policy, diagnostics, scan_reason)
            body_sql = self._compile_graph(selector_graph, params, diagnostics, scan_policy)

        domain_sql = self._apply_domain_sql(body_sql, domain.name)
        if within_relation is not None:
            domain_sql = f"""
            SELECT base.def_id
            FROM ({domain_sql}) base
            JOIN temp.{within_relation} within_ids ON within_ids.def_id = base.def_id
            """
            diagnostics.relation_strategy = "cte+temp"
        candidate_sql = f"""
        WITH candidates(def_id) AS (
            {domain_sql}
        )
        SELECT candidates.def_id
        FROM candidates
        JOIN definitions d ON d.def_id = candidates.def_id
        ORDER BY d.stable_hash, d.collision_ordinal, d.def_id
        """
        diagnostics.estimated_rows = self._estimate_rows(selector_graph, domain.name)
        diagnostics.logical_plan = self._logical_plan(selector_graph, diagnostics)
        diagnostics.physical_plan = PhysicalRelationPlan(
            strategy="inline-cte",
            root_relation_kind=diagnostics.anchor_relation_kind or "scan",
            inline_relations=diagnostics.inline_relations,
            materialized_relations=diagnostics.materialized_relations,
            fallback_reason=diagnostics.anchor_fallback_reason,
        )
        return LoweredQueryPlan(
            source_key=self.source_key,
            generation=self.generation,
            domain=domain.name,
            terminal=terminal,
            candidate_sql=candidate_sql,
            params=tuple(params),
            estimated_size=diagnostics.estimated_rows,
            scan_required=diagnostics.scan_required,
            scan_reason=diagnostics.scan_reason,
            diagnostics=diagnostics,
        )

    def iter_candidate_cdef_batches(
            self,
            plan: LoweredQueryPlan,
            *,
            after: PagedResultCursor | None = None,
            batch_size: int) -> Any:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        params = list(plan.params)
        keyset_sql = ""
        if after is not None:
            if after.source_key != plan.source_key or after.generation != plan.generation:
                raise ValueError("PagedResultCursor is not compatible with this lowered query plan.")
            keyset_sql = """
            WHERE d.stable_hash > ?
               OR (d.stable_hash = ? AND d.collision_ordinal > ?)
               OR (d.stable_hash = ? AND d.collision_ordinal = ? AND d.def_id > ?)
            """
            stable_hash = stable_hash_to_blob(after.stable_hash)
            params.extend((stable_hash, stable_hash, after.collision_ordinal, stable_hash, after.collision_ordinal, after.definition_id))
        sql = f"""
        SELECT q.def_id, d.stable_hash, d.collision_ordinal
        FROM ({plan.candidate_sql}) q
        JOIN definitions d ON d.def_id = q.def_id
        {keyset_sql}
        ORDER BY d.stable_hash, d.collision_ordinal, d.def_id
        LIMIT ?
        """
        params.append(batch_size)
        plan.diagnostics.sql_statements_executed += 1
        rows = self.con.execute(sql, tuple(params)).fetchall()
        if not rows:
            return
        ids = []
        cdefs = []
        cursor = None
        for did, stable_hash, ordinal in rows:
            ids.append(did)
            cursor = PagedResultCursor(
                source_key=plan.source_key,
                generation=plan.generation,
                stable_hash=stable_hash.hex(),
                collision_ordinal=ordinal,
                definition_id=did,
            )
        if self.cdef_loader is None:
            placeholders = ", ".join("?" for _ in ids)
            cdefs_by_id = {
                did: self.codec.decode_cdef(cdef_blob)
                for did, cdef_blob in self.con.execute(
                    f"SELECT def_id, cdef_blob FROM definitions WHERE def_id IN ({placeholders}) ORDER BY def_id",
                    tuple(ids),
                )
            }
        else:
            cdefs_by_id = self.cdef_loader(tuple(ids))
        cdefs = [cdefs_by_id[did] for did in ids]
        plan.diagnostics.candidate_rows_read += len(ids)
        plan.diagnostics.cdef_blobs_decoded += len(cdefs)
        plan.diagnostics.pages_fetched += 1
        yield CandidateBatch(tuple(ids), tuple(cdefs), cursor)

    def explain_query_plan(self, plan: LoweredQueryPlan) -> tuple[str, ...]:
        rows = self.con.execute(f"EXPLAIN QUERY PLAN {plan.candidate_sql}", plan.params).fetchall()
        plan.diagnostics.sql_statements_executed += 1
        plan.diagnostics.sqlite_plan = tuple(str(row[-1]) for row in rows)
        return plan.diagnostics.sqlite_plan

    def _compile_graph(
            self,
            selector_graph: SelectorGraph,
            params: list[Any],
            diagnostics: LoweringDiagnostics,
            scan_policy: ScanPolicy) -> str:
        anchor = self._choose_anchor(selector_graph)
        if anchor is None:
            return self._compile_root_oriented_graph(
                selector_graph,
                params,
                diagnostics,
                fallback_reason="selector graph has no indexable anchor",
            )

        path_to_root = self._path_to_root(selector_graph, anchor.node_id)
        if path_to_root is None:
            diagnostics.anchor = f"{self._anchor_reason(anchor)}:{anchor.source_path!s}"
            diagnostics.anchor_node = anchor.node_id
            diagnostics.anchor_reason = self._anchor_reason(anchor)
            diagnostics.anchor_relation_kind = self._anchor_relation_kind(anchor)
            diagnostics.anchor_estimate = self._node_estimate(anchor)
            return self._compile_root_oriented_graph(
                selector_graph,
                params,
                diagnostics,
                fallback_reason="anchor path to root is ambiguous",
            )

        ctes: list[str] = []
        compiled_subtrees: set[int] = set()
        anchor_name = f"anchor_{anchor.node_id}"

        self._compile_anchor_relation(selector_graph, anchor, anchor_name, ctes, params, compiled_subtrees)

        relation_name = anchor_name
        steps: list[LoweredEdgeStep] = []
        for edge in path_to_root:
            for sibling in selector_graph.outgoing(edge.parent):
                if sibling.child == edge.child:
                    continue
                self._compile_subtree_relation(selector_graph, sibling.child, ctes, params, compiled_subtrees)
            edge_filters = self._edge_filters(edge, "e", params)
            parent_predicates = self._node_predicates(selector_graph.node(edge.parent), params)
            parent_predicates.extend(
                self._child_exists_predicate(sibling, f"subtree_{sibling.child}", params)
                for sibling in selector_graph.outgoing(edge.parent)
                if sibling.child != edge.child
            )
            diagnostics.semijoin_steps = (*diagnostics.semijoin_steps, *(
                f"child-exists:{edge.parent}->{sibling.child}:{sibling.path!s}"
                for sibling in selector_graph.outgoing(edge.parent)
                if sibling.child != edge.child
            ))
            where_parts = [*edge_filters, *parent_predicates]
            where_sql = " AND ".join(f"({predicate})" for predicate in where_parts) if where_parts else "1 = 1"
            parent_name = f"node_{edge.parent}_from_{edge.child}"
            ctes.append(
                f"""
                {parent_name}(def_id) AS (
                    SELECT DISTINCT e.parent_def_id
                    FROM definition_edges e
                    JOIN {relation_name} child_relation ON child_relation.def_id = e.child_def_id
                    JOIN definitions d ON d.def_id = e.parent_def_id
                    WHERE {where_sql}
                )
                """
            )
            steps.append(LoweredEdgeStep(edge.child, edge.parent, edge.path, "parent", edge.unordered))
            relation_name = parent_name

        anchor_reason = self._anchor_reason(anchor)
        anchor_estimate = self._node_estimate(anchor)
        graph_plan = LoweredGraphPlan(
            anchor_node=anchor.node_id,
            anchor_reason=anchor_reason,
            anchor_estimate=anchor_estimate,
            propagation_steps=tuple(steps),
            root_node=selector_graph.root,
        )
        diagnostics.anchor = f"{graph_plan.anchor_reason}:{anchor.source_path!s}"
        diagnostics.anchor_node = graph_plan.anchor_node
        diagnostics.anchor_reason = graph_plan.anchor_reason
        diagnostics.anchor_relation_kind = self._anchor_relation_kind(anchor)
        diagnostics.anchor_estimate = graph_plan.anchor_estimate
        diagnostics.propagation_steps = tuple(
            f"{step.direction}:{step.from_node}->{step.to_node}:{step.path!s}"
            for step in graph_plan.propagation_steps
        )
        diagnostics.relation_strategy = "rare-anchor-cte"
        return f"WITH {', '.join(ctes)} SELECT def_id FROM {relation_name}"

    def _compile_root_oriented_graph(
            self,
            selector_graph: SelectorGraph,
            params: list[Any],
            diagnostics: LoweringDiagnostics,
            *,
            fallback_reason: str | None = None) -> str:
        diagnostics.relation_strategy = "root-oriented-cte"
        diagnostics.anchor_fallback_reason = fallback_reason
        ctes: list[str] = []
        for node in reversed(selector_graph.nodes):
            predicates = self._node_predicates(node, params)
            for edge in selector_graph.outgoing(node.node_id):
                predicates.append(self._child_exists_predicate(edge, f"node_{edge.child}", params))
            where_sql = " AND ".join(f"({predicate})" for predicate in predicates) if predicates else "1 = 1"
            ctes.append(f"node_{node.node_id}(def_id) AS (SELECT d.def_id FROM definitions d WHERE {where_sql})")
        diagnostics.anchor = self._anchor_description(selector_graph)
        return f"WITH {', '.join(ctes)} SELECT def_id FROM node_{selector_graph.root}"

    def _compile_anchor_relation(
            self,
            selector_graph: SelectorGraph,
            anchor: SelectorGraphNode,
            anchor_name: str,
            ctes: list[str],
            params: list[Any],
        compiled_subtrees: set[int]) -> None:
        for edge in selector_graph.outgoing(anchor.node_id):
            self._compile_subtree_relation(selector_graph, edge.child, ctes, params, compiled_subtrees)
        if anchor.exact_definition is not None:
            predicates = self._node_predicates(anchor, params)
            predicates.extend(
                self._child_exists_predicate(edge, f"subtree_{edge.child}", params)
                for edge in selector_graph.outgoing(anchor.node_id)
            )
            where_sql = " AND ".join(f"({predicate})" for predicate in predicates) if predicates else "1 = 1"
            ctes.append(f"{anchor_name}(def_id) AS (SELECT d.def_id FROM definitions d WHERE {where_sql})")
            return

        posting_req = self._posting_anchor_requirement(anchor)
        if posting_req is None:
            predicates = self._node_predicates(anchor, params)
            predicates.extend(
                self._child_exists_predicate(edge, f"subtree_{edge.child}", params)
                for edge in selector_graph.outgoing(anchor.node_id)
            )
            where_sql = " AND ".join(f"({predicate})" for predicate in predicates) if predicates else "1 = 1"
            ctes.append(f"{anchor_name}(def_id) AS (SELECT d.def_id FROM definitions d WHERE {where_sql})")
            return

        feature_id = self._feature_id(posting_req.token)
        if feature_id is None:
            predicates = ["0 = 1"]
        else:
            predicates = ["p.feature_id = ?", "p.multiplicity >= ?"]
            params.extend((feature_id, posting_req.count))
        predicates.extend(self._node_predicates(anchor, params, skip_requirement=posting_req))
        predicates.extend(
            self._child_exists_predicate(edge, f"subtree_{edge.child}", params)
            for edge in selector_graph.outgoing(anchor.node_id)
        )
        where_sql = " AND ".join(f"({predicate})" for predicate in predicates) if predicates else "1 = 1"
        ctes.append(
            f"""
            {anchor_name}(def_id) AS (
                SELECT DISTINCT p.def_id
                FROM postings p
                JOIN definitions d ON d.def_id = p.def_id
                WHERE {where_sql}
            )
            """
        )

    def _compile_subtree_relation(
            self,
            selector_graph: SelectorGraph,
            node_id: int,
            ctes: list[str],
            params: list[Any],
            compiled: set[int]) -> None:
        if node_id in compiled:
            return
        node = selector_graph.node(node_id)
        for edge in selector_graph.outgoing(node_id):
            self._compile_subtree_relation(selector_graph, edge.child, ctes, params, compiled)
        predicates = self._node_predicates(node, params)
        predicates.extend(
            self._child_exists_predicate(edge, f"subtree_{edge.child}", params)
            for edge in selector_graph.outgoing(node_id)
        )
        where_sql = " AND ".join(f"({predicate})" for predicate in predicates) if predicates else "1 = 1"
        ctes.append(f"subtree_{node_id}(def_id) AS (SELECT d.def_id FROM definitions d WHERE {where_sql})")
        compiled.add(node_id)

    def _child_exists_predicate(self, edge: SelectorGraphEdge, child_relation_name: str, params: list[Any]) -> str:
        edge_filters = self._edge_filters(edge, "e", params)
        edge_sql = " AND ".join(f"({predicate})" for predicate in edge_filters) if edge_filters else "1 = 1"
        return f"""
        EXISTS (
            SELECT 1
            FROM definition_edges e
            JOIN {child_relation_name} child ON child.def_id = e.child_def_id
            WHERE e.parent_def_id = d.def_id
              AND {edge_sql}
        )
        """

    def _edge_filters(self, edge: SelectorGraphEdge, alias: str, params: list[Any]) -> list[str]:
        filters = [f"{alias}.edge_kind = ?"]
        params.append(edge.edge_kind.value)
        if edge.unordered:
            return filters
        path_blob = self.codec.encode_graph_path(edge.path)
        params.extend((digest_blob(path_blob), path_blob))
        filters.extend([f"{alias}.path_hash = ?", f"{alias}.path_blob = ?"])
        return filters

    def _node_predicates(self, node: SelectorGraphNode, params: list[Any], *, skip_requirement=None) -> list[str]:
        predicates: list[str] = []
        if node.exact_definition is not None:
            predicates.append("d.stable_hash = ?")
            params.append(stable_hash_to_blob(node.exact_definition.stable_hash()))
        for req in node.local_requirements:
            if skip_requirement is not None and req == skip_requirement:
                continue
            feature_id = self._feature_id(req.token)
            if feature_id is None:
                predicates.append("0 = 1")
                continue
            predicates.append(
                """
                EXISTS (
                    SELECT 1
                    FROM postings p
                    WHERE p.def_id = d.def_id
                      AND p.feature_id = ?
                      AND p.multiplicity >= ?
                )
                """
            )
            params.extend((feature_id, req.count))
        return predicates

    def _choose_anchor(self, selector_graph: SelectorGraph) -> SelectorGraphNode | None:
        anchors = []
        for node in selector_graph.nodes:
            if node.exact_definition is None and not node.local_requirements:
                continue
            estimate = self._node_estimate(node)
            reason_rank = 0 if node.exact_definition is not None else 1
            anchors.append((float("inf") if estimate is None else estimate, reason_rank, node.node_id, node))
        if not anchors:
            return None
        return min(anchors, key=lambda item: (item[0], item[1], item[2]))[3]

    def _path_to_root(self, selector_graph: SelectorGraph, node_id: int) -> tuple[SelectorGraphEdge, ...] | None:
        edges: list[SelectorGraphEdge] = []
        current = node_id
        while current != selector_graph.root:
            incoming = selector_graph.incoming(current)
            if len(incoming) != 1:
                return None
            edge = incoming[0]
            edges.append(edge)
            current = edge.parent
        return tuple(edges)

    def _anchor_reason(self, node: SelectorGraphNode) -> str:
        if node.exact_definition is not None:
            return "stable-hash"
        return "local-posting"

    def _anchor_relation_kind(self, node: SelectorGraphNode) -> str:
        if node.exact_definition is not None:
            return "stable-hash"
        if node.local_requirements:
            return "posting"
        return "scan"

    def _posting_anchor_requirement(self, node: SelectorGraphNode):
        candidates = []
        for req in node.local_requirements:
            feature_id = self._feature_id(req.token)
            if feature_id is None:
                candidates.append((0, req))
                continue
            row = self.con.execute(
                "SELECT document_frequency FROM feature_tokens WHERE feature_id = ?",
                (feature_id,),
            ).fetchone()
            candidates.append((0 if row is None else row[0], req))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _node_estimate(self, node: SelectorGraphNode) -> int | None:
        estimates = []
        if node.exact_definition is not None:
            estimates.append(self.con.execute(
                "SELECT COUNT(*) FROM definitions WHERE stable_hash = ?",
                (stable_hash_to_blob(node.exact_definition.stable_hash()),),
            ).fetchone()[0])
        for req in node.local_requirements:
            feature_id = self._feature_id(req.token)
            if feature_id is None:
                return 0
            row = self.con.execute(
                "SELECT document_frequency FROM feature_tokens WHERE feature_id = ?",
                (feature_id,),
            ).fetchone()
            estimates.append(0 if row is None else row[0])
        return min(estimates) if estimates else None

    def _apply_domain_sql(self, body_sql: str, domain_name: str) -> str:
        if domain_name == "stored":
            return f"""
            SELECT base.def_id
            FROM ({body_sql}) base
            JOIN stored_roots ON stored_roots.def_id = base.def_id
            """
        if domain_name == "nested":
            return f"""
            WITH base(def_id) AS ({body_sql}),
                 ancestors(start_id, current_id) AS (
                    SELECT base.def_id, definition_edges.parent_def_id
                    FROM base
                    JOIN definition_edges ON definition_edges.child_def_id = base.def_id
                    UNION
                    SELECT ancestors.start_id, definition_edges.parent_def_id
                    FROM ancestors
                    JOIN definition_edges ON definition_edges.child_def_id = ancestors.current_id
                 )
            SELECT DISTINCT base.def_id
            FROM base
            JOIN ancestors ON ancestors.start_id = base.def_id
            JOIN stored_roots ON stored_roots.def_id = ancestors.current_id
            """
        return body_sql

    def _feature_id(self, token) -> int | None:
        token_blob = self.codec.encode_feature_token(token)
        token_hash = digest_blob(token_blob)
        for feature_id, row_blob in self.con.execute(
                "SELECT feature_id, token_blob FROM feature_tokens WHERE token_hash = ?",
                (token_hash,)):
            if feature_token_equal(self.codec.decode_feature_token(row_blob), token):
                return feature_id
        return None

    def _estimate_rows(self, selector_graph: SelectorGraph | None, domain_name: str) -> int | None:
        if selector_graph is None:
            table = "stored_roots" if domain_name == "stored" else "definitions"
            return self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        estimates = []
        for node in selector_graph.nodes:
            if node.exact_definition is not None:
                estimates.append(self.con.execute(
                    "SELECT COUNT(*) FROM definitions WHERE stable_hash = ?",
                    (stable_hash_to_blob(node.exact_definition.stable_hash()),),
                ).fetchone()[0])
            for req in node.local_requirements:
                feature_id = self._feature_id(req.token)
                if feature_id is None:
                    return 0
                row = self.con.execute(
                    "SELECT document_frequency FROM feature_tokens WHERE feature_id = ?",
                    (feature_id,),
                ).fetchone()
                estimates.append(0 if row is None else row[0])
        if estimates:
            return min(estimates)
        table = "stored_roots" if domain_name == "stored" else "definitions"
        return self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def _anchor_description(self, selector_graph: SelectorGraph) -> str | None:
        anchors = []
        for node in selector_graph.nodes:
            if node.exact_definition is not None:
                anchors.append((0, str(node.source_path), "exact"))
            elif node.local_requirements:
                anchors.append((len(node.local_requirements), str(node.source_path), "local-posting"))
        if not anchors:
            return None
        _, path, mode = min(anchors)
        return f"{mode}:{path}"

    def _apply_scan_policy(self, scan_policy: ScanPolicy, diagnostics: LoweringDiagnostics, reason: str) -> None:
        diagnostics.scan_required = True
        diagnostics.scan_reason = reason
        if scan_policy.mode == "forbid":
            raise QueryWouldScanError(reason)
        if scan_policy.mode == "warn":
            warnings.warn(f"DRYML query requires scan fallback: {reason}", RuntimeWarning, stacklevel=3)

    def _logical_plan(self, selector_graph: SelectorGraph | None, diagnostics: LoweringDiagnostics) -> LogicalRelationPlan:
        if selector_graph is None:
            return LogicalRelationPlan(
                anchor_node=None,
                anchor_reason=diagnostics.anchor_reason,
                root_node=None,
                propagation_steps=diagnostics.propagation_steps,
                residual_constraints=("selector:none",),
            )
        return LogicalRelationPlan(
            anchor_node=diagnostics.anchor_node,
            anchor_reason=diagnostics.anchor_reason,
            root_node=selector_graph.root,
            propagation_steps=diagnostics.propagation_steps,
            residual_constraints=tuple(
                f"node:{node.node_id}:local={len(node.local_requirements)}:exact={node.exact_definition is not None}"
                for node in selector_graph.nodes
            ),
        )


def _has_indexable_requirement(selector_graph: SelectorGraph) -> bool:
    for node in selector_graph.nodes:
        if node.exact_definition is not None or node.local_requirements:
            return True
    return False
