from __future__ import annotations

import warnings
from typing import Any

from ..codecs import QueryIndexCodec, digest_blob
from ..domain import DefinitionDomain
from ..lowering import CandidateBatch, LoweredQueryPlan, LoweringDiagnostics, PagedResultCursor, QueryTerminal, ScanPolicy
from ..model import QueryWouldScanError
from ..selector_graph import SelectorGraph, SelectorGraphNode
from ..utils import feature_token_equal, stable_hash_to_blob


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

        params: list[Any] = []
        if selector_graph is None:
            body_sql = "SELECT d.def_id FROM definitions d"
            scan_reason = "selector has no indexable requirements"
            self._apply_scan_policy(scan_policy, diagnostics, scan_reason)
        else:
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
        ctes: list[str] = []
        for node in reversed(selector_graph.nodes):
            predicates = self._node_predicates(node, params)
            for edge in selector_graph.outgoing(node.node_id):
                if edge.unordered:
                    predicates.append(
                        f"""
                        EXISTS (
                            SELECT 1
                            FROM definition_edges e
                            JOIN node_{edge.child} child ON child.def_id = e.child_def_id
                            WHERE e.parent_def_id = d.def_id
                        )
                        """
                    )
                else:
                    path_blob = self.codec.encode_graph_path(edge.path)
                    predicates.append(
                        f"""
                        EXISTS (
                            SELECT 1
                            FROM definition_edges e
                            JOIN node_{edge.child} child ON child.def_id = e.child_def_id
                            WHERE e.parent_def_id = d.def_id
                              AND e.path_hash = ?
                              AND e.path_blob = ?
                        )
                        """
                    )
                    params.extend((digest_blob(path_blob), path_blob))
            where_sql = " AND ".join(f"({predicate})" for predicate in predicates) if predicates else "1 = 1"
            ctes.append(f"node_{node.node_id}(def_id) AS (SELECT d.def_id FROM definitions d WHERE {where_sql})")
        diagnostics.anchor = self._anchor_description(selector_graph)
        return f"WITH {', '.join(ctes)} SELECT def_id FROM node_{selector_graph.root}"

    def _node_predicates(self, node: SelectorGraphNode, params: list[Any]) -> list[str]:
        predicates: list[str] = []
        if node.exact_definition is not None:
            predicates.append("d.stable_hash = ?")
            params.append(stable_hash_to_blob(node.exact_definition.stable_hash()))
        for req in node.local_requirements:
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
        return min(estimates) if estimates else None

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
