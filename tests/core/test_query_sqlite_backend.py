import threading

import pytest

from dryml.core2 import Definition, Object, SKIP_ARGS
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.query.fingerprint import target_local_fingerprint
from dryml.core2.query.lowering import CandidateRelation, LoweringDiagnostics, ScanPolicy
from dryml.core2.query.model import DefinitionFingerprint, FeatureRequirement, FeatureToken, QueryIndexDirty, QueryIndexError, QueryStats, QueryWouldScanError
from dryml.core2.query.domain import StoredDomain
from dryml.core2.query.graph_plan import graph_candidate_ids
from dryml.core2.query.path import GraphPath
from dryml.core2.query.selector_graph import SelectorGraph, SelectorGraphEdge, SelectorGraphNode, compile_selector_graph
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig, require_sqlite, sqlite_available
import dryml.core2.query.sqlite.index as sqlite_index_module
from dryml.core2.query.sqlite.index import SQLiteStoreQueryIndex


pytestmark = pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")


class SQLiteLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class SQLiteParent(Object):
    def __init__(self, child=None, *, name="parent"):
        super().__init__()
        self.child = child
        self.name = name


class SQLitePair(Object):
    def __init__(self, left=None, right=None, *, name="pair"):
        super().__init__()
        self.left = left
        self.right = right
        self.name = name


def sqlite_index(tmp_path):
    return SQLiteStoreQueryIndex(
        source_key="sqlite-test-store",
        path=tmp_path / "index.sqlite",
        config=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=1.0),
    )


def sqlite_rows(path, sql, params=()):
    sqlite3 = require_sqlite()
    con = sqlite3.connect(path)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def test_register_stored_roots_persists_graph_and_is_idempotent(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf("rare")
    owner = SQLiteParent(child=leaf, name="owner")
    graph = ConcreteDefinitionGraph.from_root(owner.definition)

    result = index.register_stored_roots(graph, [owner.definition])
    repeat = index.register_stored_roots(graph, [owner.definition])

    assert result.changed
    assert result.generation == 1
    assert result.definitions_added == 2
    assert result.edges_added == 1
    assert result.roots_added == 1
    assert repeat.changed is False
    assert repeat.generation == 1
    assert index.status().generation == 1

    with index.read_view() as view:
        owner_id = view.cdef_id(owner.definition)
        leaf_id = view.cdef_id(leaf.definition)
        assert owner_id is not None
        assert leaf_id is not None
        assert view.is_stored_id(owner_id)
        assert not view.is_stored_id(leaf_id)
        assert view.has_stored_ancestor(leaf_id)
        assert view.cdefs_by_id({owner_id, leaf_id}) == {
            owner_id: owner.definition,
            leaf_id: leaf.definition,
        }


def test_registered_rows_include_edge_and_root_metadata(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf("metadata")
    owner = SQLiteParent(child=leaf, name="owner")
    graph = ConcreteDefinitionGraph.from_root(owner.definition)

    result = index.register_stored_roots(graph, [owner.definition])

    assert sqlite_rows(index.path, "SELECT unordered FROM definition_edges") == [(0,)]
    stored_root = sqlite_rows(
        index.path,
        """
        SELECT storage_hash, relative_def_path, def_size, def_mtime_ns, indexed_generation
        FROM stored_roots
        """,
    )[0]
    assert stored_root[0] == bytes.fromhex(owner.definition.stable_hash())
    assert stored_root[1] == f"objects/{owner.definition.stable_hash()[:2]}/{owner.definition.stable_hash()}/def.pkl"
    assert stored_root[2] is None
    assert stored_root[3] is None
    assert stored_root[4] == result.generation


def test_register_prepares_encoded_rows_before_write_transaction(tmp_path, monkeypatch):
    index = sqlite_index(tmp_path)
    root = SQLiteLeaf("transaction-boundary")
    graph = ConcreteDefinitionGraph.from_root(root.definition)
    events = []
    original_from_cdef = sqlite_index_module._EncodedNode.from_cdef
    original_run_write = SQLiteStoreQueryIndex._run_write_transaction

    def spy_from_cdef(cls, cdef):
        events.append("encode")
        return original_from_cdef(cdef)

    def spy_run_write(self, operation):
        events.append("transaction")
        return original_run_write(self, operation)

    monkeypatch.setattr(sqlite_index_module._EncodedNode, "from_cdef", classmethod(spy_from_cdef))
    monkeypatch.setattr(SQLiteStoreQueryIndex, "_run_write_transaction", spy_run_write)

    index.register_stored_roots(graph, [root.definition])

    assert events[:2] == ["encode", "transaction"]


def test_register_graph_persists_rows_without_stored_root(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf("graph-only")
    owner = SQLiteParent(child=leaf, name="owner")
    graph = ConcreteDefinitionGraph.from_root(owner.definition)

    result = index.register_graph(graph)
    repeat = index.register_graph(graph)

    assert result.changed
    assert result.generation == 1
    assert result.definitions_added == 2
    assert result.edges_added == 1
    assert result.roots_added == 0
    assert repeat.changed is False
    assert repeat.generation == 1
    assert sqlite_rows(index.path, "SELECT COUNT(*) FROM definitions") == [(2,)]
    assert sqlite_rows(index.path, "SELECT COUNT(*) FROM definition_edges") == [(1,)]
    assert sqlite_rows(index.path, "SELECT COUNT(*) FROM stored_roots") == [(0,)]


def test_activate_and_deactivate_stored_root(tmp_path):
    index = sqlite_index(tmp_path)
    obj = SQLiteLeaf("activation")
    graph = ConcreteDefinitionGraph.from_root(obj.definition)

    index.register_graph(graph)
    activated = index.activate_stored_roots(graph, [obj.definition])
    repeat = index.activate_stored_roots(graph, [obj.definition])
    removed = index.remove_stored_roots([obj.definition])

    assert activated.changed
    assert activated.generation == 2
    assert activated.roots_added == 1
    assert repeat.changed is False
    assert repeat.generation == 2
    assert removed.changed
    assert removed.roots_removed == 1
    assert removed.generation == 3
    assert sqlite_rows(index.path, "SELECT COUNT(*) FROM stored_roots") == [(0,)]


def test_register_saved_graph_activates_stored_root(tmp_path):
    index = sqlite_index(tmp_path)
    obj = SQLiteLeaf("saved")
    graph = ConcreteDefinitionGraph.from_root(obj.definition)

    result = index.register_saved_graph(graph, [obj.definition])

    assert result.changed
    assert result.roots_added == 1
    assert sqlite_rows(index.path, "SELECT COUNT(*) FROM stored_roots") == [(1,)]


def test_hash_colliding_definitions_get_distinct_ordinals(monkeypatch, tmp_path):
    monkeypatch.setattr(sqlite_index_module, "stable_hash_to_blob", lambda stable_hash: b"same-definition-hash")
    index = sqlite_index(tmp_path)
    left = SQLiteLeaf("ordinal-left")
    right = SQLiteLeaf("ordinal-right")
    graph = ConcreteDefinitionGraph.from_roots([left.definition, right.definition])

    index.register_stored_roots(graph, [left.definition, right.definition])

    assert sqlite_rows(index.path, "SELECT collision_ordinal FROM definitions ORDER BY collision_ordinal") == [(0,), (1,)]


def test_posting_multiplicity_and_document_frequency_are_preserved(monkeypatch, tmp_path):
    token = FeatureToken("TEST_MULTIPLICITY", GraphPath(), "payload")

    def repeated_feature(_cdef):
        return DefinitionFingerprint({token: 3})

    monkeypatch.setattr(sqlite_index_module, "target_local_fingerprint", repeated_feature)
    index = sqlite_index(tmp_path)
    obj = SQLiteLeaf("multiplicity")
    graph = ConcreteDefinitionGraph.from_root(obj.definition)

    result = index.register_stored_roots(graph, [obj.definition])
    repeat = index.register_stored_roots(graph, [obj.definition])

    assert result.postings_added == 1
    assert repeat.changed is False
    assert sqlite_rows(index.path, "SELECT multiplicity FROM postings") == [(3,)]
    assert sqlite_rows(index.path, "SELECT document_frequency FROM feature_tokens") == [(1,)]


def test_exact_and_local_candidate_lookup(tmp_path):
    index = sqlite_index(tmp_path)
    wanted = SQLiteParent(child=SQLiteLeaf(name="wanted"), name="target")
    other = SQLiteParent(child=SQLiteLeaf(name="other"), name="target")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])

    selector = Definition(SQLiteParent, SKIP_ARGS, child=Definition(SQLiteLeaf, SKIP_ARGS, name="wanted"))
    selector_graph = compile_selector_graph(selector)

    with index.read_view() as view:
        domain = StoredDomain(view)
        ids = graph_candidate_ids(view, selector_graph, domain)
        cdefs = tuple(view.cdefs_by_id(ids).values())

    assert cdefs == (wanted.definition,)


def test_lowered_exact_anchor_relation_uses_hash_index(tmp_path):
    index = sqlite_index(tmp_path)
    wanted = SQLiteLeaf("wanted")
    other = SQLiteLeaf("other")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])

    selector_graph = compile_selector_graph(wanted.definition)
    with index.read_view() as view:
        diagnostics = LoweringDiagnostics()
        plan = view.lower_selector_graph(
            selector_graph,
            StoredDomain(view),
            terminal="explain",
            scan_policy=ScanPolicy(),
            diagnostics=diagnostics,
        )
        explain = view.explain_lowered_plan(plan)

    assert any("definitions_by_stable_hash" in row or "stable_hash" in row for row in explain)
    assert diagnostics.strategy == "sqlite-lowered"


def test_lowered_feature_anchor_relation_uses_postings_index(tmp_path, monkeypatch):
    token = FeatureToken("LOWERED_FEATURE", GraphPath(), "wanted")

    def controlled_fingerprint(cdef):
        return DefinitionFingerprint({token: 1})

    monkeypatch.setattr(sqlite_index_module, "target_local_fingerprint", controlled_fingerprint)
    index = sqlite_index(tmp_path)
    wanted = SQLiteLeaf("wanted")
    other = SQLiteLeaf("other")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])

    selector_graph = SelectorGraph(
        root=0,
        nodes=(SelectorGraphNode(0, GraphPath(), wanted.definition, (FeatureRequirement(token),), None),),
        edges=(),
    )
    with index.read_view() as view:
        plan = view.lower_selector_graph(
            selector_graph,
            StoredDomain(view),
            terminal="explain",
            scan_policy=ScanPolicy(),
            diagnostics=LoweringDiagnostics(),
        )
        explain = view.explain_lowered_plan(plan)

    assert "FROM postings" in plan.candidate_sql
    assert any("USING" in row and ("INDEX" in row or "PRIMARY KEY" in row) for row in explain)


def test_lowered_child_edge_join_matches_v1_path(tmp_path):
    index = sqlite_index(tmp_path)
    wanted = SQLiteParent(child=SQLiteLeaf(name="wanted"), name="root")
    other = SQLiteParent(child=SQLiteLeaf(name="other"), name="root")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])

    selector = Definition(SQLiteParent, SKIP_ARGS, child=Definition(SQLiteLeaf, SKIP_ARGS, name="wanted"))
    selector_graph = compile_selector_graph(selector)
    with index.read_view() as view:
        v1_ids = graph_candidate_ids(view, selector_graph, StoredDomain(view))
        plan = view.lower_selector_graph(
            selector_graph,
            StoredDomain(view),
            terminal="collect",
            scan_policy=ScanPolicy(),
            diagnostics=LoweringDiagnostics(),
        )
        batch = next(view.iter_candidate_cdef_batches(plan, batch_size=10))

    assert set(batch.ids) == v1_ids
    assert batch.cdefs == (wanted.definition,)


def test_lowered_sql_uses_rare_exact_nested_anchor(tmp_path):
    index = sqlite_index(tmp_path)
    rare_leaf = SQLiteLeaf(name="rare")
    wanted = SQLiteParent(child=rare_leaf, name="root")
    other = SQLiteParent(child=SQLiteLeaf(name="other"), name="root")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])
    selector = Definition(SQLiteParent, SKIP_ARGS, child=rare_leaf.definition)

    with index.read_view() as view:
        diagnostics = LoweringDiagnostics()
        plan = view.lower_selector_graph(
            compile_selector_graph(selector),
            StoredDomain(view),
            terminal="collect",
            scan_policy=ScanPolicy(),
            diagnostics=diagnostics,
        )
        batch = next(view.iter_candidate_cdef_batches(plan, batch_size=10))

    assert batch.cdefs == (wanted.definition,)
    assert "anchor_1" in plan.candidate_sql
    assert "JOIN anchor_1 child_relation" in plan.candidate_sql
    assert diagnostics.anchor_node == 1
    assert diagnostics.anchor_reason == "stable-hash"


def test_lowered_sql_uses_rare_local_posting_anchor(tmp_path):
    index = sqlite_index(tmp_path)
    wanted = SQLiteParent(child=SQLiteLeaf(name="rare-local"), name="root")
    other = SQLiteParent(child=SQLiteLeaf(name="other"), name="root")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])
    selector = Definition(SQLiteParent, SKIP_ARGS, child=Definition(SQLiteLeaf, SKIP_ARGS, name="rare-local"))

    with index.read_view() as view:
        diagnostics = LoweringDiagnostics()
        plan = view.lower_selector_graph(
            compile_selector_graph(selector),
            StoredDomain(view),
            terminal="collect",
            scan_policy=ScanPolicy(),
            diagnostics=diagnostics,
        )
        batch = next(view.iter_candidate_cdef_batches(plan, batch_size=10))

    assert batch.cdefs == (wanted.definition,)
    assert "anchor_1" in plan.candidate_sql
    assert "FROM postings" in plan.candidate_sql
    assert diagnostics.anchor_node == 1
    assert diagnostics.anchor_reason == "local-posting"


def test_lowered_plan_reports_anchor_and_direction(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf(name="direction")
    root = SQLiteParent(child=leaf, name="root")
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(root.definition), [root.definition])

    with index.read_view() as view:
        diagnostics = LoweringDiagnostics()
        view.lower_selector_graph(
            compile_selector_graph(Definition(SQLiteParent, SKIP_ARGS, child=leaf.definition)),
            StoredDomain(view),
            terminal="explain",
            scan_policy=ScanPolicy(),
            diagnostics=diagnostics,
        )

    assert diagnostics.anchor == "stable-hash:$.child"
    assert diagnostics.anchor_estimate == 1
    assert diagnostics.propagation_steps == ("parent:1->0:$.child",)


def test_lowered_parent_projection_uses_edge_index(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf(name="edge-index")
    root = SQLiteParent(child=leaf, name="root")
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(root.definition), [root.definition])

    with index.read_view() as view:
        plan = view.lower_selector_graph(
            compile_selector_graph(Definition(SQLiteParent, SKIP_ARGS, child=leaf.definition)),
            StoredDomain(view),
            terminal="explain",
            scan_policy=ScanPolicy(),
        )
        explain = view.explain_lowered_plan(plan)

    assert any("SEARCH e" in row and ("definition_edges" in row or "PRIMARY KEY" in row) for row in explain)


def test_lowered_child_projection_uses_edge_index(tmp_path):
    index = sqlite_index(tmp_path)
    wanted = SQLitePair(left=SQLiteLeaf(name="left"), right=SQLiteLeaf(name="right"), name="pair")
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(wanted.definition), [wanted.definition])
    selector = Definition(
        SQLitePair,
        SKIP_ARGS,
        left=Definition(SQLiteLeaf, SKIP_ARGS, name="left"),
        right=Definition(SQLiteLeaf, SKIP_ARGS, name="right"),
    )

    with index.read_view() as view:
        plan = view.lower_selector_graph(
            compile_selector_graph(selector),
            StoredDomain(view),
            terminal="explain",
            scan_policy=ScanPolicy(),
        )
        explain = view.explain_lowered_plan(plan)

    assert "EXISTS" in plan.candidate_sql
    assert any("SEARCH e" in row and ("definition_edges" in row or "PRIMARY KEY" in row) for row in explain)


def test_candidate_relation_pages_without_cursor_escape(tmp_path):
    index = sqlite_index(tmp_path)
    roots = [SQLiteLeaf(f"relation-{idx}").definition for idx in range(3)]
    index.register_stored_roots(ConcreteDefinitionGraph.from_roots(roots), roots)

    with index.read_view() as view:
        plan = view.lower_selector_graph(
            None,
            StoredDomain(view),
            terminal="page",
            scan_policy=ScanPolicy(),
            diagnostics=LoweringDiagnostics(),
        )
        relation = plan.relation()
        first = next(view.iter_candidate_cdef_batches(plan, batch_size=2))

    assert isinstance(relation, CandidateRelation)
    assert first.cdefs
    assert first.next_cursor is not None
    assert not hasattr(first.next_cursor, "cursor")
    assert not hasattr(first.next_cursor, "connection")


def test_candidate_relation_preserves_generation(tmp_path):
    index = sqlite_index(tmp_path)
    root = SQLiteLeaf("generation").definition
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])

    with index.read_view() as view:
        plan = view.lower_selector_graph(None, StoredDomain(view), terminal="page", scan_policy=ScanPolicy())
        relation = plan.relation()

    assert relation.source_key == index.source_key
    assert relation.generation == index.current_generation()


def test_candidate_relation_empty(tmp_path):
    index = sqlite_index(tmp_path)
    index.initialize_empty()

    with index.read_view() as view:
        plan = view.lower_selector_graph(None, StoredDomain(view), terminal="page", scan_policy=ScanPolicy())
        batch = next(view.iter_candidate_cdef_batches(plan, batch_size=2), None)

    assert batch is None


def test_candidate_relation_ordering_stable(tmp_path):
    index = sqlite_index(tmp_path)
    roots = [SQLiteLeaf(name).definition for name in ("c", "a", "b")]
    index.register_stored_roots(ConcreteDefinitionGraph.from_roots(roots), roots)

    with index.read_view() as view:
        plan = view.lower_selector_graph(None, StoredDomain(view), terminal="page", scan_policy=ScanPolicy())
        relation = plan.relation()
        first = next(view.iter_candidate_cdef_batches(plan, batch_size=2))
        second = next(view.iter_candidate_cdef_batches(plan, after=first.next_cursor, batch_size=2))

    observed = first.cdefs + second.cdefs
    assert relation.ordering == ("stable_hash", "collision_ordinal", "definition_id")
    assert observed == tuple(sorted(roots, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))


def test_lowered_unordered_edge_uses_conservative_edge_relation(tmp_path):
    index = sqlite_index(tmp_path)
    wanted_leaf = SQLiteLeaf(name="wanted")
    wanted = SQLiteParent(child=wanted_leaf, name="root")
    other = SQLiteParent(child=SQLiteLeaf(name="other"), name="root")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])
    selector_graph = SelectorGraph(
        root=0,
        nodes=(
            SelectorGraphNode(0, GraphPath(), Definition(SQLiteParent, SKIP_ARGS), (), None),
            SelectorGraphNode(1, GraphPath().child("child"), wanted_leaf.definition, (), wanted_leaf.definition),
        ),
        edges=(SelectorGraphEdge(0, GraphPath().child("child"), 1, unordered=True),),
    )

    with index.read_view() as view:
        plan = view.lower_selector_graph(
            selector_graph,
            StoredDomain(view),
            terminal="collect",
            scan_policy=ScanPolicy("forbid"),
            diagnostics=LoweringDiagnostics(),
        )
        batch = next(view.iter_candidate_cdef_batches(plan, batch_size=10))

    assert wanted.definition in batch.cdefs
    assert other.definition not in batch.cdefs


def test_temp_relation_is_dropped_on_view_exit(tmp_path):
    index = sqlite_index(tmp_path)
    obj = SQLiteLeaf("temp")
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(obj.definition), [obj.definition])

    with index.read_view() as view:
        did = view.cdef_id(obj.definition)
        name = view.create_temp_relation([did])
        assert view._con.execute(
            "SELECT name FROM sqlite_temp_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone() == (name,)

    with index.read_view() as view:
        assert view._con.execute(
            "SELECT name FROM sqlite_temp_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone() is None


def test_lowered_within_relation_restricts_candidates_with_temp_table(tmp_path):
    index = sqlite_index(tmp_path)
    wanted = SQLiteLeaf("within-wanted")
    other = SQLiteLeaf("within-other")
    index.register_stored_roots(
        ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition]),
        [wanted.definition, other.definition],
    )

    with index.read_view() as view:
        wanted_id = view.cdef_id(wanted.definition)
        within = view.create_temp_relation([wanted_id])
        plan = view.lower_selector_graph(
            None,
            StoredDomain(view),
            terminal="collect",
            scan_policy=ScanPolicy(),
            diagnostics=LoweringDiagnostics(),
            within_relation=within,
        )
        batch = next(view.iter_candidate_cdef_batches(plan, batch_size=10))

    assert batch.cdefs == (wanted.definition,)
    assert plan.diagnostics.relation_strategy == "cte+temp"


def test_lowered_stored_domain_filters_before_cdef_fetch(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf("nested-only")
    owner = SQLiteParent(child=leaf, name="owner")
    graph = ConcreteDefinitionGraph.from_root(owner.definition)
    index.register_stored_roots(graph, [owner.definition])

    selector_graph = compile_selector_graph(Definition(SQLiteLeaf, SKIP_ARGS, name="nested-only"))
    with index.read_view() as view:
        plan = view.lower_selector_graph(
            selector_graph,
            StoredDomain(view),
            terminal="collect",
            scan_policy=ScanPolicy(),
            diagnostics=LoweringDiagnostics(),
        )
        batches = tuple(view.iter_candidate_cdef_batches(plan, batch_size=10))

    assert batches == ()


def test_lowered_explain_plan_only_does_not_decode_cdefs(tmp_path, monkeypatch):
    index = sqlite_index(tmp_path)
    obj = SQLiteLeaf("plan-only")
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(obj.definition), [obj.definition])

    def fail_decode(_blob):
        raise AssertionError("plan-only explain should not decode CDefs")

    monkeypatch.setattr(sqlite_index_module._CODEC, "decode_cdef", fail_decode)
    with index.read_view() as view:
        plan = view.lower_selector_graph(
            compile_selector_graph(Definition(SQLiteLeaf, SKIP_ARGS, name="plan-only")),
            StoredDomain(view),
            terminal="explain",
            scan_policy=ScanPolicy(),
            diagnostics=LoweringDiagnostics(),
        )
        view.explain_lowered_plan(plan)


def test_scan_policy_forbid_rejects_unindexed_selector(tmp_path):
    index = sqlite_index(tmp_path)
    obj = SQLiteLeaf("scan")
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(obj.definition), [obj.definition])

    with index.read_view() as view:
        with pytest.raises(QueryWouldScanError):
            view.lower_selector_graph(
                None,
                StoredDomain(view),
                terminal="collect",
                scan_policy=ScanPolicy("forbid"),
                diagnostics=LoweringDiagnostics(),
            )


def test_scan_policy_forbid_rejects_empty_selector_graph(tmp_path):
    index = sqlite_index(tmp_path)
    root = SQLiteLeaf("scan-graph").definition
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])

    with index.read_view() as view:
        with pytest.raises(QueryWouldScanError, match="selector graph has no indexable requirements"):
            view.lower_selector_graph(
                compile_selector_graph(Definition(SQLiteLeaf, SKIP_ARGS)),
                StoredDomain(view),
                terminal="collect",
                scan_policy=ScanPolicy("forbid"),
                diagnostics=LoweringDiagnostics(),
            )


def test_explain_reports_scan_reason(tmp_path):
    index = sqlite_index(tmp_path)
    root = SQLiteLeaf("scan-reason").definition
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])

    with index.read_view() as view:
        diagnostics = LoweringDiagnostics()
        view.lower_selector_graph(
            compile_selector_graph(Definition(SQLiteLeaf, SKIP_ARGS)),
            StoredDomain(view),
            terminal="explain",
            scan_policy=ScanPolicy(),
            diagnostics=diagnostics,
        )

    assert diagnostics.scan_required
    assert diagnostics.scan_reason == "selector graph has no indexable requirements"


def test_read_view_estimates_candidates_and_query_stats(tmp_path):
    index = sqlite_index(tmp_path)
    wanted = SQLiteParent(child=SQLiteLeaf(name="wanted"), name="target")
    other = SQLiteParent(child=SQLiteLeaf(name="other"), name="target")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])
    requirements = tuple(
        FeatureRequirement(token, count)
        for token, count in target_local_fingerprint(wanted.definition).counts.items()
    )
    stats = QueryStats()

    with index.read_view() as view:
        domain = StoredDomain(view)
        exact_ids = view.exact_ids(wanted.definition)
        candidate_ids = view.local_candidates(requirements, domain=domain, stats=stats)

        assert view.estimate_exact_ids(wanted.definition) >= len(exact_ids) == 1
        assert view.estimate_local_candidates(requirements) >= len(candidate_ids) == 1
        assert tuple(view.cdefs_by_id(candidate_ids).values()) == (wanted.definition,)

    assert stats.selected_features
    assert stats.posting_sizes
    assert stats.candidate_count == 1


def test_read_view_filter_domain_replica_map_and_capture_occurrences(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf("phase4")
    owner = SQLiteParent(child=leaf, name="owner")
    graph = ConcreteDefinitionGraph.from_root(owner.definition)
    index.register_stored_roots(graph, [owner.definition])

    with index.read_view() as view:
        owner_id = view.cdef_id(owner.definition)
        leaf_id = view.cdef_id(leaf.definition)
        stored_domain = StoredDomain(view)

        assert view.filter_domain(stored_domain, {owner_id, leaf_id}) == {owner_id}
        assert view.filter_nested_ids({owner_id, leaf_id}) == {leaf_id}
        assert view.replica_map({owner_id, leaf_id}) == {owner.definition: (index.source_key,)}

        projection = view.project_owners({leaf_id})
        occurrences = view.capture_occurrences({leaf_id})
        all_occurrences = view.capture_occurrences(max_occurrences=1)

    assert projection.owner_ids == frozenset({owner_id})
    assert projection.cdefs == (owner.definition,)
    assert len(occurrences) == 1
    assert occurrences[0].owner == owner.definition
    assert occurrences[0].definition == leaf.definition
    assert all_occurrences[0].owner == owner.definition


def test_read_view_generation_snapshot_and_closed_view_contract(tmp_path):
    index = sqlite_index(tmp_path)
    first = SQLiteLeaf("first")
    second = SQLiteLeaf("second")
    index.register_stored_roots(ConcreteDefinitionGraph.from_root(first.definition), [first.definition])

    with index.read_view() as view:
        generation = view.generation
        assert generation == 1
        assert view.generation == generation
        assert view.exact_ids(second.definition) == set()

    with pytest.raises(QueryIndexError, match="closed"):
        view.all_definition_ids()

    index.register_stored_roots(ConcreteDefinitionGraph.from_root(second.definition), [second.definition])
    with index.read_view() as new_view:
        assert new_view.generation == 2
        assert len(new_view.exact_ids(second.definition)) == 1


def test_definition_hash_collision_confirms_full_cdef_equality(monkeypatch, tmp_path):
    monkeypatch.setattr(sqlite_index_module, "stable_hash_to_blob", lambda stable_hash: b"same-definition-hash")
    index = sqlite_index(tmp_path)
    left = SQLiteLeaf("left")
    right = SQLiteLeaf("right")
    graph = ConcreteDefinitionGraph.from_roots([left.definition, right.definition])

    index.register_stored_roots(graph, [left.definition, right.definition])

    with index.read_view() as view:
        left_ids = view.exact_ids(left.definition)
        right_ids = view.exact_ids(right.definition)
        left_cdefs = tuple(view.cdefs_by_id(left_ids).values())
        right_cdefs = tuple(view.cdefs_by_id(right_ids).values())

    assert len(left_ids) == 1
    assert len(right_ids) == 1
    assert left_ids != right_ids
    assert left_cdefs == (left.definition,)
    assert right_cdefs == (right.definition,)


def test_feature_hash_collision_confirms_full_token_equality(monkeypatch, tmp_path):
    monkeypatch.setattr(sqlite_index_module, "digest_blob", lambda blob: b"same-feature-hash")
    index = sqlite_index(tmp_path)
    wanted = SQLiteLeaf("wanted")
    other = SQLiteLeaf("other")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])

    requirements = tuple(
        FeatureRequirement(token, count)
        for token, count in target_local_fingerprint(wanted.definition).counts.items()
    )

    with index.read_view() as view:
        domain = StoredDomain(view)
        ids = view.local_candidates(requirements, domain=domain)
        cdefs = tuple(view.cdefs_by_id(ids).values())

    assert cdefs == (wanted.definition,)


def test_path_hash_collision_confirms_full_path_equality(monkeypatch, tmp_path):
    monkeypatch.setattr(sqlite_index_module, "digest_blob", lambda blob: b"same-path-hash")
    index = sqlite_index(tmp_path)
    left = SQLiteLeaf("left")
    right = SQLiteLeaf("right")
    owner = SQLitePair(left=left, right=right, name="owner")
    graph = ConcreteDefinitionGraph.from_root(owner.definition)
    index.register_stored_roots(graph, [owner.definition])
    left_path = next(edge.path for edge in graph.edges() if edge.child == left.definition)
    right_path = next(edge.path for edge in graph.edges() if edge.child == right.definition)

    with index.read_view() as view:
        owner_id = view.cdef_id(owner.definition)
        left_id = view.cdef_id(left.definition)
        right_id = view.cdef_id(right.definition)
        assert view.children({owner_id}, left_path, unordered=False) == {left_id}
        assert view.children({owner_id}, right_path, unordered=False) == {right_id}
        assert view.parents({left_id}, left_path, unordered=False) == {owner_id}
        assert view.parents({right_id}, right_path, unordered=False) == {owner_id}


def test_parent_child_relations_and_nested_projection(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf("nested")
    owner = SQLitePair(left=leaf, right=SQLiteLeaf("other"), name="owner")
    graph = ConcreteDefinitionGraph.from_root(owner.definition)
    index.register_stored_roots(graph, [owner.definition])

    # Use concrete graph paths for the exact relation checks.
    left_path = next(edge.path for edge in graph.edges() if edge.child == leaf.definition)
    with index.read_view() as view:
        owner_id = view.cdef_id(owner.definition)
        leaf_id = view.cdef_id(leaf.definition)
        assert view.children({owner_id}, left_path, unordered=False) == {leaf_id}
        assert view.parents({leaf_id}, left_path, unordered=False) == {owner_id}
        assert view.children({owner_id}, left_path.parent, unordered=True) >= {leaf_id}
        assert leaf_id in view.nested_ids()
        assert view.filter_nested_ids({owner_id, leaf_id}) == {leaf_id}


def test_owner_projection_and_occurrence_capture(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf("shared")
    owner1 = SQLiteParent(child=leaf, name="owner1")
    owner2 = SQLiteParent(child=leaf, name="owner2")
    graph = ConcreteDefinitionGraph.from_roots([owner1.definition, owner2.definition])
    index.register_stored_roots(graph, [owner1.definition, owner2.definition])

    with index.read_view() as view:
        leaf_id = view.cdef_id(leaf.definition)
        projection = view.project_owners({leaf_id})
        occurrences = tuple(view.occurrence_snapshot_for_nested_ids({leaf_id}).iter_occurrences())

    assert set(projection.cdefs) == {owner1.definition, owner2.definition}
    assert {occ.owner for occ in occurrences} == {owner1.definition, owner2.definition}
    assert {occ.definition for occ in occurrences} == {leaf.definition}
    assert all(str(occ.path) == "$.child" for occ in occurrences)


def test_remove_stored_roots_updates_generation_and_stored_scope(tmp_path):
    index = sqlite_index(tmp_path)
    obj = SQLiteLeaf("stored")
    graph = ConcreteDefinitionGraph.from_root(obj.definition)
    index.register_stored_roots(graph, [obj.definition])

    removed = index.remove_stored_roots([obj.definition])
    repeat = index.remove_stored_roots([obj.definition])

    assert removed.changed
    assert removed.roots_removed == 1
    assert removed.generation == 2
    assert repeat.changed is False
    assert repeat.generation == 2
    with index.read_view() as view:
        did = view.cdef_id(obj.definition)
        assert did is not None
        assert not view.is_stored_id(did)


def test_write_transaction_rolls_back_partial_graph_rows(monkeypatch, tmp_path):
    index = sqlite_index(tmp_path)
    obj = SQLiteLeaf("rollback")
    graph = ConcreteDefinitionGraph.from_root(obj.definition)
    index.initialize_empty()

    def fail_feature_resolution(con, token_blob):
        raise RuntimeError("injected feature failure")

    monkeypatch.setattr(sqlite_index_module, "_resolve_feature_id", fail_feature_resolution)

    with pytest.raises(RuntimeError, match="injected feature failure"):
        index.register_stored_roots(graph, [obj.definition])

    status = index.status()
    assert status.generation == 0
    assert status.row_counts == {
        "definitions": 0,
        "feature_tokens": 0,
        "postings": 0,
        "definition_edges": 0,
        "stored_roots": 0,
    }
    with index.read_view() as view:
        assert view.exact_ids(obj.definition) == set()


def test_building_index_reports_and_blocks_reads(tmp_path):
    index = sqlite_index(tmp_path)

    index.initialize_empty(build_state="building")

    assert index.status().state == "building"
    report = index.validate()
    assert not report.ok
    assert "not ready" in report.issues[0].message
    assert report.issues[0].detail == "build_state='building'"
    with pytest.raises(QueryIndexDirty, match="not ready"):
        index.current_generation()


def test_register_retries_transient_busy_writer(tmp_path):
    path = tmp_path / "index.sqlite"
    holder_ready = threading.Event()
    release_holder = threading.Event()
    holder_errors = []

    def hold_writer():
        holder = SQLiteStoreQueryIndex(
            source_key="sqlite-test-store",
            path=path,
            config=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=1.0),
        )
        try:
            holder.initialize_empty()
            con = holder._connections.connection(readonly=False)
            con.execute("BEGIN IMMEDIATE")
            holder_ready.set()
            release_holder.wait(timeout=2.0)
            con.execute("ROLLBACK")
        except Exception as exc:
            holder_errors.append(exc)
            holder_ready.set()
        finally:
            holder.close()

    thread = threading.Thread(target=hold_writer)
    thread.start()
    assert holder_ready.wait(timeout=2.0)
    assert not holder_errors

    index = SQLiteStoreQueryIndex(
        source_key="sqlite-test-store",
        path=path,
        config=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=0.01, max_write_retries=20),
    )
    obj = SQLiteLeaf("retry")
    graph = ConcreteDefinitionGraph.from_root(obj.definition)
    timer = threading.Timer(0.05, release_holder.set)
    timer.start()
    try:
        result = index.register_stored_roots(graph, [obj.definition])
    finally:
        release_holder.set()
        timer.cancel()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert not holder_errors
    assert result.changed
    assert index.current_generation() == result.generation


def test_batched_cdef_fetch_and_stored_filter(tmp_path):
    index = sqlite_index(tmp_path)
    roots = [SQLiteLeaf(f"leaf-{idx}").definition for idx in range(525)]
    graph = ConcreteDefinitionGraph.from_roots(roots)
    index.register_stored_roots(graph, roots)

    with index.read_view() as view:
        ids = view.all_stored_ids()
        cdefs = view.cdefs_by_id(ids)
        assert len(ids) == 525
        assert len(cdefs) == 525
        assert view.filter_stored_ids(ids | {999999}) == ids


def test_recursive_nested_filter_uses_sql_cte_for_large_candidate_set(tmp_path):
    index = sqlite_index(tmp_path)
    leaves = [SQLiteLeaf(f"nested-{idx}") for idx in range(260)]
    owners = [SQLiteParent(child=leaf, name=f"owner-{idx}") for idx, leaf in enumerate(leaves)]
    graph = ConcreteDefinitionGraph.from_roots(owner.definition for owner in owners)
    index.register_stored_roots(graph, [owner.definition for owner in owners])

    with index.read_view() as view:
        leaf_ids = {next(iter(view.exact_ids(leaf.definition))) for leaf in leaves}
        owner_ids = {next(iter(view.exact_ids(owner.definition))) for owner in owners}
        assert view.filter_nested_ids(leaf_ids | owner_ids) == leaf_ids
