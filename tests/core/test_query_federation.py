from pathlib import Path
import inspect

import pytest

from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2 import Definition, Object, Repo, SKIP_ARGS
from dryml.core2.query.model import OccurrenceTraversalSnapshot, QueryCardinalityError, QueryIndexGenerationChanged, QueryVerifyBudgetExceeded, QueryWouldScanError
import dryml.core2.query.federation as federation_module
from dryml.core2.query.federation import CACHE_SOURCE_KEY, RepoGenerationVector, StoreIndexBinding
from dryml.core2.query.query import DefinitionQuery
from dryml.core2.query.result import DefinitionResultSet, QueryBackedDefinitionResultSet
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig
from dryml.core2.query.sqlite.index import SQLiteQueryIndexReadView
from dryml.core2.query.sqlite.lowering import SQLiteOptimizerPolicy
from dryml.core2.store.dir import DirStore


class FederationLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class FederationParent(Object):
    def __init__(self, child=None, *, name="parent"):
        super().__init__()
        self.child = child
        self.name = name


class FederationPair(Object):
    def __init__(self, left=None, right=None, *, name="pair"):
        super().__init__()
        self.left = left
        self.right = right
        self.name = name


class RecordingIndex:
    def __init__(self, store):
        self.store = store
        self.registered = []
        self.closed = False

    def register_stored_roots(self, graph, roots):
        assert all(self.store.has(root) for root in roots)
        self.registered.append((graph, tuple(roots)))

    def status(self):
        from dryml.core2.query.model import QueryIndexStatus

        return QueryIndexStatus(
            backend="recording",
            store_key=self.store.catalog_key(),
            generation=len(self.registered),
            schema_version=None,
            semantic_versions={},
            state="ready",
        )

    def close(self):
        self.closed = True


class FailingIndex:
    def register_stored_roots(self, graph, roots):
        raise RuntimeError("index failed after object publish")


def test_repo_federation_bindings_follow_store_priority(tmp_path):
    store1 = DirStore(tmp_path / "store1", query_index="memory")
    store2 = DirStore(tmp_path / "store2", query_index="none")
    repo = Repo(stores=[store1, store2])

    bindings = repo._query_index.store_bindings

    assert tuple(type(binding) for binding in bindings) == (StoreIndexBinding, StoreIndexBinding)
    assert [binding.store for binding in bindings] == [store1, store2]
    assert [binding.priority for binding in bindings] == [0, 1]

    repo.set_default_store(store2)

    bindings = repo._query_index.store_bindings
    assert [binding.store for binding in bindings] == [store2, store1]
    assert [binding.priority for binding in bindings] == [0, 1]


def test_repo_federation_add_store_updates_bindings(tmp_path):
    repo = Repo()
    store1 = DirStore(tmp_path / "store1", query_index="memory")
    store2 = DirStore(tmp_path / "store2", query_index="memory")

    repo.add_store(store1)
    repo.add_store(store2, make_default=True)

    assert [binding.store for binding in repo._query_index.store_bindings] == [store2, store1]


def test_sources_for_domain(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)

    assert repo._query_index.sources_for_domain("cached") == (CACHE_SOURCE_KEY,)
    assert repo._query_index.sources_for_domain("stored") == repo._query_index.store_bindings
    assert repo._query_index.sources_for_domain("nested") == repo._query_index.store_bindings
    known_sources = repo._query_index.sources_for_domain("known")
    assert known_sources[:-1] == repo._query_index.store_bindings
    assert known_sources[-1] == CACHE_SOURCE_KEY


def test_generation_vector_includes_cache_generation(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store", query_index="memory"))
    obj = FederationLeaf("cached", repo=repo)
    repo.add_objects(obj)

    vector = repo._query_index.generation_vector()

    assert isinstance(vector, RepoGenerationVector)
    assert vector.generations[CACHE_SOURCE_KEY] == repo._query_catalog.generation


def test_index_status_reports_mixed_store_policies(tmp_path):
    memory_store = DirStore(tmp_path / "memory", query_index="memory")
    none_store = DirStore(tmp_path / "none", query_index="none")
    sqlite_store = DirStore(tmp_path / "sqlite", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[memory_store, none_store, sqlite_store])

    statuses = {status.store_key: status for status in repo.index_status()}

    assert statuses[memory_store.catalog_key()].backend == "memory"
    assert statuses[memory_store.catalog_key()].state == "ready"
    assert statuses[none_store.catalog_key()].backend == "none"
    assert statuses[none_store.catalog_key()].state == "disabled"
    assert statuses[sqlite_store.catalog_key()].backend == "sqlite"
    assert statuses[sqlite_store.catalog_key()].state == "missing"
    assert not Path(sqlite_store.query_index_path).exists()


def test_index_status_can_filter_one_store(tmp_path):
    store1 = DirStore(tmp_path / "store1", query_index="memory")
    store2 = DirStore(tmp_path / "store2", query_index="none")
    repo = Repo(stores=[store1, store2])

    statuses = repo.index_status(store=store2)

    assert len(statuses) == 1
    assert statuses[0].store_key == store2.catalog_key()


def test_save_registration_fans_out_to_custom_store_index_after_publish(tmp_path):
    opened = []

    def factory(store):
        index = RecordingIndex(store)
        opened.append(index)
        return index

    store = DirStore(tmp_path / "store", query_index=factory)
    repo = Repo(stores=store)
    obj = FederationLeaf("saved", repo=repo)

    repo.save_object(obj)

    assert len(opened) == 1
    assert len(opened[0].registered) == 1
    _, roots = opened[0].registered[0]
    assert obj.definition in roots


def test_save_registration_failure_marks_query_index_dirty(tmp_path):
    store = DirStore(tmp_path / "store", query_index=lambda store: FailingIndex())
    repo = Repo(stores=store)
    obj = FederationLeaf("dirty", repo=repo)

    import pytest

    with pytest.raises(RuntimeError, match="index failed"):
        repo.save_object(obj)

    assert store.has(obj.definition)
    assert store.query_index_is_dirty()


def test_save_registration_updates_sqlite_store_index(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = FederationLeaf("sqlite-save", repo=repo)

    repo.save_object(obj)

    statuses = repo.index_status(store=store)
    assert statuses[0].backend == "sqlite"
    assert statuses[0].state == "ready"
    assert statuses[0].generation == 1
    assert Path(store.query_index_path).exists()
    index = store.open_query_index()
    with index.read_view() as view:
        assert view.cdef_id(obj.definition) in view.all_stored_ids()


def test_repo_close_closes_opened_store_indexes(tmp_path):
    opened = []

    def factory(store):
        index = RecordingIndex(store)
        opened.append(index)
        return index

    store = DirStore(tmp_path / "store", query_index=factory)
    repo = Repo(stores=store)

    repo.index_status()
    repo.close(flush=False)

    assert opened and opened[0].closed


def test_sqlite_federated_stored_query_uses_sidecar_without_hydration(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    wanted = FederationParent(child=FederationLeaf(name="wanted", repo=repo), name="root", repo=repo)
    other = FederationParent(child=FederationLeaf(name="other", repo=repo), name="root", repo=repo)
    repo.save_object(wanted)
    repo.save_object(other)

    reopened_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo2 = Repo(stores=reopened_store)

    def fail_hydrate():
        raise AssertionError("federated SQLite query should not hydrate the Store")

    reopened_store.hydrate_index = fail_hydrate
    selector = Definition(FederationParent, SKIP_ARGS, child=Definition(FederationLeaf, SKIP_ARGS, name="wanted"))

    results = repo2.query(selector).stored().defs()

    assert list(results) == [wanted.definition]
    assert results.replicas(wanted.definition) == (reopened_store,)
    assert results.explanation.refresh_action == "federated"
    assert results.explanation.generation_vector == {reopened_store.catalog_key(): 2}
    assert len(results.explanation.source_plans) == 1
    assert results.explanation.source_plans[0].source_key == reopened_store.catalog_key()
    assert results.explanation.source_plans[0].backend == "sqlite"


def test_federated_lowered_query_pages_via_relation_api(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    for idx in range(3):
        repo.save_object(FederationLeaf(name=f"relation-api-{idx}", repo=repo))

    relation_pages = []
    original_relation = SQLiteQueryIndexReadView.iter_relation_cdef_batches
    original_candidate = SQLiteQueryIndexReadView.iter_candidate_cdef_batches

    def spy_relation_batches(self, relation, *, after=None, batch_size):
        relation_pages.append((relation.debug_label, batch_size))
        return original_relation(self, relation, after=after, batch_size=batch_size)

    def reject_direct_candidate_batches(self, plan, *, after=None, batch_size):
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_filename.endswith("query/federation.py"):
            raise AssertionError("federation must page through CandidateRelation")
        return original_candidate(self, plan, after=after, batch_size=batch_size)

    monkeypatch.setattr(SQLiteQueryIndexReadView, "iter_relation_cdef_batches", spy_relation_batches)
    monkeypatch.setattr(SQLiteQueryIndexReadView, "iter_candidate_cdef_batches", reject_direct_candidate_batches)

    results = repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().defs()

    assert len(results) == 3
    assert relation_pages


def test_relation_filter_domain_is_used_for_stored_domain_in_production(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf(name="stored-domain", repo=repo))

    domains = []
    original = SQLiteQueryIndexReadView.relation_filter_domain

    def spy_filter_domain(self, relation, domain):
        domains.append(domain.name)
        return original(self, relation, domain)

    monkeypatch.setattr(SQLiteQueryIndexReadView, "relation_filter_domain", spy_filter_domain)

    assert repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().count() == 1

    assert "stored" in domains


def test_query_backed_lowered_query_pages_via_relation_api(tmp_path, monkeypatch):
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_RESULT_THRESHOLD", 1)
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_PAGE_SIZE", 2)
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    for idx in range(4):
        repo.save_object(FederationLeaf(name=f"query-backed-relation-{idx}", repo=repo))

    relation_pages = []
    original_relation = SQLiteQueryIndexReadView.iter_relation_cdef_batches
    original_candidate = SQLiteQueryIndexReadView.iter_candidate_cdef_batches

    def spy_relation_batches(self, relation, *, after=None, batch_size):
        relation_pages.append((relation.debug_label, batch_size))
        return original_relation(self, relation, after=after, batch_size=batch_size)

    def reject_direct_candidate_batches(self, plan, *, after=None, batch_size):
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_filename.endswith("query/federation.py"):
            raise AssertionError("query-backed federation must page through CandidateRelation")
        return original_candidate(self, plan, after=after, batch_size=batch_size)

    monkeypatch.setattr(SQLiteQueryIndexReadView, "iter_relation_cdef_batches", spy_relation_batches)
    monkeypatch.setattr(SQLiteQueryIndexReadView, "iter_candidate_cdef_batches", reject_direct_candidate_batches)

    results = repo.query(None).stored().defs()

    assert isinstance(results, QueryBackedDefinitionResultSet)
    assert next(iter(results)) is not None
    assert relation_pages == [("candidate_relation:domain:stored", 2)]


def test_production_multibranch_selector_reports_semijoin_diagnostics(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    wanted = FederationPair(
        left=FederationLeaf(name="left", repo=repo),
        right=FederationLeaf(name="right", repo=repo),
        name="pair",
        repo=repo,
    )
    false_parent = FederationPair(
        left=FederationLeaf(name="left", repo=repo),
        right=FederationLeaf(name="wrong", repo=repo),
        name="pair",
        repo=repo,
    )
    repo.save_object(wanted)
    repo.save_object(false_parent)
    selector = Definition(
        FederationPair,
        SKIP_ARGS,
        left=Definition(FederationLeaf, SKIP_ARGS, name="left"),
        right=Definition(FederationLeaf, SKIP_ARGS, name="right"),
    )

    results = repo.query(selector).stored().defs()

    assert list(results) == [wanted.definition]
    assert results.explanation.lowering_diagnostics["semijoin_steps"]


def test_sqlite_lowered_require_indexed_rejects_unindexed_broad_query(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf("scan", repo=repo))

    with pytest.raises(QueryWouldScanError):
        repo.query(None).stored().require_indexed().count()


def test_sqlite_lowered_scan_warn_emits_warning(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf("scan", repo=repo))

    with pytest.warns(RuntimeWarning, match="scan fallback"):
        assert repo.query(None).stored().scan_policy("warn").count() == 1


def test_scan_policy_warns_for_graph_with_no_indexable_requirements(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf("scan-graph", repo=repo))

    with pytest.warns(RuntimeWarning, match="selector graph has no indexable requirements"):
        assert repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().scan_policy("warn").exists()


def test_scan_policy_forbid_rejects_empty_selector_graph(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf("scan-forbid", repo=repo))

    with pytest.raises(QueryWouldScanError, match="no indexable"):
        repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().require_indexed().exists()


def test_explain_reports_scan_reason(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf("scan-explain", repo=repo))

    explanation = repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().scan_policy("allow").explain()

    assert explanation.scan_required
    assert "no indexable" in explanation.scan_reason
    assert explanation.lowering_diagnostics["scan_policy"] == "allow"


def test_explain_reports_verify_budget(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf("budget-explain", repo=repo))

    explanation = repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().max_verify(3).explain()

    assert explanation.lowering_diagnostics["verify_budget"] == 3
    assert explanation.lowering_diagnostics["scan_policy"] == "allow"


def test_sqlite_lowered_max_verify_budget_is_enforced(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf("budget", repo=repo))

    with pytest.raises(QueryVerifyBudgetExceeded, match="verified 1 CDefs"):
        repo.query(None).stored().max_verify(0).count()


def test_max_verify_stops_mid_page(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    for idx in range(5):
        repo.save_object(FederationLeaf(name=f"budget-{idx}", repo=repo))

    with pytest.raises(QueryVerifyBudgetExceeded, match="verified 2 CDefs"):
        repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().max_verify(1).count()


def test_sqlite_explain_analyze_reports_lowering_counts(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf("explain", repo=repo))

    plan_only = repo.query(None).stored().explain()
    analyzed = repo.query(None).stored().explain(analyze=True)

    assert plan_only.verified_count == 0
    assert analyzed.verified_count == 1
    assert analyzed.python_verifications == 1
    assert analyzed.cdef_blobs_decoded >= 1
    assert analyzed.pages_fetched == 1
    assert analyzed.lowering_strategy == "sqlite-lowered"
    assert analyzed.terminal_stop_reason is None


def test_sqlite_plan_diagnostics_available(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf("plan", repo=repo))

    explanation = repo.query(None).stored().explain(sql=True)

    assert explanation.lowering_diagnostics["sqlite_plan"]
    assert explanation.lowering_diagnostics["strategy"] == "sqlite-lowered"
    assert explanation.lowering_diagnostics["logical_plan"] is not None
    assert explanation.lowering_diagnostics["physical_plan"] is not None


def test_sqlite_plan_diagnostics_are_opt_in(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf("plan-opt-in", repo=repo))

    explanation = repo.query(None).stored().explain()

    assert explanation.lowering_diagnostics["sqlite_plan"] == ()


def test_repo_explain_reports_local_posting_anchor_kind(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    wanted = FederationParent(child=FederationLeaf(name="posting-anchor", repo=repo), name="root", repo=repo)
    other = FederationParent(child=FederationLeaf(name="other-anchor", repo=repo), name="root", repo=repo)
    repo.save_object(wanted)
    repo.save_object(other)

    selector = Definition(
        FederationParent,
        SKIP_ARGS,
        child=Definition(FederationLeaf, SKIP_ARGS, name="posting-anchor"),
    )
    explanation = repo.query(selector).stored().explain(sql=True)

    assert explanation.lowering_diagnostics["anchor_reason"] == "local-posting"
    assert explanation.lowering_diagnostics["anchor_relation_kind"] == "posting"
    assert any("SEARCH p" in row for row in explanation.lowering_diagnostics["sqlite_plan"])


def test_production_large_relation_materializes_by_optimizer_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(
        SQLiteQueryIndexReadView,
        "optimizer_policy",
        SQLiteOptimizerPolicy(materialize_if_reused=False, materialize_if_estimate_gt=1),
    )
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    for idx in range(3):
        repo.save_object(FederationLeaf(name=f"materialize-prod-{idx}", repo=repo))

    result = repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().defs()
    diagnostics = result.explanation.lowering_diagnostics

    assert list(result)
    assert diagnostics["physical_plan"]["strategy"] == "temp-relation"
    assert diagnostics["physical_plan"]["fallback_reason"] == "estimated-large"
    assert diagnostics["materialized_relations"]
    assert diagnostics["temp_rows_inserted"] >= 3
    assert diagnostics["temp_rows_inserted"] % 3 == 0


def test_production_small_relation_stays_inline_by_optimizer_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(
        SQLiteQueryIndexReadView,
        "optimizer_policy",
        SQLiteOptimizerPolicy(materialize_if_reused=False, materialize_if_estimate_gt=100),
    )
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf(name="inline-prod", repo=repo))

    result = repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().defs()
    diagnostics = result.explanation.lowering_diagnostics

    assert list(result) == [FederationLeaf(name="inline-prod").definition]
    assert diagnostics["physical_plan"]["strategy"] == "inline-cte"
    assert diagnostics["materialized_relations"] == ()


def test_query_backed_large_relation_does_not_materialize_each_page(tmp_path, monkeypatch):
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_RESULT_THRESHOLD", 1)
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_PAGE_SIZE", 2)
    monkeypatch.setattr(
        SQLiteQueryIndexReadView,
        "optimizer_policy",
        SQLiteOptimizerPolicy(materialize_if_reused=False, materialize_if_estimate_gt=1),
    )
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    roots = []
    for idx in range(5):
        obj = FederationLeaf(name=f"query-backed-inline-{idx}", repo=repo)
        roots.append(obj.definition)
        repo.save_object(obj)

    materialize_calls = []
    original_materialize = SQLiteQueryIndexReadView.relation_materialize

    def spy_materialize(self, relation, *, reason=None):
        materialize_calls.append(reason)
        return original_materialize(self, relation, reason=reason)

    monkeypatch.setattr(SQLiteQueryIndexReadView, "relation_materialize", spy_materialize)

    results = repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().defs()

    assert isinstance(results, QueryBackedDefinitionResultSet)
    assert tuple(results) == tuple(sorted(roots, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
    assert materialize_calls == []


def test_query_backed_page_terminal_forces_inline_strategy_when_relation_is_not_persistent(tmp_path, monkeypatch):
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_RESULT_THRESHOLD", 1)
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_PAGE_SIZE", 2)
    monkeypatch.setattr(
        SQLiteQueryIndexReadView,
        "optimizer_policy",
        SQLiteOptimizerPolicy(materialize_if_reused=False, materialize_if_estimate_gt=1),
    )
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    for idx in range(4):
        repo.save_object(FederationLeaf(name=f"query-backed-strategy-{idx}", repo=repo))

    page_strategies = []
    original = SQLiteQueryIndexReadView.iter_relation_cdef_batches

    def spy_relation_batches(self, relation, *, after=None, batch_size):
        diagnostics = self.relation_diagnostics(relation)
        page_strategies.append((diagnostics.physical_plan.strategy, diagnostics.materialized_relations))
        return original(self, relation, after=after, batch_size=batch_size)

    monkeypatch.setattr(SQLiteQueryIndexReadView, "iter_relation_cdef_batches", spy_relation_batches)

    results = repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().defs()

    assert isinstance(results, QueryBackedDefinitionResultSet)
    assert next(iter(results)) is not None
    assert page_strategies == [("inline-cte", ())]


def test_eager_collect_large_relation_can_materialize_but_query_backed_page_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr(
        SQLiteQueryIndexReadView,
        "optimizer_policy",
        SQLiteOptimizerPolicy(materialize_if_reused=False, materialize_if_estimate_gt=1),
    )
    eager_store = DirStore(tmp_path / "eager", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    eager_repo = Repo(stores=eager_store)
    for idx in range(3):
        eager_repo.save_object(FederationLeaf(name=f"eager-materialize-{idx}", repo=eager_repo))

    eager_result = eager_repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().defs()
    eager_diagnostics = eager_result.explanation.lowering_diagnostics

    assert not isinstance(eager_result, QueryBackedDefinitionResultSet)
    assert eager_diagnostics["physical_plan"]["strategy"] == "temp-relation"
    assert eager_diagnostics["materialized_relations"]

    monkeypatch.setattr(federation_module, "_QUERY_BACKED_RESULT_THRESHOLD", 1)
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_PAGE_SIZE", 2)
    paged_store = DirStore(tmp_path / "paged", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    paged_repo = Repo(stores=paged_store)
    for idx in range(4):
        paged_repo.save_object(FederationLeaf(name=f"paged-inline-{idx}", repo=paged_repo))

    page_strategies = []
    original = SQLiteQueryIndexReadView.iter_relation_cdef_batches

    def spy_relation_batches(self, relation, *, after=None, batch_size):
        diagnostics = self.relation_diagnostics(relation)
        page_strategies.append((diagnostics.physical_plan.strategy, diagnostics.materialized_relations))
        return original(self, relation, after=after, batch_size=batch_size)

    monkeypatch.setattr(SQLiteQueryIndexReadView, "iter_relation_cdef_batches", spy_relation_batches)

    paged_result = paged_repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().defs()

    assert isinstance(paged_result, QueryBackedDefinitionResultSet)
    assert tuple(paged_result)
    assert page_strategies
    assert all(strategy == "inline-cte" and materialized == () for strategy, materialized in page_strategies)


def test_query_backed_resultset_fetches_first_page_only(tmp_path, monkeypatch):
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_RESULT_THRESHOLD", 2)
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_PAGE_SIZE", 2)
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    for idx in range(5):
        repo.save_object(FederationLeaf(name=f"paged-{idx}", repo=repo))

    fetched = []
    original = SQLiteQueryIndexReadView.cdefs_by_id

    def spy_cdefs_by_id(self, ids):
        result = original(self, ids)
        fetched.append(len(result))
        return result

    monkeypatch.setattr(SQLiteQueryIndexReadView, "cdefs_by_id", spy_cdefs_by_id)

    results = repo.query(None).stored().defs()
    assert isinstance(results, QueryBackedDefinitionResultSet)
    assert next(iter(results)) is not None
    assert fetched == [2]


def test_query_backed_resultset_first_and_second_iteration_same_order(tmp_path, monkeypatch):
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_RESULT_THRESHOLD", 1)
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_PAGE_SIZE", 2)
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    roots = []
    for idx in range(4):
        obj = FederationLeaf(name=f"stable-{idx}", repo=repo)
        roots.append(obj.definition)
        repo.save_object(obj)

    results = repo.query(None).stored().defs()

    assert isinstance(results, QueryBackedDefinitionResultSet)
    first = tuple(results)
    second = tuple(results)
    assert first == second
    assert first == tuple(sorted(roots, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))


def test_query_backed_resultset_holds_no_connection_or_cursor(tmp_path, monkeypatch):
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_RESULT_THRESHOLD", 1)
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    for idx in range(3):
        repo.save_object(FederationLeaf(name=f"handles-{idx}", repo=repo))

    results = repo.query(None).stored().defs()

    assert isinstance(results, QueryBackedDefinitionResultSet)
    assert not hasattr(results, "_con")
    assert not hasattr(results, "_cursor")


def test_query_backed_resultset_generation_change_fails_clearly(tmp_path, monkeypatch):
    from dryml.core2.query.model import QueryIndexGenerationChanged

    monkeypatch.setattr(federation_module, "_QUERY_BACKED_RESULT_THRESHOLD", 1)
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    for idx in range(3):
        repo.save_object(FederationLeaf(name=f"snapshot-{idx}", repo=repo))

    results = repo.query(None).stored().defs()
    assert isinstance(results, QueryBackedDefinitionResultSet)
    repo.save_object(FederationLeaf(name="later", repo=repo))

    with pytest.raises(QueryIndexGenerationChanged):
        list(results)


def test_sqlite_federated_multistore_dedup_and_replica_priority(tmp_path):
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    obj = FederationLeaf(name="shared", repo=repo)
    repo.save_object(obj, store=store1)
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[
        DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
    ])
    selector = Definition(FederationLeaf, SKIP_ARGS, name="shared")

    results = repo2.query(selector).stored().defs()

    assert list(results) == [obj.definition]
    assert tuple(store.base_dir for store in results.replicas(obj.definition)) == (store2.base_dir, store1.base_dir)
    assert set(results.explanation.generation_vector) == {repo2.stores[0].catalog_key(), repo2.stores[1].catalog_key()}
    assert [plan.backend for plan in results.explanation.source_plans] == ["sqlite", "sqlite"]


def test_query_backed_multistore_order_stable(tmp_path, monkeypatch):
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_RESULT_THRESHOLD", 1)
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    left = FederationLeaf(name="left", repo=repo)
    right = FederationLeaf(name="right", repo=repo)
    repo.save_object(left, store=store1)
    repo.save_object(right, store=store2)

    repo2 = Repo(stores=[
        DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
    ])
    results = repo2.query(None).stored().defs()

    assert isinstance(results, QueryBackedDefinitionResultSet)
    assert tuple(results) == (left.definition, right.definition)
    assert tuple(results) == (left.definition, right.definition)


def test_query_backed_multistore_source_order_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_RESULT_THRESHOLD", 1)
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    first = FederationLeaf(name="source-order-first", repo=repo)
    second = FederationLeaf(name="source-order-second", repo=repo)
    source1_obj, source2_obj = sorted(
        (first, second),
        key=lambda obj: (obj.definition.stable_hash(), repr(obj.definition)),
        reverse=True,
    )
    repo.save_object(source1_obj, store=store1)
    repo.save_object(source2_obj, store=store2)

    repo2 = Repo(stores=[
        DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
    ])
    results = repo2.query(None).stored().defs()

    assert tuple(results) == (source1_obj.definition, source2_obj.definition)
    assert tuple(results) != tuple(sorted(
        (source1_obj.definition, source2_obj.definition),
        key=lambda cdef: (cdef.stable_hash(), repr(cdef)),
    ))


def test_query_backed_paging_dedupes_same_cdef_across_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(federation_module, "_QUERY_BACKED_RESULT_THRESHOLD", 1)
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    obj = FederationLeaf(name="query-backed-duplicate", repo=repo)
    repo.save_object(obj, store=store1)
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[
        DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
    ])
    results = repo2.query(None).stored().defs()

    assert isinstance(results, QueryBackedDefinitionResultSet)
    assert tuple(results) == (obj.definition,)
    replica_dirs = {store.base_dir for store in results.replicas(obj.definition)}
    assert replica_dirs == {repo2.stores[0].base_dir, repo2.stores[1].base_dir}


def test_sqlite_objects_terminal_uses_query_replica_snapshot(tmp_path, monkeypatch):
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    obj = FederationLeaf(name="materialize-priority", repo=repo)
    repo.save_object(obj, store=store1)
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[
        DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
    ])
    results = repo2.query(Definition(FederationLeaf, SKIP_ARGS, name="materialize-priority")).stored().defs()
    repo2.set_default_store(repo2.stores[1])
    selected = []

    def spy_load(cdef, **kwargs):
        selected.append(repo2.obj_default_store[cdef])
        return obj

    monkeypatch.setattr(repo2, "load_object", spy_load)

    objects = results.objects()

    assert len(objects) == 1
    assert selected == [results.replicas(obj.definition)[0]]


def test_sqlite_federated_known_includes_cached_only_definition(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = FederationLeaf(name="cached-only", repo=repo)
    repo.add_objects(obj)
    selector = Definition(FederationLeaf, SKIP_ARGS, name="cached-only")

    cached = repo.query(selector).cached().defs()
    known = repo.query(selector).known().defs()
    stored = repo.query(selector).stored().defs()

    assert list(cached) == [obj.definition]
    assert list(known) == [obj.definition]
    assert list(stored) == []
    assert known.replicas(obj.definition) == ()
    assert known.explanation.generation_vector[CACHE_SOURCE_KEY] == repo._query_catalog.generation
    assert CACHE_SOURCE_KEY in [plan.source_key for plan in known.explanation.source_plans]


def test_sqlite_federated_known_deduplicates_cached_and_stored_cdef(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = FederationLeaf(name="cached-and-stored", repo=repo)
    repo.save_object(obj)
    selector = Definition(FederationLeaf, SKIP_ARGS, name="cached-and-stored")

    results = repo.query(selector).known().defs()

    assert list(results) == [obj.definition]
    assert results.replicas(obj.definition) == (store,)
    assert set(results.explanation.generation_vector) == {store.catalog_key(), CACHE_SOURCE_KEY}
    assert [plan.source_key for plan in results.explanation.source_plans] == [store.catalog_key(), CACHE_SOURCE_KEY]


def test_sqlite_federated_replica_removal_preserves_remaining_replicas(tmp_path):
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    obj = FederationLeaf(name="replica-removal", repo=repo)
    repo.save_object(obj, store=store1)
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[
        DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
    ])
    selector = Definition(FederationLeaf, SKIP_ARGS, name="replica-removal")

    repo2.stores[0].open_query_index().remove_stored_roots([obj.definition])
    after_one_removed = repo2.query(selector).stored().defs()

    assert list(after_one_removed) == [obj.definition]
    assert tuple(store.base_dir for store in after_one_removed.replicas(obj.definition)) == (store1.base_dir,)

    repo2.stores[1].open_query_index().remove_stored_roots([obj.definition])
    after_all_removed = repo2.query(selector).stored().defs()

    assert list(after_all_removed) == []


def test_sqlite_federated_nested_definitions_owners_and_occurrences(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    leaf = FederationLeaf(name="nested", repo=repo)
    owner = FederationParent(child=leaf, name="owner", repo=repo)
    repo.save_object(owner)

    repo2_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo2 = Repo(stores=repo2_store)
    selector = Definition(FederationLeaf, SKIP_ARGS, name="nested")

    definitions = repo2.query(selector).nested().definitions().defs()
    owners = repo2.query(selector).nested().owners().defs()
    occurrence_results = repo2.query(selector).nested().max_occurrences(10).execute()
    occurrences = tuple(occurrence_results)

    assert list(definitions) == [leaf.definition]
    assert list(owners) == [owner.definition]
    assert owners.replicas(owner.definition) == (repo2_store,)
    assert len(occurrences) == 1
    assert occurrences[0].owner == owner.definition
    assert occurrences[0].definition == leaf.definition
    assert str(occurrences[0].path) == "$.child"
    assert definitions.explanation.generation_vector == {repo2_store.catalog_key(): 1}
    assert owners.explanation.source_plans[0].result_count == 1
    assert owners.explanation.lowering_diagnostics["owners_found"] == 1
    assert occurrence_results.explanation.lowering_diagnostics["occurrence_nested_targets"] == 1
    assert occurrence_results.explanation.lowering_diagnostics["occurrence_nodes_captured"] >= 2
    assert occurrence_results.explanation.lowering_diagnostics["occurrence_incoming_edges_captured"] >= 1
    assert occurrence_results.explanation.lowering_diagnostics["occurrence_owners_found"] == 1
    assert occurrence_results.explanation.lowering_diagnostics["occurrence_path_limit"] == 10


def test_sqlite_multistore_occurrences_deduplicate_and_keep_replica_order(tmp_path):
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    leaf = FederationLeaf(name="duplicate-occurrence", repo=repo)
    owner = FederationParent(child=leaf, name="owner", repo=repo)
    repo.save_object(owner, store=store1)
    repo.save_object(owner, store=store2)

    repo2 = Repo(stores=[
        DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
    ])
    selector = Definition(FederationLeaf, SKIP_ARGS, name="duplicate-occurrence")

    occurrences = repo2.query(selector).nested().execute()
    occurrence_items = tuple(occurrences)
    owners = occurrences.owners()

    assert len(occurrence_items) == 1
    assert occurrence_items[0].owner == owner.definition
    assert occurrence_items[0].definition == leaf.definition
    assert str(occurrence_items[0].path) == "$.child"
    assert tuple(store.base_dir for store in owners.replicas(owner.definition)) == (store2.base_dir, store1.base_dir)


def test_sqlite_nested_generation_retry_is_source_local(tmp_path, monkeypatch):
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    leaf1 = FederationLeaf(name="source-local-1", repo=repo)
    owner1 = FederationParent(child=leaf1, name="owner-1", repo=repo)
    leaf2 = FederationLeaf(name="source-local-2", repo=repo)
    owner2 = FederationParent(child=leaf2, name="owner-2", repo=repo)
    repo.save_object(owner1, store=store1)
    repo.save_object(owner2, store=store2)

    repo2 = Repo(stores=[
        DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
    ])
    original = repo2._query_index._capture_lowered_nested_matches
    counts = {store.catalog_key(): 0 for store in repo2.stores}
    churn_source = repo2.stores[0].catalog_key()

    def capture_then_churn(query, binding, stats, **kwargs):
        captured = original(query, binding, stats, **kwargs)
        counts[binding.source_key] += 1
        if binding.source_key == churn_source and counts[binding.source_key] == 1:
            churn = FederationLeaf(name="source-local-churn", repo=repo2).definition
            index = repo2._query_index._source_index_for_binding(binding)
            index.register_stored_roots(ConcreteDefinitionGraph.from_root(churn), [churn])
        return captured

    monkeypatch.setattr(repo2._query_index, "_capture_lowered_nested_matches", capture_then_churn)

    owners = repo2.query(Definition(FederationLeaf, SKIP_ARGS)).nested().owners().defs()

    assert set(owners) == {owner1.definition, owner2.definition}
    assert counts[repo2.stores[0].catalog_key()] == 2
    assert counts[repo2.stores[1].catalog_key()] == 1


def test_sqlite_nested_owner_projection_retries_after_generation_change(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    leaf = FederationLeaf(name="retry-owner", repo=repo)
    owner = FederationParent(child=leaf, name="owner", repo=repo)
    repo.save_object(owner)

    repo2_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo2 = Repo(stores=repo2_store)
    original = repo2._query_index._capture_lowered_nested_matches
    calls = 0

    def capture_then_churn(query, binding, stats, **kwargs):
        nonlocal calls
        captured = original(query, binding, stats, **kwargs)
        calls += 1
        if calls == 1:
            churn = FederationLeaf(name="owner-churn", repo=repo2).definition
            index = repo2._query_index._source_index_for_binding(binding)
            index.register_stored_roots(ConcreteDefinitionGraph.from_root(churn), [churn])
        return captured

    monkeypatch.setattr(repo2._query_index, "_capture_lowered_nested_matches", capture_then_churn)

    result = repo2.query(Definition(FederationLeaf, SKIP_ARGS, name="retry-owner")).nested().owners().one()

    assert result == owner.definition
    assert calls == 2


def test_sqlite_nested_occurrence_projection_retries_after_generation_change(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    leaf = FederationLeaf(name="retry-occurrence", repo=repo)
    owner = FederationParent(child=leaf, name="owner", repo=repo)
    repo.save_object(owner)

    repo2_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo2 = Repo(stores=repo2_store)
    original = repo2._query_index._capture_lowered_nested_matches
    calls = 0

    def capture_then_churn(query, binding, stats, **kwargs):
        nonlocal calls
        captured = original(query, binding, stats, **kwargs)
        calls += 1
        if calls == 1:
            churn = FederationLeaf(name="occurrence-churn", repo=repo2).definition
            index = repo2._query_index._source_index_for_binding(binding)
            index.register_stored_roots(ConcreteDefinitionGraph.from_root(churn), [churn])
        return captured

    monkeypatch.setattr(repo2._query_index, "_capture_lowered_nested_matches", capture_then_churn)

    occurrence = repo2.query(Definition(FederationLeaf, SKIP_ARGS, name="retry-occurrence")).nested().one()

    assert occurrence.owner == owner.definition
    assert occurrence.definition == leaf.definition
    assert calls == 2


def test_mixed_memory_and_sqlite_nested_query_merges_sources(tmp_path):
    sqlite_store = DirStore(tmp_path / "sqlite", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    memory_store = DirStore(tmp_path / "memory", query_index="memory")
    repo = Repo(stores=[sqlite_store, memory_store])
    sqlite_leaf = FederationLeaf(name="sqlite", repo=repo)
    sqlite_owner = FederationParent(child=sqlite_leaf, name="sqlite-owner", repo=repo)
    memory_leaf = FederationLeaf(name="memory", repo=repo)
    memory_owner = FederationParent(child=memory_leaf, name="memory-owner", repo=repo)
    repo.save_object(sqlite_owner, store=sqlite_store)
    repo.save_object(memory_owner, store=memory_store)

    repo2 = Repo(stores=[
        DirStore(sqlite_store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(memory_store.base_dir, query_index="memory"),
    ])
    selector = Definition(FederationLeaf, SKIP_ARGS)

    definitions = repo2.query(selector).nested().definitions().defs()
    owners = repo2.query(selector).nested().owners().defs()

    assert set(definitions) == {sqlite_leaf.definition, memory_leaf.definition}
    assert set(owners) == {sqlite_owner.definition, memory_owner.definition}
    owner_dirs = {
        cdef: tuple(store.base_dir for store in owners.replicas(cdef))
        for cdef in owners
    }
    assert owner_dirs[sqlite_owner.definition] == (sqlite_store.base_dir,)
    assert owner_dirs[memory_owner.definition] == (memory_store.base_dir,)


def test_exists_stops_after_first_verified_cdef(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    for idx in range(4):
        repo.save_object(FederationLeaf(name=f"exists-{idx}", repo=repo))

    verified = 0
    original = DefinitionQuery._verify_cdefs

    def spy_verify(self, cdefs, *, stats):
        nonlocal verified
        verified += len(cdefs)
        return original(self, cdefs, stats=stats)

    monkeypatch.setattr(DefinitionQuery, "_verify_cdefs", spy_verify)

    assert repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().exists()
    assert verified == 1


def test_sqlite_federated_terminals_bound_verification_and_cdef_fetches(tmp_path, monkeypatch):
    store_dir = tmp_path / "store"
    config = SQLiteQueryIndexConfig(journal_mode="delete")
    publishing_repo = Repo(stores=DirStore(store_dir, query_index=config))
    definitions = []
    object_count = 270  # Cross the 256-row terminal page boundary.
    try:
        for idx in range(object_count):
            leaf = FederationLeaf(name=f"leaf-{idx}", repo=publishing_repo)
            definitions.append(leaf.definition)
            publishing_repo.save_object(leaf)
    finally:
        publishing_repo.close(flush=False, save=False)

    selector = Definition(FederationLeaf, SKIP_ARGS)
    original_verify = DefinitionQuery._verify_cdefs
    for domain in ("stored", "known"):
        with monkeypatch.context() as subpatch:
            verified = 0

            def spy_verify(self, cdefs, *, stats):
                nonlocal verified
                verified += len(cdefs)
                return original_verify(self, cdefs, stats=stats)

            subpatch.setattr(DefinitionQuery, "_verify_cdefs", spy_verify)
            repo = Repo(stores=DirStore(store_dir, query_index=config))
            try:
                assert getattr(repo.query(selector), domain)().exists()
                assert verified == 1
            finally:
                repo.close(flush=False, save=False)

    original_fetch = SQLiteQueryIndexReadView.cdefs_by_id
    with monkeypatch.context() as subpatch:
        fetched = []

        def spy_cdefs_by_id(self, ids):
            result = original_fetch(self, ids)
            fetched.append(len(result))
            return result

        subpatch.setattr(SQLiteQueryIndexReadView, "cdefs_by_id", spy_cdefs_by_id)
        repo = Repo(stores=DirStore(store_dir, query_index=config))
        try:
            assert repo.query(selector).stored().exists()
            assert fetched == [1]
        finally:
            repo.close(flush=False, save=False)

    with monkeypatch.context() as subpatch:
        fetched = []

        def spy_cdefs_by_id(self, ids):
            result = original_fetch(self, ids)
            fetched.append(len(result))
            return result

        subpatch.setattr(SQLiteQueryIndexReadView, "cdefs_by_id", spy_cdefs_by_id)
        repo = Repo(stores=DirStore(store_dir, query_index=config))
        try:
            with pytest.raises(QueryCardinalityError):
                repo.query(selector).stored().one()
            assert fetched == [2]
        finally:
            repo.close(flush=False, save=False)

    last = sorted(definitions, key=lambda cdef: (cdef.stable_hash(), repr(cdef)))[-1]
    target_name = last.kwargs["name"]
    from dryml.core2 import Satisfies

    last_selector = Definition(
        FederationLeaf,
        SKIP_ARGS,
        name=Satisfies(lambda value: value == target_name, name="target-name"),
    )
    with monkeypatch.context() as subpatch:
        fetched = []

        def spy_cdefs_by_id(self, ids):
            result = original_fetch(self, ids)
            fetched.append(len(result))
            return result

        subpatch.setattr(SQLiteQueryIndexReadView, "cdefs_by_id", spy_cdefs_by_id)
        repo = Repo(stores=DirStore(store_dir, query_index=config))
        try:
            assert repo.query(last_selector).stored().exists()
            assert fetched == [1] * object_count
            assert sum(fetched) == object_count
        finally:
            repo.close(flush=False, save=False)


def test_federated_query_one_and_one_or_none_stop_after_two(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf(name="left", repo=repo))
    repo.save_object(FederationLeaf(name="right", repo=repo))

    verified = 0
    original = DefinitionQuery._verify_cdefs

    def spy_verify(self, cdefs, *, stats):
        nonlocal verified
        verified += len(cdefs)
        return original(self, cdefs, stats=stats)

    monkeypatch.setattr(DefinitionQuery, "_verify_cdefs", spy_verify)

    with pytest.raises(QueryCardinalityError):
        repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().one()
    assert verified == 2

    assert repo.query(Definition(FederationLeaf, SKIP_ARGS, name="missing")).stored().one_or_none() is None
    assert repo.query(Definition(FederationLeaf, SKIP_ARGS, name="left")).stored().one().kwargs["name"] == "left"


def test_sqlite_federated_explain_does_not_verify_cdefs(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf(name="explain", repo=repo))

    def fail_verify(*args, **kwargs):
        raise AssertionError("explain() should not run final CDef verification")

    monkeypatch.setattr(DefinitionQuery, "_verify_cdefs", fail_verify)

    explanation = repo.query(None).stored().explain()

    assert explanation.refresh_action == "federated-plan"
    assert explanation.candidate_count == 1
    assert explanation.result_count is None


def test_sqlite_count_terminal_does_not_construct_resultset(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf(name="count-left", repo=repo))
    repo.save_object(FederationLeaf(name="count-right", repo=repo))

    def fail_resultset(*args, **kwargs):
        raise AssertionError("count() should not construct a DefinitionResultSet")

    monkeypatch.setattr(DefinitionResultSet, "__init__", fail_resultset)

    count = repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().count()

    assert count == 2


def test_exact_safe_count_uses_exact_index_without_final_verification(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = FederationLeaf(name="exact-safe-count", repo=repo)
    repo.save_object(obj)

    def fail_verify(*args, **kwargs):
        raise AssertionError("exact-safe count should not run final query verification")

    def fail_cdefs_by_id(*args, **kwargs):
        raise AssertionError("exact-safe count should not fetch candidate CDef pages")

    monkeypatch.setattr(DefinitionQuery, "_verify_cdefs", fail_verify)
    monkeypatch.setattr(SQLiteQueryIndexReadView, "cdefs_by_id", fail_cdefs_by_id)

    query = repo.query(obj.definition).stored()
    count, stats = repo._query_index.count_definition_domain(query)

    assert count == 1
    assert stats.verified_count == 0
    assert stats.cdef_blobs_decoded == 0
    assert stats.lowering_strategy == "exact-safe-count"
    assert stats.lowering_diagnostics["exact_safe"] is True


def test_exact_stored_exists_uses_exact_safe_backend_path(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = FederationLeaf(name="exact-safe-exists", repo=repo)
    repo.save_object(obj)

    def fail_verify(*args, **kwargs):
        raise AssertionError("exact-safe exists should not run final query verification")

    def fail_cdefs_by_id(*args, **kwargs):
        raise AssertionError("exact-safe exists should not fetch candidate CDef pages")

    monkeypatch.setattr(DefinitionQuery, "_verify_cdefs", fail_verify)
    monkeypatch.setattr(SQLiteQueryIndexReadView, "cdefs_by_id", fail_cdefs_by_id)

    assert repo.query(obj.definition).stored().exists()


def test_exact_stored_defs_uses_exact_safe_definition_path(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = FederationLeaf(name="exact-safe-defs", repo=repo)
    repo.save_object(obj)

    def fail_verify(*args, **kwargs):
        raise AssertionError("exact-safe defs should not run final query verification")

    def fail_cdefs_by_id(*args, **kwargs):
        raise AssertionError("exact-safe defs should not fetch candidate CDef pages")

    monkeypatch.setattr(DefinitionQuery, "_verify_cdefs", fail_verify)
    monkeypatch.setattr(SQLiteQueryIndexReadView, "cdefs_by_id", fail_cdefs_by_id)

    result = repo.query(obj.definition).stored().defs()

    assert list(result) == [obj.definition]
    assert result.explanation.lowering_strategy == "exact-safe-definition"
    assert result.explanation.lowering_diagnostics["exact_safe"] is True


def test_exact_stored_one_uses_exact_safe_backend_path(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = FederationLeaf(name="exact-safe-one", repo=repo)
    repo.save_object(obj)

    def fail_verify(*args, **kwargs):
        raise AssertionError("exact-safe one should not run final query verification")

    def fail_cdefs_by_id(*args, **kwargs):
        raise AssertionError("exact-safe one should not fetch candidate CDef pages")

    monkeypatch.setattr(DefinitionQuery, "_verify_cdefs", fail_verify)
    monkeypatch.setattr(SQLiteQueryIndexReadView, "cdefs_by_id", fail_cdefs_by_id)

    result = repo.query(obj.definition).stored().one()

    assert result == obj.definition


def test_exact_safe_terminals_do_not_import_ml_frameworks(tmp_path, monkeypatch):
    import builtins

    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = FederationLeaf(name="import-safe", repo=repo)
    repo.save_object(obj)
    original_import = builtins.__import__
    blocked = []

    def spy_import(name, *args, **kwargs):
        if name == "tensorflow" or name.startswith("tensorflow.") or name == "torch" or name.startswith("torch."):
            blocked.append(name)
            raise AssertionError(f"exact-safe terminal imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", spy_import)

    query = repo.query(obj.definition).stored()

    assert query.count() == 1
    assert query.exists()
    assert blocked == []


def test_lowered_count_does_not_construct_result_map(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf(name="count-no-map-left", repo=repo))
    repo.save_object(FederationLeaf(name="count-no-map-right", repo=repo))

    def fail_canonical_map(*args, **kwargs):
        raise AssertionError("count() should not use full CDef result maps")

    monkeypatch.setattr(federation_module, "_canonical_cdef_key", fail_canonical_map)

    assert repo.query(Definition(FederationLeaf, SKIP_ARGS)).stored().count() == 2


def test_lowered_count_dedupes_same_cdef_across_two_stores(tmp_path):
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    obj = FederationLeaf(name="count-dedupe", repo=repo)
    repo.save_object(obj, store=store1)
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[
        DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
    ])

    assert repo2.query(Definition(FederationLeaf, SKIP_ARGS, name="count-dedupe")).stored().count() == 1


def test_lowered_count_retries_generation_change_during_witness_load(tmp_path, monkeypatch):
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    obj = FederationLeaf(name="count-retry", repo=repo)
    repo.save_object(obj, store=store1)
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[
        DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
    ])
    original_loader = repo2._query_index._count_witness_loader
    failures = 0

    def flaky_loader(index, source_key):
        load = original_loader(index, source_key)

        def load_once_generation_changed(generation, definition_id):
            nonlocal failures
            if failures == 0:
                failures += 1
                raise QueryIndexGenerationChanged("forced count witness generation change")
            return load(generation, definition_id)

        return load_once_generation_changed

    monkeypatch.setattr(repo2._query_index, "_count_witness_loader", flaky_loader)

    assert repo2.query(Definition(FederationLeaf, SKIP_ARGS, name="count-retry")).stored().count() == 1
    assert failures == 1


def test_lowered_count_handles_hash_collision_bucket(monkeypatch):
    first = FederationLeaf(name="collision-left").definition
    second = FederationLeaf(name="collision-right").definition
    counter = federation_module._CDefDedupeCounter()
    loaded = []

    def load(_generation, definition_id):
        loaded.append(definition_id)
        return {1: first, 2: second}[definition_id]

    counter.register_source("test-source", load)

    monkeypatch.setattr(type(first), "stable_hash", lambda self: "same-stable-hash")

    assert counter.accept(first, source_key="test-source", generation=1, definition_id=1)
    assert counter._buckets == {}
    assert not counter.accept(first, source_key="test-source", generation=1, definition_id=1)
    assert counter.accept(second, source_key="test-source", generation=1, definition_id=2)
    assert counter.count == 2
    assert loaded == [1]
    assert counter.witness_reload_count == 1
    assert counter.collision_bucket_count == 1


def test_count_state_keeps_only_witness_refs_until_hash_repeats():
    first = FederationLeaf(name="unique-left").definition
    second = FederationLeaf(name="unique-right").definition
    counter = federation_module._CDefDedupeCounter()

    def fail_load(*args, **kwargs):
        raise AssertionError("unique stable hashes should not reload or retain CDefs")

    counter.register_source("test-source", fail_load)

    assert counter.accept(first, source_key="test-source", generation=1, definition_id=1)
    assert counter.accept(second, source_key="test-source", generation=1, definition_id=2)

    assert counter.count == 2
    assert set(counter._witnesses) == {first.stable_hash(), second.stable_hash()}
    assert counter._buckets == {}
    assert counter.witness_reload_count == 0
    assert counter.collision_bucket_count == 0


def test_sqlite_terminal_verification_runs_after_read_view_closes(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(FederationLeaf(name="verify-left", repo=repo))
    repo.save_object(FederationLeaf(name="verify-right", repo=repo))

    active_views = set()
    original_init = SQLiteQueryIndexReadView.__init__
    original_close = SQLiteQueryIndexReadView.close
    original_verify = DefinitionQuery._verify_cdefs

    def track_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        active_views.add(id(self))

    def track_close(self):
        active_views.discard(id(self))
        return original_close(self)

    def assert_no_active_sqlite_read_view(self, cdefs, *, stats):
        assert not active_views
        return original_verify(self, cdefs, stats=stats)

    monkeypatch.setattr(SQLiteQueryIndexReadView, "__init__", track_init)
    monkeypatch.setattr(SQLiteQueryIndexReadView, "close", track_close)
    monkeypatch.setattr(DefinitionQuery, "_verify_cdefs", assert_no_active_sqlite_read_view)

    selector = Definition(FederationLeaf, SKIP_ARGS)
    assert repo.query(selector).stored().exists()
    assert repo.query(Definition(FederationLeaf, SKIP_ARGS, name="verify-left")).stored().one().kwargs["name"] == "verify-left"
    assert repo.query(selector).stored().count() == 2
    assert not active_views


def test_sqlite_selective_query_does_not_construct_full_definition_universe(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    for idx in range(8):
        repo.save_object(FederationLeaf(name=f"leaf-{idx}", repo=repo))

    def fail_all_ids(self):
        raise AssertionError("selective SQLite query should use indexed feature postings")

    monkeypatch.setattr(SQLiteQueryIndexReadView, "all_definition_ids", fail_all_ids)

    selector = Definition(FederationLeaf, SKIP_ARGS, name="leaf-3")
    result = repo.query(selector).stored().defs()

    assert list(result) == [FederationLeaf(name="leaf-3").definition]
    assert result.explanation.universe_size is None


def test_lowered_nested_exact_query_does_not_scan_all_roots(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    owners = []
    leaves = []
    for idx in range(12):
        leaf = FederationLeaf(name=f"rare-anchor-{idx}", repo=repo)
        owner = FederationParent(child=leaf, name="owner", repo=repo)
        repo.save_object(owner)
        leaves.append(leaf)
        owners.append(owner)

    def fail_all_ids(self):
        raise AssertionError("rare nested anchor query should not enumerate all roots")

    fetched = []
    original_cdefs_by_id = SQLiteQueryIndexReadView.cdefs_by_id

    def spy_cdefs_by_id(self, ids):
        result = original_cdefs_by_id(self, ids)
        fetched.append(len(result))
        return result

    monkeypatch.setattr(SQLiteQueryIndexReadView, "all_definition_ids", fail_all_ids)
    monkeypatch.setattr(SQLiteQueryIndexReadView, "cdefs_by_id", spy_cdefs_by_id)

    selector = Definition(FederationParent, SKIP_ARGS, child=leaves[-1].definition)
    result = repo.query(selector).stored().defs()

    assert list(result) == [owners[-1].definition]
    assert fetched == [1]


def test_sqlite_nested_definitions_do_not_expand_occurrence_paths(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    leaf = FederationLeaf(name="nested-definition", repo=repo)
    owner = FederationParent(child=leaf, name="owner", repo=repo)
    repo.save_object(owner)

    def fail_occurrence_capture(*args, **kwargs):
        raise AssertionError("nested definition terminal should not capture occurrence paths")

    monkeypatch.setattr(SQLiteQueryIndexReadView, "occurrence_snapshot_for_nested_ids", fail_occurrence_capture)

    results = repo.query(Definition(FederationLeaf, SKIP_ARGS, name="nested-definition")).nested().definitions().defs()

    assert list(results) == [leaf.definition]


def test_sqlite_owner_query_uses_owner_projection_not_occurrence_capture(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    leaf = FederationLeaf(name="owner-projection", repo=repo)
    owner = FederationParent(child=leaf, name="owner", repo=repo)
    repo.save_object(owner)
    calls = 0
    original_relation_project_owners = SQLiteQueryIndexReadView.relation_project_owners

    def fail_occurrence_capture(*args, **kwargs):
        raise AssertionError("owner terminal should not build occurrence paths")

    def fail_project_owners(*args, **kwargs):
        raise AssertionError("SQLite owner terminal should use relation_project_owners")

    def spy_relation_project_owners(self, relation):
        nonlocal calls
        calls += 1
        return original_relation_project_owners(self, relation)

    monkeypatch.setattr(SQLiteQueryIndexReadView, "occurrence_snapshot_for_nested_ids", fail_occurrence_capture)
    monkeypatch.setattr(SQLiteQueryIndexReadView, "project_owners", fail_project_owners)
    monkeypatch.setattr(SQLiteQueryIndexReadView, "relation_project_owners", spy_relation_project_owners)

    owners = repo.query(Definition(FederationLeaf, SKIP_ARGS, name="owner-projection")).nested().owners().defs()

    assert list(owners) == [owner.definition]
    assert calls == 1
    assert owners.explanation.lowering_diagnostics["owner_projection_relation_ops"] == 1


def test_occurrence_limit_does_not_fetch_unrelated_roots(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    wanted_leaf = FederationLeaf(name="wanted-occ", repo=repo)
    wanted_owner = FederationParent(child=wanted_leaf, name="wanted-owner", repo=repo)
    other_leaf = FederationLeaf(name="other-occ", repo=repo)
    other_owner = FederationParent(child=other_leaf, name="other-owner", repo=repo)
    repo.save_object(wanted_owner)
    repo.save_object(other_owner)
    captured_sizes = []
    original_snapshot = SQLiteQueryIndexReadView.occurrence_snapshot_for_nested_ids

    def spy_snapshot(self, target_ids):
        captured_sizes.append(len(target_ids))
        return original_snapshot(self, target_ids)

    monkeypatch.setattr(SQLiteQueryIndexReadView, "occurrence_snapshot_for_nested_ids", spy_snapshot)

    occurrence = repo.query(Definition(FederationLeaf, SKIP_ARGS, name="wanted-occ")).nested().max_occurrences(1).one()

    assert occurrence.owner == wanted_owner.definition
    assert captured_sizes == [1]


def test_sqlite_occurrence_query_defers_path_generation_until_iteration(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    leaf = FederationLeaf(name="lazy-occurrence", repo=repo)
    owner = FederationParent(child=leaf, name="owner", repo=repo)
    repo.save_object(owner)
    yielded = 0
    original_iter = OccurrenceTraversalSnapshot.iter_occurrences

    def spy_iter(self, *, max_occurrences=None):
        nonlocal yielded
        for occurrence in original_iter(self, max_occurrences=max_occurrences):
            yielded += 1
            yield occurrence

    monkeypatch.setattr(OccurrenceTraversalSnapshot, "iter_occurrences", spy_iter)

    occurrences = repo.query(Definition(FederationLeaf, SKIP_ARGS, name="lazy-occurrence")).nested().execute()

    assert yielded == 0
    first = occurrences.first()
    assert first.owner == owner.definition
    assert first.definition == leaf.definition
    assert yielded == 1
