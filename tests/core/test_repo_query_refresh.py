import gc

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dataclasses import replace

from dryml.core import Definition, Object, Repo
from dryml.core.query.model import QueryIndexIncompatible, QueryIndexUnavailable, QueryWouldScanError
from dryml.core.query.source import SourceCapture
from dryml.core.query.sqlite import SQLiteQueryIndexConfig
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord


class RefreshLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


def _warning_diagnostics(observed):
    import tracemalloc

    details = []
    for warning in observed:
        allocation = tracemalloc.get_object_traceback(warning.source)
        details.append((warning.category.__name__, str(warning.message),
                        allocation.format() if allocation is not None else None))
    return details


def test_v3_query_observes_committed_external_writes(tmp_path):
    store = DirStore(tmp_path / "store")
    repo_a = Repo(stores=store)
    first = RefreshLeaf("first", repo=repo_a)
    repo_a.save_object(first)

    repo_view = Repo(stores=DirStore(store.base_dir))
    assert repo_view.query().cdefs().stored().count() == 1

    repo_b = Repo(stores=DirStore(store.base_dir))
    second = RefreshLeaf("second", repo=repo_b)
    repo_b.save_object(second)

    assert repo_view.query().cdefs().stored().count() == 2


def test_current_process_save_updates_v3_cached_and_stored_membership(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    first = RefreshLeaf("first", repo=repo)
    repo.add_objects(first)

    assert repo.query().cdefs().cached(scope=repo).count() == 1
    assert repo.query().cdefs().stored().count() == 0

    repo.save_object(first)

    assert repo.query().cdefs().stored().count() == 1


def _ready_store(path, *, destination=None):
    config = SQLiteQueryIndexConfig(path=destination, journal_mode="delete")
    store = DirStore(path, query_index=config)
    store.write_definition_record(DefinitionRecord(Definition(RefreshLeaf, "root").concretize()))
    store.rebuild_query_index()
    return store


def _count_cycles(monkeypatch, *stores):
    calls = []
    for store in stores:
        index = store.open_query_index()
        original = index.refresh

        def refresh(policy, *, stats=None, _original=original, _store=store):
            calls.append((_store, policy))
            return _original(policy, stats=stats)

        monkeypatch.setattr(index, "refresh", refresh)
    return calls


def test_u2_shared_branches_fresh_request_and_direct_construction(tmp_path, monkeypatch):
    from copy import copy
    from dryml.core.query.query import IdentityQuery

    store = _ready_store(tmp_path / "store")
    calls = _count_cycles(monkeypatch, store)
    base = store.query().refresh(True)
    assert base.explain().refresh_action == "none"
    assert base.take(0).count() == base.nested().take(0).count() == 0
    assert calls == []
    first = base.cdefs().explain(analyze=True)
    second = copy(base).stored().explain(analyze=True)
    assert first.refresh_action == "validate" and first.store_scan_count == 2
    assert second.refresh is True and second.refresh_action == "reused"
    assert second.store_scan_count == 1 and len(calls) == 1
    base.refresh(True).count()
    assert len(calls) == 2
    direct = IdentityQuery(base.source, refresh_policy=True)
    direct.count()
    direct.cdefs().count()
    assert len(calls) == 3
    assert base == copy(base)
    assert base == base.refresh(True)  # ephemeral state is not plan value


def test_u2_disjoint_algebra_scope_extension_and_outer_overrides(tmp_path, monkeypatch):
    from dryml.core.query import field
    from dryml.core.query.model import QueryDomainError

    a, b, c = (_ready_store(tmp_path / name) for name in ("a", "b", "c"))
    calls = _count_cycles(monkeypatch, a, b, c)
    left, right = a.query().refresh(True), b.query().refresh(True)
    left.count()
    right.count()
    for query in (left.union(right), right.intersection(left)):
        assert query.explain(analyze=True).refresh_action == "reused"
    assert calls == [(a, True), (b, True)]
    extended = left.union(right).where(field("object").missing(), scope=c)
    assert extended.count() == 0
    assert calls == [(a, True), (b, True), (c, True)]
    extended.refresh(True).count()
    assert calls[3:] == [(a, True), (b, True), (c, True)]
    left.count()
    right.count()
    assert len(calls) == 6
    pending = a.query().refresh(True)
    for policy in (False, "auto", "always"):
        outer = pending.union(pending).refresh(policy)
        outer.count()
        outer.count()
    assert calls[6:] == [(a, "always"), (a, "always")]
    pending.count()
    assert calls[-1] == (a, True)
    with pytest.raises(QueryDomainError):
        left.union(right.refresh(False))


def test_u2_partial_failure_retries_only_unresolved_and_keeps_evaluation_completion(tmp_path, monkeypatch):
    from dryml.core.query.model import QueryCardinalityError, QueryVerifyBudgetExceeded

    a, b = (_ready_store(tmp_path / name) for name in ("a", "b"))
    calls = _count_cycles(monkeypatch, a, b)
    query = a.query().union(b.query()).refresh(True)
    primary = PermissionError("injected maintenance failure")

    def fail(*args, **kwargs):
        raise primary

    with monkeypatch.context() as scoped:
        scoped.setattr(b.open_query_index(), "refresh", fail)
        with pytest.raises(PermissionError) as error:
            query.count()
        assert error.value is primary
    assert calls == [(a, True)]
    query.count()
    assert calls == [(a, True), (b, True)]
    query.max_verify(0).take(0)
    fresh = query.refresh(True)
    with pytest.raises(QueryVerifyBudgetExceeded):
        fresh.max_verify(0).count()
    before = len(calls)
    fresh.count()
    assert len(calls) == before
    empty = fresh.object_refs().refresh(True)
    with pytest.raises(QueryCardinalityError):
        empty.one()
    before = len(calls)
    assert empty.count() == 0 and len(calls) == before


def test_u2_optional_late_target_empty_store_and_completed_configuration_check(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "empty", query_index="auto")
    query = store.query().refresh(True)
    with monkeypatch.context() as scoped:
        scoped.setattr(store, "open_query_index", lambda: None)
        assert query.explain(analyze=True).refresh_action == "none"
    store.rebuild_query_index()
    calls = _count_cycles(monkeypatch, store)
    assert query.explain(analyze=True).refresh_action == "validate"
    assert query.explain(analyze=True).refresh_action == "reused" and len(calls) == 1
    index = store.open_query_index()
    index.config = replace(index.config, busy_timeout=9)
    with pytest.raises(QueryIndexIncompatible):
        query.count()


@pytest.mark.parametrize("shared_base", (False, True))
@pytest.mark.parametrize("operation", ("union", "intersection"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("completed", (False, True))
def test_u2_same_producer_operands_preserve_external_scope_bindings(
        tmp_path, monkeypatch, shared_base, operation, reverse, completed):
    """Sharing a producer does not reopen the other operand's completed scope."""
    a, b, c, d = (_ready_store(tmp_path / name) for name in ("a", "b", "c", "d"))
    calls = _count_cycles(monkeypatch, a, b, c, d)
    base = a.query()
    left = base.refresh(True).stored(scope=b)
    right = (base if shared_base else a.query()).refresh(True).stored(scope=c)
    left_request, right_request = (query._refresh_bindings[0].request for query in (left, right))
    first, second = (right, left) if reverse else (left, right)
    combined = getattr(first, operation)(second)
    if completed:
        left.count()
        right.count()
    else:
        assert combined.count() == 1
        assert len(calls) == 3 and {store for store, _ in calls} == {a, b, c}
    assert left_request.completed_domain(b.authority_fence_key())
    assert not left_request.completed_domain(c.authority_fence_key())
    assert right_request.completed_domain(c.authority_fence_key())
    assert not right_request.completed_domain(b.authority_fence_key())
    calls.clear()
    b.mark_query_index_dirty()
    c.mark_query_index_dirty()
    for query in (combined, combined.closure(), combined.nested(),
                  combined.nested().owners(), combined.nested().targets()):
        query.count()
    assert calls == []
    assert b.query_index_is_dirty() and c.query_index_is_dirty()
    combined.refresh(True).count()
    assert len(calls) == 3 and {store for store, _ in calls} == {a, b, c}
    assert all(policy is True for _, policy in calls)
    assert not b.query_index_is_dirty() and not c.query_index_is_dirty()
    calls.clear()
    combined.stored(scope=d).count()
    assert calls == [(d, True)]
    assert left_request.completed_domain(d.authority_fence_key())
    assert right_request.completed_domain(d.authority_fence_key())
    assert not left_request.completed_domain(c.authority_fence_key())
    assert not right_request.completed_domain(b.authority_fence_key())
    calls.clear()
    shared = base.refresh(True)
    shared.union(shared.cdefs()).count()
    shared.count()
    assert calls == [(a, True)]


def test_u2_completed_descriptor_is_rechecked_after_consistent_reconfiguration(tmp_path, monkeypatch):
    store = _ready_store(tmp_path / "store")
    query = store.query().refresh(True)
    query.count()
    index = store.open_query_index()
    config = replace(index.config, busy_timeout=9)
    store._query_index_config = index.config = index._connections.config = config
    monkeypatch.setattr(index, "refresh", lambda *a, **kw: pytest.fail("changed completed descriptor refreshed"))
    with pytest.raises(QueryIndexIncompatible, match="within a request"):
        query.count()


def test_u2_publication_failure_is_retryable_and_demand_recovery_outranks_reuse(tmp_path, monkeypatch):
    from dryml.core.query.model import QueryIndexError

    store = _ready_store(tmp_path / "store")
    index = store.open_query_index()
    query = store.query().refresh(True)
    store.mark_query_index_dirty()
    captures = []
    original_init = SourceCapture.__init__

    def retain(capture, *args, **kwargs):
        original_init(capture, *args, **kwargs)
        if args and args[0] is query:
            captures.append(capture)

    monkeypatch.setattr(SourceCapture, "__init__", retain)
    primary = QueryIndexError("acknowledgement failed")

    def fail(*args, **kwargs):
        raise primary

    with monkeypatch.context() as scoped:
        scoped.setattr(index, "_clear_dirty", fail)
        with pytest.raises(QueryIndexError) as error:
            query.count()
        assert error.value is primary
    assert captures[-1].store_scan_count == 1 and captures[-1].refresh_action == "rebuild"
    assert index.path.exists() and store.query_index_is_dirty()
    assert not index._connections._leases and not index._build_claim_path().exists()
    assert query.explain(analyze=True).refresh_action == "rebuild"
    assert query.explain(analyze=True).refresh_action == "reused"
    store.mark_query_index_dirty()
    bounded = replace(query.cdefs().stored(), take_limit=1)
    explanation = bounded.explain(analyze=True)
    assert explanation.refresh is True and explanation.refresh_action == "rebuild"
    assert explanation.store_scan_count == 1 and not store.query_index_is_dirty()


def test_u3_equivalent_destinations_coalesce_in_both_store_repo_orders(tmp_path, monkeypatch):
    """Physical destination, not traversal order or handle spelling, owns a cycle."""
    destination = tmp_path / "custom.sqlite"
    first = _ready_store(tmp_path / "store", destination=destination)
    second = DirStore(first.base_dir, query_index=SQLiteQueryIndexConfig(
        path=f"{destination.parent}/./{destination.name}", journal_mode="delete",
    ))
    repo = Repo((first, second))
    calls = []
    for store in (first, second):
        index = store.open_query_index()
        original = index.refresh

        def refresh(policy, *, stats=None, _original=original):
            calls.append(policy)
            return _original(policy, stats=stats)

        monkeypatch.setattr(index, "refresh", refresh)
    monkeypatch.setattr(repo._query_index, "refresh", lambda *a, **kw: pytest.fail("live Repo bulk refresh"))
    for left, right in ((first.query(), repo.query()), (repo.query(), first.query())):
        explanation = left.union(right).refresh(True).explain(analyze=True)
        assert explanation.refresh_action == "validate"
        assert explanation.store_scan_count == 2  # maintenance, then one authority domain
    assert calls == [True, True]


@pytest.mark.parametrize("within_repo", (False, True))
@pytest.mark.parametrize("conflict", ("destination", "settings", "source", "dirty", "authority"))
def test_u3_conflicting_presentations_refuse_before_maintenance(tmp_path, monkeypatch, within_repo, conflict):
    """All presentations must survive federation/source-key hiding until admission."""
    first = _ready_store(tmp_path / "store")
    config = first.open_query_index().config
    second = DirStore(first.base_dir, query_index=config)
    index = second.open_query_index()
    if conflict == "destination":
        second = DirStore(first.base_dir, query_index=replace(config, path=tmp_path / "other.sqlite"))
    elif conflict == "settings":
        second = DirStore(first.base_dir, query_index=replace(config, busy_timeout=9))
    elif conflict == "source":
        index._source_key = "incompatible-source"
    elif conflict == "dirty":
        index.dirty_path = tmp_path / "other-dirty"
    else:
        index.store = DirStore(tmp_path / "foreign", query_index="sqlite")
    for store in (first, second):
        monkeypatch.setattr(store.open_query_index(), "refresh", lambda *a, **kw: pytest.fail("conflict performed maintenance"))
    query = Repo((first, second)).query() if within_repo else first.query().union(second.query())
    with pytest.raises(QueryIndexIncompatible):
        query.refresh(True).count()


@pytest.mark.parametrize("presentation", ("backend", "manager", "config", "store"))
def test_u3_backend_manager_path_mismatch_refuses_before_maintenance(tmp_path, monkeypatch, presentation):
    store = _ready_store(tmp_path / "store")
    index = store.open_query_index()
    wrong = tmp_path / "wrong.sqlite"
    if presentation == "backend":
        index.path = wrong
    elif presentation == "manager":
        index._connections.config = replace(index.config, path=wrong)
    elif presentation == "config":
        index.config = replace(index.config, path=wrong)
    else:
        store._query_index_config = replace(index.config, path=wrong)
    monkeypatch.setattr(index, "refresh", lambda *a, **kw: pytest.fail("mismatched path refreshed"))
    with pytest.raises(QueryIndexIncompatible):
        store.query().refresh(True).count()


def test_u3_independent_domains_each_validate(tmp_path):
    stores = tuple(_ready_store(tmp_path / name) for name in ("a", "b"))
    explanation = Repo(stores).query().refresh(True).explain(analyze=True)
    assert explanation.refresh_action == "validate" and explanation.store_scan_count == 4


def test_u3_demand_consumer_rejects_conflicting_repo_presentations(tmp_path, monkeypatch):
    """Auto preflight cannot acknowledge shared dirty tokens for two destinations."""
    first = _ready_store(tmp_path / "store")
    second = DirStore(first.base_dir, query_index=SQLiteQueryIndexConfig(path=tmp_path / "second.sqlite"))
    first.mark_query_index_dirty()
    for store in (first, second):
        monkeypatch.setattr(store.open_query_index(), "refresh", lambda *a, **kw: pytest.fail("conflict maintained"))
    with pytest.raises(QueryIndexIncompatible):
        Repo((first, second)).query().cdefs().stored().take(1)


def test_u3_foreign_persisted_target_refuses_before_any_target_maintenance(tmp_path, monkeypatch):
    """A later foreign destination cannot be hidden behind an earlier valid target."""
    first = _ready_store(tmp_path / "first")
    foreign = _ready_store(tmp_path / "foreign")
    second = DirStore(tmp_path / "second", query_index=foreign.open_query_index().config)
    monkeypatch.setattr(first.open_query_index(), "refresh", lambda *a, **kw: pytest.fail("earlier target maintained"))
    with pytest.raises(QueryIndexIncompatible):
        first.query().union(second.query()).refresh(True).count()


def test_u3_optional_absence_and_explicit_configured_unavailability(tmp_path, monkeypatch):
    optional = DirStore(tmp_path / "optional", query_index="auto")
    configured = DirStore(tmp_path / "configured", query_index="sqlite")
    monkeypatch.setattr(optional, "open_query_index", lambda: None)
    monkeypatch.setattr(configured, "open_query_index", lambda: None)
    assert optional.query().refresh(True).explain(analyze=True).refresh_action == "none"
    for policy in (True, "always"):
        with pytest.raises(QueryIndexUnavailable):
            configured.query().refresh(policy).count()


@pytest.mark.parametrize("policy", (True, "always"))
@pytest.mark.parametrize("scan_policy", ("forbid", "warn"))
def test_u3_optional_absence_exact_positive_scan_controls(tmp_path, monkeypatch, policy, scan_policy):
    """Absent optional targets need no scan or credit; late ready targets still do."""
    # Finalize prior fixtures before attributing warnings to this terminal. A
    # second pass retires resources released by weak-reference finalizers.
    gc.collect()
    gc.collect()
    import warnings
    import dryml.core.store.dir as dir_module
    from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex

    store = DirStore(tmp_path / "optional", query_index="auto")
    root = Definition(RefreshLeaf, "root").concretize()
    store.write_definition_record(DefinitionRecord(root))
    query = store.query().sel(root).cdefs().stored().refresh(policy).scan_policy(scan_policy)
    request = query._refresh_bindings[0].request if policy is True else None
    with monkeypatch.context() as scoped:
        scoped.setattr(dir_module, "sqlite_available", lambda: False)
        scoped.setattr(store, "_validated_definition_root_inventory",
                       lambda: pytest.fail("optional exact positive acquired inventory"))
        scoped.setattr(store, "iter_definition_records",
                       lambda: pytest.fail("optional exact positive enumerated definitions"))
        scoped.setattr(SQLiteStoreQueryIndex, "refresh",
                       lambda *a, **kw: pytest.fail("absent optional target cycled"))
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            assert query.exists()
            explanation = query.explain(analyze=True)
        assert observed == [], _warning_diagnostics(observed)
        assert explanation.store_scan_count == 0 and explanation.refresh_action == "none"
        assert store._query_index_instance is None
        if request is not None:
            assert request.completed == request.in_flight == {}

    store.rebuild_query_index()
    calls = _count_cycles(monkeypatch, store)
    index = store.open_query_index()
    with pytest.raises(QueryWouldScanError):
        query.scan_policy("forbid").exists()
    assert calls == []
    if request is not None:
        assert request.completed == request.in_flight == {}
    with pytest.warns(RuntimeWarning, match="inventory scan") as observed:
        assert query.scan_policy("warn").exists()
    assert len(observed) == 1 and calls == [(store, policy)], _warning_diagnostics(observed)
    calls.clear()
    if policy is True:
        assert query.scan_policy("forbid").exists()
    else:
        with pytest.raises(QueryWouldScanError):
            query.scan_policy("forbid").exists()
    assert calls == []
    assert not index._connections._leases and not index._build_claim_path().exists()


def test_u3_warning_precedes_ready_validation_once(tmp_path, monkeypatch):
    import warnings

    store = _ready_store(tmp_path / "store")
    original = store._validated_definition_root_inventory
    scans = []
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")

        def inventory():
            assert len(observed) == 1
            scans.append(1)
            return original()

        monkeypatch.setattr(store, "_validated_definition_root_inventory", inventory)
        explanation = store.query().refresh(True).scan_policy("warn").explain(analyze=True)
    assert len(observed) == 1 and len(scans) == explanation.store_scan_count == 2


def test_u3_always_repeats_ready_validation_and_inert_paths_do_nothing(tmp_path, monkeypatch):
    store = _ready_store(tmp_path / "store")
    index = store.open_query_index()
    generation = index.status().generation
    query = store.query().refresh("always")
    for _ in range(2):
        explanation = query.explain(analyze=True)
        assert explanation.refresh_action == "validate" and explanation.store_scan_count == 2
    assert index.status().generation == generation
    monkeypatch.setattr(store, "open_query_index", lambda: pytest.fail("inert path opened index"))
    monkeypatch.setattr(store, "iter_definition_records", lambda: pytest.fail("inert path read authority"))
    assert query.explain().refresh_action == "none"
    assert query.take(0).count() == 0
    assert query.nested().take(0).count() == 0


def test_u3_ready_explicit_scan_admission_precedes_backend_io(tmp_path, monkeypatch):
    store = _ready_store(tmp_path / "store")
    monkeypatch.setattr(store, "open_query_index", lambda: pytest.fail("prohibited work opened backend"))
    root = Definition(RefreshLeaf, "root").concretize()
    for policy in (True, "always"):
        for query in (store.query().refresh(policy),
                      store.query().sel(root).cdefs().stored().refresh(policy)):
            with pytest.raises(QueryWouldScanError):
                query.scan_policy("forbid").exists()
            with pytest.raises(QueryIndexUnavailable):
                query.require_indexed().take(1)
    for invalid in (0, 1, None, [], {}, "invalid"):
        with pytest.raises(ValueError):
            query.refresh(invalid)


def test_u3_demand_recovery_reports_actual_rebuild_and_failed_scan(tmp_path, monkeypatch):
    store = _ready_store(tmp_path / "store")
    index = store.open_query_index()
    store.mark_query_index_dirty()
    query = replace(store.query().cdefs().stored(), take_limit=1)
    original = index.refresh

    def fail_after_scan(policy, *, stats=None):
        stats.store_scan_count += 1
        stats.refresh_action = "sqlite-rebuild"
        raise QueryIndexUnavailable("injected recovery failure")

    monkeypatch.setattr(index, "refresh", fail_after_scan)
    explanation = query.explain(analyze=True)
    assert explanation.refresh_action == "rebuild" and explanation.store_scan_count == 2
    assert store.query_index_is_dirty()
    monkeypatch.setattr(index, "refresh", original)
    explanation = query.explain(analyze=True)
    assert explanation.refresh_action == "rebuild" and explanation.store_scan_count == 1
    assert not store.query_index_is_dirty()


def test_u3_unavailable_consumer_preflight_keeps_authority_fallback(tmp_path, monkeypatch):
    """Destination conflicts fail closed; ordinary backend loss keeps fallback."""
    store = _ready_store(tmp_path / "store")

    def unavailable():
        raise QueryIndexUnavailable("compatibility reader unavailable")

    monkeypatch.setattr(store.open_query_index(), "_assert_sidecar_rebuildable", unavailable)
    query = replace(store.query().cdefs().stored(), take_limit=1)
    explanation = query.explain(analyze=True)
    assert explanation.result_count == 1 and explanation.store_scan_count == 1
    assert explanation.refresh_action == "none"
    with pytest.raises(QueryWouldScanError):
        query.scan_policy("forbid").collect()
    with pytest.raises(QueryIndexUnavailable):
        query.require_indexed().collect()


def test_u3_explicit_failure_merges_started_work_in_finally(tmp_path, monkeypatch):
    store = _ready_store(tmp_path / "store")
    captures = []
    original = SourceCapture.__init__

    def retain(capture, *a, **kw):
        original(capture, *a, **kw)
        captures.append(capture)

    monkeypatch.setattr(SourceCapture, "__init__", retain)

    def interrupt(policy, *, stats=None):
        stats.store_scan_count += 1
        stats.refresh_action = "sqlite-validate"
        raise KeyboardInterrupt("maintenance interrupted")

    monkeypatch.setattr(store.open_query_index(), "refresh", interrupt)
    with pytest.raises(KeyboardInterrupt):
        store.query().refresh(True).count()
    assert captures[-1].store_scan_count == 1 and captures[-1].refresh_action == "validate"


def test_u3_topology_attachment_after_retention_waits_until_next_terminal(tmp_path, monkeypatch):
    import threading

    first, before, after = (_ready_store(tmp_path / name) for name in ("a", "b", "c"))
    repo = Repo(first)
    query = repo.query().refresh(True)
    repo.add_store(before)
    original = first.open_query_index().refresh
    retained, resume = threading.Event(), threading.Event()

    def attach(policy, *, stats=None):
        retained.set()
        assert resume.wait(15)
        return original(policy, stats=stats)

    monkeypatch.setattr(first.open_query_index(), "refresh", attach)
    outcomes, errors = [], []

    def run():
        try:
            outcomes.append(query.explain(analyze=True))
        except BaseException as error:
            errors.append(error)
        finally:
            for store in (first, before, after):
                store.open_query_index()._connections.close_current()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert retained.wait(10)
        repo.add_store(after)
    finally:
        resume.set()
        thread.join(15)
    assert not thread.is_alive() and errors == []
    assert outcomes[0].store_scan_count == 4
    assert query.explain(analyze=True).store_scan_count == 4  # only C pending, then three fresh cuts


def test_u3_explicit_recapture_keeps_cycle_and_cumulative_ledger(tmp_path, monkeypatch):
    """Evaluation demand growth never repeats the preceding explicit cycle."""
    from dryml.core.query import field
    from dryml.core.query.model import QueryVerifyBudgetExceeded

    store = _ready_store(tmp_path / "store")
    query = store.query().refresh(True)
    index = store.open_query_index()
    original = index.refresh
    cycles = []

    def refresh(policy, *, stats=None):
        cycles.append(policy)
        return original(policy, stats=stats)

    monkeypatch.setattr(index, "refresh", refresh)

    def evaluate(capture):
        query._collect_with_capture(capture)
        return query.where(field("object").missing())._collect_with_capture(capture)

    result, capture = query.max_verify(2)._run_terminal(evaluate, return_capture=True)
    assert result.count() == 0  # CDefs are not metadata holders
    assert cycles == [True]
    assert capture.store_scan_count == 3 and capture.verified_count == 2
    assert capture.demand_recaptures == 1 and capture.refresh_action == "validate"
    with pytest.raises(QueryVerifyBudgetExceeded):
        query.max_verify(1)._run_terminal(evaluate)
    assert cycles == [True]  # later evaluation failure cannot reopen maintenance


def test_u3_target_resolution_retains_operand_and_scope_associations(tmp_path, monkeypatch):
    """U2 can bind lineage to accepted targets without another graph traversal."""
    from dryml.core.query import field

    store = _ready_store(tmp_path / "store")
    repo = Repo(store)
    left, right = store.query(), repo.query()
    query = left.union(right).where(field("object").missing(), scope=store)
    capture = SourceCapture(query)
    capture.index_presentations = query._index_presentations(capture)
    monkeypatch.setattr(store, "_validated_definition_root_inventory", lambda: pytest.fail("resolution captured authority"))
    targets = capture.resolve_index_targets(explicit=True)
    assert len(targets) == 1
    presentations = targets[0].presentations
    assert {id(item.producer) for item in presentations} == {id(left), id(right), id(query)}
    assert len(presentations) == 3 and all(item.store is store for item in presentations)
    assert capture.store_scan_count == capture.source_cuts == 0


def test_u3_inherited_repo_refuses_before_parent_configuration_lock(tmp_path):
    """A fork while another thread owns the lock must not strand child queries."""
    import multiprocessing
    import threading

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork")
    store = _ready_store(tmp_path / "store")
    repo = Repo(store)
    held, release = threading.Event(), threading.Event()

    def hold():
        with repo._configuration_lock:
            held.set()
            assert release.wait(15)

    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)

    def run():
        try:
            try:
                repo.query().count()
            except RuntimeError as error:
                refused = "process" in str(error)
            else:
                refused = False
            fresh = Repo(DirStore(store.base_dir, query_index="memory"))
            child.send((refused, fresh.query().cdefs().count()))
        finally:
            child.close()

    thread = threading.Thread(target=hold)
    thread.start()
    process = context.Process(target=run)
    try:
        assert held.wait(10)
        process.start()
        assert parent.poll(10), "child blocked on inherited Repo lock"
        assert parent.recv() == (True, 1)
        process.join(10)
        assert process.exitcode == 0
    finally:
        release.set()
        thread.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()
    assert repo.query().cdefs().count() == 1


# These preserved assertions belong to the later Repo/Query consumer units.
# Narrow strict xfail records the explicit capability refusal, not a ready claim.
from dryml.core.store.store import StoreCapabilityError

for _name, _test in tuple(globals().items()):
    if _name.startswith("test_"):
        globals()[_name] = pytest.mark.xfail(
            strict=True, raises=StoreCapabilityError,
            reason="Legacy Repo/Query V3 consumer explicitly refused until U10/U13 cutover",
        )(_test)


@pytest.mark.parametrize("refresh", [False, "auto", True, "always"])
def test_format4_repo_query_refresh_refuses_unintegrated_consumer(tmp_path, refresh):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    store.publish_node(Definition(RefreshLeaf, name="new").concretize())
    before = store.authority_cut()
    with pytest.raises(StoreCapabilityError, match="coverage contract"):
        Repo(store).query().refresh(refresh).count()
    assert store.authority_cut() == before
    assert len(store.open_query_index().projection_rows("recipe_root", refresh=False)) == 1
