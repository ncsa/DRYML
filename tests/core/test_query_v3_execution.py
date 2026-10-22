"""Focused U5 contracts for private Query V3 execution terminals."""

import pytest

from dryml.core import Definition, Object, Repo, SaveAnnotations, Serializable
from dryml.core.query import field
from dryml.core.query.identity import IdentitySet, OccurrenceSet
from dryml.core.query.model import QueryCardinalityError, QueryDomainError, QueryIndexUnavailable
from dryml.core.query.query import IdentityQuery
from dryml.core.query.source import SourceCapture, _SourceRecapture
from dryml.core.store.dir import DirStore
from dryml.core.query.model import QueryError
from dryml.core.store.records import NodeNameRecord


def _name(store, alias, node):
    root = store.publish_node(node)
    store.compare_and_set_node_name(NodeNameRecord(alias, "recipe", root.digest), expected=store.read_node_name(alias))


class ExecutionLeaf(Object):
    """Small root used to exercise detached V3 execution."""

    def __init__(self, name):
        self.name = name


class ExecutionStatefulLeaf(Serializable):
    """Stateful fixture with a root ObjectId for alias-only authority."""

    def __init__(self, name):
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


def test_u5_red_query_algebra_and_bounded_terminal_are_available(tmp_path):
    """The red proof required deferred cross-producer algebra and a prefix sink."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    first_repo = Repo(first)
    second_repo = Repo(second)
    first_state = first_repo.save_object(ExecutionLeaf("first", repo=first_repo))
    second_state = second_repo.save_object(ExecutionLeaf("second", repo=second_repo))

    query = IdentityQuery.from_store(first).union(IdentityQuery.from_store(second))

    result = query.state_refs().take(1).collect()
    assert result.count() == 1 and any(result.one().graph_equal(v) for v in (first_state, second_state))


def test_u5_algebra_reuses_one_cut_and_preserves_complete_source_evidence(tmp_path, monkeypatch):
    """Duplicate algebra branches share a cut while replicas remain one identity."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(ExecutionLeaf("shared", repo=repo))
    calls = 0
    original = store._read_node_block

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_read_node_block", counted)
    base = IdentityQuery.from_store(store).state_refs()
    union = base.union(base)

    collected = union.collect()

    assert collected.one().graph_equal(state)
    assert len(collected.evidence_for(state).sources) == 1
    assert calls == 0  # current page captures authenticated block bytes directly
    assert union.explain(analyze=True).source_cuts == 1
    assert union.intersection(base).one().graph_equal(state)


def test_u5_algebra_drops_ambiguous_default_and_validates_policies(tmp_path):
    """Algebra never selects an authority default by operand order."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    combined = IdentityQuery.from_store(first).union(IdentityQuery.from_store(second))

    with pytest.raises(QueryDomainError, match="explicit scope"):
        combined.where(field("object", "team").eq("vision"))
    with pytest.raises(QueryDomainError, match="conflicting execution policies"):
        IdentityQuery.from_store(first).scan_policy("forbid").union(
            IdentityQuery.from_store(first)
        )
    with pytest.raises(QueryIndexUnavailable, match="index coverage"):
        IdentityQuery.from_store(first).require_indexed().union(
            IdentityQuery.from_store(first).require_indexed()
        ).count()


def test_v3_query_algebra_accepts_fixed_members_without_source_adoption(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    repo.save_object(ExecutionLeaf("stored", repo=repo))
    extra = Definition(ExecutionLeaf, "extra").concretize()
    fixed = IdentitySet((extra,), bounded=True)

    combined = IdentityQuery.from_store(store).cdefs().union(fixed)
    assert extra in combined.collect()
    assert combined.collect().bounded
    assert IdentityQuery.from_store(store).cdefs().intersection(fixed).count() == 0
    assert fixed.query().union(IdentitySet((extra,))).one() == extra
    with pytest.raises(QueryDomainError, match="explicit scope"):
        combined.where(field("object", "team").eq("value"))


def test_v3_top_level_algebra_keeps_fixed_inputs_source_free(tmp_path, monkeypatch):
    from dryml.core.query.query import intersection, union

    first = Definition(ExecutionLeaf, "first").concretize()
    second = Definition(ExecutionLeaf, "second").concretize()
    left, right = IdentitySet((first,)), IdentitySet((second,))
    store = DirStore(tmp_path / "store")
    monkeypatch.setattr(
        store, "iter_definition_records",
        lambda: pytest.fail("fixed-only algebra read Store authority"),
    )

    assert isinstance(union(left, right), IdentitySet)
    assert union(left, right).count() == 2
    assert intersection(left, right).count() == 0
    assert union(IdentityQuery.from_store(store).cdefs(), right).take(0).bounded


def test_u5_scalars_do_not_call_collect_or_build_a_final_identity_set(tmp_path, monkeypatch):
    """Cardinality sinks retain only the evidence required for their answer."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first = repo.save_object(ExecutionLeaf("first", repo=repo))
    repo.save_object(ExecutionLeaf("second", repo=repo))
    query = IdentityQuery.from_store(store).state_refs()

    monkeypatch.setattr(
        IdentityQuery, "collect", lambda *_args, **_kwargs: pytest.fail("scalar called collect"),
    )
    monkeypatch.setattr(
        IdentitySet, "_from_entries",
        classmethod(lambda *_args, **_kwargs: pytest.fail("scalar built final IdentitySet")),
    )

    assert query.count() == 2
    assert query.exists()
    with pytest.raises(QueryCardinalityError):
        query.one()
    assert query.sel(first).one_or_none().graph_equal(first)


def test_u5_scalar_terminals_stop_at_distinct_cardinality_thresholds(monkeypatch):
    """Scalar sinks avoid public ordering and retain only required witnesses."""

    from dryml.core.query import identity as identity_module

    values = tuple(
        Definition(ExecutionLeaf, name).concretize()
        for name in ("first", "second", "third")
    )
    query = IdentityQuery.from_set(IdentitySet(values))
    original = IdentityQuery._iter_terminal_members
    yielded = 0

    def counted(self, capture, captured=None):
        nonlocal yielded
        for item in original(self, capture, captured):
            yielded += 1
            yield item

    monkeypatch.setattr(IdentityQuery, "_iter_terminal_members", counted)
    monkeypatch.setattr(
        identity_module,
        "_ordered_keys",
        lambda *_args, **_kwargs: pytest.fail("scalar terminal requested public order"),
    )

    assert query.exists()
    assert yielded == 1
    yielded = 0
    with pytest.raises(QueryCardinalityError):
        query.one()
    assert yielded == 2
    yielded = 0
    with pytest.raises(QueryCardinalityError):
        query.one_or_none()
    assert yielded == 2
    yielded = 0
    assert query.count() == 3
    assert yielded == 3


def test_u5_take_retains_requested_limit_and_canonical_prefix(tmp_path):
    """A requested prefix is bounded metadata, not a complete-result claim."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    states = [
        repo.save_object(ExecutionLeaf(name, repo=repo))
        for name in ("first", "second", "third")
    ]

    result = IdentityQuery.from_store(store).state_refs().take(2)

    assert isinstance(result, IdentitySet)
    assert result.bounded
    assert result.requested_limit == 2
    assert result.diagnostic().requested_limit == 2
    assert [v.exact_digest() for v in result] == sorted(v.exact_digest() for v in states)[:2]
    assert result.query().collect().requested_limit == 2
    assert IdentityQuery.from_store(store).state_refs().take(0).count() == 0
    assert not IdentityQuery.from_store(store).state_refs().take(0).exists()
    combined = (
        IdentityQuery.from_store(store).state_refs().take(1).query().union(
            IdentityQuery.from_store(store).state_refs()
        ).collect()
    )
    assert combined.bounded
    assert combined.requested_limit == 1


def test_u5_metadata_validation_precedes_scalar_early_stop(tmp_path):
    """A later invalid metadata leaf cannot be hidden by an earlier match."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    repo.save_object(
        ExecutionLeaf("invalid", repo=repo),
        annotations=SaveAnnotations(object={"score": "invalid"}),
    )
    valid = repo.save_object(
        ExecutionLeaf("valid", repo=repo),
        annotations=SaveAnnotations(object={"score": 1}),
    )

    with pytest.raises(Exception, match="numeric field"):
        (
            IdentityQuery.from_store(store)
            .sel(valid)
            .where(field("object", "score").lt(3))
            .exists()
        )


def test_u5_later_metadata_demand_recaptures_full_cut_without_using_retry_budget(tmp_path):
    """Demand growth replaces a source cut and is distinct from instability retries."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    repo.save_object(ExecutionLeaf("value", repo=repo))
    capture = SourceCapture()

    capture.capture_store(store)
    capture.capture_store(store, metadata_scopes=frozenset(("object",)))
    capture.capture_store(store, metadata_scopes=frozenset(("object", "state")))

    assert capture.source_cuts == 1
    assert capture.capture_rounds == 3
    assert capture.demand_recaptures == 2


def test_u5_occurrence_fixed_refinement_retains_count_terminal():
    result = OccurrenceSet().query().collect()
    assert result.count() == 0
    assert OccurrenceSet().query().count() == 0


def test_u5_fixed_prefix_does_not_reencode_unique_graph_digests(monkeypatch):
    from dryml.core import cdef_codec

    first = Definition(ExecutionLeaf, "first").concretize()
    second = Definition(ExecutionLeaf, "second").concretize()
    query = IdentityQuery.from_set(IdentitySet((first, second)))
    monkeypatch.setattr(
        cdef_codec, "cdef_graph_hash",
        lambda root: (_ for _ in ()).throw(AssertionError("unnecessary ordering encoding")),
    )

    assert query.take(1).count() == 1


def test_u5_zero_prefix_validates_without_opening_source_inventory(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    monkeypatch.setattr(
        store, "iter_definition_records",
        lambda: pytest.fail("take(0) opened source inventory"),
    )

    result = IdentityQuery.from_store(store).take(0).collect()
    assert result.bounded and result.requested_limit == 0
    assert result.count() == 0


@pytest.mark.parametrize("old,new", [("old", "new"), (True, 1)])
def test_u5_late_demand_restarts_both_algebra_branches_after_mutation(tmp_path, monkeypatch, old, new):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(
        ExecutionLeaf("candidate", repo=repo),
        annotations=SaveAnnotations(object={"team": old}),
    )
    lineage = repo.get_lineage_metadata(state.object_view())
    left = IdentityQuery.from_store(store).object_refs().where(
        field("object", "team").eq(old)
    )
    right = IdentityQuery.from_store(store).object_refs().where(
        field("object", "team").eq(new)
    ).where(field("lineage", "creation_status").eq(lineage.creation_status))
    original = IdentityQuery._evaluated_entries
    mutated = False

    def mutate_after_left(self, capture):
        nonlocal mutated
        result = original(self, capture)
        if self is left and not mutated:
            mutated = True
            repo.set_metadata(state.object_view(), {"team": new}, store=store, scope="object")
        return result

    monkeypatch.setattr(IdentityQuery, "_evaluated_entries", mutate_after_left)

    with pytest.raises(QueryError, match="changed"):
        left.intersection(right).count()
    assert mutated


def test_u5_late_demand_converges_without_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(
        ExecutionLeaf("candidate", repo=repo),
        annotations=SaveAnnotations(object={"team": "old"}),
    )
    lineage = repo.get_lineage_metadata(state.object_view())
    left = IdentityQuery.from_store(store).object_refs().where(
        field("object", "team").eq("old")
    )
    right = IdentityQuery.from_store(store).object_refs().where(
        field("lineage", "creation_status").eq(lineage.creation_status)
    )

    assert left.intersection(right).one().graph_equal(state.object_view())


def test_u5_four_static_metadata_demands_do_not_use_instability_retries(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(
        ExecutionLeaf("candidate", repo=repo),
        annotations=SaveAnnotations(object={"team": "old"}),
    )
    lineage = repo.get_lineage_metadata(state.object_view())
    snapshot = repo.get_snapshot_metadata(state)
    branches = (
        IdentityQuery.from_store(store).where(field("object", "team").eq("old")),
        IdentityQuery.from_store(store).where(field("state", "team").missing()),
        IdentityQuery.from_store(store).where(
            field("lineage", "creation_status").eq(lineage.creation_status)
        ),
        IdentityQuery.from_store(store).where(field("snapshot", "saved_at").eq(snapshot.saved_at)),
    )
    combined = branches[0]
    for branch in branches[1:]:
        combined = combined.union(branch)

    explanation = combined.explain(analyze=True)
    assert explanation.instability_retries == 0
    assert explanation.demand_recaptures == 3


def test_u5_repeated_cut_instability_fails_without_a_partial_answer(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    repo.save_object(ExecutionLeaf("candidate", repo=repo))
    original = IdentityQuery._iter_terminal_members
    calls = 0

    def unstable(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise _SourceRecapture(True)
        yield from original(self, *args, **kwargs)

    monkeypatch.setattr(IdentityQuery, "_iter_terminal_members", unstable)
    with pytest.raises(Exception, match="source changed during evaluation"):
        IdentityQuery.from_store(store).count()
    assert calls == 3


def test_u5_repo_attachment_does_not_enter_an_existing_terminal(tmp_path, monkeypatch):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(first)
    other = Repo(second)
    other.save_object(ExecutionLeaf("new", repo=other))
    original = IdentityQuery._evaluated_entries
    left = IdentityQuery.from_repo(repo).state_refs()
    right = IdentityQuery.from_repo(repo).state_refs()

    def attach_after_left(self, capture):
        result = original(self, capture)
        if self is left:
            repo.add_store(second)
        return result

    monkeypatch.setattr(IdentityQuery, "_evaluated_entries", attach_after_left)
    with pytest.raises(QueryError, match="topology changed"):
        left.union(right).count()


def test_u5_exact_miss_then_publication_does_not_mix_with_broad_cut(tmp_path):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo((first, second))
    value = ExecutionLeaf("candidate", repo=repo)
    state = repo.save_object(value, store=second)
    capture = SourceCapture()
    capture.active = True
    assert not capture.capture_store(first).knowledge()
    assert repo.save_object(value, store=first, source_store=second).graph_equal(state)
    with pytest.raises(QueryError, match="changed"):
        capture.capture_store(first)
    assert state in SourceCapture().capture_store(first).knowledge()


def test_v3_exact_stored_cdef_positive_does_not_enumerate_definitions(tmp_path, monkeypatch):
    original = DirStore(tmp_path / "store")
    repo = Repo(original)
    saved = repo.save_object(ExecutionLeaf("target", repo=repo))

    store = original
    store.publish_node(saved.creation_view())
    monkeypatch.setattr(store, "iter_definition_records", lambda: pytest.fail("exact lookup inventoried definitions"))
    assert IdentityQuery.from_store(store).sel(saved.creation_view()).cdefs().stored().one().graph_equal(saved.creation_view())
    assert IdentityQuery.from_store(store).sel(saved.creation_view()).cdefs().stored().scan_policy("forbid").one().graph_equal(saved.creation_view())


def test_v3_repo_exact_stored_cdef_positive_does_not_enumerate_definitions(tmp_path, monkeypatch):
    original = DirStore(tmp_path / "store")
    writer = Repo(original)
    saved = writer.save_object(ExecutionLeaf("target", repo=writer))

    store = original
    reader = Repo(store)
    store.publish_node(saved.creation_view())
    monkeypatch.setattr(store, "iter_definition_records", lambda: pytest.fail("exact lookup inventoried definitions"))
    query = IdentityQuery.from_repo(reader).sel(saved.creation_view()).cdefs().stored()

    assert query.one().graph_equal(saved.creation_view())
    assert query.scan_policy("forbid").one().graph_equal(saved.creation_view())
    explanation = query.explain(analyze=True)
    assert explanation.source_cuts == 1
    assert explanation.capture_rounds == 1


def test_v3_alias_only_stored_cdef_falls_back_to_complete_authority(tmp_path):
    from dryml.core.query.model import QueryWouldScanError

    store = DirStore(tmp_path / "store", query_index="memory")
    cdef = Definition(ExecutionStatefulLeaf, "alias-only").concretize()
    _name(store, "selected", cdef)

    query = IdentityQuery.from_store(store).sel(cdef).cdefs().stored()
    assert query.one().graph_equal(cdef)
    with pytest.raises(QueryWouldScanError, match="inventory scan"):
        query.scan_policy("forbid").count()


def test_v3_repo_exact_mixed_root_authority_retains_both_sources(tmp_path):

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    cdef = Definition(ExecutionStatefulLeaf, "shared").concretize()
    first.publish_node(cdef)
    _name(second, "selected", cdef)

    result = IdentityQuery.from_repo(Repo((first, second))).sel(cdef).cdefs().stored().collect()
    assert result.one().graph_equal(cdef)
    assert len(result.sources(cdef)) == 2


def test_exact_state_scoped_read_failure_is_visible_before_scalar_stop(tmp_path, monkeypatch,
                                                                     fixed_snapshot_environment):
    """A positive replica cannot suppress captured metadata failure in another."""

    from dryml.core.query.model import QueryError

    first = DirStore(tmp_path / "first", query_index="memory")
    second = DirStore(tmp_path / "second", query_index="memory")
    repo = Repo((first, second))
    value = ExecutionLeaf("saved", repo=repo)
    state = repo.save_object(value, store=first)
    repo.save_object(value, store=second, source_store=first)
    monkeypatch.setattr(second, "read_metadata", lambda *_a: (_ for _ in ()).throw(PermissionError("private")))
    query = repo.query().sel(state).where(field("object", "missing").missing())
    for terminal in (query.exists, query.one, query.count):
        with pytest.raises(QueryError, match="metadata read failed"):
            terminal()


def test_empty_candidates_do_not_suppress_external_alias_conflict(tmp_path):
    """Scoped obligations are normalized even when an earlier filter is empty."""

    from dryml.core import ObjectId
    from dryml.core.query.model import QueryError

    first = DirStore(tmp_path / "first", query_index="memory")
    second = DirStore(tmp_path / "second", query_index="memory")
    for store, name in ((first, "first"), (second, "second")):
        _name(store, "chosen", Definition(ExecutionStatefulLeaf, name).concretize().with_identity(ObjectId()))
    query = IdentitySet().query().alias("chosen", scope=Repo((first, second)))
    with pytest.raises(QueryError, match="alias authority conflicts"):
        query.exists()


def test_u2_completed_request_uses_fresh_scoped_conflict_proof(tmp_path, monkeypatch):
    from dryml.core import ObjectId
    from dryml.core.query.model import QueryError
    from dryml.core.query.sqlite import SQLiteQueryIndexConfig

    a, b = (DirStore(tmp_path / name, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
            for name in ("a", "b"))
    object_id = ObjectId()
    first = Definition(ExecutionStatefulLeaf, "first").concretize().with_identity(object_id)
    a.publish_node(first)
    query = a.query().refresh(True).stored(scope=b)
    assert query.count() == 0
    second = Definition(ExecutionStatefulLeaf, "second").concretize().with_identity(object_id)
    b.publish_node(second)
    for store in (a, b):
        monkeypatch.setattr(store.open_query_index(), "refresh", lambda *a, **kw: pytest.fail("completed scope refreshed"))
    with pytest.raises(QueryError, match="authority conflicts"):
        query.exists()


@pytest.mark.parametrize("bounded", (False, True))
@pytest.mark.parametrize("empty", (False, True))
def test_algebra_validates_selected_roots_before_operand_filtering(tmp_path, bounded, empty):
    """Positive and empty filtered operands retain their authority obligations."""

    from dataclasses import replace
    from dryml.core import ObjectId
    from dryml.core.query.model import QueryError

    first = DirStore(tmp_path / "first", query_index="memory")
    second = DirStore(tmp_path / "second", query_index="memory")
    object_id = ObjectId()
    refs = tuple(Definition(ExecutionStatefulLeaf, name).concretize().with_identity(object_id) for name in ("first", "second"))
    first.publish_node(refs[0])
    second.publish_node(refs[1])
    left = first.query().sel(refs[0]).object_refs()
    right = second.query().sel(refs[0] if empty else refs[1]).object_refs()
    if bounded:
        left, right = (replace(query, take_limit=1) for query in (left, right))
    for query in (left.union(right), right.union(left), left.intersection(right), right.intersection(left)):
        with pytest.raises(QueryError, match="authority conflicts"):
            query.exists()
    with pytest.raises(QueryError, match="authority conflicts"):
        left.stored(scope=second).exists()


@pytest.mark.parametrize("matching", (False, True))
def test_bounded_inventory_combines_external_stored_authority(tmp_path, monkeypatch, matching):
    """A bounded positive or empty scope match cannot hide unrelated root conflicts."""

    from dryml.core import ObjectId
    from dryml.core.query.model import QueryError

    first = DirStore(tmp_path / "first", query_index="memory")
    second = DirStore(tmp_path / "second", query_index="memory")
    target = Definition(ExecutionStatefulLeaf, "target").concretize().with_identity(ObjectId())
    first.publish_node(target)
    if matching:
        second.publish_node(target)
    object_id = ObjectId()
    for store, name in ((first, "first"), (second, "second")):
        store.publish_node(Definition(ExecutionStatefulLeaf, name).concretize().with_identity(object_id))
    query = first.query().sel(target).object_refs().stored(scope=second)
    with pytest.raises(QueryError, match="object-reference authority conflicts"):
        query.take(1)
    for store in (first, second):
        monkeypatch.setattr(store, "_validated_definition_root_inventory",
                            lambda: pytest.fail("take(0) acquired scoped inventory"))
    assert query.max_verify(0).take(0).count() == 0


def test_bounded_fallback_algebra_combines_complete_authority(tmp_path):
    """Alias-only bounded domains retain complete authority obligations in algebra."""

    from dataclasses import replace
    from dryml.core import ObjectId
    from dryml.core.query.model import QueryError

    stores = tuple(DirStore(tmp_path / name, query_index="sqlite") for name in ("first", "second"))
    object_id = ObjectId()
    for store, name in zip(stores, ("first", "second")):
        _name(store, "root", Definition(ExecutionStatefulLeaf, name).concretize().with_identity(object_id))
        store.rebuild_query_index()
    left, right = (replace(store.query().cdefs().stored().refresh(False), take_limit=1) for store in stores)
    for query in (left.union(right), right.union(left), left.intersection(right), right.intersection(left)):
        with pytest.raises(QueryError, match="object-reference authority conflicts"):
            query.collect()


def test_u6_budget_reserves_broad_domain_before_scalar_verification(tmp_path, monkeypatch):
    """Broad exists charges all three detached identities before filtering."""

    from dryml.core.query.model import QueryVerifyBudgetExceeded

    store = DirStore(tmp_path / "store", query_index="memory")
    state = Repo(store).save_object(ExecutionLeaf("member"))
    query = store.query().state_refs()
    matches = []
    original = IdentityQuery._matches

    def matched(self, *args, **kwargs):
        matches.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IdentityQuery, "_matches", matched)
    for limit in (0, 2):
        with pytest.raises(QueryVerifyBudgetExceeded):
            query.max_verify(limit).exists()
        assert matches == []
    explanation = query.max_verify(3).explain(analyze=True)
    assert explanation.verified_count == 3
    assert explanation.store_scan_count == 1
    assert query.max_verify(3).sel(state).one().graph_equal(state)


def test_u6_fixed_repeated_stages_share_admissions_and_outer_controls(monkeypatch):
    """Repeated algebra stages share identities; outer policies govern children."""

    from dryml.core.query.model import QueryVerifyBudgetExceeded

    values = tuple(Definition(ExecutionLeaf, name).concretize() for name in ("a", "b"))
    base = IdentitySet(values).query().max_verify(0)
    combined = base.union(base).max_verify(2)
    assert combined.explain(analyze=True).verified_count == 2
    assert combined.count() == 2
    with pytest.raises(QueryVerifyBudgetExceeded):
        combined.max_verify(1).exists()
    assert combined.max_verify(0).take(0).count() == 0


def test_u6_exact_paths_reserve_before_direct_verification(tmp_path, monkeypatch):
    """Exact StateRef and stored-CDef positives cannot bypass zero budgets."""

    from dryml.core.query.model import QueryVerifyBudgetExceeded

    store = DirStore(tmp_path / "store", query_index="memory")
    state = Repo(store).save_object(ExecutionLeaf("member"))
    store.publish_node(state.creation_view())
    queries = (store.query().sel(state), store.query().sel(state.creation_view()).cdefs().stored())
    reads = []
    for name in ("read_state_ref_record", "read_stored_root_record"):
        original = getattr(store, name)

        def read(*args, _original=original, **kwargs):
            reads.append(1)
            return _original(*args, **kwargs)

        monkeypatch.setattr(store, name, read)
    for query in queries:
        with pytest.raises(QueryVerifyBudgetExceeded):
            query.max_verify(0).exists()
        assert reads == []
        assert query.max_verify(3).exists()
        reads.clear()


@pytest.mark.parametrize("unstable", (False, True))
def test_u6_recapture_retains_spent_admissions(tmp_path, monkeypatch, unstable):
    """An unchanged demand replacement and an instability retry both readmit."""

    from dryml.core.query.model import QueryVerifyBudgetExceeded

    store = DirStore(tmp_path / "store", query_index="memory")
    state = Repo(store).save_object(ExecutionLeaf("member"))
    base = store.query().where(field("object").missing())
    demanded = base.where(field("lineage", "creation_status").eq("unknown"))

    def terminal(limit):
        Repo(store).set_metadata(state.object_view(), {}, scope="object")
        replaced = False

        def evaluate(capture):
            nonlocal replaced
            base._collect_with_capture(capture)
            if not replaced:
                replaced = True
                if unstable:
                    Repo(store).set_metadata(state.object_view(), {"changed": True}, scope="object")
                capture.capture_store(store, metadata_scopes=frozenset(("object", "lineage")))
            return demanded._collect_with_capture(capture)

        return base.max_verify(limit)._run_terminal(evaluate, return_capture=True)

    if unstable:
        with pytest.raises(QueryError, match="changed"):
            terminal(6)
        return
    result, capture = terminal(6)
    assert capture.verified_count == 6
    assert capture.capture_rounds == 5 and capture.demand_recaptures == 1
    assert capture.instability_retries == int(unstable)
    assert capture.store_scan_count == 1
    with pytest.raises(QueryVerifyBudgetExceeded):
        terminal(5)


def test_u6_operand_policy_conflicts_and_outer_overrides(tmp_path, monkeypatch):
    """All declared algebra policies must agree; outer replacements govern IO."""

    value = Definition(ExecutionLeaf, "member").concretize()
    fixed = IdentitySet((value,)).query()
    for changed in (fixed.refresh(False), fixed.scan_policy("forbid"), fixed.require_indexed(),
                    fixed.max_verify(1), fixed.max_depth(1), fixed.max_witnesses(1)):
        with pytest.raises(QueryDomainError, match="conflicting execution policies"):
            fixed.union(changed)

    store = DirStore(tmp_path / "store", query_index="sqlite")
    Repo(store).save_object(ExecutionLeaf("member"))
    index = store.open_query_index()
    refreshes = []
    monkeypatch.setattr(index, "refresh", lambda *a, **kw: refreshes.append(1))
    inner = store.query().refresh(True).scan_policy("forbid")
    outer = inner.union(inner).refresh(False).scan_policy("allow").max_verify(3)
    assert outer.count() == 3
    assert refreshes == []
    scans = []
    original = store._validated_definition_root_inventory

    def inventory():
        scans.append(1)
        return original()

    monkeypatch.setattr(store, "_validated_definition_root_inventory", inventory)
    from dryml.core.query.model import QueryWouldScanError

    with pytest.raises(QueryWouldScanError):
        outer.scan_policy("forbid").exists()
    assert scans == []


def test_u6_repeated_fixed_bounded_stages_do_not_replace_the_cut():
    """A source-free bounded operand is refinement, not a paging fallback."""

    from dataclasses import replace

    values = tuple(Definition(ExecutionLeaf, str(i)).concretize() for i in range(3))
    branch = replace(IdentitySet(values).query(), take_limit=1)
    query = branch.union(branch).max_verify(3)
    assert query.count() == 1
    assert query.explain(analyze=True).verified_count == 3


def test_u6_selective_state_miss_readmits_complete_inventory(tmp_path):
    """A direct miss spends one admission before complete embedded knowledge."""

    from dryml.core.query.model import QueryVerifyBudgetExceeded

    origin = DirStore(tmp_path / "origin", query_index="memory")
    state = Repo(origin).save_object(ExecutionStatefulLeaf("embedded"))
    store = DirStore(tmp_path / "store", query_index="memory")
    # One definition embeds the StateRef, its ObjectRef, and its definition.
    from dryml.core.links import DefLink
    from dryml.core.cdef_graph import EdgeKind
    store.publish_node(Definition(ExecutionLeaf, DefLink.finalized(EdgeKind.REF, state)).concretize())
    query = store.query().sel(state)
    answer, capture = query.max_verify(4)._run_terminal(
        lambda cut: next(query._iter_terminal_members(cut), None) is not None,
        return_capture=True,
    )
    assert answer and capture.verified_count == 4
    assert capture.store_scan_count == 1
    with pytest.raises(QueryVerifyBudgetExceeded):
        query.max_verify(3).exists()


def test_u6_selective_cdef_miss_readmits_complete_inventory(tmp_path):
    """A missing direct marker cannot refund its abandoned selective admission."""

    from dryml.core.query.model import QueryVerifyBudgetExceeded

    store = DirStore(tmp_path / "store", query_index="memory")
    value = Definition(ExecutionLeaf, "alias-only").concretize()
    _name(store, "chosen", value)
    query = store.query().sel(value).cdefs().stored()
    explanation = query.max_verify(3).explain(analyze=True)
    assert explanation.verified_count == 1
    assert explanation.store_scan_count == 1
    with pytest.raises(QueryVerifyBudgetExceeded):
        query.max_verify(0).exists()


def test_u7_bounded_stage_reuses_existing_complete_cut(tmp_path, monkeypatch):
    """A later bounded stage must not inspect newer authority than retained facts."""

    from dataclasses import replace

    store = DirStore(tmp_path / "store", query_index="sqlite")
    root = Definition(ExecutionLeaf, "root").concretize()
    main = Definition(ExecutionLeaf, "main-only").concretize()
    store.publish_node(root)
    store.install_creation(main)
    store.rebuild_query_index()
    query = store.query().cdefs().stored().refresh(False)
    bounded = replace(query, take_limit=1)

    def evaluate(capture):
        assert query._collect_with_capture(capture).one().graph_equal(root)
        _name(store, "main", main)
        return bounded._collect_with_capture(capture)

    monkeypatch.setattr(store.open_query_index(), "_retained_read_view",
                        lambda: pytest.fail("bounded stage reopened an existing complete cut"))
    with pytest.raises(QueryError, match="changed"):
        query.max_verify(2)._run_terminal(evaluate, return_capture=True)
