"""Focused U5 contracts for private Query V3 execution terminals."""

import pytest

from dryml.core import Definition, Object, Repo, SaveAnnotations, Serializable
from dryml.core.query import field
from dryml.core.query.identity import IdentitySet, OccurrenceSet
from dryml.core.query.model import QueryCardinalityError, QueryDomainError, QueryIndexUnavailable
from dryml.core.query.query import IdentityQuery
from dryml.core.query.source import SourceCapture, _SourceRecapture
from dryml.core.store.dir import DirStore


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

    assert set(query.state_refs().take(1).collect()) <= {first_state, second_state}


def test_u5_algebra_reuses_one_cut_and_preserves_complete_source_evidence(tmp_path, monkeypatch):
    """Duplicate algebra branches share a cut while replicas remain one identity."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(ExecutionLeaf("shared", repo=repo))
    calls = 0
    original = store.iter_definition_records

    def counted():
        nonlocal calls
        calls += 1
        yield from original()

    monkeypatch.setattr(store, "iter_definition_records", counted)
    base = IdentityQuery.from_store(store).state_refs()
    union = base.union(base)

    collected = union.collect()

    assert collected.one() == state
    assert len(collected.evidence_for(state).sources) == 1
    assert calls == 1
    assert union.explain(analyze=True).source_cuts == 1
    assert union.intersection(base).one() == state


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
    assert query.sel(first).one_or_none() == first


def test_u5_take_retains_requested_limit_and_canonical_prefix(tmp_path):
    """A requested prefix is bounded metadata, not a complete-result claim."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    states = [
        repo.save_object(ExecutionLeaf(name, repo=repo))
        for name in ("first", "second", "third")
    ]

    result = IdentityQuery.from_store(store).state_refs().take(2).collect()

    assert result.bounded
    assert result.requested_limit == 2
    assert result.diagnostic().requested_limit == 2
    assert tuple(result) == tuple(sorted(states, key=lambda state: state.digest()))[:2]
    assert result.query().collect().requested_limit == 2
    assert IdentityQuery.from_store(store).state_refs().take(0).count() == 0
    assert not IdentityQuery.from_store(store).state_refs().take(0).exists()
    assert (
        IdentityQuery.from_store(store).state_refs().take(1).union(
            IdentityQuery.from_store(store).state_refs()
        ).collect().bounded
    )


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
    query = IdentityQuery.from_set(IdentitySet((first, second))).take(1)
    monkeypatch.setattr(
        cdef_codec, "cdef_graph_hash",
        lambda root: (_ for _ in ()).throw(AssertionError("unnecessary ordering encoding")),
    )

    assert query.collect().count() == 1


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
    lineage = repo.get_lineage_metadata(state.object)
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
            repo.set_metadata(state.object, {"team": new}, store=store)
        return result

    monkeypatch.setattr(IdentityQuery, "_evaluated_entries", mutate_after_left)

    assert left.intersection(right).count() == 0
    assert mutated


def test_u5_late_demand_converges_without_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(
        ExecutionLeaf("candidate", repo=repo),
        annotations=SaveAnnotations(object={"team": "old"}),
    )
    lineage = repo.get_lineage_metadata(state.object)
    left = IdentityQuery.from_store(store).object_refs().where(
        field("object", "team").eq("old")
    )
    right = IdentityQuery.from_store(store).object_refs().where(
        field("lineage", "creation_status").eq(lineage.creation_status)
    )

    assert left.intersection(right).one() == state.object


def test_u5_four_static_metadata_demands_do_not_use_instability_retries(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(
        ExecutionLeaf("candidate", repo=repo),
        annotations=SaveAnnotations(object={"team": "old"}),
    )
    lineage = repo.get_lineage_metadata(state.object)
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
    original = SourceCapture.capture_store
    calls = 0

    def unstable(self, source, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise _SourceRecapture(True)
        return original(self, source, **kwargs)

    monkeypatch.setattr(SourceCapture, "capture_store", unstable)
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
    assert left.union(right).count() == 0


def test_u5_exact_miss_then_publication_does_not_mix_with_broad_cut(tmp_path, monkeypatch):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo((first, second))
    value = ExecutionLeaf("candidate", repo=repo)
    state = repo.save_object(value, store=second)
    left = IdentityQuery.from_store(first).sel(state).state_refs()
    right = IdentityQuery.from_store(first).state_refs()
    original = IdentityQuery._evaluated_entries
    published = False

    def publish_after_miss(self, capture):
        nonlocal published
        result = original(self, capture)
        if self is left and not published:
            published = True
            assert repo.save_object(value, store=first, source_store=second) == state
        return result

    monkeypatch.setattr(IdentityQuery, "_evaluated_entries", publish_after_miss)
    assert left.intersection(right).one() == state


def test_v3_exact_stored_cdef_positive_does_not_enumerate_definitions(tmp_path):
    original = DirStore(tmp_path / "store")
    repo = Repo(original)
    saved = repo.save_object(ExecutionLeaf("target", repo=repo))

    class CountingStore(DirStore):
        def iter_definition_records(self):
            pytest.fail("exact stored CDef lookup enumerated definition records")

    store = CountingStore(original.base_dir)
    assert IdentityQuery.from_store(store).sel(saved.definition).cdefs().stored().one() == saved.definition
    assert IdentityQuery.from_store(store).sel(saved.definition).cdefs().stored().scan_policy("forbid").one() == saved.definition


def test_v3_alias_only_stored_cdef_falls_back_to_complete_authority(tmp_path):
    from dryml.core import ObjectId, ObjectRef
    from dryml.core.query.model import QueryWouldScanError
    from dryml.core.store.records import DefinitionRecord, ObjectAliasRecord
    from dryml.core.utils.graph.path import GraphPath

    store = DirStore(tmp_path / "store")
    cdef = Definition(ExecutionStatefulLeaf, "alias-only").concretize()
    store.write_definition_record(DefinitionRecord(cdef), stored_root=False)
    store.write_object_alias(ObjectAliasRecord("selected", ObjectRef(cdef, {GraphPath(): ObjectId()})))

    query = IdentityQuery.from_store(store).sel(cdef).cdefs().stored()
    assert query.one() == cdef
    with pytest.raises(QueryWouldScanError, match="inventory scan"):
        query.scan_policy("forbid").count()
