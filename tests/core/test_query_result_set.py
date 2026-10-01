import pytest

from dryml.core import Definition, Generator, Object, ObjectId, ObjectRef, Repo, Serializable, SKIP_ARGS, StateRef
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.domains import UniformFromSet
from dryml.core.template import Par
from dryml.core.query import QueryCardinalityError, QueryDomainError
from dryml.core.query.model import (
    ContainmentContext,
    ContainmentHop,
    ReferenceOccurrence,
)
from dryml.core.query.result import OccurrenceResultSet, QueryBackedDefinitionResultSet
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Key

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


class ResultLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class ResultParent(Serializable):
    def __init__(self, child):
        super().__init__()
        self.child = child


class ResultReferenceLeaf(Serializable):
    def __init__(self, name):
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


def _result_exact_selector():
    """Return a minimal exact selector used to reject reference-value refinement."""

    return Generator(
        Definition(ResultLeaf, Par("name")),
        {"name": UniformFromSet(("target",))},
    ).support_selector()


def test_result_set_cardinality_helpers(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    first = ResultLeaf("first", repo=repo)
    second = ResultLeaf("second", repo=repo)
    repo.save_object(first)
    repo.save_object(second)

    many = repo.find_defs(None)
    empty = repo.find_defs(Definition(ResultLeaf, "missing"), refresh=False)
    one = repo.find_defs(Definition(ResultLeaf, "first"), refresh=False)

    assert many.count() == len(many)
    assert many.exists()
    with pytest.raises(QueryCardinalityError):
        many.one()
    with pytest.raises(QueryCardinalityError):
        many.one_or_none()
    with pytest.raises(QueryCardinalityError):
        empty.one()
    assert empty.one_or_none() is None
    assert one.one() == first.definition


def test_refine_preserves_nested_nonmaterializable_domain(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = ResultLeaf("child", repo=repo)
    parent = ResultParent(child, repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    nested_defs = repo2.query(Definition(ResultLeaf, SKIP_ARGS)).nested().definitions().defs()
    refined = nested_defs.refine(Definition(ResultLeaf, "child"))

    assert refined.domain == "nested-definitions"
    assert not refined.materializable
    assert list(refined) == [child.definition]
    with pytest.raises(QueryDomainError):
        refined.objects()


def test_result_snapshot_does_not_gain_later_indexed_results(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    first = ResultLeaf("first", repo=repo)
    repo.save_object(first)
    snapshot = repo.find_defs(None)

    second = ResultLeaf("second", repo=repo)
    repo.save_object(second)

    assert list(snapshot) == [first.definition]
    assert repo.find_defs(None, refresh=False).count() == 2


def test_result_set_refinement_never_expands_original_set(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    first = ResultLeaf("first", repo=repo)
    second = ResultLeaf("second", repo=repo)
    repo.save_object(first)
    repo.save_object(second)

    broad = repo.find_defs(Definition(ResultLeaf, "first"))
    refined = broad.refine(Definition(ResultLeaf, SKIP_ARGS))

    assert list(refined) == [first.definition]


def test_union_and_intersection_are_deterministic_and_deduplicate(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    first = ResultLeaf("first", repo=repo)
    second = ResultLeaf("second", repo=repo)
    repo.save_object(first)
    repo.save_object(second)

    first_rs = repo.find_defs(Definition(ResultLeaf, "first"))
    all_rs = repo.find_defs(Definition(ResultLeaf, SKIP_ARGS))

    assert first_rs.union(first_rs).count() == 1
    assert list(all_rs.intersection(first_rs)) == [first.definition]


def test_union_rejects_incompatible_result_domains(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = ResultLeaf("child", repo=repo)
    parent = ResultParent(child, repo=repo)
    repo.save_object(parent)

    stored = repo.find_defs(None, scope="stored")
    nested = repo.query(Definition(ResultLeaf, SKIP_ARGS)).nested().definitions().defs()

    with pytest.raises(ValueError, match="different domains"):
        stored.union(nested)
    with pytest.raises(ValueError, match="different domains"):
        nested.intersection(stored)


def test_resultset_replica_metadata_is_snapshotted(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    obj = ResultLeaf("snap", repo=repo)
    repo.save_object(obj, store=store1)

    snapshot = repo.find_defs(None)
    assert len(snapshot.replicas(obj.definition)) == 1

    repo.save_object(obj, store=store2)
    current = repo.find_defs(None, refresh=False)

    assert len(snapshot.replicas(obj.definition)) == 1
    assert len(current.replicas(obj.definition)) == 2


def test_resultset_union_and_intersection_replica_metadata_is_commutative(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    obj = ResultLeaf("snap", repo=repo)
    repo.save_object(obj, store=store1)
    one_replica = repo.find_defs(None)
    repo.save_object(obj, store=store2)
    two_replicas = repo.find_defs(None, refresh=False)

    assert len(one_replica.union(two_replicas).replicas(obj.definition)) == 2
    assert len(two_replicas.union(one_replica).replicas(obj.definition)) == 2
    assert one_replica.union(two_replicas).replicas(obj.definition) == two_replicas.union(one_replica).replicas(obj.definition)
    assert one_replica.intersection(two_replicas).replicas(obj.definition) == two_replicas.intersection(one_replica).replicas(obj.definition)


def test_query_backed_resultset_preserves_adversarial_page_order(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    first = ResultLeaf("stream-first", repo=repo).definition
    second = ResultLeaf("stream-second", repo=repo).definition
    page_order = (second, first)

    def page_factory():
        for cdef in page_order:
            yield cdef, ()

    results = QueryBackedDefinitionResultSet(repo, page_factory, materializable=False)

    first_iteration = tuple(results)
    second_iteration = tuple(results)

    assert first_iteration == page_order
    assert second_iteration == first_iteration


def test_fixed_resultset_universe_rejects_domain_switch(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = ResultLeaf("child", repo=repo)
    parent = ResultParent(child, repo=repo)
    repo.save_object(parent)

    nested_defs = repo.query(Definition(ResultLeaf, SKIP_ARGS)).nested().definitions().defs()

    with pytest.raises(QueryDomainError, match="Cannot switch"):
        nested_defs.query(Definition(ResultLeaf, SKIP_ARGS)).stored().defs()


def test_containment_reference_projections_keep_complete_typed_identities(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    first = repo.save_object(ResultReferenceLeaf("same", repo=repo))
    second = ObjectRef(
        first.definition,
        {path: ObjectId() for path in first.object.objects},
    )
    first_state = StateRef(first.object, first.states)
    other_first_state = StateRef(
        first.object,
        {path: "pkl-" + "0" * 64 for path in first.states},
    )
    owner = ResultParent("owner", repo=repo).definition
    object_occurrences = OccurrenceResultSet(
        repo,
        (
            ReferenceOccurrence(
                owner, GraphPath((Key("first"),)), first.object,
                (ContainmentHop(GraphPath((Key("first"),)), EdgeKind.MATERIALIZE),),
            ),
            ReferenceOccurrence(
                owner, GraphPath((Key("second"),)), second,
                (ContainmentHop(GraphPath((Key("second"),)), EdgeKind.REF),),
            ),
        ),
        owner_replicas={owner: ()},
        containment=ContainmentContext(target_kind="object_ref", edges="all"),
    )
    state_occurrences = OccurrenceResultSet(
        repo,
        (
            ReferenceOccurrence(owner, GraphPath((Key("state"),)), first_state),
            ReferenceOccurrence(owner, GraphPath((Key("other-state"),)), other_first_state),
        ),
        owner_replicas={owner: ()},
        containment=ContainmentContext(target_kind="state_ref", edges="all"),
    )

    object_results = object_occurrences.object_refs()
    state_results = state_occurrences.state_refs()

    assert set(object_results) == {first.object, second}
    assert set(state_results) == {first_state, other_first_state}
    assert object_results.count() == 2
    assert state_results.count() == 2
    with pytest.raises(QueryCardinalityError):
        object_results.one()
    with pytest.raises(QueryCardinalityError):
        state_results.one_or_none()
    with pytest.raises(QueryDomainError, match="StateRef"):
        state_occurrences.object_refs()


@pytest.mark.parametrize("projection", ["object_refs", "state_refs"])
def test_fixed_exact_reference_results_reject_exact_selector_refinement(
        tmp_path, projection):
    """Reference targets cannot become exact-selector CDef candidates."""

    repo = Repo(DirStore(tmp_path / "store"))
    saved = repo.save_object(ResultReferenceLeaf("target", repo=repo))
    target = saved.object if projection == "object_refs" else StateRef(saved.object, saved.states)
    owner = ResultParent("owner", repo=repo).definition
    occurrences = OccurrenceResultSet(
        repo,
        (ReferenceOccurrence(owner, GraphPath((Key("target"),)), target),),
        owner_replicas={owner: ()},
        containment=ContainmentContext(
            target_kind="object_ref" if projection == "object_refs" else "state_ref",
            edges="all",
        ),
    )

    with pytest.raises(QueryDomainError, match="Exact selector"):
        getattr(occurrences, projection)().query(_result_exact_selector())


def test_containment_result_unions_keep_owner_witness_ledgers_and_fixed_refinement(tmp_path, monkeypatch):
    repo = Repo(DirStore(tmp_path / "store"))
    first = repo.save_object(ResultLeaf("first", repo=repo))
    second = repo.save_object(ResultLeaf("second", repo=repo))
    owner = ResultParent("owner", repo=repo).definition
    context = ContainmentContext(target_kind="object_ref", edges="all", source_scope=("source",))
    first_replica = object()
    second_replica = object()

    def result_for(value, name, replica):
        occurrence = ReferenceOccurrence(
            owner,
            GraphPath((Key(name),)),
            value,
            (ContainmentHop(GraphPath((Key(name),)), EdgeKind.REF),),
        )
        return OccurrenceResultSet(
            repo,
            (occurrence,),
            owner_replicas={owner: (replica,)},
            containment=context,
        ).owners()

    first_owner = result_for(first.object, "first", first_replica)
    second_owner = result_for(second.object, "second", second_replica)
    combined = first_owner.union(second_owner)
    intersected = first_owner.intersection(second_owner)
    monkeypatch.setattr(
        repo._query_catalog,
        "refresh",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fixed universe scanned Store")),
    )
    refined = combined.query(first.object).nested(edges="all").owners().defs()

    assert list(combined) == [owner]
    assert set(combined.replicas(owner)) == {first_replica, second_replica}
    assert list(refined) == [owner]
    assert refined._containment_witnesses[0].target == first.object
    assert {item.target for item in intersected._containment_witnesses} == {
        first.object, second.object,
    }


def test_containment_result_context_rejects_incompatible_union_and_keeps_bounded_projection(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    target = repo.save_object(ResultLeaf("target", repo=repo)).object
    owner = ResultParent("owner", repo=repo).definition
    occurrence = ReferenceOccurrence(owner, GraphPath((Key("target"),)), target)
    all_edges = OccurrenceResultSet(
        repo,
        (occurrence,),
        owner_replicas={owner: ()},
        containment=ContainmentContext(target_kind="object_ref", edges="all"),
    )
    ref_edges = OccurrenceResultSet(
        repo,
        (occurrence,),
        owner_replicas={owner: ()},
        containment=ContainmentContext(target_kind="object_ref", edges="ref"),
    )

    projected = all_edges.object_refs()
    empty = all_edges.query(target).nested(edges="all").max_occurrences(0).object_refs()

    assert not projected._containment.bounded
    assert empty._containment.bounded
    assert empty.count() == 0
    assert empty.one_or_none() is None
    with pytest.raises(QueryCardinalityError):
        empty.one()
    with pytest.raises(ValueError, match="containment contexts"):
        all_edges.union(ref_edges)


def test_lazy_raw_containment_projection_requery_retains_visible_witnesses(tmp_path):
    """A capped lazy raw result remains a fixed view of its visible witnesses."""

    repo = Repo(DirStore(tmp_path / "store"))
    child = ResultLeaf("child", repo=repo)
    owner = ResultParent(child, repo=repo)
    repo.save_object(owner)

    raw = repo.query(child.definition).nested().max_occurrences(1).execute()
    definitions = raw.definitions()
    owners = raw.owners()

    assert definitions._containment.bounded
    assert list(definitions.query(child.definition).nested().definitions().defs()) == [
        child.definition,
    ]
    assert list(owners.query(owner.definition).nested().owners().defs()) == [
        owner.definition,
    ]

    reference = repo.save_object(ResultReferenceLeaf("reference", repo=repo))
    reference_owner = ResultParent(
        DefLink.finalized(EdgeKind.REF, reference.object), repo=repo,
    )
    repo.save_object(reference_owner)
    reference_raw = repo.query(reference.object).nested(edges="all").max_occurrences(1).execute()

    assert list(reference_raw.owners().query(reference.object).nested(edges="all").owners().defs()) == [
        reference_owner.definition,
    ]


def test_direct_owner_projection_retains_carrier_and_target_ledgers(tmp_path):
    """Direct owner projections support both owner and exact-target requery."""

    repo = Repo(DirStore(tmp_path / "store"))
    target = repo.save_object(ResultReferenceLeaf("target", repo=repo))
    owner = ResultParent(
        DefLink.finalized(EdgeKind.REF, target.object), repo=repo,
    )
    other = ResultParent("other", repo=repo)
    repo.save_object(owner)
    repo.save_object(other)

    owners = repo.query(target.object).nested(edges="all").owners().defs()

    assert list(owners.query(owner.definition).nested(edges="all").owners().defs()) == [
        owner.definition,
    ]
    assert list(owners.query(other.definition).nested(edges="all").owners().defs()) == []
    assert list(owners.query(target.object).nested(edges="all").owners().defs()) == [
        owner.definition,
    ]
    with pytest.raises(ValueError, match="containment contexts"):
        owners.union(repo.query(target.object).nested(edges="ref").owners().defs())


def test_direct_definition_projection_refines_the_target_carrier(tmp_path):
    """A direct definition projection never treats its enclosing owner as target."""

    repo = Repo(DirStore(tmp_path / "store"))
    target = ResultLeaf("target", repo=repo)
    owner = ResultParent(
        DefLink.finalized(EdgeKind.REF, target.definition), repo=repo,
    )
    repo.save_object(owner)

    definitions = repo.query(target.definition).nested(edges="all").definitions().defs()

    assert list(definitions.query(target.definition).nested(edges="all").definitions().defs()) == [
        target.definition,
    ]
    assert list(definitions.query(owner.definition).nested(edges="all").definitions().defs()) == []


def test_direct_definition_projection_keeps_owner_target_ledgers_across_replicas(tmp_path):
    """A target projection retains one witness for every contributing owner."""

    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    target = ResultLeaf("target", repo=repo)
    first_owner = ResultParent(
        DefLink.finalized(EdgeKind.REF, target.definition), repo=repo,
    )
    second_owner = ResultParent(
        {"target": DefLink.finalized(EdgeKind.REF, target.definition)}, repo=repo,
    )
    repo.save_object(first_owner, store=store1)
    repo.save_object(first_owner, store=store2)
    repo.save_object(second_owner, store=store2)

    definitions = repo.query(target.definition).nested(edges="all").definitions().defs()
    expected_owners = {first_owner.definition, second_owner.definition}

    assert {witness.owner for witness in definitions._containment_witnesses} == expected_owners
    assert {witness.target for witness in definitions._containment_witnesses} == {target.definition}
    assert len(definitions._containment_witnesses) == 2

    for result in (definitions.union(definitions), definitions.intersection(definitions)):
        refined = result.query(target.definition).nested(edges="all").definitions().defs()

        assert {witness.owner for witness in refined._containment_witnesses} == expected_owners
        assert {witness.target for witness in refined._containment_witnesses} == {target.definition}
        assert len(refined._containment_witnesses) == 2


def test_fixed_direct_containment_requery_rejects_projection_changes(tmp_path):
    """Fixed direct containment results cannot change their result carrier."""

    repo = Repo(DirStore(tmp_path / "store"))
    target = ResultLeaf("target", repo=repo)
    owner = ResultParent(
        DefLink.finalized(EdgeKind.REF, target.definition), repo=repo,
    )
    repo.save_object(owner)

    definitions = repo.query(target.definition).nested(edges="all").definitions().defs()
    owners = repo.query(target.definition).nested(edges="all").owners().defs()

    with pytest.raises(QueryDomainError, match="projection"):
        definitions.query(target.definition).nested(edges="all").owners()
    with pytest.raises(QueryDomainError, match="projection"):
        owners.query(owner.definition).nested(edges="all").definitions()


def test_direct_containment_results_retain_source_scope_for_unions_and_explanations(tmp_path):
    """Fixed direct projections neither widen source scope nor report live sources."""

    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    target = repo.save_object(ResultReferenceLeaf("target", repo=repo))
    owner = ResultParent(
        DefLink.finalized(EdgeKind.REF, target.object), repo=repo,
    )
    repo.save_object(owner, store=store1)
    repo.save_object(owner, store=store2)

    first = repo.query(target.object).nested(edges="all").in_store(store1).owners().defs()
    second = repo.query(target.object).nested(edges="all").in_store(store2).owners().defs()
    explanation = first.query(owner.definition).nested(edges="all").explain()

    assert explanation.containment_source_scope == (store1.catalog_key(),)
    with pytest.raises(ValueError, match="containment contexts"):
        first.union(second)
