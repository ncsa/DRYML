"""Focused U4 coverage for private typed Query V3 relationship traversal."""

from __future__ import annotations

import pytest

from dryml.core import Definition, Object, ObjectId, ObjectRef, Serializable, StateRef
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.query.identity import IdentitySet
from dryml.core.query import field
from dryml.core.query.identity import Occurrence, OccurrenceSet
from dryml.core.query.query import IdentityQuery
from dryml.core.query.relationships import (
    EdgePolicy,
    RelationshipDepthExceeded,
    RelationshipKind,
    RelationshipPath,
)
from dryml.core.utils.graph.path import GraphPath, Parameter


class V3TraversalLeaf(Serializable):
    """Minimal owned stateful child used by detached reference fixtures."""

    def __init__(self, name):
        self.name = name


class V3TraversalParent(Object):
    """Minimal parent retaining one direct materialized child definition."""

    def __init__(self, child):
        self.child = child


def _references():
    """Build a parent StateRef and its direct owned child projections."""

    child = Definition(V3TraversalLeaf, "child").concretize()
    root = Definition(V3TraversalParent, child).concretize()
    child_path = GraphPath((Parameter("child"),))
    object_ref = ObjectRef(root, {child_path: ObjectId(("root", "child"))})
    state = StateRef(object_ref, {child_path: "pkl-" + "1" * 64})
    return root, child, state, object_ref, state.at(child_path), object_ref.at(child_path), child_path


def test_v3_closure_uses_typed_associations_and_owned_state_adjacency():
    root, child, state, object_ref, child_state, child_object, _ = _references()
    query = IdentityQuery.from_set(IdentitySet((state,)))

    assert set(query.closure().collect()) == {
        state, object_ref, root, child, child_state, child_object,
    }
    assert set(query.closure(EdgePolicy.ASSOCIATIONS).collect()) == {
        state, object_ref, root,
    }
    assert set(query.closure(EdgePolicy.STATE_OBJECT).collect()) == {state, object_ref}


def test_v3_nested_requires_a_non_empty_typed_path_and_keeps_association_hops():
    root, _, state, object_ref, child_state, _, child_path = _references()
    query = IdentityQuery.from_set(IdentitySet((state,)))

    assert query.nested(state).count() == 0
    association = query.nested(object_ref, edges=EdgePolicy.STATE_OBJECT).one()
    assert association.path == RelationshipPath.from_hops(
        ((RelationshipKind.STATE_OBJECT, GraphPath()),),
    )
    occurrence = query.nested(child_state).one()
    assert occurrence.owner == state
    assert occurrence.target == child_state
    assert occurrence.path == RelationshipPath.from_hops(
        ((RelationshipKind.MATERIALIZE, child_path),),
    )
    assert occurrence.path.graph_path == child_path
    assert root not in {occurrence.target}
    with pytest.raises(RelationshipDepthExceeded, match="depth budget"):
        query.nested(child_state).max_depth(0).count()
    assert query.nested(object_ref).path(GraphPath()).count() == 1
    assert query.nested(object_ref).path(RelationshipPath()).count() == 0
    assert IdentityQuery.from_set(IdentitySet((object_ref,))).closure(
        EdgePolicy.STATE_OBJECT
    ).one() == object_ref


def test_v3_all_policy_keeps_reference_edges_and_shared_paths_distinct():
    child = Definition(V3TraversalLeaf, "child").concretize()
    reference_root = Definition(
        V3TraversalParent, DefLink.finalized(EdgeKind.REF, child),
    ).concretize()
    shared_root = Definition(V3TraversalParent, [child, child]).concretize()

    reference_query = IdentityQuery.from_set(IdentitySet((reference_root,))).nested(child)
    shared = IdentityQuery.from_set(IdentitySet((shared_root,))).nested(child).collect()

    assert reference_query.count() == 1
    assert reference_query.through(RelationshipKind.REFERENCE).count() == 1
    assert reference_query.max_occurrences(1).collect().bounded
    assert IdentityQuery.from_set(IdentitySet((reference_root,))).nested(
        child, edges=EdgePolicy.OWNED,
    ).count() == 0
    assert shared.count() == 2
    assert len({item.path for item in shared}) == 2


def test_v3_direct_projections_ignore_raw_caps_without_enumerating_occurrences(monkeypatch):
    root, child, _, _, _, _, _ = _references()
    query = IdentityQuery.from_set(IdentitySet((root,))).nested(child).max_occurrences(0)

    monkeypatch.setattr(
        "dryml.core.query.relationships.iter_relationship_occurrences",
        lambda *_args, **_kwargs: pytest.fail("direct projection enumerated raw occurrences"),
    )

    assert query.owners().one() == root
    assert query.targets().one() == child


def test_v3_bounded_roots_stay_bounded_through_refinement_and_relationships():
    root, child, state, _, _, _, _ = _references()
    roots = IdentityQuery.from_set(IdentitySet((state,), bounded=True))

    assert roots.state_refs().collect().bounded
    assert roots.closure().collect().bounded
    assert roots.nested(child).collect().bounded
    assert roots.nested(child).owners().collect().bounded
    assert roots.nested(child).targets().collect().bounded


def test_v3_occurrence_metadata_filters_targets_with_shared_authority(tmp_path):
    from dryml.core import Repo, SaveAnnotations
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = V3TraversalLeaf("child", repo=repo)
    state = repo.save_object(
        child, annotations=SaveAnnotations(object={"team": "selected"}),
    )
    roots = IdentityQuery.from_set(IdentitySet((state,)))
    nested = roots.nested().where(field("object", "team").eq("selected"), scope=store)

    assert nested.targets().one() == state.object
    assert nested.one().target == state.object
    assert roots.nested().where(field("object", "team").eq("missing"), scope=store).count() == 0


def test_v3_fixed_occurrence_metadata_requires_explicit_scope(tmp_path):
    from dryml.core import Repo, SaveAnnotations
    from dryml.core.query.model import QueryDomainError
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(
        V3TraversalLeaf("child", repo=repo),
        annotations=SaveAnnotations(object={"team": "selected"}),
    )
    result = OccurrenceSet((Occurrence(state, GraphPath((Parameter("edge"),)), state.object),))

    with pytest.raises(QueryDomainError, match="explicit scope"):
        result.query().where(field("object", "team").eq("selected"))
    assert result.query().where(field("object", "team").eq("selected"), scope=store).one().target == state.object


def test_v3_occurrence_selector_cannot_hide_invalid_later_metadata(tmp_path):
    from dryml.core import Repo, SaveAnnotations
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    invalid = repo.save_object(
        V3TraversalLeaf("invalid", repo=repo),
        annotations=SaveAnnotations(object={"score": "wrong-type"}),
    )
    selected = repo.save_object(
        V3TraversalLeaf("selected", repo=repo),
        annotations=SaveAnnotations(object={"score": 1}),
    )
    owner = Definition(V3TraversalParent, None).concretize()
    fixed = OccurrenceSet((
        Occurrence(owner, GraphPath((Parameter("first"),)), invalid.object),
        Occurrence(owner, GraphPath((Parameter("second"),)), selected.object),
    ))

    with pytest.raises(Exception, match="numeric field"):
        fixed.query().sel(selected.object).where(field("object", "score").lt(3), scope=store).exists()

    assert fixed.query().sel(selected.object).sel(invalid.object).count() == 0


def test_v3_occurrence_late_metadata_demand_restarts_root_evidence(tmp_path, monkeypatch):
    from dryml.core import Repo, SaveAnnotations
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(
        V3TraversalLeaf("candidate", repo=repo),
        annotations=SaveAnnotations(object={"team": "old"}),
    )
    lineage = repo.get_lineage_metadata(state.object)
    roots = IdentityQuery.from_store(store).state_refs().where(
        field("object", "team").eq("old")
    )
    original = IdentityQuery._collect_with_capture
    mutated = False

    def move_after_root(self, capture):
        nonlocal mutated
        result = original(self, capture)
        if self is roots and not mutated:
            mutated = True
            repo.set_metadata(state.object, {"team": "new"}, store=store)
        return result

    monkeypatch.setattr(IdentityQuery, "_collect_with_capture", move_after_root)
    nested = roots.nested().where(field("object", "team").eq("new")).where(
        field("lineage", "creation_status").eq(lineage.creation_status)
    )

    assert nested.count() == 0
    assert mutated
