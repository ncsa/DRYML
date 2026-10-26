"""Current Node recipe selection and exact quotation projection contracts."""

from dryml.core import Definition, IdentitySet, LocalState, ObjectId, QueryPredicate, Serializable, StateKey


class ProjectionLeaf(Serializable):
    """Stateful leaf used to retain exact Node facets in projection tests."""

    def __init__(self, value):
        self.value = value


def test_node_query_concrete_recipe_omits_facets_but_quoted_checkpoint_is_exact():
    """Recipe anchors do not pin identities; quotation retains exact saved data."""
    creation = Definition(ProjectionLeaf, 2).concretize()
    state = LocalState(StateKey(creation.creation_digest(), "pkl-" + "a" * 64), creation)
    candidate = creation.with_identity(ObjectId(("selected",))).with_state(state)
    assert QueryPredicate(creation).matches(candidate)
    assert QueryPredicate(Definition(value=2)).matches(candidate)
    assert not QueryPredicate(Definition(recipe=creation.quote())).matches(
        Definition(recipe=candidate.quote()))
    assert not QueryPredicate(Definition(child=creation, recipe=creation.quote())).matches(
        Definition(child=candidate, recipe=candidate.quote()))
    assert not QueryPredicate(candidate).matches(creation)
    changed = candidate.without_state()
    assert candidate.object_projection_equal(changed)
    assert not QueryPredicate(candidate.quote().value).matches(changed)
    categorical = candidate.categorical(drop=("value",))
    assert categorical.object_id == candidate.object_id and categorical.state == candidate.state
    assert QueryPredicate(categorical).matches(candidate)
    assert not QueryPredicate(categorical).matches(creation)


def test_node_object_projection_merges_state_replicas_and_preserves_evidence():
    """Fixed Node projections keep identity distinctions and both source witnesses."""
    from dryml.core import SourceEvidence

    creation = Definition(ProjectionLeaf, 2).concretize()
    identified = creation.with_identity(ObjectId())
    first = identified.with_state(LocalState(StateKey(creation.creation_digest(), "pkl-" + "a" * 64), creation))
    second = identified.with_state(LocalState(StateKey(creation.creation_digest(), "pkl-" + "b" * 64), creation))
    other = second.replace_bindings(identities={"$": ObjectId()})
    first_source, second_source = (SourceEvidence.from_source(name) for name in ("first", "second"))
    projected = IdentitySet(((first, first_source), (second, second_source), other)).object_projection()

    assert projected.count() == 2
    assert projected.evidence_for(identified.object_view()).sources == frozenset((first_source, second_source))
    assert all(not node.state_spec() for node in projected)
