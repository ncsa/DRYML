"""Metadata-only proofs for creation, compatibility and operation projections."""

import json

import pytest

from dryml.core import Definition, ObjectId, ObjectRef, Serializable, StateRef, Ref, QuotedDef
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.node_codec import decode_node_graph, encode_node_graph
from dryml.core.node_prepare import NodePreparation
from dryml.core.state_values import LocalState, StateKey
from dryml.core.utils.graph.path import GraphPath, Parameter, Index

CHILD = GraphPath((Parameter("child"),))


class ViewModel(Serializable):
    def __init__(self, width=4, child=None, initializer: Ref[QuotedDef] | None = None):
        self.width = width
        self.child = child
        self.initializer = initializer


def model(**kwargs):
    return Definition(ViewModel, **kwargs).concretize()


def selection(node, char="a", codec="pkl"):
    return LocalState(StateKey(node.creation_digest(), codec + "-" + char * 64),
                      node.creation_view())


def test_three_views_have_independent_authority_and_domains():
    from dryml.core.node_views import CREATION_VIEW, COMPATIBILITY_VIEW, OBJECT_PROJECTION_VIEW

    child = model().with_identity(ObjectId())
    first = model(child=child.with_state(selection(child)))
    second = model(child=child.with_state(selection(child, "b")))
    assert first.creation_digest() != second.creation_digest()
    assert first.compatibility_fingerprint() == second.compatibility_fingerprint()
    assert first.compatibility_view().graph_equal(second.compatibility_view())
    assert first.object_projection_digest() == second.object_projection_digest()
    assert first.object_projection_equal(second)
    assert [view.name for view in (CREATION_VIEW, COMPATIBILITY_VIEW, OBJECT_PROJECTION_VIEW)] == [
        "dryml-creation-cdef/v1", "dryml-state-compatibility/v1", "dryml-object-projection/v2"]
    for view in (CREATION_VIEW, COMPATIBILITY_VIEW, OBJECT_PROJECTION_VIEW):
        eager = NodePreparation((first,), view=view, scheduler="eager")
        deferred = NodePreparation((first,), view=view)
        assert eager.exact_key(eager.roots[0]) == deferred.exact_key(deferred.roots[0])


def test_foreign_origin_equal_bytes_keep_full_key_and_destination_creation():
    child = model().with_identity(ObjectId())
    donor = model(child=child.with_state(selection(child)))
    destination = model(child=child.with_state(selection(child, "b")))
    foreign, own = selection(donor), selection(destination)
    assert foreign.key.codec_hash == own.key.codec_hash
    assert foreign.key != own.key
    assert foreign.validate_for(destination, codec="pkl") is foreign
    applied = destination.with_realization(states={"$": foreign})
    assert applied.state is foreign
    assert applied.creation_digest() == destination.creation_digest()
    assert selection(applied).key == own.key
    assert foreign.creation_cdef.graph_equal(donor.creation_view())


@pytest.mark.parametrize("change", ["width", "sharing", "role", "quote", "class"])
def test_incompatible_configuration_is_denied_before_payload_hooks(change):
    child = model().with_identity(ObjectId())
    quote = child.with_state(selection(child)).quote()
    donor = model(child=[child, child], initializer=quote)
    if change == "width":
        destination = model(width=8, child=[child, child], initializer=quote)
    elif change == "sharing":
        destination = model(child=[child, model()], initializer=quote)
    elif change == "role":
        destination = model(child=[DefLink.finalized(EdgeKind.REF, child), child], initializer=quote)
    elif change == "quote":
        destination = model(child=[child, child], initializer=child.with_state(selection(child, "b")).quote())
    else:
        destination = Definition(Serializable).concretize()
    state = selection(donor)
    retained = destination.with_state(state)
    assert retained.state is state  # Binding is metadata, not admission.
    hooks = []
    with pytest.raises(ValueError, match="incompatible"):
        state.validate_for(destination, codec="pkl")
        hooks.append("restore")
    assert hooks == []


def test_malformed_creation_and_codec_rejected_without_payload_access():
    target = model()
    state = selection(target)
    with pytest.raises(ValueError, match="codec"):
        state.validate_for(target, codec="other")
    forged = LocalState(StateKey("0" * 64, state.key.codec_hash), state.creation_cdef)
    with pytest.raises(ValueError, match="creation.*digest"):
        forged.validate_for(target, codec="pkl")
    dirty = LocalState(state.key, target.with_state(state))
    with pytest.raises(ValueError, match="normalized creation"):
        dirty.validate_for(target, codec="pkl")
    with pytest.raises(TypeError, match="concrete|Concrete"):
        state.validate_for(Definition(ViewModel), codec="pkl")


def test_prefilter_collision_never_authorizes_incompatible_state_or_projection(monkeypatch):
    import dryml.core.node_prepare as keys

    donor, wrong = model(), model(width=8)
    state = selection(donor)
    monkeypatch.setattr(keys, "_digest", lambda *args: "0" * 64)
    # Recompute a valid origin address under the injected hash implementation.
    state = selection(donor)
    assert donor.compatibility_fingerprint() == wrong.compatibility_fingerprint()
    assert donor.object_projection_digest() == wrong.object_projection_digest()
    with pytest.raises(ValueError, match="incompatible"):
        state.validate_for(wrong, codec="pkl")
    assert not donor.object_projection_equal(wrong)


def test_capture_retains_construction_pins_and_excludes_own_recursive_key():
    child = model()
    s0, s1 = selection(child), selection(child, "b")
    parent = model(child=child.with_state(s0))
    captured = parent.with_realization(
        identities={"$": ObjectId(), CHILD: ObjectId()},
        states={CHILD: s1})
    own = selection(captured)
    captured = captured.with_realization(states={"$": own})
    assert captured.creation_digest() == parent.creation_digest()
    assert captured.creation_view().at(CHILD).state is s0
    assert captured.at(CHILD).state is s1
    assert captured.creation_view().object_id is None
    assert captured.creation_view().state is None
    assert captured.creation_view().at(CHILD).object_id is None
    decoded = decode_node_graph(json.loads(json.dumps(encode_node_graph(NodePreparation((captured,))))))
    assert decoded.creation_digest() == parent.creation_digest()
    assert decoded.creation_view().at(CHILD).state == s0
    assert decoded.at(CHILD).state == s1


def test_restore_adoption_checks_original_projection_then_recapture_uses_adopted_pins():
    child = model().with_identity(ObjectId())
    child0 = child.with_state(selection(child))
    child1 = child.with_state(selection(child, "b"))
    identity = ObjectId()
    receiver = model(child=DefLink.finalized(EdgeKind.REF, child0)).with_identity(identity)
    incoming = model(child=DefLink.finalized(EdgeKind.REF, child1)).with_identity(identity)
    saved = incoming.with_realization(states={"$": selection(incoming)})
    adopted = receiver.adopt_saved_view(saved)
    assert adopted.object_id == receiver.object_id
    assert adopted._occurrence is receiver._occurrence
    assert adopted.at(CHILD).state == child1.state
    assert adopted.creation_digest() == incoming.creation_digest()
    assert adopted.creation_digest() != receiver.creation_digest()
    assert selection(adopted).creation_cdef.graph_equal(incoming.creation_view())
    restored = decode_node_graph(encode_node_graph(NodePreparation((adopted,))))
    assert restored.creation_digest() == incoming.creation_digest()
    wrong = saved.replace_bindings(identities={"$": ObjectId()})
    with pytest.raises(ValueError, match="original.*projection"):
        receiver.adopt_saved_view(wrong)
    assert receiver.at(CHILD).state == child0.state


def test_legacy_reference_states_weaken_recursively_but_quotes_and_ids_stay_exact():
    child = model()
    reference = ObjectRef(child, {GraphPath(): ObjectId()})
    first = StateRef(reference, {GraphPath(): "pkl-" + "a" * 64})
    second = StateRef(reference, {GraphPath(): "pkl-" + "b" * 64})
    a = model(child=DefLink.finalized(EdgeKind.REF, first))
    b = model(child=DefLink.finalized(EdgeKind.REF, second))
    assert a.object_projection_equal(b)
    assert a.creation_digest() != b.creation_digest()
    assert selection(a).validate_for(b, codec="pkl")
    assert not model(initializer=Definition(ViewModel, child=first).quote()).object_projection_equal(
        model(initializer=Definition(ViewModel, child=second).quote()))
    other = ObjectRef(child, {GraphPath(): ObjectId()})
    assert not a.object_projection_equal(model(child=DefLink.finalized(EdgeKind.REF, other)))


def test_named_views_and_admission_are_pure_and_quote_preserves_ordinary_ids(monkeypatch):
    from dryml.core.symbol import ImportRef

    donor = model(initializer=Definition(ViewModel, child=ObjectId()).quote())
    state = selection(donor)
    same = donor.with_realization(identities={"$": ObjectId()}, states={"$": state})
    different = model(initializer=Definition(ViewModel, child="other").quote())

    def unexpected(*args, **kwargs):
        pytest.fail("metadata view/admission entered runtime effects")

    monkeypatch.setattr(ImportRef, "resolve", unexpected)
    monkeypatch.setattr(ObjectId, "__init__", unexpected)
    monkeypatch.setattr(Serializable, "restore_state_from_dir_imp", unexpected)
    assert state.validate_for(same, codec="pkl") is state
    assert same.creation_digest() == donor.creation_digest()
    assert same.compatibility_view().graph_equal(donor.compatibility_view())
    assert not same.object_projection_equal(donor)
    assert not donor.object_projection_equal(different)


def test_adoption_preserves_owned_aliases_and_rejects_quoted_checkpoint_changes():
    child = model().with_identity(ObjectId())
    child0, child1 = child.with_state(selection(child)), child.with_state(selection(child, "b"))
    receiver = model(child=[child0, child0]).with_identity(ObjectId())
    supplied = receiver.with_realization(states={GraphPath((Parameter("child"), Index(0))): child1.state})
    supplied = supplied.with_realization(states={"$": selection(supplied)})
    adopted = receiver.adopt_saved_view(supplied)
    left, right = adopted.parameters["child"]
    assert left is right
    assert left.object_id == child.object_id
    assert left._occurrence is receiver.parameters["child"][0]._occurrence
    assert left.state == child1.state
    assert adopted.creation_view().parameters["child"][0].state == child0.state
    round_trip = decode_node_graph(encode_node_graph(NodePreparation((adopted,))))
    assert round_trip.creation_digest() == adopted.creation_digest()
    assert round_trip.parameters["child"][0] is round_trip.parameters["child"][1]
    quote_receiver = model(initializer=child0.quote()).with_identity(ObjectId())
    quote_supplied = model(initializer=child1.quote()).with_identity(quote_receiver.object_id)
    quote_supplied = quote_supplied.with_realization(states={"$": selection(quote_supplied)})
    with pytest.raises(ValueError, match="original.*projection"):
        quote_receiver.adopt_saved_view(quote_supplied)


def test_unordered_adoption_and_compatibility_preserve_cross_member_aliases_on_hash_ties(monkeypatch):
    import dryml.core.node_prepare as keys
    from dryml.core.node import _owned_nodes

    a, b = model().with_identity(ObjectId()), model().with_identity(ObjectId())
    receiver = model(child=[{a, b}, a]).with_identity(ObjectId())
    supplied = receiver.copy_graph()
    supplied = supplied.with_realization(states={
        path: selection(node) for path, node in _owned_nodes(supplied)})
    state = selection(receiver)
    wrong = model(child=[{a, b}, model()])
    monkeypatch.setattr(keys, "_digest", lambda *args: "0" * 64)
    assert receiver.compatibility_fingerprint() == wrong.compatibility_fingerprint()
    assert not receiver.compatibility_view().graph_equal(wrong.compatibility_view())
    adopted = receiver.adopt_saved_view(supplied)
    members, alias = adopted.parameters["child"]
    assert alias in members
    assert alias._occurrence is a._occurrence
    assert {item.object_id for item in members} == {a.object_id, b.object_id}
    assert state.creation_cdef.graph_equal(receiver.creation_view())


def test_nested_factory_reference_states_weaken_but_ordinary_identity_data_does_not():
    from dryml.core.factory import FactorySpec
    from dryml.core.config import ConfigRef

    reference = ObjectRef(model(), {GraphPath(): ObjectId()})
    first = StateRef(reference, {GraphPath(): "pkl-" + "a" * 64})
    second = StateRef(reference, {GraphPath(): "pkl-" + "b" * 64})
    a = model(child=FactorySpec("builtins:tuple", ConfigRef("state", first)))
    b = model(child=FactorySpec("builtins:tuple", ConfigRef("state", second)))
    assert a.object_projection_equal(b)
    assert selection(a).validate_for(b, codec="pkl")
    first_id, second_id = model(child=ObjectId()), model(child=ObjectId())
    assert not first_id.compatibility_view().graph_equal(second_id.compatibility_view())
    assert not first_id.object_projection_equal(second_id)
