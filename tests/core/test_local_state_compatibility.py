"""Metadata-only proofs for creation, compatibility and operation projections."""

import json

import pytest

from dryml.core import Definition, ObjectId, Serializable, Ref, QuotedDef
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


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("edit", ["identity", "state", "unpin"])
def test_realization_overlay_keeps_cross_quoted_aliases_exact(pinned, edit):
    child = model()
    initial = selection(child)
    if pinned:
        child = child.with_identity(ObjectId()).with_state(initial)
    bundle = Definition(ViewModel, child=[child, child], initializer=child.quote()).quote()
    root = model(child=[child, child], initializer=bundle)
    root = root.with_identity(ObjectId()).with_state(selection(root))
    child_path = GraphPath((Parameter("child"), Index(0)))
    identities = {child_path: ObjectId()} if edit == "identity" else None
    states = ({child_path: selection(child, "b")} if edit == "state" else
              {child_path: None} if edit == "unpin" else None)

    changed = root.with_realization(identities=identities, states=states)

    left, right = changed.parameters["child"]
    assert left is right
    assert changed.parameters["initializer"] is root.parameters["initializer"]
    assert changed.at(GraphPath((Parameter("initializer"),))) is bundle.value
    assert bundle.value.kwargs["child"][0] is bundle.value.kwargs["child"][1] is child
    assert left.object_id == (identities[child_path] if identities else child.object_id)
    assert left.state == (states[child_path] if states else child.state)
    assert changed.object_id == root.object_id and changed.state == root.state
    assert changed.construction_view is root
    assert changed.creation_digest() == root.creation_digest()
    if edit != "identity":
        assert changed.object_projection_equal(root)


@pytest.mark.parametrize("edit", ["identity", "state"])
def test_realization_collection_ignores_distinct_quoted_view_of_same_occurrence(edit):
    child = model()
    owned = child.with_identity(ObjectId()).with_state(selection(child))
    quoted = child.with_identity(ObjectId()).with_state(selection(child, "b"))
    root = model(child=[owned, owned], initializer=quoted.quote())
    path = GraphPath((Parameter("child"), Index(0)))
    identity, state = ObjectId(), selection(child, "c")

    changed = root.with_realization(
        identities={path: identity} if edit == "identity" else None,
        states={path: state} if edit == "state" else None)

    left, right = changed.parameters["child"]
    assert left is right
    assert left.object_id == (identity if edit == "identity" else owned.object_id)
    assert left.state == (state if edit == "state" else owned.state)
    assert changed.at(GraphPath((Parameter("initializer"),))) is quoted
    assert owned._occurrence is quoted._occurrence is left._occurrence
    assert changed.construction_view is root
    if edit == "state":
        assert changed.object_projection_equal(root)
    eager = NodePreparation((changed,), scheduler="eager")
    deferred = NodePreparation((changed,))
    assert eager.exact_key(changed) == deferred.exact_key(changed)
    restored = decode_node_graph(encode_node_graph(eager))
    assert restored.graph_equal(changed)
    assert restored.exact_key() == changed.exact_key()
    restored_owned = restored.parameters["child"][0]
    restored_quoted = restored.at(GraphPath((Parameter("initializer"),)))
    assert restored_owned is restored.parameters["child"][1]
    assert restored_owned is not restored_quoted
    assert restored_owned._occurrence is restored_quoted._occurrence
    assert restored_owned.object_id == left.object_id and restored_owned.state == left.state
    assert restored_quoted.object_id == quoted.object_id and restored_quoted.state == quoted.state


def test_explicit_authoring_bindings_still_edit_cross_quoted_aliases():
    child = model()
    root = model(child=[child, child], initializer=child.quote())
    path = GraphPath((Parameter("child"), Index(0)))
    identity, state = ObjectId(), selection(child)
    bound = root.bind(identities={path: identity}, states={path: state})
    assert bound.at(GraphPath((Parameter("initializer"),))) is bound.at(path)
    assert bound.at(path).object_id == identity and bound.at(path).state == state
    replaced = bound.replace_bindings(states={path: None})
    assert replaced.at(GraphPath((Parameter("initializer"),))) is replaced.at(path)
    assert replaced.at(path).object_id == identity and replaced.at(path).state is None


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

    reference = model().with_identity(ObjectId())
    first = reference.with_state(selection(reference))
    second = reference.with_state(selection(reference, "b"))
    a = model(child=FactorySpec("builtins:tuple", ConfigRef("state", first)))
    b = model(child=FactorySpec("builtins:tuple", ConfigRef("state", second)))
    assert a.object_projection_equal(b)
    assert selection(a).validate_for(b, codec="pkl")
    first_id, second_id = model(child=ObjectId()), model(child=ObjectId())
    assert not first_id.compatibility_view().graph_equal(second_id.compatibility_view())
    assert not first_id.object_projection_equal(second_id)


@pytest.fixture
def preparation_work(monkeypatch):
    """Count actual production discovery and hashing, including disposable owners."""
    import dryml.core.node_codec as codec
    import dryml.core.node_prepare as preparation

    work = {"discovery": 0, "folds": 0, "hashes": 0}
    describe, digest = codec._describe_node, preparation._digest

    def counted_describe(*args, **kwargs):
        work["discovery"] += 1
        return describe(*args, **kwargs)

    def counted_digest(*args, **kwargs):
        work["hashes"] += 1
        work["folds"] += args[0] == "dryml-node-value/v1"
        return digest(*args, **kwargs)

    monkeypatch.setattr(codec, "_describe_node", counted_describe)
    monkeypatch.setattr(preparation, "_digest", counted_digest)
    return work


@pytest.mark.parametrize("method", ["creation_digest", "compatibility_fingerprint", "object_projection_digest"])
def test_ordinary_named_digest_reuses_resident_preparation(method, preparation_work):
    child = model().with_identity(ObjectId())
    root = model(child=[child, child], initializer=child.quote())
    expected = getattr(root, method)()
    cold = dict(preparation_work)
    assert cold["discovery"] > 0 and cold["hashes"] > 0
    assert getattr(root, method)() == expected
    assert preparation_work == cold


def test_ordinary_named_equality_reuses_resident_preparation(preparation_work):
    child = model().with_identity(ObjectId())
    root = model(child=[child, child], initializer=child.quote())
    copied = root.copy_graph()
    assert root.object_projection_equal(copied)
    cold = dict(preparation_work)
    assert root.object_projection_equal(copied)
    assert preparation_work == cold


def test_same_cdef_multistate_validation_reuses_facts_not_admission(preparation_work):
    child = model().with_identity(ObjectId())
    origin = model(child=[child, child])
    destination = origin.with_realization(identities={"$": ObjectId()})
    first = selection(origin)
    second = LocalState(StateKey(first.key.cdef_digest, "pkl-" + "b" * 64), first.creation_cdef)
    assert first.validate_for(destination, codec="pkl") is first
    cold = dict(preparation_work)
    assert second.validate_for(destination, codec="pkl") is second
    assert first.validate_for(destination, codec="pkl") is first
    assert preparation_work == cold
    with pytest.raises(ValueError, match="codec"):
        second.validate_for(destination, codec="other")
    forged = LocalState(StateKey("0" * 64, second.key.codec_hash), first.creation_cdef)
    with pytest.raises(ValueError, match="creation.*digest"):
        forged.validate_for(destination, codec="pkl")
    with pytest.raises(ValueError, match="incompatible"):
        second.validate_for(model(width=8), codec="pkl")


def test_named_helpers_clear_eviction_and_schedulers_keep_exact_faceted_keys(preparation_work):
    from dryml.core.node_prepare import _owned_preparation
    from dryml.core.node_views import CREATION_VIEW, COMPATIBILITY_VIEW, OBJECT_PROJECTION_VIEW, view_equal

    child = model().with_identity(ObjectId())
    pinned = child.with_state(selection(child))
    root = model(child=[{pinned, pinned.copy_graph()}, pinned], initializer=pinned.quote())
    root = root.with_realization(identities={"$": ObjectId()}, states={"$": selection(root)})
    copied = root.copy_graph()
    source_key = root.exact_key()
    views = (CREATION_VIEW, COMPATIBILITY_VIEW, OBJECT_PROJECTION_VIEW)
    methods = ("creation_digest", "compatibility_fingerprint", "object_projection_digest")
    expected = []
    for view, method in zip(views, methods):
        eager = NodePreparation((root,), view=view, scheduler="eager")
        deferred = NodePreparation((root,), view=view, max_cached_keys=1)
        key = eager.exact_key(eager.roots[0])
        assert deferred.exact_key(deferred.roots[0]) == key
        deferred.clear()
        assert deferred.exact_key(deferred.roots[0]) == key
        digest = eager.digest(eager.roots[0], domain=view.name)
        assert getattr(root, method)() == digest
        assert view_equal(root, copied, view)
        expected.append(digest)

    owner = _owned_preparation(root)
    assert owner._cache_bytes <= owner.max_cached_bytes
    cold = dict(preparation_work)
    assert [getattr(root, method)() for method in methods] == expected
    for view in views:
        assert view_equal(root, copied, view)
    assert preparation_work == cold
    owner.clear()
    assert owner._cache_bytes == 0 and not owner._facts
    assert [getattr(root, method)() for method in methods] == expected
    assert preparation_work["discovery"] > cold["discovery"]
    assert preparation_work["folds"] > cold["folds"]
    assert root.exact_key() == source_key

    # Both count and byte eviction change only work, including oversized tables
    # that cannot remain resident at all. The immutable source graph is untouched.
    owner.clear()
    owner.max_cached_keys = 1
    for view, method, digest in zip(views, methods, expected):
        assert getattr(root, method)() == digest
        assert view_equal(root, copied, view)
        assert len(owner._facts) <= 1
        assert owner._cache_bytes <= owner.max_cached_bytes
    owner.clear()
    owner.max_cached_bytes = 1
    for view, method, digest in zip(views, methods, expected):
        assert getattr(root, method)() == digest
        assert view_equal(root, copied, view)
    assert not owner._facts and owner._cache_bytes == 0


def test_same_cdef_validation_after_clear_and_eviction(preparation_work):
    from dryml.core.node_prepare import _owned_preparation

    origin = model(child=model())
    destination = origin.with_realization(identities={"$": ObjectId()})
    first = selection(origin)
    second = LocalState(StateKey(first.key.cdef_digest, "pkl-" + "b" * 64), first.creation_cdef)
    assert first.validate_for(destination, codec="pkl") is first
    cold = dict(preparation_work)
    assert second.validate_for(destination, codec="pkl") is second
    assert preparation_work == cold
    for root in (first.creation_cdef, destination):
        _owned_preparation(root).clear()
    assert second.validate_for(destination, codec="pkl") is second
    assert preparation_work["discovery"] > cold["discovery"]
    for root in (first.creation_cdef, destination):
        owner = _owned_preparation(root)
        owner.clear()
        owner.max_cached_bytes = 1
    assert first.validate_for(destination, codec="pkl") is first
    assert second.validate_for(destination, codec="pkl") is second
    forged = LocalState(StateKey("0" * 64, first.key.codec_hash), first.creation_cdef)
    with pytest.raises(ValueError, match="creation.*digest"):
        forged.validate_for(destination, codec="pkl")


def test_handle_owned_views_concurrent_clear_and_lifetime():
    from concurrent.futures import ThreadPoolExecutor
    import gc
    import weakref
    from dryml.core.node_prepare import _owned_preparation

    root = model(child=[model(), model()])
    copied = root.copy_graph()
    owner = _owned_preparation(root)
    expected = (root.creation_digest(), root.compatibility_fingerprint(), root.object_projection_digest())

    def request(index, owner=owner, root=root):
        if index % 3 == 0:
            owner.clear()
        assert root.object_projection_equal(copied)
        return root.creation_digest(), root.compatibility_fingerprint(), root.object_projection_digest()

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(request, range(12))) == [expected] * 12
    assert owner._cache_bytes <= owner.max_cached_bytes
    reference = weakref.ref(root)
    del request, root, owner
    gc.collect()
    assert reference() is None


def test_warm_named_helpers_do_not_transport_owners_or_freeze_codec_admission():
    import dill

    origin = model(child=model())
    state = selection(origin)
    assert state.validate_for(origin, codec="pkl") is state
    origin.compatibility_fingerprint()
    origin.object_projection_digest()
    restored = dill.loads(dill.dumps(origin))
    assert "_pure_preparation" not in vars(restored)
    assert restored.creation_digest() == origin.creation_digest()
    assert restored.object_projection_equal(origin)
    assert state.validate_for(restored, codec="pkl") is state
    with pytest.raises(ValueError, match="codec"):
        state.validate_for(restored, codec="json")


def test_named_preparation_separates_same_name_transforms_and_failed_work():
    from dryml.core.node_prepare import PreparationView, _owned_preparation
    from dryml.core.node_views import view_digest, view_equal

    root, alternate = model(), model(width=8)
    first = PreparationView("policy-test", lambda node: node)
    second = PreparationView("policy-test", lambda node: alternate)
    expected = [view_digest(root, view) for view in (first, second)]
    assert expected[0] != expected[1]
    assert [view_digest(root, view) for view in (first, second)] == expected
    assert not view_equal(root, alternate, first)
    assert view_equal(root, alternate, second)
    owner = _owned_preparation(root)
    before = len(owner._facts), owner._cache_bytes

    def interrupted(node):
        raise RuntimeError("view preparation interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        view_digest(root, PreparationView("policy-test", interrupted))
    assert (len(owner._facts), owner._cache_bytes) == before
    assert [view_digest(root, view) for view in (first, second)] == expected


def test_legacy_creation_adaptation_reuses_one_bounded_origin_owner(preparation_work):
    from dryml.core.definition import ConcreteDefinition
    from dryml.core.node_prepare import _owned_preparation
    from dryml.core.node_views import CREATION_VIEW, view_digest

    destination = model(child=model())
    legacy = ConcreteDefinition._from_bound_record(
        destination.cls, destination._bound_args, stateful_role=True)
    first = LocalState(StateKey(view_digest(legacy, CREATION_VIEW), "pkl-" + "a" * 64), legacy)
    second = LocalState(StateKey(first.key.cdef_digest, "pkl-" + "b" * 64), legacy)
    assert first.validate_for(destination, codec="pkl") is first
    cold = dict(preparation_work)
    assert second.validate_for(destination, codec="pkl") is second
    assert preparation_work == cold
    owner = _owned_preparation(legacy)
    normalized = owner._prepared_view(legacy, CREATION_VIEW).roots[0]
    assert "_pure_preparation" not in vars(normalized)
    assert owner._cache_bytes <= owner.max_cached_bytes
    owner.clear()
    assert first.validate_for(destination, codec="pkl") is first
    assert preparation_work["discovery"] > cold["discovery"]
