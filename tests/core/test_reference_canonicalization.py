import pytest

from dryml.core import Definition, ObjectId, ObjectRef, StateRef
from dryml.core.cdef_graph import EdgeKind
from dryml.core.cdef_codec import CDefGraphCodecError, decode_cdef_graph, encode_cdef_graph
from dryml.core.links import DefLink
from dryml.core.object import Object, Serializable
from dryml.core.utils.graph.path import GraphPath, Index, Parameter


class CanonicalLeaf(Serializable):
    pass


class CanonicalOwner(Object):
    def __init__(self, value):
        self.value = value


def _state_ref():
    leaf = Definition(CanonicalLeaf).concretize()
    object_ref = ObjectRef(leaf, {GraphPath(): ObjectId(("canonical",))})
    return StateRef(object_ref, {GraphPath(): "pkl-" + "b" * 64})


def test_exact_references_are_atomic_canonical_values_and_codec_round_trip():
    state = _state_ref()
    cdef = Definition(
        CanonicalOwner,
        {"state": state, "ref": DefLink.finalized(EdgeKind.REF, state)},
    ).concretize()

    assert cdef.parameters["value"]["state"] is state
    assert cdef.parameters["value"]["ref"].target is state
    decoded = type(state).from_data(state.to_data())
    assert decoded == state
    assert decoded.definition.graph_equal(state.definition)
    assert decoded.definition is not state.definition


def test_soft_state_selector_resolves_once_before_cdef_identity():
    state = _state_ref()
    selector = state.object.state("best")

    class FakeRepo:
        calls = 0

        def resolve_state_selector(self, value):
            self.calls += 1
            assert value == selector
            return state

    repo = FakeRepo()
    cdef = Definition(
        CanonicalOwner,
        [
            selector,
            DefLink.finalized(EdgeKind.REF, selector),
            DefLink.finalized(EdgeKind.MATERIALIZE, selector),
        ],
    ).concretize(repo=repo)
    assert repo.calls == 1
    assert cdef.parameters["value"][0] is state
    assert cdef.parameters["value"][1].target is state
    assert cdef.parameters["value"][2] is state


def test_soft_state_selector_fails_without_or_outside_repo_scope():
    state = _state_ref()
    selector = state.object.state("best")
    with pytest.raises(KeyError, match="no alias"):
        Definition(CanonicalOwner, selector).concretize()

    class BadRepo:
        def resolve_state_selector(self, value):
            leaf = Definition(CanonicalLeaf).concretize()
            other = ObjectRef(leaf, {GraphPath(): ObjectId()})
            return StateRef(other, {GraphPath(): "pkl-" + "c" * 64})

    with pytest.raises(ValueError, match="outside its ObjectRef scope"):
        Definition(CanonicalOwner, selector).concretize(repo=BadRepo())


def test_soft_state_selector_is_a_stable_definition_and_selector_leaf():
    selector = _state_ref().object.state("best")
    definition = Definition(CanonicalOwner, selector)

    assert definition.stable_hash()
    assert definition.as_selector().root.parameters["value"] is selector


def test_cdef_graph_codec_rejects_state_selector_authority():
    cdef = Definition(CanonicalOwner, "value").concretize()
    authority = encode_cdef_graph(cdef)
    authority["nodes"][0]["parameters"]["items"][0][1] = {
        "kind": "state_selector_ref",
        "object": _state_ref().object.to_data(),
        "alias": "best",
    }

    with pytest.raises(CDefGraphCodecError, match="Unknown CDef graph value kind"):
        decode_cdef_graph(authority)


def test_legacy_definition_surface_export_refuses_node_facet_loss():
    """A V2 call-surface projection cannot silently erase identity-3 constraints."""
    from dryml.core.canonical import thaw_definition_surface_value

    pinned = Definition(CanonicalLeaf).concretize().with_identity(ObjectId())
    for export in (lambda: thaw_definition_surface_value(pinned),
                   lambda: pinned.object_projection(),
                   lambda: ObjectRef(pinned, {GraphPath(): pinned.object_id}).to_data()):
        with pytest.raises(CDefGraphCodecError, match="identity-3"):
            export()


def test_v2_reference_round_trip_keeps_value_scopes_without_weakening_node_equality():
    """Transported reference values do not promise cross-value occurrence aliases."""
    from dryml.core.store.records import DefinitionRecord

    seed = _state_ref()
    root = Definition(CanonicalOwner, [seed, seed]).concretize()
    reference = ObjectRef(root, {GraphPath((Parameter("value"), Index(0))): seed.object.object_id})
    restored = ObjectRef.from_data(reference.to_data())
    record = DefinitionRecord.from_bytes(DefinitionRecord(root).to_bytes())

    assert reference == restored and restored == reference
    assert hash(reference) == hash(restored)
    assert record._matches_definition(root)
    assert not root.graph_equal(restored.definition)
    assert not restored.definition.graph_equal(root)


def test_v2_definition_boundary_requires_exact_topology_roles_and_reference_pins(monkeypatch):
    """Matching never substitutes structural hashes for scoped graph evidence."""
    from dryml.core.definition import ConcreteDefinition, ConcreteNode
    from dryml.core.quoted import QuotedDef
    from dryml.core.store.records import DefinitionRecord
    import dryml.core.node_prepare as preparation

    seed = _state_ref()
    child = Definition(CanonicalOwner, 1).concretize()
    root = Definition(CanonicalOwner, [child, child, seed]).concretize()
    record = DefinitionRecord.from_bytes(DefinitionRecord(root).to_bytes())
    changed_state = StateRef(seed.object, {"$": "pkl-" + "c" * 64})
    changed_identity = StateRef(ObjectRef(seed.definition, {"$": ObjectId()}), seed.states)
    wrong_values = (
        [child, child.copy_graph(), seed],
        [child, Definition(CanonicalOwner, 2).concretize(), seed],
        [child, DefLink.finalized(EdgeKind.REF, child), seed],
        [child, QuotedDef(child), seed],
        [child, child, changed_state],
        [child, child, changed_identity],
    )
    wrong = [Definition(CanonicalOwner, value).concretize() for value in wrong_values]
    monkeypatch.setattr(preparation, "_digest", lambda *args: "0" * 64)
    for cls in (ConcreteDefinition, ConcreteNode):
        monkeypatch.setattr(cls, "graph_hash", lambda self: "0" * 64)
        monkeypatch.setattr(cls, "stable_hash", lambda self: "0" * 64)

    assert record._matches_definition(root)
    for candidate in wrong:
        assert not record._matches_definition(candidate)
    path = GraphPath((Parameter("value"), Index(2)))
    assert ObjectRef(root, {path: seed.object.object_id}) != ObjectRef(
        wrong[-1], {path: changed_identity.object.object_id})


def test_v2_definition_boundary_rejects_nested_node_facets_and_retained_construction():
    """Legacy matching must not admit new evidence that its codec cannot retain."""
    from dryml.core.store.records import DefinitionRecord

    seed = _state_ref()
    root = Definition(CanonicalOwner, seed).concretize()
    record = DefinitionRecord.from_bytes(DefinitionRecord(root).to_bytes())
    pinned = seed.definition.with_identity(ObjectId())
    retained = pinned.with_realization(identities={"$": None})
    for child in (pinned, retained):
        candidate = Definition(
            CanonicalOwner, ObjectRef(child, seed.object.objects)
        ).concretize()
        with pytest.raises(CDefGraphCodecError, match="identity-3"):
            record._matches_definition(candidate)
        with pytest.raises(CDefGraphCodecError, match="identity-3"):
            encode_cdef_graph(candidate)
