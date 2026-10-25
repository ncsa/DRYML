"""Pre-dispatch identity allocation preserves occurrence and construction semantics."""

from pathlib import Path

import pytest

from dryml.core import (
    ConcreteNode, Definition, LocalState, Object, ObjectId, ObjectRef, Repo,
    Serializable, StateKey, object_namespace,
)
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.node_codec import decode_node_graph, encode_node_graph
from dryml.core.node_prepare import NodePreparation
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath


class IdentityNode(Serializable):
    """Stateful graph fixture with an independently persisted integer payload."""

    def __init__(self, children=(), value=0):
        self.children = children
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class IdentityEnvelope(Object):
    """Stateless owner used to check that only stateful descendants receive IDs."""

    def __init__(self, children=()):
        self.children = children


def _state(node, marker):
    return LocalState(StateKey(node.creation_digest(), "pkl-" + marker * 64), node.creation_view())


def test_allocate_identities_preserves_shared_and_independent_occurrences():
    shared, independent = Definition(IdentityNode), Definition(IdentityNode)
    recipe = Definition(IdentityNode, {
        "sequence": [shared, shared, independent],
        "unordered": {shared, independent},
        "nested": Definition(IdentityEnvelope, [shared]),
    }).concretize()

    with object_namespace("jobs"):
        with object_namespace("data"):
            identified = recipe.allocate_identities()

    children = identified.parameters["children"]
    first, alias, equal = children["sequence"]
    assert first is alias
    assert first is children["nested"].parameters["children"][0]
    assert first is not equal and first.recipe_equal(equal)
    assert children["unordered"] == {first, equal}
    assert children["nested"].object_id is None
    identities = tuple(identified.identity_spec().values())
    assert len(identities) == len(set(identities)) == 3
    assert all(identity.namespace == ("jobs", "data") for identity in identities)
    assert not recipe.identity_spec() and not identified.state_spec()
    assert identified.construction_view is recipe
    assert identified.creation_digest() == recipe.creation_digest()
    assert identified.allocate_identities() is identified

    another = recipe.allocate_identities()
    assert set(another.identity_spec().values()).isdisjoint(identities)


def test_allocate_identities_preserves_existing_state_and_construction_pins():
    child = Definition(IdentityNode, value=1).concretize()
    initial, later = _state(child, "a"), _state(child, "b")
    pinned = child.with_state(initial)
    selected = pinned.with_realization(states={"$": later})
    existing = ObjectId(("existing",))
    recipe = Definition(IdentityNode, [selected]).concretize().with_identity(existing)

    identified = recipe.allocate_identities()

    assigned_child = identified.parameters["children"][0]
    assert identified.object_id == existing
    assert assigned_child.object_id is not None
    assert assigned_child.state is later
    assert assigned_child.construction_view is pinned
    assert assigned_child.construction_view.state is initial
    assert identified.construction_view is recipe
    assert identified.creation_digest() == recipe.creation_digest()
    assert selected.object_id is None and selected.state is later


def test_allocate_identities_respects_references_and_exact_quotations():
    owned = Definition(IdentityNode).concretize()
    referenced = Definition(IdentityNode, [Definition(IdentityNode)]).concretize()
    recipe = Definition(IdentityEnvelope, {
        "owned": owned,
        "alias": DefLink.finalized(EdgeKind.REF, owned),
        "reference": DefLink.finalized(EdgeKind.REF, referenced),
        "quoted": owned.quote(),
        "future": Definition(IdentityNode).quote(),
    }).concretize()
    original = recipe.parameters["children"]

    identified = recipe.allocate_identities()

    children = identified.parameters["children"]
    assert identified.object_id is None
    assert children["owned"].object_id is not None
    assert children["alias"].target is children["owned"]
    assert children["reference"] is original["reference"]
    assert children["reference"].target.object_id is None
    assert children["reference"].target.parameters["children"][0].object_id is None
    assert children["quoted"] is original["quoted"]
    assert children["quoted"].value is owned and owned.object_id is None
    assert children["future"] is original["future"]
    assert identified.allocate_identities() is identified


def test_allocate_identities_needs_no_construction_resolution_or_store(monkeypatch):
    from dryml.core import symbol
    from dryml.core import materialization

    recipe = Definition(IdentityNode, [Definition(IdentityNode)]).concretize()

    def forbidden(*args, **kwargs):
        pytest.fail("identity allocation attempted construction, resolution or Store access")

    monkeypatch.setattr(IdentityNode, "__init__", forbidden)
    monkeypatch.setattr(ConcreteNode, "concretize", forbidden)
    monkeypatch.setattr(symbol, "resolve_symbol", forbidden)
    monkeypatch.setattr(materialization, "resolve_symbol", forbidden)
    monkeypatch.setattr(DirStore, "__init__", forbidden)
    monkeypatch.setattr(DirStore, "publish_node", forbidden)
    monkeypatch.setattr(DirStore, "read_state", forbidden)
    identified = recipe.allocate_identities()
    assert len(identified.identity_spec()) == 2
    assert not identified.state_spec()


@pytest.mark.parametrize("invalid", ["soft", "collision", "stateless_id", "stateless_state", "legacy"])
def test_allocate_identities_rejects_invalid_graph_before_allocating(monkeypatch, invalid):
    from dryml.core import reference_values

    expected, match = ValueError, "ObjectId"
    if invalid == "soft":
        recipe = Definition(IdentityNode)
        expected, match = TypeError, "concret"
    elif invalid == "collision":
        identity = ObjectId()
        recipe = Definition(IdentityNode, [
            Definition(IdentityNode).with_identity(identity),
            Definition(IdentityNode).with_identity(identity),
        ]).concretize()
    elif invalid == "stateless_id":
        recipe = Definition(IdentityEnvelope).concretize().with_identity(ObjectId())
        match = "Stateless"
    elif invalid == "stateless_state":
        recipe = Definition(IdentityEnvelope).concretize()
        recipe = recipe.with_state(_state(recipe, "a"))
        match = "Stateless"
    else:
        legacy = ObjectRef(Definition(IdentityNode).concretize(), {GraphPath(): ObjectId()})
        recipe = Definition(IdentityNode, [legacy]).concretize()
        expected, match = TypeError, "legacy"

    def forbidden():
        pytest.fail("invalid graph consumed a new identity")

    monkeypatch.setattr(reference_values, "uuid4", forbidden)
    with pytest.raises(expected, match=match):
        recipe.allocate_identities()


def test_allocate_identities_complete_graphs_are_noops(monkeypatch):
    from dryml.core import reference_values

    stateless = Definition(IdentityEnvelope).concretize()
    complete = Definition(IdentityNode).concretize().with_identity(ObjectId())
    complete = complete.with_state(_state(complete, "a"))

    def forbidden():
        pytest.fail("complete graph consumed a new identity")

    monkeypatch.setattr(reference_values, "uuid4", forbidden)
    with object_namespace("unused"):
        assert stateless.allocate_identities() is stateless
        assert complete.allocate_identities() is complete


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_allocated_graph_transport_and_separate_builds_keep_save_identity(tmp_path):
    child = Definition(IdentityNode, value=7)
    recipe = Definition(IdentityNode, [child, child, Definition(IdentityNode, value=7)]).concretize()
    identified = recipe.allocate_identities()
    packet = encode_node_graph(NodePreparation((identified,)))
    first_node, second_node = decode_node_graph(packet), decode_node_graph(packet)
    assert first_node is not second_node
    assert first_node.graph_equal(identified) and second_node.graph_equal(identified)

    first_store = DirStore(tmp_path / "store", query_index="none")
    second_store = DirStore.open_existing(tmp_path / "store", query_index="none")
    first_repo, second_repo = Repo(first_store), Repo(second_store)
    try:
        first = first_repo.build(first_node, cache="none")
        second = second_repo.build(second_node, cache="none")
        for live in (first, second):
            assert live.object_id == identified.object_id
            assert live.children[0] is live.children[1]
            assert live.children[0] is not live.children[2]
        first_view, second_view = first_repo.describe(first), second_repo.describe(second)
        assert first_view.object_projection_equal(identified)
        assert second_view.object_projection_equal(identified)
        assert first_view.creation_digest() == second_view.creation_digest() == recipe.creation_digest()
        first_saved, second_saved = first_repo.save_object(first), second_repo.save_object(second)
        assert first_saved.graph_equal(second_saved)
        restored = second_repo.load_state_ref(first_saved, reuse_live="never")
        assert restored.children[0] is restored.children[1]
        assert [child.value for child in restored.children] == [7, 7, 7]
        assert second_repo.describe(restored).object_projection_equal(identified)
    finally:
        first_repo.close()
        second_repo.close()
        first_store.close()
        second_store.close()
