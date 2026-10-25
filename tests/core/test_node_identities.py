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
from dryml.core.store import StoreAuthorityError, StoreCapabilityError
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


def test_unchecked_allocation_ignores_repo_and_does_not_read_authority(monkeypatch):
    repo = Repo()
    try:
        def forbidden(*args, **kwargs):
            pytest.fail("unchecked allocation consulted Repo authority")

        monkeypatch.setattr(repo, "_allocate_unique_object_ids", forbidden)
        assert Definition(IdentityNode).concretize().allocate_identities(repo=repo).object_id is not None
        assert Definition(IdentityNode).concretize().allocate_identities(check_unique=False, repo=repo).object_id is not None
    finally:
        repo.close()


def test_checked_allocation_retries_published_child_identity_across_all_stores(tmp_path, monkeypatch):
    from dryml.core import reference_values

    empty = DirStore(tmp_path / "empty", query_index="none")
    occupied = DirStore(tmp_path / "occupied", query_index="none")
    parent = Definition(IdentityNode, [Definition(IdentityNode)]).concretize().allocate_identities()
    occupied.publish_node(parent)
    identity = parent.parameters["children"][0].object_id
    repo = Repo([empty, occupied])
    try:
        cuts = empty.authority_cut(), occupied.authority_cut()
        fresh = ObjectId()
        candidates = iter((identity.nonce, fresh.nonce))
        monkeypatch.setattr(reference_values, "uuid4", lambda: next(candidates))
        recipe = Definition(IdentityNode).concretize()
        identified = recipe.allocate_identities(check_unique=True, repo=repo)
        assert identified.object_id == fresh
        assert identified.construction_view is recipe and recipe.object_id is None
        assert (empty.authority_cut(), occupied.authority_cut()) == cuts
        assert not repo._lineage_candidates
    finally:
        repo.close()
        empty.close()
        occupied.close()


@pytest.mark.parametrize("cache", ["strong", "weak"])
def test_checked_allocation_retries_unsaved_cached_root_and_child_ids(monkeypatch, cache):
    from dryml.core import reference_values

    repo = Repo()
    try:
        live = repo.build(Definition(IdentityNode, [Definition(IdentityNode)]).concretize(), cache="none")
        getattr(repo, "cache_" + cache)(live)
        fresh = ObjectId()
        candidates = iter((live.object_id.nonce, live.children[0].object_id.nonce, fresh.nonce))
        monkeypatch.setattr(reference_values, "uuid4", lambda: next(candidates))
        identified = Definition(IdentityNode).concretize().allocate_identities(check_unique=True, repo=repo)
        assert identified.object_id == fresh
    finally:
        repo.close()


def test_checked_allocation_avoids_graph_and_batch_ids_with_default_repo(monkeypatch):
    from dryml.core import reference_values
    from dryml.core.repo import default_repo

    repo = Repo()
    try:
        existing, first, second = ObjectId(), ObjectId(), ObjectId()
        child = Definition(IdentityNode)
        reference = Definition(IdentityNode).concretize().with_identity(existing)
        recipe = Definition(IdentityNode, [child, child, reference.quote()]).concretize()
        candidates = iter((existing.nonce, first.nonce, first.nonce, second.nonce))
        monkeypatch.setattr(reference_values, "uuid4", lambda: next(candidates))
        with default_repo(repo):
            identified = recipe.allocate_identities(check_unique=True)
        assert identified.object_id == first
        assert identified.parameters["children"][0].object_id == second
        assert identified.parameters["children"][0] is identified.parameters["children"][1]
        assert identified.parameters["children"][2].value is reference
        assert not repo._lineage_candidates
    finally:
        repo.close()


def test_checked_allocation_compares_full_namespace_and_preserves_existing_ids(monkeypatch):
    from dryml.core import reference_values

    repo = Repo()
    try:
        existing = ObjectId(("existing",))
        recipe = Definition(IdentityNode, [Definition(IdentityNode)]).concretize().with_identity(existing)
        monkeypatch.setattr(reference_values, "uuid4", lambda: existing.nonce)
        with object_namespace("new"):
            identified = recipe.allocate_identities(check_unique=True, repo=repo)
        assert identified.object_id == existing
        child = identified.parameters["children"][0]
        assert child.object_id.nonce == existing.nonce
        assert child.object_id.namespace == ("new",)
        assert child.object_id != existing
        assert identified.creation_digest() == recipe.creation_digest()

        def forbidden(*args, **kwargs):
            pytest.fail("identity-complete graph performed a uniqueness lookup")

        monkeypatch.setattr(repo, "_allocate_unique_object_ids", forbidden)
        assert identified.allocate_identities(check_unique=True, repo=repo) is identified
    finally:
        repo.close()


@pytest.mark.parametrize("mode", ["nonbool", "missing_repo", "wrong_repo", "closed_repo"])
def test_checked_allocation_validates_options_before_uuid_allocation(monkeypatch, mode):
    from dryml.core import reference_values
    from dryml.core import repo as repo_module

    def forbidden():
        pytest.fail("invalid options consumed a UUID")

    monkeypatch.setattr(reference_values, "uuid4", forbidden)
    monkeypatch.setattr(repo_module, "get_default_repo", lambda: None)
    kwargs = {"check_unique": True}
    error = ValueError
    if mode == "nonbool":
        kwargs["check_unique"] = 1
        error = TypeError
    elif mode == "wrong_repo":
        kwargs["repo"] = object()
        error = TypeError
    elif mode == "closed_repo":
        repo = Repo()
        repo.close()
        kwargs["repo"] = repo
        error = RuntimeError
    with pytest.raises(error):
        Definition(IdentityNode).concretize().allocate_identities(**kwargs)


def test_checked_allocation_bounds_collision_retries_without_publication(monkeypatch):
    from dryml.core import reference_values

    repo = Repo()
    try:
        existing = ObjectId()
        recipe = Definition(IdentityNode, [Definition(IdentityNode)]).concretize().with_identity(existing)
        attempts = []

        def collide():
            attempts.append(1)
            return existing.nonce

        monkeypatch.setattr(reference_values, "uuid4", collide)
        with pytest.raises(RuntimeError, match="32 candidates"):
            recipe.allocate_identities(check_unique=True, repo=repo)
        assert len(attempts) == 32
        assert recipe.object_id == existing and recipe.parameters["children"][0].object_id is None
        assert not repo._lineage_candidates
    finally:
        repo.close()


@pytest.mark.parametrize("error", [
    OSError("unreadable"), StoreAuthorityError("invalid authority"),
    StoreCapabilityError("unsupported declaration anchors"),
])
def test_checked_allocation_propagates_lookup_failure(tmp_path, monkeypatch, error):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    try:
        cut = store.authority_cut()

        def fail_lookup(identity):
            raise error

        monkeypatch.setattr(store, "read_declaration_anchor", fail_lookup)
        recipe = Definition(IdentityNode).concretize()
        with pytest.raises(type(error)) as raised:
            recipe.allocate_identities(check_unique=True, repo=repo)
        assert raised.value is error
        assert recipe.object_id is None and store.authority_cut() == cut
    finally:
        repo.close()
        store.close()
