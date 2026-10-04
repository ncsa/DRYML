"""Integration tests for V3 reference-identity restrictions and traversal."""

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.query import EdgePolicy, RelationshipKind, field
from dryml.core.links import DefLink
from dryml.core.cdef_graph import EdgeKind
from dryml.core.store.dir import DirStore


class QueryIdentityLeaf(Serializable):
    """Small stateful identity fixture for V3 producer tests."""

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class QueryIdentityWrapper(Object):
    """Root that embeds one saved reference identity."""

    def __init__(self, child):
        self.child = child


def test_reference_identity_filters_use_the_unified_producer(tmp_path):
    """ObjectId, namespace, selector, and state-hash filters retain V3 identities."""

    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    state = repo.save_object(QueryIdentityLeaf(3, repo=repo))

    assert repo.query().object_refs().object_id(state.object_id).one() == state.object
    assert repo.query().object_refs().namespace(state.object_id.namespace).one() == state.object
    assert repo.query().object_refs().sel(state.definition).one() == state.object
    assert repo.query().state_refs().state_hash(next(iter(state.states.values()))).one() == state


def test_reference_metadata_and_alias_restrictions_stay_in_one_query(tmp_path):
    """Metadata and aliases compose with reference identities without loading payloads."""

    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    state = repo.save_object(QueryIdentityLeaf(3, repo=repo))
    repo.set_metadata(state.object, {"project": "forecasting"})
    repo.set_alias("selected", state)

    assert (
        repo.query()
        .object_refs()
        .alias("selected")
        .where(field("object", "project").eq("forecasting"))
        .one()
        == state.object
    )


def test_reference_containment_uses_explicit_roots_and_nonempty_paths(tmp_path):
    """Traversal starts from selected roots and owners are a composable V3 universe."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(QueryIdentityLeaf(3, repo=repo))
    owner = repo.save_object(
        QueryIdentityWrapper(DefLink.finalized(EdgeKind.REF, state), repo=repo)
    )

    occurrences = (
        repo.query()
        .cdefs()
        .stored(scope=store)
        .nested(state, edges=EdgePolicy.ALL)
        .through(RelationshipKind.REFERENCE)
        .collect()
    )

    assert occurrences.one().target == state
    assert occurrences.query().owners().cdefs().one() == owner.definition


def test_reference_contains_restricts_aggregate_object_identities(tmp_path):
    """A proper owned exact reference selects its aggregate ObjectRef identity."""

    repo = Repo(DirStore(tmp_path / "store", query_index="sqlite"))
    child = repo.save_object(QueryIdentityLeaf(3, repo=repo))
    aggregate = repo.save_object(QueryIdentityWrapper(child, repo=repo))

    assert repo.query().object_refs().contains(child.object).one() == aggregate.object
