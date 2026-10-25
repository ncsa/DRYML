"""Integration tests for V3 reference-identity restrictions and traversal."""

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Object, Repo, Serializable
from dryml.core.query import EdgePolicy, RelationshipKind, field
from dryml.core.links import DefLink
from dryml.core.cdef_graph import EdgeKind
from dryml.core.store.dir import DirStore
from dryml.core.store.records import NodeNameRecord
from tests.core.test_repo_aliases import _saved_digest


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

    assert repo.query().object_refs().object_id(state.object_id).one().graph_equal(state.object_view())
    assert repo.query().object_refs().namespace(state.object_id.namespace).one().graph_equal(state.object_view())
    assert repo.query().object_refs().sel(state.creation_view()).one().graph_equal(state.object_view())
    assert repo.query().saved().state_hash(state.state.key.codec_hash).one().graph_equal(state)


def test_reference_metadata_and_alias_restrictions_stay_in_one_query(tmp_path):
    """Metadata and aliases compose with reference identities without loading payloads."""

    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    state = repo.save_object(QueryIdentityLeaf(3, repo=repo))
    repo.set_metadata(state.object_view(), {"project": "forecasting"})
    repo.default_store.compare_and_set_node_name(NodeNameRecord("selected", "saved", _saved_digest(state)), expected=None)

    assert (
        repo.query()
        .saved()
        .alias("selected")
        .where(field("object", "project").eq("forecasting"))
        .one().graph_equal(state)
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

    assert occurrences.one().target.graph_equal(state)
    assert occurrences.query().owners().cdefs().one().graph_equal(owner.creation_view())


def test_reference_contains_restricts_aggregate_object_identities(tmp_path):
    """A proper owned exact reference selects its aggregate ObjectRef identity."""

    repo = Repo(DirStore(tmp_path / "store", query_index="sqlite"))
    child = repo.save_object(QueryIdentityLeaf(3, repo=repo))
    aggregate = repo.save_object(QueryIdentityWrapper(child, repo=repo))

    assert repo.query().object_refs().contains(child.object_view()).one().graph_equal(aggregate.object_view())
