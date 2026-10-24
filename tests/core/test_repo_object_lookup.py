import pytest

from dryml.core import Definition, Repo, Serializable, object_namespace
from dryml.core.store.dir import DirStore
from dryml.core.query.model import QueryError

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


class LookupValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class LookupAggregate(Serializable):
    def __init__(self, child):
        self.child = child

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


def test_authoritative_scans_find_ids_namespaces_and_containing_refs(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    with object_namespace("experiment", "blue"):
        state = repo.save_object(LookupValue(1, repo=repo))

    assert repo.query().object_refs().object_id(state.object_id).one().graph_equal(state.object_view())
    assert repo.query().object_refs().namespace(("experiment",)).one().graph_equal(state.object_view())
    assert not repo.query().object_refs().contains(state.object_view()).exists()


def test_incompatible_object_id_authority_rejects_without_a_query_index(tmp_path):
    first = DirStore(tmp_path / "first", query_index="memory")
    second = DirStore(tmp_path / "second", query_index="memory")
    repo = Repo([first, second])
    state = repo.save_object(LookupValue(1, repo=repo), store=first)
    incompatible = Definition(LookupValue, 2).with_identity(state.object_id).concretize()
    second.publish_node(incompatible)

    with pytest.raises(QueryError, match="conflicts"):
        repo.query().object_refs().object_id(state.object_id).one()


def test_lookup_returns_closed_child_subtree_and_containment_returns_aggregate(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    child = repo.save_object(LookupValue(1, repo=repo))
    aggregate = repo.save_object(LookupAggregate(child, repo=repo))

    assert repo.query().object_refs().sel(child.object_view()).one().graph_equal(child.object_view())
    owners = tuple(repo.query().object_refs().contains(child.object_view()).collect())
    assert any(node.graph_equal(aggregate.object_view()) for node in owners)
