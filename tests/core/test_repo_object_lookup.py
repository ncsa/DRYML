import pytest

from dryml.core import ObjectRef, Repo, Serializable, object_namespace
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DeclarationRecord


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

    assert repo.lookup_object_ref(state.object_id) == state.object
    assert repo.find_object_refs(namespace=("experiment",)) == ((repo.default_store, state.object),)
    assert repo.find_object_refs(contains=state.object) == ((repo.default_store, state.object),)


def test_incompatible_object_id_authority_rejects_without_a_query_index(tmp_path):
    first = DirStore(tmp_path / "first", query_index="memory")
    second = DirStore(tmp_path / "second", query_index="memory")
    repo = Repo([first, second])
    state = repo.save_object(LookupValue(1, repo=repo), store=first)
    incompatible = ObjectRef(
        LookupValue(2).definition,
        {"$": state.object_id},
    )
    second.write_declaration_record(DeclarationRecord(incompatible))

    with pytest.raises(RepoLoadError, match="incompatible closed-subtree authority"):
        repo.lookup_object_ref(state.object_id)


def test_lookup_returns_closed_child_subtree_and_containment_returns_aggregate(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    child = repo.save_object(LookupValue(1, repo=repo))
    aggregate = repo.save_object(LookupAggregate(child, repo=repo))

    assert repo.lookup_object_ref(child.object_id) == child.object
    assert (repo.default_store, aggregate.object) in repo.find_object_refs(contains=child.object)
