"""DefinitionRecord closure coverage for graphs with ephemeral nodes."""

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Object, QueryDomainError, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord


class QueryLeaf(Object):
    """Ephemeral child retained structurally by its enclosing definition."""

    def __init__(self, name):
        self.name = name


class QueryParent(Serializable):
    """Stateful root used to publish an enclosing definition closure."""

    def __init__(self, child, *, label="parent"):
        self.child = child
        self.label = label

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Publish no payload files for this structural test value."""


def test_save_records_definition_closure_for_ephemeral_child(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = QueryLeaf("child", repo=repo)
    parent = QueryParent(child, repo=repo)

    repo.save_object(parent)

    assert store.read_definition_record(DefinitionRecord(parent.definition).digest)
    assert store.read_definition_record(DefinitionRecord(child.definition).digest)


def test_repeated_ephemeral_child_has_one_definition_record(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = QueryLeaf("shared", repo=repo)
    parent = QueryParent([child, child], repo=repo)

    repo.save_object(parent)

    records = tuple(store.iter_definition_records())
    assert {record.definition for record in records} == {parent.definition, child.definition}


def test_nested_source_restriction_is_immutable_and_validated(tmp_path):
    store = DirStore(tmp_path / "store")
    other = DirStore(tmp_path / "other")
    repo = Repo(stores=store)
    child = QueryLeaf("child", repo=repo)

    unrestricted = repo.query(child.definition).nested()
    restricted = unrestricted.in_store(store)

    assert unrestricted.source_store is None
    assert restricted.source_store is store
    with pytest.raises(ValueError, match="connected Store"):
        unrestricted.in_store(other)


def test_nested_source_restriction_does_not_claim_unimplemented_execution(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = QueryLeaf("child", repo=repo)

    with pytest.raises(QueryDomainError, match="containment execution"):
        repo.query(child.definition).nested().in_store(store).count()
