"""DefinitionRecord closure coverage for graphs with ephemeral nodes."""

from dryml.core import Object, Repo, Serializable
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
