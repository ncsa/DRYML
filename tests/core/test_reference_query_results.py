from dryml.core import Definition, ObjectRef, Repo, Serializable
from dryml.core.store.dir import DirStore


class ReferenceResultLeaf(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class ReferenceResultParent(Serializable):
    def __init__(self, selected):
        self.selected = selected

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


def test_reference_occurrences_retain_complete_owner_and_typed_path(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = repo.save_object(ReferenceResultLeaf(1, repo=repo))
    first = repo.save_object(ReferenceResultParent(child, repo=repo))
    second = repo.save_object(ReferenceResultParent(child, repo=repo))

    occurrences = repo.references().object_id(child.object_id).state_refs().occurrences()
    owners = {item.owner for item in occurrences if item.owner in {first.object, second.object}}

    assert owners == {first.object, second.object}
    assert all(item.path.__class__.__name__ == "GraphPath" for item in occurrences)


def test_reference_values_dedupe_identical_store_replicas(tmp_path):
    from dryml.core.store.records import DefinitionRecord, StateRefRecord

    first_store = DirStore(tmp_path / "first")
    repo = Repo(first_store)
    state = repo.save_object(ReferenceResultLeaf(1, repo=repo))
    second_store = DirStore(tmp_path / "second")
    second_store.write_definition_record(
        DefinitionRecord(state.definition), stored_root=False
    )
    second_store.write_state_ref_record(StateRefRecord(state))
    repo.add_store(second_store)

    assert list(repo.references().object_id(state.object_id).object_refs()) == [state.object]
