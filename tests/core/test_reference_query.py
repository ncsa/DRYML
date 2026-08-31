from dryml.core import Definition, Object, Ref, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter


class ReferenceQueryLeaf(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class ReferenceQueryWrapper(Object):
    def __init__(self, child):
        self.child = child


class ReferenceQueryRefParent(Object):
    def __init__(self, selected):
        self.selected = selected


def test_reference_filters_scan_authority_without_materializing(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    state = repo.save_object(ReferenceQueryLeaf(3, repo=repo))

    assert repo.references().object_id(state.object_id).object_refs().one() == state.object
    assert repo.references().namespace(state.object_id.namespace).object_refs().one() == state.object
    assert repo.references().definition(state.definition).object_refs().one() == state.object
    assert repo.references().state_hash(next(iter(state.states.values()))).state_refs().one() == state


def test_object_id_lookup_is_closed_but_reference_query_returns_aggregate(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    child = repo.save_object(ReferenceQueryLeaf(3, repo=repo))
    aggregate = repo.save_object(ReferenceQueryWrapper(child, repo=repo))

    assert repo.lookup_object_ref(child.object_id) == child.object
    assert aggregate.object in repo.references().object_id(child.object_id).object_refs()


def test_reference_filters_keep_exact_paths_aliases_and_all_ephemeral_refs(tmp_path):
    repo = Repo(DirStore(tmp_path / "store", query_index="sqlite"))
    child = repo.save_object(ReferenceQueryLeaf(3, repo=repo))
    aggregate = repo.save_object(ReferenceQueryWrapper(child.object, repo=repo))
    repo.set_alias("aggregate", aggregate)

    path = GraphPath((Parameter("child"),))
    assert list(repo.references().contains(child.object).object_refs()) == [aggregate.object]
    assert list(repo.references().alias("aggregate").object_refs()) == [aggregate.object]
    assert list(repo.references().path(path).object_refs()) == [child.object]

    ephemeral = repo.save_object(ReferenceQueryWrapper("value", repo=repo))
    assert ephemeral.object.objects == {}
    assert repo.references().exact(ephemeral.object).object_refs().one() == ephemeral.object
    assert repo.references().definition(ephemeral.definition).state_refs().one() == ephemeral


def test_object_terminal_preserves_nested_ref_state_reference(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(ReferenceQueryLeaf(3, repo=repo))
    parent = repo.save_object(ReferenceQueryRefParent(Ref(state), repo=repo))

    loaded = repo.query(parent.definition).stored().objects(cache="none").one()

    assert loaded.selected == state
    assert loaded.definition.parameters["selected"].target == state
