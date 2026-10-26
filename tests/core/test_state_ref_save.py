from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Object, Repo, Serializable, require_saved
from dryml.core.node_prepare import NodePreparation
from dryml.core.utils.graph.path import GraphPath, Parameter
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore
from dryml.core.store.store import StoreCapabilityError


class CountingState(Serializable):
    saves = 0

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).saves += 1
        Path(dest_dir, "value.txt").write_text(str(self.value))


class EphemeralRoot(Object):
    def __init__(self, children):
        self.children = children


class ImportedWrapper(Object):
    def __init__(self, child):
        self.child = child


def test_imported_state_ref_rebases_live_nested_bindings_without_owning_ref_edges(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    leaf = CountingState("nested", repo=repo)
    imported = ImportedWrapper(EphemeralRoot(leaf, repo=repo), repo=repo)
    imported_state = imported.save(repo=repo)
    outer = ImportedWrapper(imported_state, repo=repo)
    imported_path = GraphPath((Parameter("child"),))
    nested_path = imported_path.child(Parameter("child")).child(Parameter("children"))

    state = outer.save(repo=repo, deep_capture=True)

    assert outer.graph_at(imported_path) is outer.child
    assert outer.graph_at(nested_path) is outer.child.child.children
    assert state.at(nested_path).state.key.codec_hash == outer.child.child.children._last_state_hash


def test_stateful_save_publishes_verified_exact_state_ref(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = CountingState("value", repo=repo)

    state = obj.save(repo=repo)

    assert require_saved(state) is state
    assert state.object_projection_equal(repo.describe(obj))
    assert state.state.key.codec_hash == obj._last_state_hash
    digest = NodePreparation((state,)).digest(state, domain="dryml-saved-root/v1")
    assert store.read_node(digest, kind="saved").graph_equal(state)
    assert obj.last_state_ref.graph_equal(state)
    payload = Path(store.state_data_path(state.state.key))
    assert payload.parent.parent.name == state.state.key.cdef_digest
    assert (payload / "value.txt").read_text() == "value"
    assert not (tmp_path / "store" / "local-state").exists()


def test_ephemeral_root_has_topology_but_only_child_state_paths(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = CountingState("child", repo=repo)
    root = EphemeralRoot(child, repo=repo)

    state = root.save(repo=repo)

    assert state.object_id is None
    assert len(state.state_spec()) == 1
    assert next(iter(state.state_spec())) in state.identity_spec()


def test_all_ephemeral_graph_publishes_an_empty_exact_state_mapping(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    root = EphemeralRoot("value", repo=repo)

    state = root.save(repo=repo)

    assert not state.identity_spec()
    assert not state.state_spec()
    digest = NodePreparation((state,)).digest(state, domain="dryml-saved-root/v1")
    assert store.read_node(digest, kind="saved").graph_equal(state)


def test_shared_node_is_saved_once_and_independent_equal_nodes_twice(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    CountingState.saves = 0
    shared = CountingState("same", repo=repo)
    EphemeralRoot([shared, shared], repo=repo).save(repo=repo)
    assert CountingState.saves == 1

    CountingState.saves = 0
    first = CountingState("same", repo=repo)
    second = CountingState("same", repo=repo)
    state = EphemeralRoot([first, second], repo=repo).save(repo=repo)
    assert CountingState.saves == 2
    assert len(state.state_spec()) == 2


def test_deep_capture_serializes_reused_owned_nodes_again(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = CountingState("child", repo=repo)
    root = EphemeralRoot(child, repo=repo)
    CountingState.saves = 0

    root.save(repo=repo)
    root.save(repo=repo, deep_capture=True)

    assert CountingState.saves == 2


def test_zip_retained_save_refuses_before_capture(tmp_path):
    archive = tmp_path / "state.zip"
    store = ZipStore(archive)
    repo = Repo(store)
    before = CountingState.saves
    try:
        with pytest.raises(StoreCapabilityError, match="durable"):
            CountingState("zip", repo=repo).save(repo=repo)
        assert CountingState.saves == before
        assert not archive.exists()
    finally:
        store.close()
