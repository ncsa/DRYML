import pytest

from dryml.core import ConcreteDefinition, Definition, Object, Repo, Serializable
from dryml.core.cdef_identity import cdef_node_key
from dryml.core.object import WorkspaceCapable
from dryml.core.query.path import GraphPath, Index, Key, Parameter, SetMember, get_subtree
from dryml.core.repo import RepoGraphError, RepoSaveError, default_repo, get_default_repo
from dryml.core.repo_plan import GraphApplyResult, collect_runtime_roots
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord
from dryml.core.utils.general import pickle_load, pickle_save
from dryml.core.workspaces import WorkspaceManager


class GraphLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class GraphNode(Object):
    def __init__(self, name, *children):
        super().__init__()
        self.name = name
        self.children = children


class WorkspaceLeaf(Object, WorkspaceCapable):
    def __init__(self, name):
        super().__init__()
        self.name = name


class RepoAware(Object):
    def __init__(self):
        super().__init__()
        self.seen_repo = get_default_repo()


class PersistentLeaf(Serializable):
    def __init__(self, name, *, value=0):
        super().__init__()
        self.name = name
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pickle_save(self.value, f"{dest_dir}/value.pkl")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = pickle_load(f"{src_dir}/value.pkl")


class PersistentNode(Serializable):
    def __init__(self, name, child):
        super().__init__()
        self.name = name
        self.child = child


def test_iter_graph_walks_rooted_object_graph_postorder():
    repo = Repo()
    left = GraphLeaf("left", repo=repo)
    right = GraphLeaf("right", repo=repo)
    root = GraphNode("root", left, right, repo=repo)

    assert [obj.name for obj in repo.iter_graph(root)] == ["left", "right", "root"]


def test_iter_graph_can_exclude_root_and_dedupes_shared_children():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", leaf, leaf, repo=repo)

    assert [obj.name for obj in repo.iter_graph(root, include_root=False)] == ["leaf"]


def test_iter_graph_occurrence_mode_visits_shared_child_twice():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", leaf, leaf, repo=repo)

    assert [obj.name for obj in repo.iter_graph(root, include_root=False, dedupe=False)] == ["leaf", "leaf"]


def test_iter_graph_preorder_visits_parent_before_children():
    repo = Repo()
    left = GraphLeaf("left", repo=repo)
    right = GraphLeaf("right", repo=repo)
    root = GraphNode("root", left, right, repo=repo)

    assert [obj.name for obj in repo.iter_graph(root, order="pre")] == ["root", "left", "right"]


def test_apply_graph_returns_results_by_definition_in_visit_order():
    repo = Repo()
    left = GraphLeaf("left", repo=repo)
    right = GraphLeaf("right", repo=repo)
    root = GraphNode("root", left, right, repo=repo)

    results = repo.apply_graph(root, lambda obj: obj.name)

    assert list(results.values()) == ["left", "right", "root"]


def test_apply_graph_occurrence_mode_preserves_repeated_results():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", leaf, leaf, repo=repo)

    results = repo.apply_graph(root, lambda obj: obj.name, include_root=False, dedupe=False)

    assert all(isinstance(result, GraphApplyResult) for result in results)
    assert [result.value for result in results] == ["leaf", "leaf"]
    assert [str(result.path) for result in results] == ["$[@param(\"children\")][0]", "$[@param(\"children\")][1]"]


def test_repo_occurrence_path_preserves_arg_segment():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", leaf, repo=repo)

    result = repo.apply_graph(root, lambda obj: obj.name, include_root=False, dedupe=False)[0]

    assert isinstance(result.path[0], Parameter)
    assert get_subtree(root.definition, result.path) == leaf.definition


def test_repo_occurrence_path_preserves_mapping_key_segment():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", {5: leaf}, repo=repo)

    result = repo.apply_graph(root, lambda obj: obj.name, include_root=False, dedupe=False)[0]

    assert isinstance(result.path[0], Parameter)
    assert result.path[1] == Index(0)
    assert result.path[2] == Key(5)
    assert get_subtree(root.definition, result.path) == leaf.definition


def test_repo_occurrence_path_preserves_set_member_segment():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", {leaf}, repo=repo)

    result = repo.apply_graph(root, lambda obj: obj.name, include_root=False, dedupe=False)[0]

    assert isinstance(result.path[0], Parameter)
    assert isinstance(result.path[1], Index)
    assert isinstance(result.path[2], SetMember)
    assert get_subtree(root.definition, result.path) == leaf.definition


def test_runtime_root_path_for_integer_dict_key_is_key_not_index():
    leaf = GraphLeaf("leaf")

    roots = collect_runtime_roots({5: leaf})

    assert roots[0].path == GraphPath((Key(5),))


def test_runtime_root_set_paths_are_stable_set_members():
    leaf = GraphLeaf("leaf")

    first = collect_runtime_roots({leaf})
    second = collect_runtime_roots({leaf})

    assert isinstance(first[0].path[0], SetMember)
    assert first[0].path == second[0].path


def test_add_objects_retains_independent_equal_instances():
    repo = Repo()
    first = GraphLeaf("same", repo=repo)
    second = GraphLeaf("same")
    repo.add_objects(first)

    repo.add_objects(second)

    assert repo.strong_obj_cache.candidates(first.definition) == (first,)
    assert repo.strong_obj_cache.candidates(second.definition) == (second,)


def test_binding_keeps_two_live_instances_with_independent_private_nodes():
    repo = Repo()
    first = GraphLeaf("same")
    second = GraphLeaf("same")

    repo.add_objects([first, second])

    assert first in repo.strong_obj_cache.candidates(first.definition)
    assert second in repo.strong_obj_cache.candidates(second.definition)


def test_workspace_labels_distinguish_private_nodes_without_exposing_tokens(tmp_path):
    repo = Repo()
    repo.workspace_manager = WorkspaceManager(tmp_path / "workspaces")

    direct_first = WorkspaceLeaf("same", repo=repo)
    direct_second = WorkspaceLeaf("same", repo=repo)
    shared = Definition(WorkspaceLeaf, "same")
    aliased = repo.load_or_build(
        Definition(GraphNode, "aliased", shared, shared), cache="none"
    )
    first = Definition(WorkspaceLeaf, "same")
    second = Definition(WorkspaceLeaf, "same")
    distinct = repo.load_or_build(
        Definition(GraphNode, "distinct", first, second), cache="none"
    )

    assert direct_first.workspace != direct_second.workspace
    assert aliased.children[0].workspace == aliased.children[1].workspace
    assert distinct.children[0].workspace != distinct.children[1].workspace
    assert direct_first.workspace != aliased.children[0].workspace
    assert aliased.children[0].workspace != distinct.children[0].workspace
    for obj in (
            direct_first, direct_second, aliased.children[0],
            distinct.children[0], distinct.children[1]
    ):
        assert repr(cdef_node_key(obj.definition)) not in obj.workspace
        assert obj._realization_scope.token not in obj.workspace


def test_get_cached_is_exact_and_rejects_cross_tier_ambiguity():
    repo = Repo()
    first = GraphLeaf("same", repo=repo)
    second = GraphLeaf("same", repo=repo)
    repo.cache_strong(first)
    repo.cache_strong(second)

    assert first.definition == second.definition
    assert repo.get_cached(first.definition) is first
    assert repo.get_cached(second.definition) is second

    duplicate = GraphLeaf("same", repo=repo, __cdef__=first.definition)
    repo.cache_weak(duplicate)

    assert repo.get_cached(first.definition) is None
    assert not repo.has_cached(first.definition)


def test_independent_current_nodes_keep_workspace_and_store_bindings(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)

    class WorkspaceHandle:
        def __init__(self, key):
            self.key = key

        def path(self):
            return str(tmp_path / "workspaces" / self.key)

    class WorkspaceManager:
        def alloc(self, key, *, scope=None, node_key=None):
            return WorkspaceHandle(
                f"{key}/{scope.workspace_label}" if scope else key
            )

    repo.workspace_manager = WorkspaceManager()
    first = Definition(WorkspaceLeaf, "same").concretize(repo=repo)
    second = Definition(WorkspaceLeaf, "same").concretize(repo=repo)

    first_obj = repo.load_or_build(first)
    second_obj = repo.load_or_build(second)
    repo.pin(first_obj)
    repo.set_object_store(first, store)
    repo.set_object_store(second, store)

    assert first == second
    assert cdef_node_key(first) is not cdef_node_key(second)
    assert repo.strong_obj_cache[first] is first_obj
    assert repo.weak_obj_cache[second] is second_obj
    assert repo.obj_default_store[first] is store
    assert repo.obj_default_store[second] is store
    assert first_obj.workspace != second_obj.workspace


def test_add_objects_assigns_store_to_unique_graph_nodes(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo()
    child = GraphLeaf("child", repo=repo)
    root = GraphNode("root", child, repo=repo)

    repo.add_objects(root, store=store)

    assert repo.obj_default_store[root.definition] is store
    assert repo.obj_default_store[child.definition] is store


def test_iter_graph_missing_cdef_raises_by_default():
    repo = Repo()
    leaf = GraphLeaf("leaf")

    with pytest.raises(RepoGraphError):
        list(repo.iter_graph(leaf.definition))


def test_iter_graph_missing_cdef_can_skip():
    repo = Repo()
    leaf = GraphLeaf("leaf")

    assert list(repo.iter_graph(leaf.definition, missing="skip")) == []


def test_iter_graph_missing_cdef_can_load_from_store(tmp_path):
    store = DirStore(tmp_path / "store")
    repo1 = Repo(stores=store)
    leaf = GraphLeaf("leaf", repo=repo1)
    repo1.save_object(leaf)

    repo2 = Repo(stores=DirStore(store.base_dir))
    loaded = list(repo2.iter_graph(leaf.definition, missing="load"))

    assert len(loaded) == 1
    assert loaded[0].name == "leaf"


def test_object_init_sees_explicit_construction_repo_and_restores_outer_repo():
    outer_repo = Repo()
    construction_repo = Repo()

    with default_repo(outer_repo):
        obj = RepoAware(repo=construction_repo)
        assert obj.seen_repo is construction_repo
        assert get_default_repo() is outer_repo




def test_save_object_uses_direct_state_ref_controls(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo()
    obj = GraphLeaf("saved", repo=repo)

    state = repo.save_object(obj, store=store, alias="saved")

    assert store.read_definition_record(DefinitionRecord(obj.definition).digest) is not None
    assert store.read_object_alias("saved").object_ref == state.object


def test_save_object_without_store_raises_repo_save_error():
    repo = Repo()
    obj = GraphLeaf("unsaved", repo=repo)

    with pytest.raises(RepoSaveError, match="No Store available"):
        repo.save_object(obj)


def test_save_uses_direct_controls_for_one_loaded_object(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo()
    obj = GraphLeaf("saved", repo=repo)
    repo.pin(obj)

    repo.save(obj, store=store)

    assert store.read_definition_record(DefinitionRecord(obj.definition).digest) is not None


def test_save_skips_ephemeral_child_under_serializable_parent(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = GraphLeaf("child", repo=repo)
    parent = PersistentNode("parent", child, repo=repo)

    state = repo.save_object(parent)

    assert store.read_state_ref_record(state.digest()).state_ref == state
    assert len(state.states) == 1


def test_save_traverses_skipped_ephemeral_to_save_serializable_descendant(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    leaf = PersistentLeaf("leaf", repo=repo)
    leaf.value = 42
    child = GraphNode("ephemeral", leaf, repo=repo)
    parent = PersistentNode("parent", child, repo=repo)

    state = repo.save_object(parent)

    assert len(state.states) == 2
    assert leaf._last_state_hash in state.states.values()


def test_explicit_ephemeral_root_save_creates_object_dir(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = GraphLeaf("root", repo=repo)

    state = repo.save_object(obj)

    assert state.states == {}


def test_live_ephemeral_alias_forces_save(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = GraphLeaf("aliased", repo=repo)

    state = repo.save_object(obj, alias="leaf")

    assert store.read_object_alias("leaf").object_ref == state.object


def test_retired_ephemeral_depth_is_not_a_save_option(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = GraphLeaf("child", repo=repo)
    parent = GraphNode("parent", child, repo=repo)

    with pytest.raises(TypeError):
        repo.save_object(parent, ephemeral_depth=None)
