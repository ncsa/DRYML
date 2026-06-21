import pytest

from dryml.core2 import Object, Repo, Serializable
from dryml.core2.policies import RepoLoadOptions, RepoSaveOptions
from dryml.core2.repo import RepoGraphError, default_repo, get_default_repo
from dryml.core2.store.dir import DirStore
from dryml.core2.utils.general import pickle_load, pickle_save


class GraphLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class GraphNode(Object):
    def __init__(self, name, *children):
        super().__init__()
        self.name = name
        self.children = children


class RepoAware(Object):
    def __init__(self):
        super().__init__()
        self.seen_repo = get_default_repo()


class PersistentLeaf(Serializable):
    def __init__(self, name, *, value=0):
        super().__init__()
        self.name = name
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        pickle_save(self.value, f"{dest_dir}/value.pkl")

    def restore_state_from_dir_imp(self, src_dir, revision=None):
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


def test_apply_graph_returns_results_by_definition_in_visit_order():
    repo = Repo()
    left = GraphLeaf("left", repo=repo)
    right = GraphLeaf("right", repo=repo)
    root = GraphNode("root", left, right, repo=repo)

    results = repo.apply_graph(root, lambda obj: obj.name)

    assert list(results.values()) == ["left", "right", "root"]


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


def test_get_and_apply_share_load_options():
    repo = Repo()
    weak = GraphLeaf("weak", repo=repo)
    strong = GraphLeaf("strong", repo=repo)
    repo.pin(strong)

    options = RepoLoadOptions(reuse_weak=False)

    selected = repo.get(options=options)
    applied = repo.apply(lambda obj: obj.name, options=options)

    assert weak.definition not in selected
    assert strong.definition in selected
    assert applied == {strong.definition: "strong"}


def test_save_object_uses_save_options(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo()
    obj = GraphLeaf("saved", repo=repo)

    repo.save_object(obj, options=RepoSaveOptions(store=store, alias="saved"))

    assert store.has(obj.definition)
    assert repo.get_alias("saved") == obj.definition


def test_save_uses_save_options_for_loaded_objects(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo()
    obj = GraphLeaf("saved", repo=repo)
    repo.pin(obj)

    repo.save(options=RepoSaveOptions(store=store))

    assert store.has(obj.definition)


def test_save_skips_ephemeral_child_under_serializable_parent(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = GraphLeaf("child", repo=repo)
    parent = PersistentNode("parent", child, repo=repo)

    repo.save_object(parent)

    assert store.has(parent.definition)
    assert not store.has(child.definition)

    repo2 = Repo(stores=DirStore(store.base_dir))
    loaded = repo2.load_object(parent.definition)

    assert loaded.child.name == "child"


def test_save_traverses_skipped_ephemeral_to_save_serializable_descendant(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    leaf = PersistentLeaf("leaf", repo=repo)
    leaf.value = 42
    child = GraphNode("ephemeral", leaf, repo=repo)
    parent = PersistentNode("parent", child, repo=repo)

    repo.save_object(parent)

    assert store.has(parent.definition)
    assert not store.has(child.definition)
    assert store.has(leaf.definition)

    leaf.value = 0
    repo2 = Repo(stores=DirStore(store.base_dir))
    loaded = repo2.load_object(parent.definition)

    assert loaded.child.children[0].value == 42


def test_explicit_ephemeral_root_save_creates_object_dir(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = GraphLeaf("root", repo=repo)

    repo.save_object(obj)

    assert store.has(obj.definition)


def test_live_ephemeral_alias_forces_save(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = GraphLeaf("aliased", repo=repo)

    repo.set_alias("leaf", obj)
    repo.close(flush=True)

    assert store.has(obj.definition)
    repo2 = Repo(stores=DirStore(store.base_dir))
    loaded = repo2.load_alias("leaf")
    assert loaded.name == "aliased"


def test_ephemeral_depth_none_saves_all_ephemeral_descendants(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = GraphLeaf("child", repo=repo)
    parent = GraphNode("parent", child, repo=repo)

    repo.save_object(parent, options=RepoSaveOptions(store=store, ephemeral_depth=None))

    assert store.has(parent.definition)
    assert store.has(child.definition)
