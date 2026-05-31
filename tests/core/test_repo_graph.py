import pytest

from dryml.core2 import Object, Repo
from dryml.core2.repo import RepoGraphError, default_repo, get_default_repo
from dryml.core2.store.dir import DirStore


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
