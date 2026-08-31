import pytest

from dryml.core import Definition, Object, Repo
from dryml.core.links import Ref
from dryml.core.utils.graph.path import GraphPathError


class GraphAtLeaf(Object):
    def __init__(self, value):
        super().__init__()
        self.value = value


class GraphAtParent(Object):
    def __init__(self, child, ref, cls):
        super().__init__()
        self.child = child
        self.ref = ref
        self.cls = cls


def test_graph_at_returns_completed_endpoint_forms_without_lookup(monkeypatch):
    repo = Repo()
    child = GraphAtLeaf("child", repo=repo)
    target = Definition(GraphAtLeaf, "target").concretize(repo=repo)
    parent = GraphAtParent(child, Ref(target), GraphAtLeaf, repo=repo)

    monkeypatch.setattr(repo, "get_cached", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    assert parent.graph_at() is parent
    assert parent.graph_at('$[@param("child")]') is child
    assert parent.graph_at('$[@param("ref")]') is target
    assert parent.graph_at('$[@param("cls")]') is GraphAtLeaf
    with pytest.raises(GraphPathError):
        parent.graph_at('$[@param("missing")]')
