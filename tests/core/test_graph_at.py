import pytest

from dryml.core import Definition, Object, Repo
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
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
    parent = GraphAtParent(
        child, DefLink.finalized(EdgeKind.REF, target), GraphAtLeaf, repo=repo
    )

    monkeypatch.setattr(repo, "get_cached", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    assert parent.graph_at() is parent
    assert parent.graph_at('$[@param("child")]') is child
    assert parent.graph_at('$[@param("ref")]') is target
    assert parent.graph_at('$[@param("cls")]') is GraphAtLeaf
    with pytest.raises(GraphPathError):
        parent.graph_at('$[@param("missing")]')


def test_node_at_boundaries_are_independent_and_endpoints_are_nodes():
    leaf = Definition(value=1)
    middle = Definition(child=leaf)
    root = Definition(ref=middle.ref(), quoted=middle.quote(), scalar=3, container=[leaf])
    assert root.at("$.ref") is middle
    assert root.at("$.quoted") is middle
    with pytest.raises(GraphPathError, match="reference"):
        root.at("$.ref.child")
    with pytest.raises(GraphPathError, match="quotation"):
        root.at("$.quoted.child", traverse_refs=True)
    assert root.at("$.ref.child", traverse_refs=True) is leaf
    assert root.at("$.quoted.child", traverse_quoted=True) is leaf
    for path in ("$.scalar", "$.container", "$.container[0].value"):
        with pytest.raises(GraphPathError, match="Node endpoint"):
            root.at(path)
    with pytest.raises(TypeError, match="bool"):
        root.at("$", traverse_refs=1)
    both = Definition(both=DefLink.finalized(EdgeKind.REF, middle.quote()))
    assert both.at("$.both") is middle
    with pytest.raises(GraphPathError, match="reference"):
        both.at("$.both.child", traverse_quoted=True)
    with pytest.raises(GraphPathError, match="quotation"):
        both.at("$.both.child", traverse_refs=True)
    assert both.at("$.both.child", traverse_refs=True, traverse_quoted=True) is leaf
