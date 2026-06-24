import pytest

from dryml.core2 import ConcreteDefinitionGraph, Definition, Object, Repo, SKIP_ARGS, SetMember
from dryml.core2.cdef_graph import ConcreteDefinitionGraphCycleError, iter_direct_cdef_edges
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.query.path import get_subtree


class GraphLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class GraphParent(Object):
    def __init__(self, child=None, *, label="parent"):
        super().__init__()
        self.child = child
        self.label = label


class GraphContainer(Object):
    def __init__(self, value):
        super().__init__()
        self.value = value


def test_single_cdef_graph_has_one_root_and_no_edges():
    repo = Repo()
    leaf = GraphLeaf("x", repo=repo)

    graph = ConcreteDefinitionGraph.from_root(leaf.definition)

    assert graph.roots == (leaf.definition,)
    assert [node.definition for node in graph.nodes()] == [leaf.definition]
    assert graph.edges() == ()


def test_direct_child_creates_two_nodes_and_one_edge():
    repo = Repo()
    child = GraphLeaf("child", repo=repo)
    parent = GraphParent(child, repo=repo)

    graph = ConcreteDefinitionGraph.from_root(parent.definition)

    edge = graph.edges()[0]

    assert len(graph.nodes()) == 2
    assert edge.parent == parent.definition
    assert str(edge.path) == "$.args[0]"
    assert edge.child == child.definition
    assert graph.resolve(parent.definition, edge.path) == child.definition


def test_grandchild_uses_two_direct_edges_not_one_transitive_edge():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    mid = GraphParent(leaf, repo=repo)
    root = GraphParent(mid, repo=repo)

    graph = ConcreteDefinitionGraph.from_root(root.definition)

    edge_pairs = {(edge.parent, edge.child) for edge in graph.edges()}


    assert len(graph.edges()) == 2
    assert (root.definition, mid.definition) in edge_pairs
    assert (mid.definition, leaf.definition) in edge_pairs
    assert (root.definition, leaf.definition) not in edge_pairs


def test_same_child_at_two_paths_has_one_node_and_two_edges():
    repo = Repo()
    child = GraphLeaf("shared", repo=repo)
    parent = GraphContainer([child, child], repo=repo)

    graph = ConcreteDefinitionGraph.from_root(parent.definition)
    occurrences = tuple(graph.iter_occurrences(target=child.definition))

    assert len([node for node in graph.nodes() if node.definition == child.definition]) == 1
    assert len([edge for edge in graph.edges() if edge.child == child.definition]) == 2
    assert {str(occ.path) for occ in occurrences} == {"$.args[0][0]", "$.args[0][1]"}


def test_multi_root_graph_deduplicates_shared_nodes():
    repo = Repo()
    shared = GraphLeaf("shared", repo=repo)
    left = GraphParent(shared, label="left", repo=repo)
    right = GraphParent(shared, label="right", repo=repo)

    graph = ConcreteDefinitionGraph.from_roots((left.definition, right.definition))

    definitions = [node.definition for node in graph.nodes()]

    assert definitions.count(shared.definition) == 1
    assert graph.roots == (left.definition, right.definition)


def test_child_inside_set_uses_semantic_set_member_path():
    repo = Repo()
    child = GraphLeaf("set", repo=repo)
    parent = GraphContainer({child}, repo=repo)

    graph = ConcreteDefinitionGraph.from_root(parent.definition)
    edge = graph.edges()[0]

    assert isinstance(edge.path[-1], SetMember)
    assert get_subtree(parent.definition, edge.path) == child.definition


def test_walk_nodes_postorder_dependencies_first():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    mid = GraphParent(leaf, repo=repo)
    root = GraphParent(mid, repo=repo)

    graph = ConcreteDefinitionGraph.from_root(root.definition)

    assert [node.definition for node in graph.walk_nodes(order="post")] == [leaf.definition, mid.definition, root.definition]
    assert graph.topological_order(dependencies_first=True) == (leaf.definition, mid.definition, root.definition)


def test_repo_definition_graph_rejects_plain_definition():
    repo = Repo()

    with pytest.raises(TypeError, match="concretize"):
        repo.definition_graph(Definition(GraphLeaf, "x"))


def test_iter_direct_cdef_edges_stops_at_child_boundary():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    mid = GraphParent(leaf, repo=repo)
    root = GraphParent(mid, repo=repo)

    edges = tuple(iter_direct_cdef_edges(root.definition))

    assert edges == ((edges[0][0], mid.definition),)
    assert str(edges[0][0]) == "$.args[0]"


def test_exact_graph_builder_rejects_plain_definition_in_cdef():
    cdef = ConcreteDefinition(GraphContainer, FrozenTuple((Definition(GraphLeaf, "x"),)), FrozenDict({}))

    with pytest.raises(Exception, match="Plain Definition"):
        ConcreteDefinitionGraph.from_root(cdef)
