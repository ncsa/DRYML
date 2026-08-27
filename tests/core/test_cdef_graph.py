import pytest

from dryml.core import ConcreteDefinitionGraph, Definition, Object, Repo, SKIP_ARGS, SetMember
from dryml.core.cdef_graph import (
    CDefEdge,
    CDefNode,
    ConcreteDefinitionGraphCycleError,
    ConcreteDefinitionGraphError,
    EdgeKind,
    GraphClosure,
    as_query_index_graph,
    iter_direct_cdef_edges,
)
from dryml.core.bound_args import BoundArguments, bind_complete_arguments
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.query.path import Arg, GraphPath, Kwarg, Parameter, get_subtree


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
    assert str(edge.path) == '$[@param("child")]'
    assert edge.child == child.definition
    assert graph.resolve(parent.definition, edge.path) == child.definition


def test_v2_parent_uses_semantic_parameter_edge_and_keeps_legacy_child_identity():
    child = ConcreteDefinition._from_persisted_record(GraphLeaf, ("child",), {})
    parent = ConcreteDefinition._from_persisted_record(
        GraphParent,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("child", child),)),
    )

    graph = ConcreteDefinitionGraph.from_root(parent)
    edge = graph.edges()[0]

    assert edge.path == GraphPath((Parameter("child"),))
    assert graph.resolve(parent, edge.path) == child
    assert {node.definition for node in graph.nodes()} == {parent, child}


def test_equivalent_v2_call_spellings_produce_the_same_semantic_edge_path():
    child = ConcreteDefinition._from_persisted_record(GraphLeaf, ("child",), {})
    positional = ConcreteDefinition._from_persisted_record(
        GraphParent,
        identity_version=V2_IDENTITY_VERSION,
        parameters=bind_complete_arguments(GraphParent, (child,), {}),
    )
    keyword = ConcreteDefinition._from_persisted_record(
        GraphParent,
        identity_version=V2_IDENTITY_VERSION,
        parameters=bind_complete_arguments(GraphParent, (), {"child": child}),
    )

    assert positional == keyword
    assert ConcreteDefinitionGraph.from_root(positional).edges()[0].path == GraphPath((Parameter("child"),))
    assert ConcreteDefinitionGraph.from_root(keyword).edges()[0].path == GraphPath((Parameter("child"),))


def test_mixed_version_parent_paths_follow_each_parent_identity_version():
    leaf = ConcreteDefinition._from_persisted_record(GraphLeaf, ("leaf",), {})
    v2_parent = ConcreteDefinition._from_persisted_record(
        GraphParent,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("child", leaf),)),
    )
    v1_root = ConcreteDefinition._from_persisted_record(GraphParent, (v2_parent,), {})

    graph = ConcreteDefinitionGraph.from_root(v1_root)

    assert {str(edge.path) for edge in graph.edges()} == {"$.args[0]", '$[@param("child")]'}
    assert {node.definition for node in graph.nodes()} == {v1_root, v2_parent, leaf}


def test_v2_variadic_parameter_buckets_produce_stable_child_edges():
    first = ConcreteDefinition._from_persisted_record(GraphLeaf, ("first",), {})
    second = ConcreteDefinition._from_persisted_record(GraphLeaf, ("second",), {})
    parent = ConcreteDefinition._from_persisted_record(
        GraphContainer,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((
            ("sources", FrozenTuple((first, second))),
            ("capabilities", FrozenDict({"encoder": second})),
        )),
    )

    graph = ConcreteDefinitionGraph.from_root(parent)

    assert {str(edge.path) for edge in graph.edges()} == {
        '$[@param("sources")][0]',
        '$[@param("sources")][1]',
        '$[@param("capabilities")][\'encoder\']',
    }


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
    assert {str(occ.path) for occ in occurrences} == {
        '$[@param("value")][0]',
        '$[@param("value")][1]',
    }


def test_primary_path_of_repeated_child_returns_first_path():
    repo = Repo()
    child = GraphLeaf("shared", repo=repo)
    parent = GraphContainer([child, child], repo=repo)

    graph = ConcreteDefinitionGraph.from_root(parent.definition)

    assert str(graph.primary_path(parent.definition, child.definition)) == '$[@param("value")][0]'


def test_multi_root_graph_deduplicates_shared_nodes():
    repo = Repo()
    shared = GraphLeaf("shared", repo=repo)
    left = GraphParent(shared, label="left", repo=repo)
    right = GraphParent(shared, label="right", repo=repo)

    graph = ConcreteDefinitionGraph.from_roots((left.definition, right.definition))

    definitions = [node.definition for node in graph.nodes()]

    assert definitions.count(shared.definition) == 1
    assert graph.roots == (left.definition, right.definition)


def test_query_index_graph_rebuilds_for_requested_root_order():
    repo = Repo()
    shared = GraphLeaf("shared", repo=repo)
    left = GraphParent(shared, label="left", repo=repo)
    right = GraphParent(shared, label="right", repo=repo)
    graph = ConcreteDefinitionGraph.for_query_index_roots((left.definition, right.definition))

    reordered = as_query_index_graph(graph, (right.definition, left.definition))

    assert reordered is not graph
    assert reordered.closure is GraphClosure.QUERY_INDEX
    assert reordered.roots == (right.definition, left.definition)


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

    assert edges == ((edges[0][0], mid.definition, EdgeKind.MATERIALIZE),)
    assert str(edges[0][0]) == '$[@param("child")]'


def test_public_exact_graph_builder_binds_nested_plain_definitions():
    cdef = ConcreteDefinition(GraphContainer, FrozenTuple((Definition(GraphLeaf, "x"),)), FrozenDict({}))

    assert cdef.identity_version == V2_IDENTITY_VERSION
    assert isinstance(cdef.parameters["value"], ConcreteDefinition)


def test_graph_rejects_missing_root_node():
    cdef = ConcreteDefinition(GraphLeaf, FrozenTuple(("x",)), FrozenDict({}))

    with pytest.raises(ConcreteDefinitionGraphError, match="root"):
        ConcreteDefinitionGraph((cdef,), (), ())


def test_graph_rejects_missing_edge_endpoint():
    parent = ConcreteDefinition(GraphParent, FrozenTuple(()), FrozenDict({}))
    child = ConcreteDefinition(GraphLeaf, FrozenTuple(("x",)), FrozenDict({}))

    with pytest.raises(ConcreteDefinitionGraphError, match="child"):
        ConcreteDefinitionGraph(
            (parent,),
            (CDefNode(parent, parent.stable_hash()),),
            (CDefEdge(parent, GraphPath((Kwarg("child"),)), child),),
        )


def test_graph_rejects_direct_cycle():
    left = ConcreteDefinition(GraphLeaf, FrozenTuple(("left",)), FrozenDict({}))

    with pytest.raises(ConcreteDefinitionGraphCycleError, match="cycle"):
        ConcreteDefinitionGraph(
            (left,),
            (CDefNode(left, left.stable_hash()),),
            (CDefEdge(left, GraphPath(), left),),
        )


def test_graph_rejects_inconsistent_node_key():
    cdef = ConcreteDefinition(GraphLeaf, FrozenTuple(("x",)), FrozenDict({}))

    with pytest.raises(ConcreteDefinitionGraphError, match="stable_hash"):
        ConcreteDefinitionGraph((cdef,), (CDefNode(cdef, "wrong"),), ())


def test_graph_rejects_edge_whose_path_resolves_to_different_child():
    child1 = ConcreteDefinition(GraphLeaf, FrozenTuple(("one",)), FrozenDict({}))
    child2 = ConcreteDefinition(GraphLeaf, FrozenTuple(("two",)), FrozenDict({}))
    parent = ConcreteDefinition(GraphParent, FrozenTuple((child1,)), FrozenDict({}))

    with pytest.raises(ConcreteDefinitionGraphError, match="different child"):
        ConcreteDefinitionGraph(
            (parent,),
            (
                CDefNode(parent, parent.stable_hash()),
                CDefNode(child1, child1.stable_hash()),
                CDefNode(child2, child2.stable_hash()),
            ),
            (CDefEdge(parent, GraphPath((Parameter("child"),)), child2),),
        )


def test_graph_rejects_edge_path_resolving_to_scalar():
    child = ConcreteDefinition(GraphLeaf, FrozenTuple(("child",)), FrozenDict({}))
    parent = ConcreteDefinition(GraphParent, FrozenTuple(("scalar",)), FrozenDict({}))

    with pytest.raises(ConcreteDefinitionGraphError, match="ConcreteDefinition boundary"):
        ConcreteDefinitionGraph(
            (parent,),
            (CDefNode(parent, parent.stable_hash()), CDefNode(child, child.stable_hash())),
            (CDefEdge(parent, GraphPath((Parameter("child"),)), child),),
        )


def test_graph_rejects_node_unreachable_from_all_roots():
    root = ConcreteDefinition(GraphLeaf, FrozenTuple(("root",)), FrozenDict({}))
    unreachable = ConcreteDefinition(GraphLeaf, FrozenTuple(("unreachable",)), FrozenDict({}))

    with pytest.raises(ConcreteDefinitionGraphError, match="not reachable"):
        ConcreteDefinitionGraph(
            (root,),
            (CDefNode(root, root.stable_hash()), CDefNode(unreachable, unreachable.stable_hash())),
            (),
        )
