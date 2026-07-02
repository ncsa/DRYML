import pytest

from dryml.core2 import Definition, Object, Repo, SKIP_ARGS
from dryml.core2.query.path import DefinitionPath, Kwarg
from dryml.core2.query.fingerprint import selector_local_requirements
from dryml.core2.query.selector_graph import (
    SelectorGraph,
    SelectorGraphCycleError,
    SelectorGraphEdge,
    SelectorGraphError,
    SelectorGraphNode,
    compile_selector_graph,
)


class SelectorLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class SelectorParent(Object):
    def __init__(self, child=None, *, name="parent"):
        super().__init__()
        self.child = child
        self.name = name


def test_selector_without_nested_definitions_has_one_node():
    graph = compile_selector_graph(Definition(SelectorLeaf, SKIP_ARGS, name="x"))

    assert graph is not None
    assert len(graph.nodes) == 1
    assert graph.edges == ()


def test_nested_definition_creates_selector_edge():
    selector = Definition(SelectorParent, SKIP_ARGS, child=Definition(SelectorLeaf, SKIP_ARGS, name="x"))

    graph = compile_selector_graph(selector)

    assert graph is not None
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert str(graph.edges[0].path) == "$.child"
    assert str(graph.node(graph.edges[0].child).source_path) == "$.child"


def test_nested_exact_cdef_creates_exact_node():
    repo = Repo()
    child = SelectorLeaf("exact", repo=repo)
    selector = Definition(SelectorParent, SKIP_ARGS, child=child.definition)

    graph = compile_selector_graph(selector)

    assert graph is not None
    child_node = graph.node(graph.edges[0].child)
    assert child_node.exact_definition == child.definition


def test_two_equal_selector_occurrences_remain_distinct_nodes():
    child_selector = Definition(SelectorLeaf, SKIP_ARGS)
    selector = Definition(SelectorParent, [child_selector, child_selector])

    graph = compile_selector_graph(selector)

    assert graph is not None
    assert len(graph.nodes) == 3
    assert {str(edge.path) for edge in graph.edges} == {"$.args[0][0]", "$.args[0][1]"}


def test_nested_exact_cdef_inside_set_has_path_edge():
    repo = Repo()
    child = SelectorLeaf("set", repo=repo)
    selector = Definition(SelectorParent, {child.definition})

    graph = compile_selector_graph(selector)
    assert graph is not None
    assert len(graph.edges) == 1
    assert graph.edges[0].unordered is False
    assert str(graph.edges[0].path).startswith('$.args[0][@set("')


def test_selector_graph_local_requirements_use_shared_walker():
    selector = Definition(SelectorParent, SKIP_ARGS, child=Definition(SelectorLeaf, SKIP_ARGS, name="x"))

    graph = compile_selector_graph(selector)

    assert graph is not None
    assert graph.node(graph.root).local_requirements == selector_local_requirements(selector)


def test_selector_graph_rejects_cycle_with_source_path():
    selector = Definition(SelectorParent, SKIP_ARGS)
    with pytest.raises(TypeError):
        selector.kwargs["child"] = selector


def test_selector_graph_rejects_self_referential_list():
    items = []
    items.append(items)
    with pytest.raises(Exception, match="Cycle"):
        Definition(SelectorParent, SKIP_ARGS, child=items)


def test_selector_graph_rejects_self_referential_mapping():
    mapping = {}
    mapping["self"] = mapping
    with pytest.raises(Exception, match="Cycle"):
        Definition(SelectorParent, SKIP_ARGS, child=mapping)


def test_shared_acyclic_container_is_allowed():
    shared = ["x"]
    selector = Definition(SelectorParent, [shared, shared])

    graph = compile_selector_graph(selector)

    assert graph is not None


def test_selector_graph_rejects_invalid_edge_endpoint():
    selector = Definition(SelectorLeaf, SKIP_ARGS)
    node = SelectorGraphNode(0, DefinitionPath(), selector, ())

    with pytest.raises(SelectorGraphError, match="child"):
        SelectorGraph(0, (node,), (SelectorGraphEdge(0, DefinitionPath((Kwarg("x"),)), 1),))


def test_selector_graph_rejects_direct_cycle():
    parent = Definition(SelectorParent, SKIP_ARGS)
    child = Definition(SelectorLeaf, SKIP_ARGS)
    nodes = (
        SelectorGraphNode(0, DefinitionPath(), parent, ()),
        SelectorGraphNode(1, DefinitionPath((Kwarg("child"),)), child, ()),
    )
    edges = (
        SelectorGraphEdge(0, DefinitionPath((Kwarg("child"),)), 1),
        SelectorGraphEdge(1, DefinitionPath((Kwarg("parent"),)), 0),
    )

    with pytest.raises(SelectorGraphCycleError, match="cycle"):
        SelectorGraph(0, nodes, edges)
