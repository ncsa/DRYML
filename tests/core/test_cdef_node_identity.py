from dryml.core import ConcreteDefinitionGraph, Definition, Object
from dryml.core.cdef_identity import cdef_node_key
from dryml.core.utils.graph import GraphPath, Index, Parameter


class NodeLeaf(Object):
    def __init__(self, value):
        self.value = value


class NodeParent(Object):
    def __init__(self, children):
        self.children = children


def test_independent_equal_definition_sources_remain_distinct_graph_nodes():
    """Canonicalization must preserve source identity before structural equality."""

    left = Definition(NodeLeaf, "same")
    right = Definition(NodeLeaf, "same")
    root = Definition(NodeParent, [left, right]).concretize()

    graph = ConcreteDefinitionGraph.from_root(root)

    assert len(graph.nodes()) == 3


def test_shared_and_independent_nodes_have_separate_graph_identity_projections():
    """Graph identity distinguishes alias topology without changing structural identity."""

    shared_leaf = Definition(NodeLeaf, "same")
    shared = Definition(NodeParent, [shared_leaf, shared_leaf]).concretize()
    independent = Definition(
        NodeParent,
        [Definition(NodeLeaf, "same"), Definition(NodeLeaf, "same")],
    ).concretize()

    assert shared == independent
    assert hash(shared) == hash(independent)
    assert shared.stable_hash() == independent.stable_hash()
    assert not shared.graph_equal(independent)
    assert shared.graph_hash() != independent.graph_hash()


def test_copy_graph_rekeys_every_node_and_cow_rekeys_only_changed_ancestors():
    """Explicit copies rekey whole graphs while path replacement preserves leaves."""

    leaf = Definition(NodeLeaf, "same")
    root = Definition(NodeParent, [leaf, leaf]).concretize()
    copied = root.copy_graph()
    changed = root.at(
        GraphPath((Parameter("children"), Index(0), Parameter("value")))
    ).set("changed")

    assert root.graph_equal(copied)
    assert cdef_node_key(root) is not cdef_node_key(copied)
    assert cdef_node_key(root.parameters["children"][0]) is not cdef_node_key(
        copied.parameters["children"][0]
    )
    assert cdef_node_key(changed) is not cdef_node_key(root)
    assert cdef_node_key(changed.parameters["children"][1]) is cdef_node_key(
        root.parameters["children"][1]
    )
