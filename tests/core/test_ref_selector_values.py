from typing import Annotated, Optional

import dryml
import pytest
from dryml.core2 import (
    ConcreteDefinition,
    Definition,
    EdgeKind,
    QuotedDef,
    Ref,
    RefCDef,
    RefCDefArg,
    Selector,
    SelectorArg,
    SelectorSpec,
)
from dryml.core2.arg_roles import resolve_arg_roles
from dryml.core2.cdef_graph import ConcreteDefinitionGraph, ConcreteDefinitionGraphError
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.links import DefLink
from dryml.core2.object import Object
from dryml.core2.query.selector_graph import compile_selector_graph
from dryml.core2.utils.graph.path import GraphPath, Index, Parameter

import core2_objects as objects


class RefOwner(Object):
    def __init__(self, child: RefCDef, *, label="owner"):
        self.child = child
        self.label = label


class OptionalRefOwner(Object):
    def __init__(self, child: Optional[Annotated[ConcreteDefinition, RefCDefArg()]] = None):
        self.child = child


class SelectorOwner(Object):
    def __init__(self, selector: SelectorArg):
        self.selector = selector


class OptionalSelectorOwner(Object):
    def __init__(self, selector: Optional[Annotated[object, SelectorArg()]] = None):
        self.selector = selector


def test_ref_wrapper_is_non_materializing_cdef_edge():
    child = Definition(objects.TestClass1, 1).concretize()
    parent = Definition(objects.TestNest3, Ref(child)).concretize()

    assert parent.args[0].kind is EdgeKind.REF
    assert parent.args[0].target == child
    edges = ConcreteDefinitionGraph.from_root(parent).edges()
    assert (parent, GraphPath((Parameter("args"), Index(0))), child, EdgeKind.REF) == (
        edges[0].parent,
        edges[0].path,
        edges[0].child,
        edges[0].kind,
    )


def test_ref_role_canonicalizes_object_and_cdef_values():
    child = Definition(objects.TestClass1, 2).concretize()
    cdef = Definition(RefOwner, child).concretize()

    assert cdef.parameters["child"].kind is EdgeKind.REF
    assert cdef.parameters["child"].target == child
    assert isinstance(resolve_arg_roles(RefOwner)["child"], RefCDefArg)


def test_optional_ref_role_resolution():
    child = Definition(objects.TestClass1, 3).concretize()
    cdef = Definition(OptionalRefOwner, child).concretize()

    assert cdef.parameters["child"].kind is EdgeKind.REF
    assert cdef.parameters["child"].target == child


def test_optional_role_default_and_explicit_none_match_in_private_bound_records():
    omitted = Definition(OptionalRefOwner).concretize()
    explicit = Definition(OptionalRefOwner, None).concretize()

    assert omitted == explicit
    assert omitted["parameters"]["child"] is None

    selector_omitted = Definition(OptionalSelectorOwner).concretize()
    selector_explicit = Definition(OptionalSelectorOwner, None).concretize()
    assert selector_omitted == selector_explicit
    assert selector_omitted["parameters"]["selector"] is None


def test_selector_arg_stores_quoted_selector_data_not_edge():
    selector = Selector(Definition(objects.TestClass1, test=dryml.Present()))
    cdef = Definition(SelectorOwner, selector).concretize()

    assert isinstance(cdef.parameters["selector"], SelectorSpec)
    assert ConcreteDefinitionGraph.from_root(cdef).edges() == ()
    assert Selector(Definition(SelectorOwner, selector=selector)).matches(cdef)


def test_selector_arg_runtime_constructor_receives_wrapper():
    selector = Selector(Definition(objects.TestClass1, test=dryml.Present()))
    owner = Definition(SelectorOwner, selector).build()

    assert isinstance(owner.selector, SelectorSpec)
    assert owner.selector.selector == selector


def test_deflink_rejects_invalid_edge_kind():
    child = Definition(objects.TestClass1, 1).concretize()

    with pytest.raises(TypeError, match="EdgeKind"):
        DefLink("ref", child)


def test_quoted_def_stores_expression_data_not_edge():
    quoted = Definition(objects.TestClass1, test=dryml.Present()).quote()
    cdef = Definition(objects.TestNest3, quoted).concretize()

    assert isinstance(cdef.parameters["args"][0], QuotedDef)
    assert ConcreteDefinitionGraph.from_root(cdef).edges() == ()


def test_same_child_can_be_ref_and_materialize_edges():
    child = Definition(objects.TestClass1, 4).concretize()
    parent = Definition(objects.TestNest3, child, ref=Ref(child)).concretize()
    edges = ConcreteDefinitionGraph.from_root(parent).edges()

    assert {edge.kind for edge in edges} == {EdgeKind.MATERIALIZE, EdgeKind.REF}


def test_materialize_only_containment_ignores_ref_only_child():
    ref_child = Definition(objects.TestClass1, 5).concretize()
    parent = Definition(objects.TestNest3, ref=Ref(ref_child)).concretize()
    graph = ConcreteDefinitionGraph.from_root(parent)

    assert not graph.contains(parent, ref_child)
    assert graph.paths_to(parent, ref_child) == ()


def test_selector_graph_ref_edge_kind():
    selector = Definition(objects.TestNest3, ref=Definition(objects.TestClass1, dryml.AnyValue()).ref())
    graph = compile_selector_graph(selector)

    assert graph.edges[0].edge_kind is EdgeKind.REF


def test_invalid_ref_graph_edge_validation_message():
    child = Definition(objects.TestClass1, 6).concretize()
    parent = ConcreteDefinition(objects.TestNest3, FrozenTuple((child,)), FrozenDict({}))
    bad_edge = next(iter(ConcreteDefinitionGraph.from_root(parent).edges()))

    from dryml.core2.cdef_graph import CDefEdge, CDefNode

    with pytest.raises(ConcreteDefinitionGraphError):
        ConcreteDefinitionGraph(
            (parent,),
            (CDefNode(parent, parent.stable_hash()), CDefNode(child, child.stable_hash())),
            (CDefEdge(parent, bad_edge.path, child, EdgeKind.REF),),
        )


def test_public_ref_exports():
    for name in ("Ref", "Mat", "RefCDef", "RefCDefArg", "SelectorArg", "SelectorSpec", "QuotedDef"):
        assert hasattr(dryml, name)
