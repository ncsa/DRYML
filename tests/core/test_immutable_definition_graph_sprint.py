import pytest

import dryml
from dryml.core2 import (
    AnyValue,
    Choice,
    ConcreteDefinition,
    Definition,
    EdgeKind,
    Mat,
    Missing,
    Present,
    QuotedDef,
    Ref,
    SearchSpace,
    Selector,
    SelectorSpec,
    UniformFromSet,
    UniformIntRange,
    definition_mode,
    selector_mode,
    space_mode,
)
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.errors import CannotConcretizeParameterizedDefinition
from dryml.core2.freeze import FrozenDict, FrozenList

import core2_objects as objects

Cls1 = objects.TestClass1
Cls2 = objects.TestClass2
Nest3 = objects.TestNest3


def test_definition_is_immutable_hashable_and_structural():
    d1 = Definition(Cls1, 1, test={"x": [1, 2]})
    d2 = Definition(Cls1, 1, test={"x": [1, 2]})
    d3 = Definition(Cls1, 2, test={"x": [1, 2]})

    assert d1 == d2
    assert d1 != d3
    assert hash(d1) == hash(d2)
    assert {d1: "value"}[d2] == "value"
    with pytest.raises(Exception):
        d1._kwargs = {}


def test_definition_deep_freezes_user_containers_and_updates():
    values = [1, [2, 3]]
    d1 = Definition(Cls1, values, test={"a": [4]})
    values[1].append(99)

    assert isinstance(d1.args[0], FrozenList)
    assert d1.args[0][1] == FrozenList([2, 3])
    assert isinstance(d1.kwargs, FrozenDict)

    d2 = d1.with_arg(0, [9])
    d3 = d2.with_kwarg("test", {"b": [10]})

    assert d1.args[0][0] == 1
    assert d2.args[0] == FrozenList([9])
    assert d3.kwargs["test"] == FrozenDict({"b": FrozenList([10])})


def test_concretize_ref_mat_quoted_and_par_behaviour():
    child = Definition(Cls1, 3)
    quoted = Definition(Cls2, model=Definition(Cls1, test=Present())).quote()
    parent = Definition(Nest3, child.mat(), ref=child.ref(), quoted=quoted)

    cdef = parent.concretize()

    assert isinstance(cdef, ConcreteDefinition)
    assert isinstance(cdef.args[0], ConcreteDefinition)
    assert cdef.kwargs["ref"].kind is EdgeKind.REF
    assert isinstance(cdef.kwargs["ref"].target, ConcreteDefinition)
    assert isinstance(cdef.kwargs["quoted"], QuotedDef)

    with pytest.raises(CannotConcretizeParameterizedDefinition):
        Definition(Cls1, Present()).concretize()


def test_cdef_graph_uses_ref_edges_and_materialize_only_containment():
    child = Definition(Cls1, 3)
    parent = Definition(Nest3, child, ref=Definition(Cls1, 4).ref()).concretize()
    graph = ConcreteDefinitionGraph.from_root(parent)

    edge_kinds = {(str(edge.path), edge.kind) for edge in graph.edges()}
    assert any(kind is EdgeKind.MATERIALIZE for _, kind in edge_kinds)
    assert any(kind is EdgeKind.REF for _, kind in edge_kinds)

    child_cdef = parent.args[0]
    ref_child_cdef = parent.kwargs["ref"].target
    assert graph.contains(parent, child_cdef)
    assert not graph.contains(parent, ref_child_cdef)
    assert graph.paths_to(parent, ref_child_cdef) == ()


def test_selector_matches_ref_edges_and_matchers():
    child = Definition(Cls1, 3)
    target = Definition(Nest3, ref=child.ref(), value=10).concretize()

    assert Selector(Definition(Nest3, ref=Definition(Cls1, AnyValue()).ref())).matches(target)
    assert not Selector(Definition(Nest3, ref=Definition(Cls1, AnyValue()).mat())).matches(target)
    assert Selector(Definition(Nest3, missing=Missing())).matches(target)
    assert Selector(Definition(Nest3, value=Choice([9, 10]))).matches(target)


def test_quoted_selector_is_data_not_graph_edge():
    quoted = SelectorSpec(Selector(Definition(Cls1, test=Present())))
    cdef = Definition(Nest3, models=quoted).concretize()
    graph = ConcreteDefinitionGraph.from_root(cdef)

    assert cdef.kwargs["models"] == quoted
    assert graph.edges() == ()
    assert Selector(Definition(Nest3, models=quoted)).matches(cdef)


def test_search_space_sample_grid_and_support_selector():
    space = Definition(Cls1, UniformIntRange(1, 2), test=UniformFromSet(["a", "b"])).as_space()

    assert isinstance(space, SearchSpace)
    assert isinstance(space.sample(), Definition)
    assert len(list(space.grid())) == 4
    support = space.support_selector()
    assert isinstance(support, Selector)
    assert support.matches(Definition(Cls1, 1, test="a").concretize())


def test_definition_selector_and_space_modes():
    with definition_mode():
        d = Cls1(1)
    with selector_mode():
        s = Cls1(1)
    with space_mode():
        sp = Cls1(UniformIntRange(1, 2))

    assert isinstance(d, Definition)
    assert isinstance(s, Selector)
    assert isinstance(sp, SearchSpace)


def test_public_exports():
    for name in ("Definition", "ConcreteDefinition", "Ref", "Mat", "Selector", "SelectorSpec", "QuotedDef", "Par", "SearchSpace"):
        assert hasattr(dryml, name)
