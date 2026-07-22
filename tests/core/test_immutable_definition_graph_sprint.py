import pytest

import dryml
from dryml.core import (
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
    RefCDef,
    Repo,
    SearchSpace,
    Selector,
    SelectorSpec,
    Satisfies,
    UniformFromSet,
    UniformIntRange,
    definition_mode,
    selector_mode,
    space_mode,
)
from dryml.core.object import Object
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.errors import CannotConcretizeParameterizedDefinition, CannotConcretizeSelectorReference, CycleError
from dryml.core.freeze import FrozenDict, FrozenList, FrozenTuple
from dryml.core.query.path import Arg, Index
from dryml.core.utils.graph.path import GraphPath

import core_objects as objects

Cls1 = objects.TestClass1
Cls2 = objects.TestClass2
Nest3 = objects.TestNest3


class AuditChainNode(Object):
    def __init__(self, name, child=None, ref=None, width=None):
        self.name = name
        self.child = child
        self.ref = ref
        self.width = width


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


def test_query_index_graph_expands_ref_targets_without_changing_containment():
    d = Definition(AuditChainNode, "D", width=512)
    c = Definition(AuditChainNode, "C", ref=d.ref())
    b = Definition(AuditChainNode, "B", child=c.mat())
    a = Definition(AuditChainNode, "A", ref=b.ref()).concretize()
    b_cdef = a.kwargs["ref"].target
    c_cdef = b_cdef.kwargs["child"]
    d_cdef = c_cdef.kwargs["ref"].target

    materialize_graph = ConcreteDefinitionGraph.from_root(a)
    query_graph = ConcreteDefinitionGraph.for_query_index(a)

    assert {(edge.parent, edge.child, edge.kind) for edge in materialize_graph.edges()} == {
        (a, b_cdef, EdgeKind.REF),
    }
    assert {(edge.parent, edge.child, edge.kind) for edge in query_graph.edges()} == {
        (a, b_cdef, EdgeKind.REF),
        (b_cdef, c_cdef, EdgeKind.MATERIALIZE),
        (c_cdef, d_cdef, EdgeKind.REF),
    }
    assert not query_graph.contains(a, b_cdef)
    assert not query_graph.contains(a, c_cdef)
    assert not query_graph.contains(a, d_cdef)


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


class FakeStore:
    def catalog_key(self):
        return "immutable-definition-graph-audit"


class AuditRefOwner(Object):
    def __init__(self, child: RefCDef):
        self.child = child


def test_indexed_query_can_inspect_ref_target_subgraph():
    repo = Repo()
    d = Definition(AuditChainNode, "D", width=512)
    c = Definition(AuditChainNode, "C", ref=d.ref())
    b = Definition(AuditChainNode, "B", child=c.mat())
    a = Definition(AuditChainNode, "A", ref=b.ref()).concretize()
    repo._query_catalog.register_stored(a, FakeStore())

    selector = Definition(
        AuditChainNode,
        "A",
        ref=Ref(Selector(Definition(
            AuditChainNode,
            "B",
            child=Definition(
                AuditChainNode,
                "C",
                ref=Definition(AuditChainNode, "D", width=512).ref(),
            ),
        ))),
    )

    assert list(repo.query(selector).stored(refresh=False).defs()) == [a]


def test_repo_query_selector_preserves_policy():
    repo = Repo()
    child = Definition(objects.TestClassA, item=1).concretize(repo=repo)
    repo._query_catalog.register_stored(child, FakeStore())

    broad = Selector(Definition(objects.TestBase), cls_policy="selector")
    exact = Selector(Definition(objects.TestBase), cls_policy="exact")

    assert repo.query(broad).stored(refresh=False).count() == 1
    assert repo.query(exact).stored(refresh=False).count() == 0


def test_lens_set_deep_freezes_replacement_inside_frozen_container():
    d1 = Definition(Nest3, [[1]])
    replacement = [2, 3]
    d2 = d1.at(GraphPath((Arg(0), Index(0)))).set(replacement)

    replacement.append(4)

    assert d2.args[0][0] == FrozenList([2, 3])


def test_user_supplied_frozen_containers_are_revalidated():
    inner = [1]
    supplied = FrozenList([inner])
    d = Definition(Nest3, supplied)

    inner.append(2)

    assert d.args[0][0] == FrozenList([1])


def test_definition_match_uses_selector_semantics_for_par_and_ref():
    target = Definition(Nest3, ref=Definition(Cls1, 10).ref()).concretize()
    selector = Definition(Nest3, ref=Definition(Cls1, AnyValue()).ref())

    assert selector.match(target)


def test_refcdef_runtime_constructor_receives_cdef_target():
    child = Definition(Cls1, 11).concretize()
    owner = Definition(AuditRefOwner, child).build()

    assert owner.child == child
    assert isinstance(owner.child, ConcreteDefinition)


def test_ref_selector_cannot_concretize_with_clear_error():
    selector = Selector(Definition(Cls1, test=Present()))

    with pytest.raises(CannotConcretizeSelectorReference):
        Definition(Nest3, Ref(selector)).concretize()

    with pytest.raises(CannotConcretizeSelectorReference):
        Definition(Nest3, Mat(selector)).concretize()


def test_nested_mapping_missing_matches_absent_key():
    selector = Selector(Definition(Nest3, cfg={"x": Missing()}))
    target = Definition(Nest3, cfg={}).concretize()

    assert selector.matches(target)


def test_indexed_query_missing_does_not_require_presence():
    repo = Repo()
    root_missing = Definition(Cls2).concretize()
    nested_missing = Definition(Nest3, cfg={}).concretize()
    ref_child = Definition(Cls1).concretize()
    ref_parent = Definition(Nest3, ref=ref_child.ref()).concretize()
    for cdef in (root_missing, nested_missing, ref_parent):
        repo._query_catalog.register_stored(cdef, FakeStore())

    assert list(repo.query(Definition(Cls2, missing=Missing())).stored(refresh=False).defs()) == [root_missing]
    assert list(repo.query(Definition(Nest3, cfg={"x": Missing()})).stored(refresh=False).defs()) == [nested_missing]

    ref_selector = Definition(Nest3, ref=Ref(Selector(Definition(Cls1, test=Missing()))))
    assert list(repo.query(ref_selector).stored(refresh=False).defs()) == [ref_parent]


def test_concrete_definition_collapses_materialize_links():
    child = Definition(Cls1, 12).concretize()
    raw = ConcreteDefinition(Nest3, FrozenTuple((child,)), FrozenDict({}))
    linked = ConcreteDefinition(Nest3, FrozenTuple((Mat(child),)), FrozenDict({}))

    assert linked.args[0] == child
    assert linked == raw
    assert hash(linked) == hash(raw)


def test_unstable_selector_data_fails_at_concrete_boundary():
    selector = Selector(Definition(Cls1, test=Satisfies(lambda value: True)))

    with pytest.raises(TypeError, match="stable-hashable"):
        Definition(Nest3, SelectorSpec(selector)).concretize()


def test_concretize_nested_definition_inside_container_becomes_cdef():
    child = Definition(Cls1, 13)
    cdef = Definition(Nest3, xs=[child]).concretize()

    assert isinstance(cdef.kwargs["xs"][0], ConcreteDefinition)


def test_concretize_rejects_nested_par_inside_container():
    with pytest.raises(CannotConcretizeParameterizedDefinition):
        Definition(Nest3, xs=[Missing()]).concretize()


def test_concrete_definition_rejects_nested_definition_inside_container():
    with pytest.raises(TypeError, match="unresolved Definition"):
        ConcreteDefinition(Nest3, FrozenTuple((FrozenList([Definition(Cls1)]),)), FrozenDict({}))


def test_concrete_definition_rejects_cyclic_container():
    xs = []
    xs.append(xs)

    with pytest.raises(CycleError):
        ConcreteDefinition(Nest3, FrozenTuple((xs,)), FrozenDict({}))


def test_nested_concrete_boundary_error_reports_path():
    with pytest.raises(CannotConcretizeParameterizedDefinition, match="kwargs/xs/0"):
        ConcreteDefinition(Nest3, FrozenTuple(()), FrozenDict({"xs": FrozenList([Missing()])}))


def test_concrete_definition_collapses_nested_materialize_links():
    child = Definition(Cls1, 14).concretize()
    cdef = ConcreteDefinition(Nest3, FrozenTuple((FrozenList([Mat(child)]),)), FrozenDict({}))

    assert cdef.args[0][0] == child


def test_unstable_selector_data_fails_inside_container():
    selector = Selector(Definition(Cls1, test=Satisfies(lambda value: True)))

    with pytest.raises(TypeError, match="stable-hashable"):
        Definition(Nest3, specs=[SelectorSpec(selector)]).concretize()


def test_satisfies_lambda_requires_stable_name_for_hash():
    anon = Definition(Cls1, test=Satisfies(lambda value: True))
    named1 = Definition(Cls1, test=Satisfies(lambda value: True, name="always"))
    named2 = Definition(Cls1, test=Satisfies(lambda value: False, name="always"))

    with pytest.raises(TypeError, match="stable-hashable"):
        hash(anon)
    assert hash(named1) == hash(named2)


def test_raw_function_value_in_definition_is_rejected():
    with pytest.raises(TypeError, match="Anonymous function"):
        Definition(Cls1, test=lambda value: True)
