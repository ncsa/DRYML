import pytest

import dryml
from dryml.core import (
    AnyValue,
    Choice,
    ConcreteDefinition,
    Definition,
    EdgeKind,
    IdentitySet,
    Missing,
    Par,
    Present,
    QuotedDef,
    Ref,
    Repo,
    Selector,
    SelectorSpec,
    Satisfies,
    Generator,
    UniformFromSet,
    UniformIntRange,
    definition_mode,
    selector_mode,
)
from dryml.core.object import Object
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.links import DefLink
from dryml.core.errors import CannotConcretizeParameterizedDefinition, CannotConcretizeSelectorReference, CycleError
from dryml.core.freeze import FrozenDict, FrozenList, FrozenTuple
from dryml.core.query.path import Arg, Index
from dryml.core.utils.graph.path import GraphPath, Kwarg

from tests.core import core_objects as objects

Cls1 = objects.TestClass1
Cls2 = objects.TestClass2
Nest3 = objects.TestNest3


def _ref(value):
    return DefLink.finalized(EdgeKind.REF, value)


def _mat(value):
    return DefLink.finalized(EdgeKind.MATERIALIZE, value)


class AuditChainNode(Object):
    def __init__(self, name, child=None, ref=None, width=None):
        self.name = name
        self.child = child
        self.ref = ref
        self.width = width


def test_definition_is_immutable_hashable_occurrence_with_named_comparison():
    d1 = Definition(Cls1, 1, test={"x": [1, 2]})
    d2 = Definition(Cls1, 1, test={"x": [1, 2]})
    d3 = Definition(Cls1, 2, test={"x": [1, 2]})

    assert d1 != d2
    assert d1.recipe_equal(d2)
    assert d1 != d3
    assert not d1.recipe_equal(d3)
    assert len({d1, d2, d1}) == 2
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
    parent = Definition(Nest3, _mat(child), ref=_ref(child), quoted=quoted)

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
    parent = Definition(Nest3, child, ref=_ref(Definition(Cls1, 4))).concretize()
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
    c = Definition(AuditChainNode, "C", ref=_ref(d))
    b = Definition(AuditChainNode, "B", child=_mat(c))
    a = Definition(AuditChainNode, "A", ref=_ref(b)).concretize()
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
    target = Definition(Nest3, ref=_ref(child), value=10).concretize()

    assert Selector(Definition(Nest3, ref=_ref(Definition(Cls1, AnyValue())))).matches(target)
    assert not Selector(Definition(Nest3, ref=_mat(Definition(Cls1, AnyValue())))).matches(target)
    assert Selector(Definition(Nest3, missing=Missing())).matches(target)
    assert Selector(Definition(Nest3, value=Choice([9, 10]))).matches(target)


def test_quoted_selector_is_data_not_graph_edge():
    quoted = SelectorSpec(Selector(Definition(Cls1, test=Present())))
    cdef = Definition(Nest3, models=quoted).concretize()
    graph = ConcreteDefinitionGraph.from_root(cdef)

    assert cdef.kwargs["models"] == quoted
    assert graph.edges() == ()
    assert Selector(Definition(Nest3, models=quoted)).matches(cdef)


def test_template_generation_replaces_search_space():
    generator = Generator(
        Definition(Cls1, Par("value"), test=Par("label")),
        {"value": UniformIntRange(1, 2), "label": UniformFromSet(["a", "b"])},
    )

    assert isinstance(generator.sample(), Definition)
    assert len(generator.grid()) == 4
    support = generator.support_selector()
    assert support.matches(Definition(Cls1, 1, test="a").concretize())


def test_definition_and_selector_modes():
    with definition_mode():
        d = Cls1(1)
    with selector_mode():
        s = Cls1(1)

    assert isinstance(d, Definition)
    assert isinstance(s, Selector)


def test_public_exports():
    for name in ("Definition", "ConcreteDefinition", "Ref", "Mat", "Selector", "SelectorSpec", "QuotedDef", "Par", "Template", "Generator"):
        assert hasattr(dryml, name)
    assert not hasattr(dryml, "TemplateGenerator")
    assert not hasattr(dryml, "SearchSpace")
    assert not hasattr(dryml, "space_mode")


class FakeStore:
    def catalog_key(self):
        return "immutable-definition-graph-audit"


class AuditRefOwner(Object):
    def __init__(self, child: Ref[ConcreteDefinition]):
        self.child = child


def test_indexed_query_can_inspect_ref_target_subgraph():
    repo = Repo()
    d = Definition(AuditChainNode, "D", width=512)
    c = Definition(AuditChainNode, "C", ref=_ref(d))
    b = Definition(AuditChainNode, "B", child=_mat(c))
    a = Definition(AuditChainNode, "A", ref=_ref(b)).concretize()
    repo._query_catalog.register_stored(a, FakeStore())

    selector = Definition(
        AuditChainNode,
        "A",
        ref=_ref(Definition(
            AuditChainNode,
            "B",
            child=Definition(
                AuditChainNode,
                "C",
                ref=_ref(Definition(AuditChainNode, "D", width=512)),
            ),
        )),
    )

    assert list(IdentitySet((a,)).query().sel(selector).collect()) == [a]


def test_repo_query_selector_preserves_policy():
    repo = Repo()
    child = Definition(objects.TestClassA, item=1).concretize(repo=repo)
    repo._query_catalog.register_stored(child, FakeStore())

    broad = Selector(Definition(objects.TestBase), cls_policy="selector")
    exact = Selector(Definition(objects.TestBase), cls_policy="exact")

    universe = IdentitySet((child,)).query()
    assert universe.sel(broad).count() == 1
    assert universe.sel(exact).count() == 0


def test_lens_set_deep_freezes_replacement_inside_frozen_container():
    d1 = Definition(Nest3, [[1]])
    replacement = [2, 3]
    d2 = d1.lens(GraphPath((Arg(0), Index(0)))).set(replacement)

    replacement.append(4)

    assert d2.args[0][0] == FrozenList([2, 3])


def test_user_supplied_frozen_containers_are_revalidated():
    inner = [1]
    supplied = FrozenList([inner])
    d = Definition(Nest3, supplied)

    inner.append(2)

    assert d.args[0][0] == FrozenList([1])


def test_definition_match_uses_selector_semantics_for_par_and_ref():
    target = Definition(Nest3, ref=_ref(Definition(Cls1, 10))).concretize()
    selector = Definition(Nest3, ref=_ref(Definition(Cls1, AnyValue())))

    assert selector.match(target)


def test_refcdef_runtime_constructor_receives_cdef_target():
    child = Definition(Cls1, 11).concretize()
    owner = Definition(AuditRefOwner, child).build()

    assert owner.child == child
    assert isinstance(owner.child, ConcreteDefinition)


def test_ref_selector_cannot_concretize_with_clear_error():
    selector = Selector(Definition(Cls1, test=Present()))

    with pytest.raises(CannotConcretizeSelectorReference):
        Definition(Nest3, _ref(selector)).concretize()

    with pytest.raises(CannotConcretizeSelectorReference):
        Definition(Nest3, _mat(selector)).concretize()


def test_nested_mapping_missing_matches_absent_key():
    selector = Selector(Definition(Nest3, cfg={"x": Missing()}))
    target = Definition(Nest3, cfg={}).concretize()

    assert selector.matches(target)


def test_indexed_query_missing_does_not_require_presence():
    repo = Repo()
    root_missing = Definition(Cls2).concretize()
    nested_missing = Definition(Nest3, cfg={}).concretize()
    ref_child = Definition(Cls1, {}).concretize()
    ref_parent = Definition(Nest3, ref=_ref(ref_child)).concretize()
    for cdef in (root_missing, nested_missing, ref_parent):
        repo._query_catalog.register_stored(cdef, FakeStore())

    universe = IdentitySet((root_missing, nested_missing, ref_parent)).query()
    assert list(universe.sel(Definition(Cls2, missing=Missing())).collect()) == [root_missing]
    assert list(universe.sel(Definition(Nest3, cfg={"x": Missing()})).collect()) == [nested_missing]

    ref_selector = Definition(
        Nest3,
        ref=_ref(Definition(Cls1, x={"missing": Missing()})),
    )
    assert list(universe.sel(ref_selector).collect()) == [ref_parent]


def test_concrete_definition_collapses_materialize_links():
    child = Definition(Cls1, 12).concretize()
    raw = ConcreteDefinition(Nest3, FrozenTuple((child,)), FrozenDict({}))
    linked = ConcreteDefinition(Nest3, FrozenTuple((_mat(child),)), FrozenDict({}))

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


def test_concrete_definition_concretizes_nested_definition_inside_container():
    cdef = ConcreteDefinition(
        Nest3,
        FrozenTuple((FrozenList([Definition(Cls1, 1)]),)),
        FrozenDict({}),
    )

    assert isinstance(cdef.parameters["args"][0][0], ConcreteDefinition)


def test_concrete_definition_rejects_cyclic_container():
    xs = []
    xs.append(xs)

    with pytest.raises(CycleError):
        ConcreteDefinition(Nest3, FrozenTuple((xs,)), FrozenDict({}))


def test_nested_concrete_boundary_error_reports_path():
    with pytest.raises(CannotConcretizeParameterizedDefinition, match=r"kwargs/xs/\[0\]"):
        ConcreteDefinition(Nest3, FrozenTuple(()), FrozenDict({"xs": FrozenList([Missing()])}))


def test_concrete_definition_collapses_nested_materialize_links():
    child = Definition(Cls1, 14).concretize()
    cdef = ConcreteDefinition(Nest3, FrozenTuple((FrozenList([_mat(child)]),)), FrozenDict({}))

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
        anon.stable_hash()
    assert named1.stable_hash() == named2.stable_hash()


def test_node_bind_alias_constraints_replacement_and_extraction(monkeypatch):
    from dryml.core import LocalState, Node, ObjectId, StateKey

    child = Definition(Cls1, 7)
    independent = Definition(Cls1, 7)
    root = Definition(left=child, right=child, independent=independent)
    identity = ObjectId()
    state = LocalState(StateKey("a" * 64, "pkl-" + "b" * 64), child.concretize())
    monkeypatch.setattr(ObjectId, "__init__", lambda *a, **k: pytest.fail("identity allocation"))
    monkeypatch.setattr(Cls1, "__init__", lambda *a, **k: pytest.fail("constructor"))

    bound = root.bind(identities={"$.left": identity, "$.right": identity},
                      states={"$.right": state})
    assert isinstance(bound, Node)
    assert bound.at("$.left") is bound.at("$.right")
    assert bound.at("$.left").object_id == identity
    assert bound.at("$.right").state == state
    assert bound.at("$.independent") is independent
    assert child.object_id is None and child.state is None
    assert bound.bind(identities={"$.right": identity}) is bound
    assert dict(bound.identity_spec()) == {GraphPath((Kwarg("left"),)): identity}
    assert dict(bound.at("$.right").state_spec()) == {GraphPath(): state}
    assert bound.at("$.right").identity_spec().root is bound.at("$.left")
    with pytest.raises(ValueError, match="conflict"):
        bound.bind(states={"$.left": LocalState(StateKey("a" * 64, "pkl-" + "c" * 64), state.creation_cdef)})
    with pytest.raises(TypeError, match="LocalState"):
        root.bind(states={"$.left": [state, state]})
    removed = bound.replace_bindings(identities={"$.right": None})
    assert removed.at("$.left").object_id is None
    assert removed.at("$.right").state == state
    assert bound.at("$.left").object_id == identity
    replacement = Definition(Cls1, 99)
    edited = bound.replace("$.right", replacement)
    assert edited.at("$.left") is replacement
    assert edited.at("$.right") is replacement
    assert edited.at("$.independent") is independent


def test_node_embedded_views_compose_facets_and_reject_conflicts():
    from dryml.core import LocalState, ObjectId, StateKey

    child = Definition(Cls1, 4)
    identity = ObjectId()
    state = LocalState(StateKey("a" * 64, "pkl-" + "b" * 64), child.concretize())
    composed = Definition(left=child.with_identity(identity), right=child.with_state(state))
    assert composed.at("$.left") is composed.at("$.right")
    assert composed.at("$.left").object_id == identity
    assert composed.at("$.left").state == state
    with pytest.raises(ValueError, match="conflict"):
        Definition(left=child.with_identity(identity), right=child.with_identity(ObjectId()))


def test_node_pickle_sets_and_semantic_topology_preserve_occurrences():
    import pickle

    first, second = Definition(Cls1, 1), Definition(Cls1, 1)
    root = Definition(members={first, second}, alias=first)
    restored = pickle.loads(pickle.dumps(root))
    assert len(root.kwargs["members"]) == len(restored.kwargs["members"]) == 2
    assert restored.at("$.alias") in restored.kwargs["members"]
    assert restored != root
    assert restored.graph_equal(root)
    assert root.recipe_equal(Definition(members={first, second}, alias=second))
    assert not Definition(a=first, b=first).graph_equal(Definition(a=first, b=second))


def test_concrete_node_facets_saved_refinement_and_retained_construction_view():
    from dryml.core import ConcreteNode, LocalState, ObjectId, StateKey, require_saved
    from dryml.core.object import Serializable

    concrete = Definition(Serializable).concretize()
    assert isinstance(concrete, ConcreteNode)
    with pytest.raises(ValueError, match="identity|state"):
        require_saved(concrete)
    state0 = LocalState(StateKey("a" * 64, "pkl-" + "b" * 64), concrete)
    state1 = LocalState(StateKey("a" * 64, "pkl-" + "c" * 64), concrete)
    pinned = concrete.with_state(state0)
    captured = pinned.with_realization(identities={"$": ObjectId()}, states={"$": state1})
    assert require_saved(captured) is captured
    assert captured.construction_view is pinned
    assert captured.state == state1 and pinned.state == state0
    assert captured.without_state().construction_view.state == state0
    assert len({concrete, Definition(Serializable).concretize()}) == 2
    with pytest.raises(AttributeError):
        captured._facets = concrete._facets
    with pytest.raises(TypeError):
        require_saved(Definition(Serializable))


def test_nested_concretization_keeps_facets_aliases_and_named_paths():
    from dryml.core import ConcreteNode, ObjectId, Parameter, Key

    identity = ObjectId()
    child = Definition(Cls1, 3).with_identity(identity)
    authored = Definition(Nest3, child, other=child)
    concrete = authored.concretize()
    first = concrete.at(GraphPath((Parameter("args"), Index(0))))
    assert isinstance(concrete, ConcreteNode) and isinstance(first, ConcreteNode)
    assert first is concrete.at(GraphPath((Parameter("kwargs"), Key("other"))))
    assert first.object_id == identity
    assert authored.at(GraphPath((Arg(0),))).object_id == identity
    with pytest.raises(ValueError, match="Stateless Nodes"):
        authored.build()


def test_node_projection_and_pickle_preserve_aliases_and_quoted_configuration():
    import pickle
    from dryml.core import LocalState, StateKey, ObjectId

    concrete = Definition(Cls1, 1).concretize()
    state = LocalState(StateKey("a" * 64, "pkl-" + "b" * 64), concrete)
    concrete = concrete.with_identity(ObjectId()).with_state(state)
    root = Definition(members={concrete, Definition(Cls1, 1).concretize()},
                      alias=concrete, quoted=concrete.quote())
    restored = pickle.loads(pickle.dumps(root))
    assert restored.graph_equal(root)
    assert len(restored.kwargs["members"]) == 2
    assert restored.at("$.alias") in restored.kwargs["members"]
    projected = restored.object_view()
    assert projected.at("$.alias").state is None
    assert projected.at("$.alias").object_id == concrete.object_id
    assert projected.at("$.alias") in projected.kwargs["members"]
    assert projected.at("$.quoted").state.key == state.key
    embedded = Definition(projected=projected)
    assert embedded.at("$.projected.alias").state is None
    assert embedded.at("$.projected.quoted").state.key == state.key
    assert root.at("$.alias").state is state
    copied = concrete.copy_graph()
    assert copied.state == state and copied.object_id == concrete.object_id
    assert copied != concrete and copied.graph_equal(concrete)


def test_conflicting_bulk_alias_edit_fails_atomically_and_independent_nodes_survive():
    from dryml.core import ObjectId

    first, independent = Definition(Cls1, 1), Definition(Cls1, 1)
    root = Definition(left=first, right=first, independent=independent)
    one, two = ObjectId(), ObjectId()
    with pytest.raises(ValueError, match="conflict"):
        root.bind(identities={"$.left": one, "$.right": two})
    with pytest.raises(ValueError, match="conflict"):
        root.replace_bindings(identities={"$.left": None, "$.right": one})
    assert first.object_id is None and root.at("$.left") is root.at("$.right")
    bound = root.bind(identities={"$.left": one, "$.independent": two})
    assert bound.at("$.left") is bound.at("$.right")
    assert bound.at("$.independent") is not bound.at("$.left")


def test_node_value_tags_and_array_snapshots_are_immutable():
    import pickle
    import numpy as np

    values = np.arange(4, dtype=">i4").reshape(2, 2)
    first, second = Definition(Cls1, 1), Definition(Cls1, 1)
    node = Definition(Nest3, values=values, mutable={first, second},
                      frozen=frozenset({first, second}), ordered=[first, second])
    values[0, 0] = 999
    assert node.kwargs["values"][0, 0] == 0
    assert node.kwargs["values"].dtype == np.dtype(">i4")
    assert not node.kwargs["values"].flags.writeable
    assert type(node.kwargs["frozen"]) is frozenset
    assert type(node.kwargs["mutable"]) is not frozenset
    restored = pickle.loads(pickle.dumps(node))
    assert restored.graph_equal(node)
    concrete = node.concretize()
    assert type(concrete.parameters["kwargs"]["frozen"]) is frozenset
    assert len(concrete.parameters["kwargs"]["frozen"]) == 2
    with pytest.raises(TypeError, match="(?i)object.dtype|object-dtype"):
        Definition(Nest3, values={"nested": np.array([object()], dtype=object)})
    with pytest.raises(TypeError, match="str and int"):
        Definition(Nest3, values={True: "not-an-integer-key"})


def test_raw_function_value_in_definition_is_rejected():
    with pytest.raises(TypeError, match="Anonymous function"):
        Definition(Cls1, test=lambda value: True)
