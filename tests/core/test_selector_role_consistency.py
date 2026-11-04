"""Role-aware ordinary predicates must agree across query interpretation paths."""

import json

import pytest

from dryml.core import (
    ConcreteNode, Definition, Generator, Node, Object, QueryPredicate, QuotedDef, Ref,
    Repo, Selector, SelectorSpec, Template, UniformFromSet,
)
from dryml.core.query.selector_graph import compile_selector_graph
from dryml.core.repo_plan import SaveRouting
from dryml.core.store.dir import DirStore
from dryml.core.template import Par
from tests.core.test_template_projection import ProjectionChild, ProjectionTemplateOwner


class RoleTemplate(Object):
    """Retain a Template argument without constructing it during a query."""

    def __init__(self, value: Template):
        raise AssertionError("query constructed an Object")


class RoleDefinition(Object):
    """Receive expression data through an exact Definition role."""

    def __init__(self, value: Ref[Definition]):
        raise AssertionError("query constructed an Object")


class RoleNode(Object):
    """Receive exact Node data through a quotation-bearing role."""

    def __init__(self, value: Ref[Node]):
        raise AssertionError("query constructed an Object")


class RoleConcrete(Object):
    """Receive a concrete graph through an ordinary reference edge."""

    def __init__(self, value: Ref[ConcreteNode]):
        raise AssertionError("query constructed an Object")


class RoleQuoted(Object):
    """Receive a quotation wrapper as data rather than a graph edge."""

    def __init__(self, value: Ref[QuotedDef]):
        raise AssertionError("query constructed an Object")


class RoleSelector(Object):
    """Receive Selector data through a quotation-bearing role."""

    def __init__(self, value: Ref[Selector]):
        raise AssertionError("query constructed an Object")


class RoleSelectorSpec(Object):
    """Receive a selector wrapper without adding a graph edge."""

    def __init__(self, value: Ref[SelectorSpec]):
        raise AssertionError("query constructed an Object")


class RoleVariadic(Object):
    """Distinguish variadic positional names from actual keyword arguments."""

    def __init__(self, *recipes: Template, **items: Ref[ConcreteNode]):
        raise AssertionError("query constructed an Object")


class RolePositionalOnly(Object):
    """Allow a positional-only name to occur independently in keyword extras."""

    def __init__(self, recipe: Template, /, **items: Ref[ConcreteNode]):
        raise AssertionError("query constructed an Object")


class RolePair(Object):
    """Retain alias constraints across two typed reference slots."""

    def __init__(self, left: Ref[ConcreteNode], right: Ref[ConcreteNode]):
        raise AssertionError("query constructed an Object")


def _request(width=32, value=7):
    recipe = Definition(ProjectionChild, value=Ref(Definition(ProjectionChild, value=Par("this.width"))))
    return Definition(ProjectionTemplateOwner, width, "extra", recipes={"score": recipe}, optimizer=value)


@pytest.mark.parametrize("owner,value", (
    (RoleTemplate, Definition(ProjectionChild, value=Par("this.width"))),
    (RoleTemplate, Definition(ProjectionChild, value=7).concretize()),
    (RoleDefinition, Definition(ProjectionChild, value=7)),
    (RoleNode, Definition(ProjectionChild, value=7)),
    (RoleNode, Definition(ProjectionChild, value=7).concretize()),
    (RoleConcrete, Definition(ProjectionChild, value=7).concretize()),
    (RoleQuoted, Definition(ProjectionChild, value=7)),
    (RoleSelector, Selector(Definition(ProjectionChild, value=7))),
    (RoleSelectorSpec, Selector(Definition(ProjectionChild, value=7))),
    (RoleSelector, Selector(_request().with_kwargs(recipes={"score": Definition(ProjectionChild, value=7)}))),
    (RoleSelectorSpec, Selector(_request().with_kwargs(recipes={"score": Definition(ProjectionChild, value=7)}))),
))
def test_ordinary_predicate_interprets_receiving_role(owner, value):
    pattern = Definition(owner, value)
    candidate = pattern.concretize()
    assert Selector(pattern).matches(candidate)
    assert QueryPredicate(pattern).matches(candidate)
    assert QueryPredicate(pattern).compile().matches(candidate)
    assert compile_selector_graph(pattern).exact_residual.matches(candidate)
    restored = QueryPredicate.from_data(json.loads(json.dumps(QueryPredicate(pattern).to_data())))
    assert restored.matches(candidate)


@pytest.mark.parametrize("query_index", ("memory", "sqlite"))
def test_notebook_chained_raw_definition_filter_is_exact_and_portable(tmp_path, query_index):
    template = _request(Par("width"))
    first, second, different = (_request(width, value).concretize() for width, value in ((32, 7), (64, 7), (32, 8)))
    repo = Repo(DirStore(tmp_path / "store", query_index=query_index))
    try:
        for candidate in (first, second, different):
            repo.default_store.publish_node(candidate)
        universe = repo.query().refresh(True).sel(template.loose_selector())
        assert universe.count() == 2
        assert universe.sel(template.sub(width=32)).one().graph_equal(first)
        assert universe.sel(template.sub(width=64)).one().graph_equal(second)
        a, b = QueryPredicate(_request(32)), QueryPredicate(_request(64))
        assert repo.query().sel(a | b).count() == 2
        assert repo.query().sel(a & ~b).one().graph_equal(first)
        restored = QueryPredicate.from_data(json.loads(json.dumps((a | b).to_data())))
        assert repo.query().sel(restored).count() == 2
        owner = Definition(ProjectionChild, value=first).concretize()
        repo.default_store.publish_node(owner)
        assert repo.query().sel(owner).nested(_request(32)).targets().one().graph_equal(first)
    finally:
        repo.close(flush=False)


def test_role_pattern_preserves_negative_constraints_and_quotation():
    pattern = _request()
    assert Selector(pattern).matches(pattern.concretize())
    assert not Selector(pattern).matches(_request(64).concretize())
    assert not Selector(pattern).matches(_request(32, 8).concretize())
    assert not Selector(pattern).matches(pattern.with_kwargs(recipes={"score": Definition(ProjectionChild, value=0)}).concretize())


def test_template_mapping_preserves_concrete_node_kind_and_fixed_defaults():
    child = Definition(ProjectionChild, value=7).concretize()
    pattern = _request().with_kwargs(recipes={"score": child})
    candidate = pattern.concretize()
    assert QueryPredicate(pattern).matches(candidate)
    assert not QueryPredicate(pattern).matches(_request().with_kwargs(recipes={"score": child.thaw()}).concretize())


def test_preparation_preserves_soft_candidate_and_quotation_payload():
    pattern = _request()
    before = pattern.kwargs["recipes"]["score"]
    compiled = QueryPredicate(pattern).compile()
    assert compiled.matches(pattern.copy_graph())
    assert pattern.kwargs["recipes"]["score"] is before
    assert compiled.predicate.value.kwargs["recipes"]["score"].target.value is before


@pytest.mark.parametrize("owner,key", ((RoleVariadic, "recipes"), (RolePositionalOnly, "recipe")))
def test_keyword_extras_do_not_inherit_positional_slot_roles(owner, key):
    leaf = Definition(ProjectionChild, value=7).concretize()
    pattern = Definition(owner, Definition(ProjectionChild, value=7), **{key: leaf})
    candidate = pattern.concretize()
    assert QueryPredicate(pattern).matches(candidate)
    assert QueryPredicate.from_data(QueryPredicate(pattern).to_data()).matches(candidate)


def test_role_preparation_preserves_shared_and_independent_reference_partitions():
    leaf = Definition(ProjectionChild, value=7).concretize()
    shared = Definition(RolePair, leaf, leaf)
    independent = Definition(RolePair, leaf.copy_graph(), leaf.copy_graph())
    assert QueryPredicate(shared).matches(shared.concretize())
    assert not QueryPredicate(shared).matches(independent.concretize())
    assert not QueryPredicate(independent).matches(shared.concretize())
    assert QueryPredicate.from_data(QueryPredicate(shared).to_data()).matches(shared.concretize())


def test_generator_role_predicate_round_trip_keeps_exact_support():
    template = _request(Par("width"))
    generator = Generator(template, {"width": UniformFromSet((32, 64))})
    selector = generator.support_selector()
    predicate = QueryPredicate.support(generator)
    restored = QueryPredicate.from_data(json.loads(json.dumps(predicate.to_data())))
    restored_selector = type(selector).from_data(json.loads(json.dumps(selector.to_data())))
    for candidate, expected in ((_request(32), True), (_request(64), True), (_request(128), False), (_request(32, 8), False)):
        candidate = candidate.concretize()
        assert selector.matches(candidate) is expected
        assert restored.matches(candidate) is expected
        assert restored_selector.matches(candidate) is expected


def test_generator_transport_cannot_silently_hide_active_roots_in_template_roles():
    from dryml.core.template_codec import TemplateCodecError

    template = Definition(RoleTemplate, Definition(ProjectionChild, value=Par("value")))
    generator = Generator(template, {"value": UniformFromSet((7, 8))})
    with pytest.raises(TemplateCodecError, match="active roots"):
        QueryPredicate.support(generator).to_data()
    with pytest.raises(TemplateCodecError, match="active roots"):
        generator.support_selector().to_data()
    traversing = Generator(template, {"value": UniformFromSet((7, 8))}, traverse_refs=True)
    with pytest.raises(TemplateCodecError, match="active roots"):
        QueryPredicate.support(traversing).to_data()


def test_query_role_preparation_does_not_construct_resolve_or_access_repo(monkeypatch):
    from dryml.core import symbol
    from dryml.core import object as object_module

    pattern = _request()
    candidate = pattern.concretize()
    def forbidden(*args, **kwargs):
        pytest.fail("role projection crossed a construction or resolution boundary")
    monkeypatch.setattr(symbol, "resolve_symbol", forbidden)
    monkeypatch.setattr(object_module, "_normalize_definition_arguments", forbidden)
    monkeypatch.setattr(Repo, "materialize_boundary", forbidden)
    monkeypatch.setattr(Repo, "describe", forbidden)
    assert QueryPredicate(pattern).compile().matches(candidate)
    assert Selector(pattern).matches(candidate)
    assert QueryPredicate.from_data(QueryPredicate(pattern).to_data()).matches(candidate)


def test_portable_repo_routing_preserves_role_pattern(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store, save_routing=SaveRouting(((Selector(_request()), store),)))
    restored = None
    try:
        definition = repo.to_definition()
        restored = Repo.from_definition(definition)
        selector, _ = restored.save_routing.routes[0]
        assert selector.matches(_request().concretize())
        assert not selector.matches(_request(64).concretize())
    finally:
        if restored is not None:
            restored.close(flush=False)
        repo.close(flush=False)


def test_portable_repo_routing_keeps_selector_as_data_exact(tmp_path):
    inner = _request().with_kwargs(recipes={"score": Definition(ProjectionChild, value=7)})
    pattern = Definition(RoleSelector, Selector(inner))
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store, save_routing=SaveRouting(((Selector(pattern), store),)))
    restored = None
    try:
        restored = Repo.from_definition(repo.to_definition())
        selector, _ = restored.save_routing.routes[0]
        assert selector.matches(pattern.concretize())
        assert not selector.matches(Definition(RoleSelector, Selector(inner.with_args(64, "extra"))).concretize())
    finally:
        if restored is not None:
            restored.close(flush=False)
        repo.close(flush=False)
