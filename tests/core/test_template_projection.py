"""Focused tests for deliberately loose template selector projection."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from dryml.core import Definition, F, Generator, Object, Ref, Repo, Serializable, Template, UniformFromSet
from dryml.core.store.dir import DirStore
from dryml.core.template import Par


class ProjectionModel:
    """Constructor declaration used to test preserved projection positions."""

    def __init__(self, values, *, activation, flag=False):
        self.values = values
        self.activation = activation
        self.flag = flag


class ProjectionChild(Object):
    """Fixed child definition used as an exact selector anchor."""

    def __init__(self, value):
        self.value = value


class ProjectionOwner(Serializable):
    """Own a fixed child and a variable width without constructing either."""

    def __init__(self, child, width):
        self.child = child
        self.width = width

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class ProjectionTemplateOwner(Object):
    """Exercise Template mappings alongside every constructor argument bucket."""

    def __init__(self, width, /, *extras, recipes: Mapping[str, Template] | None = None, **options):
        raise AssertionError("selector projection constructed the owner")


class ProjectionTemplateSlot(Object):
    """Exercise a positional Template argument without activating its recipe."""

    def __init__(self, recipe: Template, width, *, flag=False):
        raise AssertionError("selector projection constructed the owner")


@pytest.mark.parametrize("symbolic_recipe", (False, True))
def test_loose_projection_matches_admitted_template_mapping_and_preserves_call_shape(symbolic_recipe):
    recipe = Definition(ProjectionChild, value=Par("this.value") if symbolic_recipe else 7)
    authored = Definition(ProjectionTemplateOwner, Par("width"), "extra", recipes={"score": recipe}, optimizer="adam")
    candidate = authored.sub(width=32).concretize()
    selector = authored.loose_selector()

    assert selector.matches(candidate)
    assert selector.matches(authored.sub(width=64).concretize())
    assert not selector.matches(authored.sub(width=32).with_kwargs(optimizer="sgd").concretize())
    assert not selector.matches(Definition(ProjectionTemplateOwner, 32, "wrong", recipes={"score": recipe}, optimizer="adam").concretize())
    assert authored.parameters["recipes"]["score"] is recipe
    if not symbolic_recipe:
        different = authored.sub(width=32).with_kwargs(recipes={"score": Definition(ProjectionChild, value=8)}).concretize()
        assert not selector.matches(different)
        assert Generator(authored, {"width": UniformFromSet((32, 64))}).support_selector().matches(candidate)


def test_loose_projection_matches_positional_template_slot_without_defaults():
    recipe = Definition(ProjectionChild, value=7)
    authored = Definition(ProjectionTemplateSlot, recipe, Par("width"))

    assert authored.loose_selector().matches(authored.sub(width=32).concretize())
    assert authored.loose_selector().matches(authored.sub(width=32).with_kwargs(flag=True).concretize())
    assert not authored.loose_selector().matches(Definition(ProjectionTemplateSlot, Definition(ProjectionChild, value=8), 32).concretize())


@pytest.mark.parametrize("reference", (False, True))
def test_loose_projection_keeps_explicit_symbolic_quotation_exact(reference):
    recipe = Definition(ProjectionChild, value=Par("this.value"))
    other = Definition(ProjectionChild, value=Par("this.width"))
    quoted = Ref(recipe.quote()) if reference else recipe.quote()
    authored = Definition(ProjectionTemplateSlot, quoted, Par("width"))

    assert authored.loose_selector().matches(authored.sub(width=32).concretize())
    assert not authored.loose_selector().matches(Definition(ProjectionTemplateSlot, other, 32).concretize())


@pytest.mark.parametrize("query_index", ("memory", "sqlite"))
def test_loose_projection_rediscovers_template_role_candidate(tmp_path, query_index):
    recipe = Definition(ProjectionChild, value=Par("this.width"))
    authored = Definition(ProjectionTemplateOwner, Par("width"), recipes={"score": recipe}, optimizer="adam")
    candidate = authored.sub(width=32).concretize()
    repo = Repo(DirStore(tmp_path / "store", query_index=query_index))
    try:
        repo.default_store.publish_node(candidate)
        selected = repo.query().sel(authored.loose_selector()).cdefs().one()
        assert selected.graph_equal(candidate)
    finally:
        repo.close(flush=False)


def test_loose_projection_preserves_known_values_and_sequence_positions():
    """Unknown values become local wildcards while fixed facts still constrain."""
    definition = Definition(
        ProjectionModel,
        (Par("width"), "fixed", Par("depth")),
        activation="relu",
        flag=False,
    )
    selector = definition.loose_selector()

    assert selector.matches(Definition(ProjectionModel, (999, "fixed", 0), activation="relu", flag=False))
    assert not selector.matches(Definition(ProjectionModel, (999, "wrong", 0), activation="relu", flag=False))
    assert not selector.matches(Definition(ProjectionModel, (999, "fixed", 0), activation="relu", flag=True))


def test_loose_projection_keeps_partial_factory_call_shape_without_building():
    """Projected factories retain target, arity, known values, and keyword presence."""
    definition = Definition(
        ProjectionModel,
        (F("builtins:tuple", Par("width"), "fixed", flag=Par("flag")),),
        activation="relu",
    )
    selector = definition.loose_selector()

    assert selector.matches(Definition(
        ProjectionModel,
        (F("builtins:tuple", 64, "fixed", flag=False),),
        activation="relu",
    ))
    assert not selector.matches(Definition(
        ProjectionModel,
        (F("builtins:tuple", 64, "other", flag=False),),
        activation="relu",
    ))


@pytest.mark.parametrize("query_index", ("memory", "sqlite"))
def test_loose_projection_preserves_fixed_cdef_anchors_in_queries(tmp_path, query_index, monkeypatch):
    """Mapping-like CDefs stay exact in loose selectors and Generator prefilters."""

    child = Definition(ProjectionChild, value=7).concretize()
    other_child = Definition(ProjectionChild, value=8).concretize()
    authored = Definition(ProjectionOwner, child=child, width=Par("width"))
    candidate = authored.sub(width=32).concretize()
    other = Definition(ProjectionOwner, child=other_child, width=32).concretize()
    loose = authored.loose_selector()
    support = Generator(authored, {"width": UniformFromSet((32, 64))}).support_selector()

    assert loose.root.child is child
    assert support.prefilter.root.child is child
    assert loose.matches(candidate)
    assert authored.loose_selector(strict=True).matches(candidate)
    assert support.matches(candidate)
    assert not loose.matches(other)
    assert not support.matches(other)

    store = DirStore(tmp_path / "store", query_index=query_index)
    repo = Repo(store)
    store.publish_node(candidate)
    store.publish_node(other)

    assert repo.query().sel(loose).cdefs().one().graph_equal(candidate)
    assert repo.query().sel(support).cdefs().one().graph_equal(candidate)

    if query_index == "sqlite":
        from dryml.core.store import authority
        monkeypatch.setattr(authority._DetachedRead, "directory", lambda *args: pytest.fail("fixed anchor inventoried the Store"))
    selected = tuple(repo.query().refresh(False).sel(loose).cdefs().collect())
    assert len(selected) == 1
    assert selected[0].graph_equal(candidate)
    from dryml.core import DefinitionPath, Parameter
    assert selected[0].at(DefinitionPath((Parameter("child"),))).graph_equal(child)


@pytest.mark.parametrize("query_index", ("memory", "sqlite"))
def test_fixed_cdef_query_anchor_keeps_exact_topology_after_decode(tmp_path, query_index):
    """A structural index hit cannot admit a different fixed-anchor alias graph."""
    leaf = Definition(ProjectionChild, value=7).concretize()
    child = Definition(ProjectionChild, value=(leaf, leaf)).concretize()
    different_topology = Definition(ProjectionChild, value=(leaf, leaf.copy_graph())).concretize()
    assert child.stable_hash() == different_topology.stable_hash()
    assert not child.graph_equal(different_topology)

    authored = Definition(ProjectionOwner, child=child, width=Par("width"))
    candidate = authored.sub(width=32).concretize()
    other = Definition(ProjectionOwner, child=different_topology, width=32).concretize()
    repo = Repo(DirStore(tmp_path / "store", query_index=query_index))
    repo.default_store.publish_node(candidate)
    repo.default_store.publish_node(other)

    support = Generator(authored, {"width": UniformFromSet((32, 64))}).support_selector()
    for selector in (authored.loose_selector(), support):
        assert not selector.matches(other)
        assert repo.query().sel(selector).cdefs().one().graph_equal(candidate)
