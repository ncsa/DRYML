"""Focused API tests for inert template authoring vocabulary."""

from __future__ import annotations

import random
import importlib

import pytest

from dryml import Expr, Match, Par, Shared, Template, TemplateGenerator, repeat
from dryml.core import Definition, F, Object, Ref
from dryml.core.domains import UniformFromSet, UniformIntRange
from dryml.core.errors import ParameterizationError, UnresolvedDefinitionError
from dryml.core.params import PresentMatcher
from dryml.core.utils.graph.path import GraphPath, Parameter


class TemplateModel:
    """Minimal inert target used to prove templates do not construct classes."""

    calls = 0

    def __init__(self, value, *, label="default"):
        type(self).calls += 1
        self.value = value
        self.label = label


class FactoryTarget:
    """Trusted target used to prove factory expression preflight is inert."""

    calls: list[object] = []

    def __init__(self, value):
        type(self).calls.append(value)
        self.value = value


class DocumentedModel:
    """Inert target used by the executable templates guide example."""

    def __init__(self, layers, width):
        self.layers = layers
        self.width = width


class DocumentedRecipeConsumer(Object):
    """Template guide consumer proving Ref recipe admission stays inert."""

    def __init__(self, recipe: Ref[Template]):
        self.recipe = recipe


def test_template_direct_authoring_does_not_restore_definition_conversion():
    """The retired Definition-to-Template conversion surface remains absent."""
    TemplateModel.calls = 0
    expression = Par("width") * 2

    template = Template(TemplateModel, expression, label="template")

    assert template.root.args == (expression,)
    assert template.root.kwargs["label"] == "template"
    assert not template.is_resolved
    assert template.names == ("width",)
    assert not hasattr(Definition(TemplateModel, expression, label="template"), "as_template")
    assert TemplateModel.calls == 0

    with pytest.raises(TypeError, match="class, ImportRef, or SourceSpec"):
        Template(Definition(TemplateModel, 1))
    assert TemplateModel.calls == 0


def test_template_from_value_handles_generic_roots_without_constructor_execution():
    """Generic template roots remain inert and definitions round-trip directly."""
    TemplateModel.calls = 0

    template = Template.from_value({"recipe": F(TemplateModel, Par("width"))})

    assert template.root["recipe"].args == (Par("width"),)
    assert template.names == ("width",)
    assert TemplateModel.calls == 0


def test_par_normalizes_qualified_names_and_rejects_ambiguous_spelling():
    """Parameters split convenience paths into a normalized root and GraphPath."""
    parameter = Par("namespace/this.model.encoder")

    assert parameter.name == "namespace/this"
    assert parameter.path == GraphPath((Parameter("model"), Parameter("encoder")))
    assert parameter == Par(
        "namespace/this",
        path=GraphPath((Parameter("model"), Parameter("encoder"))),
    )

    for name in ("", "/width", "width/", "width//depth", "width..child", "width/1bad"):
        with pytest.raises(ParameterizationError):
            Par(name)
    with pytest.raises(ParameterizationError, match="dotted"):
        Par("width.child", path=GraphPath())


def test_expression_operators_and_reflected_repetition_are_inert():
    """Arithmetic and repetition syntax captures operands without evaluating them."""
    parameter = Par("count")

    expressions = (
        parameter * 2,
        2 * parameter,
        parameter / 2,
        2 / parameter,
        parameter // 2,
        2 // parameter,
        ["layer"] * parameter,
        ("layer",) * parameter,
        ["layer"] * Shared(parameter),
        repeat(["layer"], parameter),
    )

    assert all(isinstance(expression, Expr) for expression in expressions)
    assert ["layer"] * 3 == ["layer", "layer", "layer"]
    assert ("layer",) * 2 == ("layer", "layer")
    with pytest.raises(TypeError):
        _ = parameter * "unsupported"


def test_factory_rejects_unresolved_expressions_before_target_invocation():
    """Factory builds fail before target resolution can invoke an unresolved recipe."""
    FactoryTarget.calls = []
    unresolved = F(FactoryTarget, Par("width") * 2)

    with pytest.raises(UnresolvedDefinitionError):
        unresolved.build()
    assert FactoryTarget.calls == []
    assert F(FactoryTarget, 6).build().value == 6
    assert FactoryTarget.calls == [6]


def test_domains_are_immutable_indexed_capabilities_with_bounded_validation():
    """Built-in domains validate support without allocating integer ranges."""
    values = ["one", "two"]
    choices = UniformFromSet(values)
    values.append("three")
    huge = UniformIntRange(0, 10**100)

    assert choices.cardinality() == 2
    assert choices.value_at(1) == "two"
    assert choices.contains("one")
    assert choices.sample(random.Random(0)) in {"one", "two"}
    assert huge.cardinality() == 10**100 + 1
    assert huge.value_at(10**100) == 10**100

    with pytest.raises(ParameterizationError):
        UniformFromSet(())
    with pytest.raises(ParameterizationError):
        UniformFromSet((1, 1))
    with pytest.raises(ParameterizationError):
        UniformIntRange(True, 2)
    with pytest.raises(ParameterizationError):
        UniformIntRange(1, False)


def test_match_is_a_query_leaf_distinct_from_template_parameters():
    """Matcher leaves retain legacy predicate behavior without becoming expressions."""
    match = Match(PresentMatcher(), name="present")

    assert match.matches("value")
    assert not match.matches(None, present=False)
    assert not isinstance(match, Expr)
    assert not isinstance(match, Par)


def test_documented_authoring_example_captures_complete_definitions():
    """The Templates guide's authoring flow remains executable and inert."""

    group = [F("builtins:tuple", Par("width")), F("builtins:tuple")]
    model = Template(DocumentedModel, group * Par("depth"), width=Par("width"))
    bound = model.sub(width=64, depth=2)
    generator = TemplateGenerator(
        model,
        width=UniformFromSet((32, 64)),
        depth=UniformFromSet((1, 2)),
    )

    assert isinstance(bound.to_definition(), Definition)
    assert len(generator.grid()) == 4
    sample = generator.sample(random.Random(7))
    assert generator.support_selector().matches(sample)
    assert model.as_selector().matches(sample)
    assert model.sub(sub_dict={"width": 64}, depth=2).is_resolved
    with pytest.raises(TypeError, match="Template"):
        Definition(DocumentedRecipeConsumer, recipe=model)


def test_retired_search_space_and_predicate_parameter_apis_are_absent():
    """The pre-beta cutover rejects retired imports and constructor shapes."""

    import dryml
    import dryml.core.params as params

    assert not hasattr(params, "Par")
    assert not hasattr(dryml, "SearchSpace")
    assert not hasattr(dryml, "space_mode")
    assert not hasattr(Definition(TemplateModel, 1), "as_space")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dryml.core.search_space")
    with pytest.raises(TypeError):
        Par("width", PresentMatcher())
    with pytest.raises(TypeError):
        UniformIntRange(1, 2, name="width")
