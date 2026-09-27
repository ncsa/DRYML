"""Focused API tests for inert template authoring vocabulary."""

from __future__ import annotations

import random

import pytest

from dryml import Expr, Match, Shared, Template, repeat
from dryml.core import Definition, F
from dryml.core.domains import UniformFromSet, UniformIntRange
from dryml.core.errors import TemplateError, UnresolvedTemplateError
from dryml.core.params import PresentMatcher
from dryml.core.template import Par
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


def test_template_direct_authoring_and_definition_conversion_are_inert():
    """Class-first templates retain the same frozen Definition recipe."""
    TemplateModel.calls = 0
    expression = Par("width") * 2

    template = Template(TemplateModel, expression, label="template")
    converted = Definition(TemplateModel, expression, label="template").as_template()

    assert template.root == converted.root
    assert template.root.args == (expression,)
    assert template.root.kwargs["label"] == "template"
    assert not template.is_resolved
    assert template.names == ("width",)
    assert template.stable_hash() == converted.stable_hash()
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
        with pytest.raises(TemplateError):
            Par(name)
    with pytest.raises(TemplateError, match="dotted"):
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

    with pytest.raises(UnresolvedTemplateError):
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

    with pytest.raises(TemplateError):
        UniformFromSet(())
    with pytest.raises(TemplateError):
        UniformFromSet((1, 1))
    with pytest.raises(TemplateError):
        UniformIntRange(True, 2)
    with pytest.raises(TemplateError):
        UniformIntRange(1, False)


def test_match_is_a_query_leaf_distinct_from_template_parameters():
    """Matcher leaves retain legacy predicate behavior without becoming expressions."""
    match = Match(PresentMatcher(), name="present")

    assert match.matches("value")
    assert not match.matches(None, present=False)
    assert not isinstance(match, Expr)
