"""Focused captured-generation tests for definition templates."""

from __future__ import annotations

import random

import pytest

from dryml.core import Definition, Object
from dryml.core.domains import UniformFromSet
from dryml.core.errors import ParameterizationError, UnresolvedDefinitionError
from dryml.core.template import Par, Template
from dryml.core.template_selector import TemplateGenerator


class GeneratedModel:
    """Inert constructor shape used to inspect generated Definitions."""

    def __init__(self, width, *, scale, product):
        self.width = width
        self.scale = scale
        self.product = product


class StaticObject(Object):
    """Object-shaped value used to prove static capture detaches live inputs."""

    def __init__(self):
        pass


class DynamicRootDistribution:
    """Runtime provider whose sampled value introduces an active root."""

    def sample(self, rng):
        return Par("later")

    def cardinality(self):
        return 1

    def value_at(self, index):
        return Par("later")

    def contains(self, value):
        return None

    def bounds(self):
        return None


class OversizeDomain:
    """Indexed provider proving grid limits are checked before lookup."""

    lookups = 0

    def sample(self, rng):
        return 1

    def cardinality(self):
        return 4_097

    def value_at(self, index):
        type(self).lookups += 1
        return index

    def contains(self, value):
        return True

    def bounds(self):
        return None


class OrderedDomain:
    """Provider sentinel recording the generator's deterministic draw order."""

    calls: list[str] = []

    def __init__(self, name):
        self.name = name

    def sample(self, rng):
        type(self).calls.append(self.name)
        return 1

    def cardinality(self):
        return 1

    def value_at(self, index):
        return 1

    def contains(self, value):
        return value == 1

    def bounds(self):
        return 1, 1


def test_generator_captures_complete_bindings_and_samples_once_per_root():
    """Sampling uses caller RNG once per lexical root and returns a Definition."""
    template = Template(
        GeneratedModel,
        Par("encoder/width"),
        scale=Par("scale"),
        product=Par("encoder/width") * Par("scale"),
    )
    widths = UniformFromSet((32, 64))
    generator = TemplateGenerator(template, sub_dict={"encoder/width": widths, "scale": 2})

    first = generator.sample(random.Random(7))
    second = generator.sample(random.Random(7))

    assert isinstance(first, Definition)
    assert first == second
    assert first.parameters["product"] == first.parameters["width"] * 2
    assert generator.template.names == ("encoder/width",)
    assert generator.domains["encoder/width"] is widths


def test_generator_samples_qualified_roots_in_lexical_order():
    """Equivalent template layout does not change provider draw ordering."""

    OrderedDomain.calls = []
    generator = TemplateGenerator(
        Template(GeneratedModel, Par("zeta"), scale=Par("alpha"), product=1),
        zeta=OrderedDomain("zeta"),
        alpha=OrderedDomain("alpha"),
    )

    generator.sample(random.Random(0))

    assert tuple(generator.domains) == ("alpha", "zeta")
    assert OrderedDomain.calls == ["alpha", "zeta"]


def test_generator_grid_is_complete_and_static_capture_detaches_live_objects():
    """Grid returns every root assignment only after all results are complete."""
    template = Template(
        GeneratedModel,
        Par("width"),
        scale=Par("scale"),
        product=Par("width") * Par("scale"),
    )
    captured = StaticObject()
    generator = TemplateGenerator(
        template,
        width=UniformFromSet((32, 64)),
        scale=UniformFromSet((1, 2)),
    )

    values = generator.grid()

    assert len(values) == 4
    assert {item.parameters["product"] for item in values} == {32, 64, 128}
    static = TemplateGenerator(
        Template(GeneratedModel, Par("width"), scale=1, product=1),
        width=captured,
    )
    assert static.template.root.args[0] == captured.object_ref


def test_generator_rejects_incomplete_or_invalid_root_contracts_before_sampling():
    """Capture rejects missing roots and unsupported generation roots eagerly."""
    template = Template(GeneratedModel, Par("width"), scale=1, product=Par("width"))

    with pytest.raises(ParameterizationError, match="missing"):
        TemplateGenerator(template)
    with pytest.raises(ParameterizationError, match="unknown"):
        TemplateGenerator(template, width=1, extra=2)
    with pytest.raises(ParameterizationError, match="Definition root"):
        TemplateGenerator(Template.from_value([Par("width")]), width=1)

    introduced = Template(GeneratedModel, Par("width"), scale=1, product=1)
    with pytest.raises(ParameterizationError, match="uncovered"):
        TemplateGenerator(introduced, width=UniformFromSet((Par("later"),)))
    generator = TemplateGenerator(introduced, width=DynamicRootDistribution())
    with pytest.raises(UnresolvedDefinitionError):
        generator.sample(random.Random(0))


def test_grid_checks_cardinality_before_provider_indexing():
    """A product over the cap fails without allocating or indexing support."""
    OversizeDomain.lookups = 0
    template = Template(GeneratedModel, Par("width"), scale=1, product=1)

    with pytest.raises(ParameterizationError):
        TemplateGenerator(template, width=OversizeDomain()).grid()
    assert OversizeDomain.lookups == 0
