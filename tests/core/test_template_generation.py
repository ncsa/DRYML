"""Focused captured-generation tests for definition templates."""

from __future__ import annotations

import random

import pytest

from dryml.core import Definition, Generator, Object
from dryml.core.domains import UniformFromSet
from dryml.core.errors import ParameterizationError, UnresolvedDefinitionError
from dryml.core.template import Par


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


class DistributionResult:
    """Runtime provider whose result attempts to leak generation policy."""

    def sample(self, rng):
        return UniformFromSet((1,))

    def cardinality(self):
        return 1

    def value_at(self, index):
        return UniformFromSet((1,))

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


def test_generator_captures_complete_distributions_and_samples_once_per_root():
    """Sampling uses caller RNG once per lexical root and returns a Definition."""
    definition = Definition(
        GeneratedModel,
        Par("encoder/width"),
        scale=Par("scale"),
        product=Par("encoder/width") * Par("scale"),
    )
    widths = UniformFromSet((32, 64))
    generator = Generator(definition.sub(scale=2), {"encoder/width": widths})

    first = generator.sample(random.Random(7))
    second = generator.sample(random.Random(7))

    assert isinstance(first, Definition)
    assert first == second
    assert first.parameters["product"] == first.parameters["width"] * 2
    assert generator.definition.names == ("encoder/width",)
    assert generator.distributions["encoder/width"] is widths


def test_generator_samples_qualified_roots_in_lexical_order():
    """Equivalent template layout does not change provider draw ordering."""

    OrderedDomain.calls = []
    generator = Generator(
        Definition(GeneratedModel, Par("zeta"), scale=Par("alpha"), product=1),
        {"zeta": OrderedDomain("zeta"), "alpha": OrderedDomain("alpha")},
    )

    generator.sample(random.Random(0))

    assert tuple(generator.distributions) == ("alpha", "zeta")
    assert OrderedDomain.calls == ["alpha", "zeta"]


def test_generator_grid_is_complete_after_static_substitution():
    """Grid returns every root assignment after callers bind static values."""
    definition = Definition(
        GeneratedModel,
        Par("width"),
        scale=Par("scale"),
        product=Par("width") * Par("scale"),
    )
    generator = Generator(
        definition,
        {"width": UniformFromSet((32, 64)), "scale": UniformFromSet((1, 2))},
    )

    values = generator.grid()

    assert len(values) == 4
    assert {item.parameters["product"] for item in values} == {32, 64, 128}
def test_generator_rejects_incomplete_or_invalid_root_contracts_before_sampling():
    """Construction rejects non-exact and non-provider mappings before sampling."""
    definition = Definition(GeneratedModel, Par("width"), scale=1, product=Par("width"))

    with pytest.raises(ParameterizationError, match="missing"):
        Generator(definition, {})
    with pytest.raises(ParameterizationError, match="unknown"):
        Generator(definition, {"width": UniformFromSet((1,)), "extra": UniformFromSet((2,))})
    with pytest.raises(ParameterizationError, match="bind static"):
        Generator(definition, {"width": 64})
    with pytest.raises(ParameterizationError, match="requires a Definition"):
        Generator(object(), {})

    introduced = Definition(GeneratedModel, Par("width"), scale=1, product=1)
    with pytest.raises(ParameterizationError, match="uncovered"):
        Generator(introduced, {"width": UniformFromSet((Par("later"),))})
    generator = Generator(introduced, {"width": DynamicRootDistribution()})
    with pytest.raises(UnresolvedDefinitionError):
        generator.sample(random.Random(0))
    with pytest.raises(UnresolvedDefinitionError):
        generator.grid()
    with pytest.raises(ParameterizationError, match="Distribution"):
        Generator(introduced, {"width": DistributionResult()}).sample(random.Random(0))


def test_grid_checks_cardinality_before_provider_indexing():
    """A product over the cap fails without allocating or indexing support."""
    OversizeDomain.lookups = 0
    definition = Definition(GeneratedModel, Par("width"), scale=1, product=1)

    with pytest.raises(ParameterizationError):
        Generator(definition, {"width": OversizeDomain()}).grid()
    assert OversizeDomain.lookups == 0


def test_definition_rejects_distribution_values_at_public_boundaries():
    """Generation policy cannot be embedded directly in Definition structure."""

    with pytest.raises(ParameterizationError, match="Distribution"):
        Definition(GeneratedModel, UniformFromSet((32, 64)), scale=1, product=1)
    with pytest.raises(ParameterizationError, match="Distribution"):
        Definition(GeneratedModel, {"width": [UniformFromSet((32, 64))]}, scale=1, product=1)
