"""Focused exact-support tests for captured template generators."""

from __future__ import annotations

import pytest

from dryml.core import Definition, Object
from dryml.core.domains import UniformFromSet, UniformIntRange
from dryml.core.errors import ParameterizationError, ParameterizationLimitError, UnsupportedGeneratorVerificationError
from dryml.core.template import Par, Shared, Template, repeat
from dryml.core.template_selector import TemplateGenerator


class SupportModel:
    """Small constructor declaration used by exact support tests."""

    def __init__(self, width=None, *, scale=None, product=None):
        self.width = width
        self.scale = scale
        self.product = product


class SupportLeaf(Object):
    """Owned node used to distinguish exact shared and independent topology."""

    def __init__(self, value):
        self.value = value


class SupportParent(Object):
    """Root node retaining an ordered group of owned children."""

    def __init__(self, children):
        self.children = children


class UnindexedDomain:
    """Provider sentinel whose unknown support cannot be exactly verified."""

    def sample(self, rng):
        return 1

    def cardinality(self):
        return None

    def value_at(self, index):
        raise AssertionError("unindexed providers must not be indexed")

    def contains(self, value):
        return None

    def bounds(self):
        return 0, 2


class InvalidBranchDomain:
    """Finite runtime support with one valid and one invalid arithmetic branch."""

    def sample(self, rng):
        return 1

    def cardinality(self):
        return 2

    def value_at(self, index):
        return (1, 0)[index]

    def contains(self, value):
        return value in {0, 1}

    def bounds(self):
        return 0, 1


def test_support_selector_enforces_correlated_and_hidden_finite_products():
    """Exact support retains joint assignments rather than projected value sets."""
    visible = Template(
        SupportModel,
        Par("width"),
        scale=2,
        product=Par("width") * 2,
    )
    selector = TemplateGenerator(visible, width=UniformFromSet((32, 64))).support_selector()

    assert selector.matches(Definition(SupportModel, 64, scale=2, product=128))
    assert not selector.matches(Definition(SupportModel, 32, scale=2, product=128))

    hidden = Template(SupportModel, product=Par("width") * Par("scale"))
    hidden_selector = TemplateGenerator(
        hidden,
        width=UniformFromSet((32, 64)),
        scale=UniformFromSet((1, 2)),
    ).support_selector()
    assert hidden_selector.matches(Definition(SupportModel, product=64))
    assert not hidden_selector.matches(Definition(SupportModel, product=96))


def test_support_selector_enforces_shared_and_independent_graph_topology():
    """Exact support uses a two-way node mapping rather than structural equality."""

    block = Definition(SupportLeaf, Par("width"))
    shared_selector = TemplateGenerator(
        Template(SupportParent, repeat([block], Shared(2))),
        width=UniformFromSet((64,)),
    ).support_selector()
    independent_selector = TemplateGenerator(
        Template(SupportParent, repeat([block], 2)),
        width=UniformFromSet((64,)),
    ).support_selector()
    shared_leaf = Definition(SupportLeaf, 64)
    shared = Definition(SupportParent, [shared_leaf, shared_leaf])
    independent = Definition(
        SupportParent,
        [Definition(SupportLeaf, 64), Definition(SupportLeaf, 64)],
    )

    assert shared_selector.matches(shared)
    assert not shared_selector.matches(independent)
    assert independent_selector.matches(independent)
    assert not independent_selector.matches(shared)


def test_direct_wide_range_membership_does_not_require_enumeration():
    """A visible range root uses exact membership before finite expansion."""
    template = Template(SupportModel, Par("width"), product=Par("width") * 2)
    selector = TemplateGenerator(template, width=UniformIntRange(1, 1_000_000)).support_selector()

    assert selector.matches(Definition(SupportModel, 999_999, product=1_999_998))
    assert not selector.matches(Definition(SupportModel, 999_999, product=1_999_999))


def test_hidden_wide_support_exceeding_budget_is_not_approximated():
    """Unknown finite roots exceed proof budget rather than accepting bounds."""
    template = Template(SupportModel, product=Par("width") * 2)
    selector = TemplateGenerator(template, width=UniformIntRange(1, 1_000_000)).support_selector(max_assignments=8)

    with pytest.raises(ParameterizationLimitError):
        selector.matches(Definition(SupportModel, product=8))


def test_support_never_hides_invalid_branches_or_unavailable_exact_proofs():
    """A later invalid branch and unindexed hidden root are explicit failures."""
    invalid = Template(SupportModel, product=10 / Par("divisor"))
    selector = TemplateGenerator(invalid, divisor=InvalidBranchDomain()).support_selector()

    with pytest.raises(ParameterizationError, match="division"):
        selector.matches(Definition(SupportModel, product=10))

    unindexed = Template(SupportModel, product=Par("width") * 2)
    unsupported = TemplateGenerator(unindexed, width=UnindexedDomain()).support_selector()
    with pytest.raises(UnsupportedGeneratorVerificationError):
        unsupported.matches(Definition(SupportModel, product=2))
