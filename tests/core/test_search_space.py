import pytest

from dryml.core2 import Definition, UniformIntRange
from dryml.core2.freeze import FrozenDict
from dryml.core2.object import Object


class SearchSpaceFixture(Object):
    def __init__(self, value, optional="default"):
        self.value = value
        self.optional = optional


def test_search_space_exposes_template_semantic_parameters_without_defaults():
    space = Definition(SearchSpaceFixture, UniformIntRange(1, 2)).as_space()

    assert space.parameters == FrozenDict({"value": space.template.value})
    assert space.value is space.template.value
    assert not hasattr(space, "optional")
    with pytest.raises(AttributeError):
        space.missing
