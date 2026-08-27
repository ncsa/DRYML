import pytest

from dryml.core import Definition, Selector
from dryml.core.freeze import FrozenDict
from dryml.core.object import Object


class SelectorFixture(Object):
    def __init__(self, value, optional="default"):
        self.value = value
        self.optional = optional


def test_selector_exposes_only_supplied_semantic_parameters():
    selector = Selector(Definition(SelectorFixture, 4))

    assert selector.parameters == FrozenDict({"value": 4})
    assert selector.value == 4
    assert not hasattr(selector, "optional")
    with pytest.raises(AttributeError):
        selector.missing
