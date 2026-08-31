import pytest

from dryml.core import Definition, ObjectId, ObjectRef, StateRef, UniformIntRange
from dryml.core.freeze import FrozenDict
from dryml.core.object import Object, Serializable
from dryml.core.utils.graph.path import GraphPath


class SearchSpaceFixture(Object):
    def __init__(self, value, optional="default"):
        self.value = value
        self.optional = optional


class SearchSpaceStateful(Serializable):
    pass


def test_search_space_exposes_template_semantic_parameters_without_defaults():
    space = Definition(SearchSpaceFixture, UniformIntRange(1, 2)).as_space()

    assert space.parameters == FrozenDict({"value": space.template.value})
    assert space.value is space.template.value
    assert not hasattr(space, "optional")
    with pytest.raises(AttributeError):
        space.missing


def test_search_space_keeps_exact_references_atomic():
    leaf = Definition(SearchSpaceStateful).concretize()
    object_ref = ObjectRef(leaf, {GraphPath(): ObjectId()})
    state = StateRef(object_ref, {GraphPath(): "pkl-" + "d" * 64})
    search = Definition(SearchSpaceFixture, {"reference": state, "value": UniformIntRange(1, 2)}).as_space()

    assert len(search.params) == 1
    assert all(sample.parameters["value"]["reference"] is state for sample in search.grid())
