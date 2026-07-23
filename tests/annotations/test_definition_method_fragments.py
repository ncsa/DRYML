from __future__ import annotations

import pytest

import dryml
import dryml.annotations as ann
from dryml.core.definition import ConcreteDefinition, Definition


def _requirements(fragments):
    values: list[str] = []
    for fragment in fragments:
        values.extend(fragment.fragment.get("requirements", ()))
    return tuple(values)


@dryml.env.req(requirements=("def-class>=1",))
class DefinitionModel:
    built = False

    def __init__(self):
        type(self).built = True
        raise AssertionError("definition method collection must not build objects")

    @dryml.env.req(requirements=("def-method>=1",))
    def train(self):
        return None


def test_definition_method_collection_uses_definition_cls_without_building():
    defn = Definition(DefinitionModel)

    fragments = ann.fragments_for_definition_method(defn, "train", namespace="environment")

    assert _requirements(fragments) == ("def-class>=1", "def-method>=1")
    assert DefinitionModel.built is False


def test_concrete_definition_method_resolution_uses_cls_without_building():
    cdef = ConcreteDefinition(DefinitionModel)

    resolution = ann.resolve_definition_method_requirements(cdef, "train")

    assert tuple(resolution.environment_requirement.requirements) == ("def-class>=1", "def-method>=1")
    assert DefinitionModel.built is False


def test_object_exposing_definition_is_supported_without_building():
    class Holder:
        definition = Definition(DefinitionModel)

    assert _requirements(ann.fragments_for_definition_method(Holder(), "train", namespace="environment")) == ("def-class>=1", "def-method>=1")


def test_unresolvable_definition_class_raises_type_error():
    with pytest.raises(TypeError):
        ann.fragments_for_definition_method(object(), "train")


def test_missing_definition_method_raises_attribute_error():
    with pytest.raises(AttributeError):
        ann.fragments_for_definition_method(Definition(DefinitionModel), "missing")
