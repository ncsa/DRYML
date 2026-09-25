import inspect
from abc import ABC, abstractmethod

import pytest

from dryml.core import Definition, Object, Repo
from dryml.core.object import (
    _register_class_transformer,
    _register_class_validator,
    definition_mode,
    selector_mode,
    space_mode,
)


class AbstractObject(Object):
    @abstractmethod
    def instance_method(self):
        """Return the implementation-specific result."""

    @property
    @abstractmethod
    def property_value(self):
        """Return the implementation-specific property value."""

    @classmethod
    @abstractmethod
    def class_method(cls):
        """Return the implementation-specific class result."""

    @staticmethod
    @abstractmethod
    def static_method():
        """Return the implementation-specific static result."""


class PartialAbstractObject(AbstractObject):
    def instance_method(self):
        return "instance"


class ConcreteAbstractObject(PartialAbstractObject):
    @property
    def property_value(self):
        return "property"

    @classmethod
    def class_method(cls):
        return "class"

    @staticmethod
    def static_method():
        return "static"


def test_object_abstractness_uses_standard_abc_finalization():
    class ABCMixin(ABC):
        @abstractmethod
        def mixed_method(self):
            """Return the mixin result."""

    class MixedAbstractObject(Object, ABCMixin):
        pass

    class MROConcreteMixin:
        def mixed_method(self):
            return "mixed"

    class MROConcreteObject(MROConcreteMixin, MixedAbstractObject):
        pass

    assert inspect.isabstract(AbstractObject)
    assert AbstractObject.__abstractmethods__ == {
        "instance_method", "property_value", "class_method", "static_method",
    }
    assert inspect.isabstract(PartialAbstractObject)
    assert PartialAbstractObject.__abstractmethods__ == {
        "property_value", "class_method", "static_method",
    }
    assert not inspect.isabstract(ConcreteAbstractObject)
    assert not ConcreteAbstractObject.__abstractmethods__
    assert inspect.isabstract(MixedAbstractObject)
    assert not inspect.isabstract(MROConcreteObject)
    assert MROConcreteObject.mixed_method(None) == "mixed"


def test_class_finalization_callbacks_are_class_local_and_phased():
    events = []

    class CallbackBase(Object):
        pass

    def base_transformer(cls):
        events.append(("base-transform", cls))
        cls.transformed = True

    def base_validator(cls):
        events.append(("base-validate", cls))
        assert cls.transformed

    _register_class_transformer(CallbackBase, base_transformer)
    _register_class_transformer(CallbackBase, base_transformer)
    _register_class_validator(CallbackBase, base_validator)

    class CallbackChild(CallbackBase):
        pass

    assert events == [
        ("base-transform", CallbackChild),
        ("base-validate", CallbackChild),
    ]


def test_abstract_object_inert_modes_do_not_construct_or_resolve():
    class AbstractProbe(Object):
        effects = 0

        @abstractmethod
        def required(self):
            """Return the required implementation result."""

        def __init__(self, value):
            type(self).effects += 1
            self.value = value

    with definition_mode():
        definition = AbstractProbe("definition")
    with definition_mode(concrete=True):
        cdef = AbstractProbe("concrete")
    with selector_mode():
        selector = AbstractProbe("selector")
    with space_mode():
        space = AbstractProbe("space")

    assert isinstance(definition, Definition)
    assert cdef is not None
    assert selector is not None
    assert space is not None
    assert AbstractProbe.effects == 0


def test_abstract_object_direct_construction_rejects_before_effects():
    class AbstractProbe(Object):
        effects = []

        @abstractmethod
        def required(self):
            """Return the required implementation result."""

        @classmethod
        def __pre_init__(cls):
            cls.effects.append("pre-init")

        def __new__(cls):
            cls.effects.append("new")
            return super().__new__(cls)

        def __init__(self):
            type(self).effects.append("init")

    with pytest.raises(TypeError, match="required"):
        AbstractProbe(repo=Repo())

    assert AbstractProbe.effects == []

    cdef = Definition(AbstractProbe).concretize(repo=Repo())
    with pytest.raises(TypeError, match="required"):
        AbstractProbe(repo=Repo(), __cdef__=cdef)

    assert AbstractProbe.effects == []


def test_abstract_object_definition_build_rejects_before_concrete_dependency():
    class ConcreteDependency(Object):
        constructions = 0

        def __init__(self):
            type(self).constructions += 1

    class AbstractRoot(Object):
        @abstractmethod
        def required(self):
            """Return the required implementation result."""

        def __init__(self, dependency):
            self.dependency = dependency

    repo = Repo()
    root = Definition(AbstractRoot, Definition(ConcreteDependency)).concretize(repo=repo)

    with pytest.raises(TypeError, match="required"):
        repo.load_or_build(root)

    assert ConcreteDependency.constructions == 0
    assert repo._num_constructions == 0
