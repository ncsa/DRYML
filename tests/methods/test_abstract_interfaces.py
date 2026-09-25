"""Abstract logical-call contracts for direct and trait-authored Methods."""

from __future__ import annotations

import inspect
from abc import abstractmethod

import pytest

from dryml.methods import ImplementationDeclarationError, Method, traits


class AbstractLogicalCall(Method):
    """Require one logical Method call without requiring output inference."""

    @abstractmethod
    def __call__(self, value):
        """Return the implementation-specific result for ``value``."""


def test_trait_catalog_discharges_only_an_abstract_logical_call():
    """One concrete trait lets a trait-only child satisfy ``__call__`` statically."""

    calls = []

    class TraitOnly(AbstractLogicalCall):
        @traits()
        def implementation(self, value):
            calls.append(value)
            return value + 1

    assert not inspect.isabstract(TraitOnly)
    assert object.__new__(TraitOnly)(2) == 3
    assert calls == [2]


def test_abstract_trait_alternatives_do_not_discharge_or_execute():
    """Abstract alternatives remain obligations and never become catalog targets."""

    from dryml.methods.method import _catalog_for_class

    class AbstractAlternative(AbstractLogicalCall):
        @traits()
        @abstractmethod
        def implementation(self, value):
            """Return the backend-specific implementation result."""

    assert inspect.isabstract(AbstractAlternative)
    assert AbstractAlternative.__abstractmethods__ == {"__call__", "implementation"}
    assert _catalog_for_class(AbstractAlternative, receiver=None) == ()
    with pytest.raises(TypeError, match="__call__"):
        AbstractAlternative()


def test_abstract_method_catalog_keeps_declaration_errors_and_runtime_classes_concrete():
    """Trait validation remains strict while root and runtime-only Methods stay usable."""

    class TraitBase(AbstractLogicalCall):
        @traits()
        def implementation(self, value):
            return value

    class UnannotatedShadow(TraitBase):
        def implementation(self, value):
            return value

    class RuntimeOnly(Method):
        pass

    assert not inspect.isabstract(Method)
    assert not inspect.isabstract(RuntimeOnly)
    assert object.__new__(RuntimeOnly).call_mode == "eager"
    with pytest.raises(ImplementationDeclarationError, match="unannotated shadow"):
        object.__new__(UnannotatedShadow).implementations()


def test_abstract_overrides_remove_inherited_catalog_slots_without_changing_mro_rules():
    """Abstract declarations block older same-name targets until a concrete override restores one."""

    from dryml.methods.method import _catalog_for_class

    class ConcreteBase(Method):
        def __call__(self, value):
            """Return the concrete base result."""

            return "base", value

    class AbstractChild(ConcreteBase):
        @abstractmethod
        def __call__(self, value):
            """Require a replacement logical call."""

    class ConcreteGrandchild(AbstractChild):
        def __call__(self, value):
            """Restore the logical call below the abstract intermediate."""

            return "grandchild", value

    class TraitBase(Method):
        @traits()
        def implementation(self, value):
            """Provide one concrete trait alternative."""

            return value

    class AbstractTraitOverride(TraitBase):
        @traits()
        @abstractmethod
        def implementation(self, value):
            """Require a replacement trait alternative."""

    class MroConcrete(Method):
        def __call__(self, value):
            """Return the concrete MRO branch result."""

            return "mro-concrete", value

    class MroAbstract(Method):
        @abstractmethod
        def __call__(self, value):
            """Require the MRO-selected logical call."""

    class AbstractFirst(MroAbstract, MroConcrete):
        pass

    class ConcreteFirst(MroConcrete, MroAbstract):
        pass

    class Left(Method):
        @traits()
        def implementation(self, value):
            """Declare an unrelated trait slot."""

            return value

    class Right(Method):
        @traits()
        def implementation(self, value):
            """Declare a conflicting unrelated trait slot."""

            return value

    class UnrelatedConflict(Left, Right):
        pass

    assert inspect.isabstract(AbstractChild)
    assert _catalog_for_class(AbstractChild, receiver=None) == ()
    assert inspect.isabstract(AbstractTraitOverride)
    assert _catalog_for_class(AbstractTraitOverride, receiver=None) == ()
    assert not inspect.isabstract(ConcreteGrandchild)
    assert object.__new__(ConcreteGrandchild)(3) == ("grandchild", 3)
    assert inspect.isabstract(AbstractFirst)
    assert _catalog_for_class(AbstractFirst, receiver=None) == ()
    assert not inspect.isabstract(ConcreteFirst)
    assert object.__new__(ConcreteFirst)(4) == ("mro-concrete", 4)
    with pytest.raises(ImplementationDeclarationError, match="conflict"):
        _catalog_for_class(UnrelatedConflict, receiver=None)
