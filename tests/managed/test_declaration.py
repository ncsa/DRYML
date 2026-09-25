"""Declaration and descriptor tests for managed operations."""

from __future__ import annotations

import asyncio
import inspect
from abc import abstractmethod

import pytest

from dryml.annotations import Annotation, attach_annotation, own_annotations
from dryml.managed import (
    ManagedConfig,
    ManagedConfigError,
    ManagedControlError,
    ManagedDeclarationError,
    managed_operation,
)


@pytest.mark.parametrize(
    "written_order",
    (
        ("abstract", "annotation", "function", "managed"),
        ("annotation", "function", "abstract", "managed"),
        ("function", "abstract", "annotation", "managed"),
        ("managed", "annotation", "function", "abstract"),
        ("annotation", "managed", "function", "abstract"),
        ("function", "managed", "annotation", "abstract"),
    ),
)
def test_abstract_managed_declarations_preserve_the_standard_marker(written_order):
    """Both abstract/managed orders preserve one abstract managed obligation."""

    from dryml.core import Serializable, function

    def annotation(target):
        return attach_annotation(target, Annotation("test.abstract", "present"))

    def operation(self, *, managed):
        raise AssertionError("abstract declarations must not run")

    decorators = {
        "abstract": abstractmethod,
        "annotation": annotation,
        "function": function,
        "managed": managed_operation(),
    }
    member = operation
    for name in reversed(written_order):
        member = decorators[name](member)

    subject = type("AbstractManagedSubject", (Serializable,), {"operation": member})

    assert inspect.isabstract(subject)
    assert subject.__abstractmethods__ == {"operation"}
    assert getattr(subject.__dict__["operation"], "__isabstractmethod__", False)


def test_managed_validator_rejects_concrete_bare_overrides_across_mro():
    """Inherited managed obligations cannot be discharged by a bare method."""

    from dryml.core import Serializable

    class AbstractManaged(Serializable):
        @abstractmethod
        @managed_operation()
        def operation(self, *, managed):
            """Provide one operation that concrete children must keep managed."""

    class StillAbstract(AbstractManaged):
        @abstractmethod
        def operation(self, *, managed):
            """Leave the inherited managed operation abstract."""

    class ConcreteManaged(AbstractManaged):
        @managed_operation()
        def operation(self, *, managed):
            """Supply the required managed implementation."""

    base_validators = AbstractManaged.__dict__["__dryml_class_validators__"]
    concrete_validators = ConcreteManaged.__dict__["__dryml_class_validators__"]

    class SiblingManaged(AbstractManaged):
        @managed_operation()
        def operation(self, *, managed):
            """Supply an independent managed implementation."""

    class OtherImplementation:
        def operation(self, *, managed):
            """Deliberately provide a non-managed competing implementation."""

    assert inspect.isabstract(StillAbstract)
    assert not inspect.isabstract(ConcreteManaged)
    assert AbstractManaged.__dict__["__dryml_class_validators__"] is base_validators
    assert ConcreteManaged.__dict__["__dryml_class_validators__"] is concrete_validators
    assert SiblingManaged.__dict__["__dryml_class_validators__"] is not base_validators

    with pytest.raises(ManagedDeclarationError, match=r"BareOverride\.operation"):
        class BareOverride(AbstractManaged):
            def operation(self, *, managed):
                """Incorrectly drop the managed declaration."""

    with pytest.raises(ManagedDeclarationError, match=r"CrossBaseOverride\.operation"):
        class CrossBaseOverride(OtherImplementation, AbstractManaged):
            pass


def test_declaration_requires_a_native_instance_function_and_keyword_only_slot():
    """The decorator rejects unsupported callables before any class binding."""

    def missing(self, value):
        return value

    def catch_all(self, value, **kwargs):
        return value

    async def asynchronous(self, *, managed):
        return None

    def generator(self, *, managed):
        yield None

    for target in (missing, catch_all, asynchronous, generator, staticmethod(missing), classmethod(missing), property(missing)):
        with pytest.raises(ManagedDeclarationError):
            managed_operation()(target)

    assert inspect.iscoroutinefunction(asynchronous)
    assert inspect.isgeneratorfunction(generator)
    assert not asyncio.iscoroutinefunction(missing)
    with pytest.raises(TypeError):
        managed_operation(version=1)


def test_declaration_rejects_forged_inspection_metadata_without_invoking_target():
    """Signature and wrapping metadata cannot replace native function authority."""

    def target(receiver, *, managed):
        raise AssertionError("declaration must not invoke the target")

    target.__signature__ = inspect.Signature()
    with pytest.raises(ManagedDeclarationError):
        managed_operation()(target)

    def wrapped(receiver, *, managed):
        raise AssertionError("declaration must not invoke the target")

    wrapped.__wrapped__ = target
    with pytest.raises(ManagedDeclarationError):
        managed_operation()(wrapped)


def test_descriptor_is_inert_on_class_access_and_records_stable_member_identity():
    """Binding creates an invocation view but cannot run the raw method in U3."""

    calls = []

    class Base:
        @managed_operation(resumable=True)
        def train(receiver, value=3, *, managed):
            calls.append((receiver, value, managed))
            return value

    class Child(Base):
        pass

    assert Base.train is Base.__dict__["train"]
    assert Child.train is Base.__dict__["train"]
    assert Base.train.member == "train"
    assert Child.train.member == "train"
    assert Base.train.resumable is True

    bound = Base().train
    signature = inspect.signature(bound)
    assert signature.parameters["value"].default == 3
    assert signature.parameters["managed"].default is None
    assert "ManagedConfig" in str(signature.parameters["managed"].annotation)
    # U6 owns lifecycle execution; this declaration-only test must not assume an
    # unavailable placeholder instead of the real Store-backed behavior.
    assert calls == []


def test_descriptor_rejects_same_member_reuse_by_an_unrelated_owner():
    """A copied declaration cannot retain stale owner evidence for transport."""

    class First:
        @managed_operation()
        def operation(self, *, managed):
            """Provide one source declaration that must remain single-owner."""

    with pytest.raises((ManagedDeclarationError, RuntimeError)) as caught:
        class Second:
            operation = First.__dict__["operation"]

    error = caught.value
    if not isinstance(error, ManagedDeclarationError):
        error = error.__cause__
    assert isinstance(error, ManagedDeclarationError)
    assert "multiple owners" in str(error)


def test_descriptor_rejects_different_member_names_and_preserves_passive_annotations():
    """One declaration has one stable key and remains annotation-compatible."""

    copied = Annotation("test.copied", "copied")
    later = Annotation("test.later", "later")

    def copy_then_manage(target):
        return attach_annotation(target, copied)

    @managed_operation()
    @copy_then_manage
    def first(receiver, *, managed):
        return None

    assert own_annotations(first) == (copied,)
    attach_annotation(first, later)
    assert own_annotations(first) == (copied, later)

    class Subject:
        operation = first

    with pytest.raises(ManagedDeclarationError, match="multiple"):
        first.__set_name__(Subject, "other")

    attached_first = Annotation("test.first", "first")

    def attach_after_manage(target):
        return attach_annotation(target, attached_first)

    @attach_after_manage
    @managed_operation()
    def second(receiver, *, managed):
        return None

    assert own_annotations(second) == (attached_first,)


def test_invalid_snapshot_or_arguments_do_not_run_the_authored_method():
    """U3 completes every supported validation before lifecycle delegation."""

    calls = []

    class Subject:
        @managed_operation()
        def operation(self, value, *, managed):
            calls.append(value)

    callbacks = [lambda object_, context: None]
    config = ManagedConfig(callbacks=callbacks)
    callbacks.append(object())
    with pytest.raises(ManagedConfigError):
        Subject().operation("value", managed=config)
    with pytest.raises(ManagedConfigError):
        Subject().operation({"unsupported": object()})
    assert calls == []
