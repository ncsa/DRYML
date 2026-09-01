"""Tests for deterministic static annotation collection."""

from dataclasses import dataclass

import pytest

from dryml.annotations import (
    Annotation,
    AnnotationValidationError,
    UnsupportedAnnotationTargetError,
    annotations_for_class,
    annotations_for_members,
    attach_annotation,
    collect_annotations,
)


def _attach(key, value):
    """Return a consumer-owned decorator using the passive attachment seam."""

    def decorate(target):
        return attach_annotation(target, Annotation(key, value))

    return decorate


def test_class_collection_uses_reversed_c3_and_direct_declaration_order():
    """Diamond declarations are ordered base-to-subclass through reversed C3."""

    @_attach("requirements.entry", "root")
    class Root:
        pass

    @_attach("requirements.entry", "left")
    class Left(Root):
        pass

    @_attach("requirements.entry", "right")
    class Right(Root):
        pass

    @_attach("requirements.entry", "leaf-first")
    @_attach("requirements.entry", "leaf-second")
    class Leaf(Left, Right):
        pass

    assert [item.value for item in annotations_for_class(Leaf)] == [
        "root",
        "right",
        "left",
        "leaf-second",
        "leaf-first",
    ]
    assert [item.value for item in collect_annotations(Leaf, key="requirements.entry")] == [
        "root",
        "right",
        "left",
        "leaf-second",
        "leaf-first",
    ]


def test_class_collection_bypasses_hostile_metaclass_mro_hook():
    """Class traversal reads the native MRO without metaclass interception."""

    class HostileMeta(type):
        @property
        def __mro__(cls):
            raise AssertionError("metaclass MRO hook must not run")

    class Subject(metaclass=HostileMeta):
        pass

    entry = Annotation("consumer.class", object())
    attach_annotation(Subject, entry)
    assert annotations_for_class(Subject) == (entry,)


def test_collection_filters_by_exact_key_without_interpreting_consumer_values():
    """Independent consumer types select their own annotations by owner key."""

    @dataclass(frozen=True)
    class MethodTrait:
        label: str

    @dataclass(frozen=True)
    class Requirement:
        capability: str

    class Subject:
        pass

    method_trait = MethodTrait("vectorized")
    requirement = Requirement("gpu")
    attach_annotation(Subject, Annotation("method.trait", method_trait))
    attach_annotation(Subject, Annotation("requirements.capability", requirement))

    assert collect_annotations(Subject, key="method.trait") == (Subject.__dict__["__dryml_annotations__"][0],)
    assert [item.value.capability for item in collect_annotations(Subject, key="requirements.capability")] == ["gpu"]


@pytest.mark.parametrize("key", [0, "", "1invalid", "x" * 129, "not-ascii-\u00f1"])
def test_collection_rejects_invalid_filter_keys(key):
    """Filtering validates only the generic annotation-key grammar."""

    class Subject:
        pass

    with pytest.raises(AnnotationValidationError, match="key"):
        collect_annotations(Subject, key=key)


def test_collection_rejects_hostile_string_filter_subclasses_without_running_hooks():
    """Filter validation rejects subclasses before their string hooks can run."""

    class HostileString(str):
        def __len__(self):
            raise AssertionError("filter length hook must not run")

        def isascii(self):
            raise AssertionError("filter ASCII hook must not run")

        def __eq__(self, other):
            raise AssertionError("filter comparison hook must not run")

    class Subject:
        pass

    with pytest.raises(AnnotationValidationError, match="key"):
        collect_annotations(Subject, key=HostileString("consumer.hostile"))


def test_collection_rejects_unsupported_targets_and_corrupted_metadata():
    """Collection fails rather than returning incomplete generic declarations."""

    class DefinitionLike:
        pass

    with pytest.raises(UnsupportedAnnotationTargetError):
        collect_annotations(DefinitionLike())
    with pytest.raises(AnnotationValidationError):
        annotations_for_class(DefinitionLike())

    class Corrupt:
        pass

    Corrupt.__dryml_annotations__ = ("bad",)
    with pytest.raises(AnnotationValidationError, match="malformed"):
        annotations_for_class(Corrupt)


def test_member_collection_preserves_base_to_subclass_declarations_and_shadows():
    """Member evidence retains raw declarations, including unannotated shadows."""

    class Root:
        def root_member(self):
            return "root"

        def shared(self):
            return "root"

    class Left(Root):
        def left_member(self):
            return "left"

    class Right(Root):
        def right_member(self):
            return "right"

    class Leaf(Left, Right):
        def shared(self):
            return "leaf"

        def leaf_member(self):
            return "leaf"

    root_entry = Annotation("consumer.member", "root")
    left_entry = Annotation("consumer.member", "left")
    right_entry = Annotation("consumer.member", "right")
    leaf_entry = Annotation("consumer.member", "leaf")
    attach_annotation(Root.__dict__["root_member"], root_entry)
    attach_annotation(Root.__dict__["shared"], root_entry)
    attach_annotation(Left.__dict__["left_member"], left_entry)
    attach_annotation(Right.__dict__["right_member"], right_entry)
    attach_annotation(Leaf.__dict__["leaf_member"], leaf_entry)

    members = annotations_for_members(Leaf)

    assert [(member.owner, member.name, member.descriptor, member.annotations) for member in members] == [
        (Root, "root_member", Root.__dict__["root_member"], (root_entry,)),
        (Root, "shared", Root.__dict__["shared"], (root_entry,)),
        (Right, "right_member", Right.__dict__["right_member"], (right_entry,)),
        (Left, "left_member", Left.__dict__["left_member"], (left_entry,)),
        (Leaf, "shared", Leaf.__dict__["shared"], ()),
        (Leaf, "leaf_member", Leaf.__dict__["leaf_member"], (leaf_entry,)),
    ]


def test_member_collection_filters_exact_keys_without_losing_shadows_or_identity_deduplication():
    """Filtering applies to direct member entries before shadow evidence is retained."""

    class Base:
        def method(self):
            return "base"

    class Leaf(Base):
        def method(self):
            return "leaf"

    shared = Annotation("consumer.match", "shared")
    other = Annotation("consumer.other", "other")
    attach_annotation(Base.__dict__["method"], shared)
    attach_annotation(Base.__dict__["method"], shared)
    attach_annotation(Base.__dict__["method"], other)

    members = annotations_for_members(Leaf, key="consumer.match")

    assert [(member.owner, member.name, member.annotations) for member in members] == [
        (Base, "method", (shared,)),
        (Leaf, "method", ()),
    ]


def test_member_collection_preserves_non_descriptor_shadows():
    """A later ordinary value remains visible when it shadows an annotated member."""

    class Base:
        def method(self):
            return "base"

    entry = Annotation("consumer.member", "base")
    attach_annotation(Base.__dict__["method"], entry)

    class Leaf(Base):
        method = None

    members = annotations_for_members(Leaf)

    assert [(member.owner, member.name, member.descriptor, member.annotations) for member in members] == [
        (Base, "method", Base.__dict__["method"], (entry,)),
        (Leaf, "method", None, ()),
    ]


def test_member_collection_rejects_invalid_inputs_and_corrupt_member_metadata():
    """Member discovery fails atomically through the existing annotation errors."""

    class Subject:
        def valid(self):
            return "valid"

        def corrupt(self):
            return "corrupt"

    entry = Annotation("consumer.member", "valid")
    attach_annotation(Subject.__dict__["valid"], entry)
    Subject.__dict__["corrupt"].__dryml_annotations__ = ("bad",)

    with pytest.raises(AnnotationValidationError, match="requires a class"):
        annotations_for_members(Subject())
    with pytest.raises(AnnotationValidationError, match="key"):
        annotations_for_members(Subject, key=0)
    with pytest.raises(AnnotationValidationError, match="malformed"):
        annotations_for_members(Subject)
