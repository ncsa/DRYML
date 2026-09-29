"""Tests for shared static requirement declaration adapters."""

import pytest

from dryml.annotations import Annotation, attach_annotation
from dryml.requirements import RequirementDeclaration, RequirementError, RequirementSource
from dryml.requirements.collection import attach_declaration, collect_declarations


KEY = "example.requirement"


def _declaration(value):
    """Create an exact declaration with a stable source for adapter tests."""

    return RequirementDeclaration(value, source=RequirementSource("test"))


def test_adapters_collect_selected_owner_key_in_static_annotation_order():
    """Class and selected method collection reuse the passive annotation ordering."""

    class Base:
        def work(self):
            return None

    class Child(Base):
        def work(self):
            return None

    attach_declaration(Base, key=KEY, declaration=_declaration("base"))
    attach_declaration(Child, key=KEY, declaration=_declaration("child"))
    attach_declaration(Child.__dict__["work"], key=KEY, declaration=_declaration("method"))
    attach_annotation(Child, Annotation("foreign.requirement", _declaration("foreign")))

    assert [item.value for item in collect_declarations(Child, key=KEY, value_type=str)] == ["base", "child"]
    assert [item.value for item in collect_declarations(Child(), key=KEY, value_type=str, method_name="work")] == [
        "base",
        "child",
        "method",
    ]


def test_adapters_reject_corrupt_selected_entries_without_partial_collection():
    """Selected-key corruption and invalid type contracts fail before return."""

    def target():
        return None

    attach_annotation(target, Annotation(KEY, "not a declaration"))
    with pytest.raises(RequirementError):
        collect_declarations(target, key=KEY, value_type=str)
    with pytest.raises(RequirementError):
        collect_declarations(target, key=KEY, value_type="str")


def test_instance_method_collection_uses_only_the_exact_class():
    """Instance normalization never performs dynamic lookup on the instance."""

    class Subject:
        def __getattribute__(self, name):
            raise AssertionError("instance lookup must not run")

        def work(self):
            return None

    attach_declaration(Subject, key=KEY, declaration=_declaration("class"))
    attach_declaration(Subject.__dict__["work"], key=KEY, declaration=_declaration("method"))

    assert [item.value for item in collect_declarations(Subject(), key=KEY, value_type=str, method_name="work")] == [
        "class",
        "method",
    ]
