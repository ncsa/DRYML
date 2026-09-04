"""Tests for passive world requirement declarations."""

import pytest

from dryml.annotations import Annotation, attach_annotation
from dryml.requirements import RequirementDeclaration, RequirementSource
import dryml.worlds as worlds


def test_decorators_preserve_target_identity_and_static_method_safety() -> None:
    """World declarations attach without wrapping or binding live targets."""

    def function(value):
        return value + 1

    assert worlds.req(cpus=2)(function) is function
    assert function(1) == 2

    class HostileDescriptor:
        def __get__(self, instance, owner):
            raise AssertionError("descriptor binding must not run")

    class Subject:
        work = worlds.req(cpus=2)(HostileDescriptor())

        @worlds.req(memory="1GiB")
        @staticmethod
        def first():
            return "first"

        @staticmethod
        @worlds.req(cpus=1)
        def second():
            return "second"

    assert worlds.req(replicas=2)(Subject) is Subject
    assert Subject.first() == "first"
    assert Subject.second() == "second"
    assert worlds.requirements_for_method(Subject(), "work").value.roles["main"].resources.cpus.min == 2


def test_flattened_declaration_omits_every_unsupplied_constraint() -> None:
    """A flattened CPU requirement does not invent replicas or other resources."""

    @worlds.req(cpus=2)
    class Target:
        pass

    role = worlds.requirements_for(Target).value.roles["main"]
    assert role.replicas == worlds.CountConstraint()
    assert role.resources.cpus == worlds.CountConstraint(2, 2)
    assert role.resources.memory == worlds.CountConstraint()
    assert not role.resources.accelerators
    assert not role.resources.accelerator_memory
    assert not role.resources.devices
    assert not role.resources.named


def test_complete_roles_grammar_is_exclusive_and_never_invents_main() -> None:
    """Multi-role input is complete and cannot be mixed with flattened fields."""

    @worlds.req(roles={"worker": {"replicas": {"min": 2, "max": None}, "resources": {"memory": "1GiB"}}})
    class Target:
        pass

    result = worlds.requirements_for(Target)
    assert tuple(result.value.roles) == ("worker",)
    assert result.value.roles["worker"].resources.memory == worlds.CountConstraint(1024**3, 1024**3)

    for kwargs in (
        {"roles": {"worker": {}}, "cpus": 1},
        {"roles": {"worker": {}}, "topology": {"rack": "a"}},
        {"roles": {"worker": {}}, "role": "worker"},
    ):
        with pytest.raises(worlds.WorldRequirementError):
            worlds.req(**kwargs)
    worlds.req(roles={"worker": {}}, cpus=None, role="main")
    with pytest.raises(worlds.WorldRequirementError):
        worlds.req(topology={"note": "x" * 4097})


def test_world_collection_ignores_foreign_annotation_keys() -> None:
    """Only the world key contributes to world combination."""

    @worlds.req(cpus=1)
    class Target:
        pass

    attach_annotation(Target, Annotation("dryml.environments.requirement", RequirementDeclaration(object(), source=RequirementSource("environment"))))
    assert worlds.requirements_for(Target).value.roles["main"].resources.cpus.min == 1
