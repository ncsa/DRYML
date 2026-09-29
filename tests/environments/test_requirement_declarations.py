"""Tests for passive environment requirement declarations."""

import traceback

import pytest

from dryml.annotations import Annotation, attach_annotation
from dryml.requirements import RequirementDeclaration, RequirementSource
import dryml.environments as envs


def test_decorators_preserve_supported_target_identity_and_calls() -> None:
    """Declarations attach without wrapping classes, functions, or descriptors."""

    def function(value):
        return value + 1

    assert envs.req(tags=("unit",))(function) is function
    assert function(1) == 2

    class Subject:
        @envs.req(capabilities=("feature",))
        def method(self):
            return "method"

        @envs.req(tags=("static",))
        @staticmethod
        def first():
            return "first"

        @staticmethod
        @envs.req(tags=("second",))
        def second():
            return "second"

    decorated = envs.req(python=">=3.10", requirements=("dryml>=0.3",), excludes=("tensorflow",), capabilities=("feature",), tags=("unit",), dryml_protocol=">=1", schema_versions={"environment_record": ">=1"})(Subject)
    assert decorated is Subject
    assert Subject().method() == "method"
    assert Subject.first() == "first"
    assert Subject.second() == "second"


def test_method_resolution_is_static_for_classes_and_instances() -> None:
    """Selected method resolution neither binds descriptors nor reads instance state."""

    class HostileDescriptor:
        def __get__(self, instance, owner):
            raise AssertionError("descriptor binding must not run")

    class Subject:
        def __getattribute__(self, name):
            raise AssertionError("instance lookup must not run")

        work = envs.req(tags=("method",))(HostileDescriptor())

    envs.req(tags=("class",))(Subject)
    from_class = envs.requirements_for_method(Subject, "work")
    from_instance = envs.requirements_for_method(Subject(), "work")
    assert from_class == from_instance
    assert from_class.value.tags == ("class", "method")


def test_declarations_use_only_the_environment_annotation_key() -> None:
    """Foreign annotations and retired fragment attributes cannot affect resolution."""

    @envs.req(tags=("environment",))
    class Target:
        pass

    attach_annotation(Target, Annotation("dryml.worlds.requirement", RequirementDeclaration(object(), source=RequirementSource("world"))))
    Target.__dryml_environment_fragments__ = (object(),)
    result = envs.requirements_for(Target)
    assert result.value.tags == ("environment",)
    assert not hasattr(envs, "add_req")
    assert not hasattr(envs, "requirements_for_class")
    with pytest.raises(ModuleNotFoundError):
        __import__("dryml.environments.fragments")


def test_declaration_rejects_unverifiable_package_forms() -> None:
    """Hard declarations reject package forms absent from environment evidence."""

    for requirement in ("demo[extra]>=1", "demo @ https://example.test/demo.whl", "demo; extra == 'test'"):
        with pytest.raises(envs.EnvironmentRequirementError):
            envs.req(requirements=(requirement,))


def test_declaration_bounds_one_shot_inputs_before_attachment() -> None:
    """The sixty-fifth input is consumed only to reject the declaration."""

    seen: list[int] = []

    def values():
        for index in range(65):
            seen.append(index)
            yield f"tag-{index}"

    with pytest.raises(envs.EnvironmentRequirementError):
        envs.req(tags=values())
    assert seen == list(range(65))


def test_declaration_treats_exact_string_fields_as_single_entries() -> None:
    """Exact strings retain the scalar behavior used by environment requirements."""

    class Target:
        pass

    envs.req(
        requirements="dryml>=0.3",
        excludes="tensorflow",
        capabilities="feature",
        tags="gpu",
    )(Target)

    requirement = envs.requirements_for(Target).value
    assert requirement.requirements == ("dryml>=0.3",)
    assert requirement.excludes == ("tensorflow",)
    assert requirement.capabilities == ("feature",)
    assert requirement.tags == ("gpu",)

    with pytest.raises(envs.EnvironmentRequirementError):
        envs.req(tags="x" * 4097)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"python": "/private/token=secret"},
        {"dryml_protocol": "/private/token=secret"},
        {"schema_versions": {"environment_record": "/private/token=secret"}},
    ),
)
def test_declaration_hides_specifier_validation_tracebacks(kwargs: dict[str, object]) -> None:
    """Specifier validation exposes only the fixed environment declaration error."""

    with pytest.raises(envs.EnvironmentRequirementError) as excinfo:
        envs.req(**kwargs)

    assert type(excinfo.value) is envs.EnvironmentRequirementError
    assert str(excinfo.value) == "environment requirement declaration is invalid"
    assert excinfo.value.__cause__ is None
    formatted = "".join(traceback.format_exception(excinfo.value))
    assert "/private" not in formatted
    assert "secret" not in formatted


@pytest.mark.parametrize("source", ("", "bad\n", "x" * 257))
def test_declaration_normalizes_shared_source_failures(source: str) -> None:
    """Malformed shared source values use the environment exception contract."""

    with pytest.raises(envs.EnvironmentRequirementError) as excinfo:
        envs.req(source=source)

    assert type(excinfo.value) is envs.EnvironmentRequirementError
    assert str(excinfo.value) == "environment requirement declaration is invalid"
    assert excinfo.value.__cause__ is None
