"""Tests for diagnostic-first environment requirement combination."""

import pytest

from dryml.annotations import Annotation, attach_annotation
from dryml.requirements import RequirementDeclaration, RequirementSource
import dryml.environments as envs
from dryml.environments import combination


def test_combines_inherited_and_selected_method_declarations_deterministically() -> None:
    """Reversed-C3 collection produces one stable value and source explanation."""

    @envs.req(requirements=("base>=1",), tags=("base",), source="base")
    class Base:
        pass

    @envs.req(requirements=("left>=1",), source="left")
    class Left(Base):
        pass

    @envs.req(tags=("right",), source="right")
    class Right(Base):
        pass

    class Target(Left, Right):
        @envs.req(capabilities=("method",), source="method")
        def work(self):
            return None

    direct = envs.requirements_for(Target)
    selected = envs.requirements_for_method(Target(), "work")
    assert direct.value.requirements == ("base>=1", "left>=1")
    assert direct.value.details["sources"] == ("1: base", "2: right", "3: left")
    assert selected.value.capabilities == ("method",)
    assert selected.value.details["sources"][-1] == "4: method"


def test_reports_every_conflict_without_partial_value() -> None:
    """Python, package, protocol, and schema conflicts remain independently visible."""

    @envs.req(python="<3", requirements=("demo<1",), dryml_protocol="<1", schema_versions={"record": "<1"}, source="first")
    class First:
        pass

    @envs.req(python=">=3", requirements=("demo>=1",), dryml_protocol=">=1", schema_versions={"record": ">=1"}, source="second")
    class Second(First):
        pass

    result = envs.requirements_for(Second)
    assert result.value is None
    assert [issue.path for issue in result.report.issues] == ["dryml_protocol", "python", "requirements.demo", "schema_versions.record"]
    assert all(tuple(source.label for source in issue.sources) == ("1: first", "2: second") for issue in result.report.issues)
    assert envs.requirements_for(Second) == result


def test_package_exclusion_conflict_attributes_both_declarations() -> None:
    """A package/exclusion conflict reports the requirement and exclusion sources."""

    @envs.req(requirements=("demo",), source="required")
    class Required:
        pass

    @envs.req(excludes=("demo",), source="excluded")
    class Excluded(Required):
        pass

    issue = envs.requirements_for(Excluded).report.issues[0]
    assert issue.path == "requirements.demo"
    assert tuple(source.label for source in issue.sources) == ("1: required", "2: excluded")


def test_method_conflict_attributes_inherited_class_and_method_sources() -> None:
    """Selected-method conflicts retain every deterministic declaration source."""

    @envs.req(python="<3", source="inherited")
    class Base:
        pass

    @envs.req(python=">=3", source="class")
    class Target(Base):
        @envs.req(python="<2", source="method")
        def work(self):
            return None

    result = envs.requirements_for_method(Target(), "work")

    assert result.value is None
    assert [issue.path for issue in result.report.issues] == ["python"]
    assert tuple(source.label for source in result.report.issues[0].sources) == (
        "1: inherited",
        "2: class",
        "3: method",
    )


def test_conflict_source_attribution_constructs_paths_once_per_declaration(monkeypatch) -> None:
    """Conflict reporting reuses preflight paths rather than rebuilding them per issue."""

    paths = tuple(f"package-{index}" for index in range(64))
    declarations = tuple(
        RequirementDeclaration(
            envs.EnvironmentRequirement(requirements=tuple(f"{path}{'<1' if index % 2 else '>=1'}" for path in paths)),
            source=RequirementSource(str(index)),
        )
        for index in range(64)
    )
    original_paths = combination._paths
    calls = 0

    def counted_paths(value):
        nonlocal calls
        calls += 1
        return original_paths(value)

    monkeypatch.setattr(combination, "_paths", counted_paths)

    result = combination._EnvironmentCombiner().combine(declarations)

    assert len(result.report.issues) == 64
    assert calls == len(declarations)


def test_malformed_attached_values_and_declaration_limit_fail_before_results() -> None:
    """Corrupt or oversized environment declarations cannot yield partial results."""

    def target():
        return None

    attach_annotation(target, Annotation(envs.ENVIRONMENT_REQUIREMENT_KEY, "bad"))
    with pytest.raises(envs.EnvironmentRequirementError):
        envs.requirements_for(target)

    class Subject:
        pass

    for index in range(65):
        attach_annotation(Subject, Annotation(envs.ENVIRONMENT_REQUIREMENT_KEY, RequirementDeclaration(envs.EnvironmentRequirement(tags=(str(index),)), source=RequirementSource(str(index)))))
    with pytest.raises(envs.EnvironmentRequirementError):
        envs.requirements_for(Subject)


def test_manual_declaration_is_preflighted_for_unverifiable_package_forms() -> None:
    """Manual annotation cannot bypass hard-package admission constraints."""

    class Subject:
        pass

    value = envs.EnvironmentRequirement(requirements=("demo[extra]>=1",))
    attach_annotation(Subject, Annotation(envs.ENVIRONMENT_REQUIREMENT_KEY, RequirementDeclaration(value, source=RequirementSource("manual"))))
    with pytest.raises(envs.EnvironmentRequirementError):
        envs.requirements_for(Subject)


def test_merge_keeps_success_and_conflict_exception_behavior() -> None:
    """The two-value API shares the diagnostic semantic merge implementation."""

    merged = envs.EnvironmentRequirement(requirements=("demo>=1",)).merge(envs.EnvironmentRequirement(requirements=("demo<2",)))
    assert merged.requirements == ("demo<2,>=1",)
    with pytest.raises(envs.EnvironmentRequirementError):
        envs.EnvironmentRequirement(requirements=("demo<1",)).merge(envs.EnvironmentRequirement(requirements=("demo>=1",)))


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("demo[feature]>=1", "demo[feature]<2", "demo[feature]<2,>=1"),
        ("demo @ https://example.invalid/demo-1.0.whl", "demo @ https://example.invalid/demo-1.0.whl", "demo @ https://example.invalid/demo-1.0.whl"),
        ("demo>=1; extra == 'feature'", "demo<2; extra == 'feature'", 'demo<2,>=1; extra == "feature"'),
    ],
)
def test_public_merge_retains_pep_508_package_forms(left: str, right: str, expected: str) -> None:
    """The retained two-value API accepts public PEP 508 requirement forms."""

    merged = envs.EnvironmentRequirement(requirements=(left,)).merge(envs.EnvironmentRequirement(requirements=(right,)))

    assert merged.requirements == (expected,)


def test_post_release_intersection_is_not_reported_as_a_conflict() -> None:
    """A valid post-release witness remains usable during declaration combination."""

    @envs.req(python=">1.post0", source="lower")
    class Lower:
        pass

    @envs.req(python="<1.post2", source="upper")
    class Upper(Lower):
        pass

    result = envs.requirements_for(Upper)

    assert result.has_value
    assert result.value.python == "<1.post2,>1.post0"


def test_multibyte_aggregate_byte_cap_is_complete_or_fails_before_a_result(monkeypatch) -> None:
    """UTF-8 aggregate accounting accepts complete boundary inputs and rejects excess."""

    @envs.req(tags=("é",), source="x")
    @envs.req(tags=("ü",), source="y")
    class Target:
        pass

    for limit in (26, 27):
        monkeypatch.setattr(combination, "_MAX_BYTES", limit)
        result = envs.requirements_for(Target)
        assert result.has_value
        assert result.value.tags == ("é", "ü")

    monkeypatch.setattr(combination, "_MAX_BYTES", 25)
    with pytest.raises(envs.EnvironmentRequirementError) as raised:
        envs.requirements_for(Target)
    assert str(raised.value) == "environment requirement collection or combination failed"
