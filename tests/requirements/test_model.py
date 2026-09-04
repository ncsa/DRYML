"""Tests for immutable shared requirement value contracts."""

from dataclasses import FrozenInstanceError
import inspect

import pytest

from dryml.requirements import (
    RequirementDeclaration,
    RequirementError,
    RequirementIssue,
    RequirementReport,
    RequirementResult,
    RequirementSource,
)


def test_shared_values_are_immutable_and_have_legal_result_states():
    """Shared carriers preserve values while result states remain unambiguous."""

    source = RequirementSource("declared", module="example", qualname="Task.run")
    declaration = RequirementDeclaration("value", source=source)
    issue = RequirementIssue("example.conflict", "cannot use token=secret at /private/file", sources=(source,))
    report = RequirementReport((issue,))

    assert declaration == RequirementDeclaration("value", source=source)
    assert report.ok is False
    assert RequirementReport().ok is True
    assert RequirementResult().ok is True
    assert RequirementResult().has_value is False
    assert RequirementResult("value").ok is True
    assert RequirementResult("value").has_value is True
    assert RequirementResult(report=report).ok is False
    assert RequirementResult(report=report).has_value is False
    assert "secret" not in issue.message
    assert "/private/file" not in issue.message
    with pytest.raises(FrozenInstanceError):
        source.label = "changed"


@pytest.mark.parametrize(
    "factory",
    (
        lambda: RequirementSource(""),
        lambda: RequirementSource("x\n"),
        lambda: RequirementSource("x" * 257),
        lambda: RequirementSource("ok", module="x" * 513),
        lambda: RequirementIssue("not qualified", "message"),
        lambda: RequirementIssue("example.conflict", "message", sources=(RequirementSource("ok"),) * 4097),
        lambda: RequirementReport((RequirementIssue("example.conflict", "message"),) * 1025),
        lambda: RequirementResult("value", RequirementReport((RequirementIssue("example.conflict", "message"),))),
    ),
)
def test_shared_values_reject_malformed_or_ambiguous_states(factory):
    """Invalid source, report, and result shapes cannot reach consumers."""

    with pytest.raises(RequirementError):
        factory()


def test_errors_project_context_without_formatting_untrusted_values():
    """Errors retain bounded immutable diagnostics without invoking value hooks."""

    class Unprintable:
        def __str__(self):
            raise AssertionError("diagnostic projection must not format unknown values")

    error = RequirementError(
        "failed password=secret at https://user:pass@example.invalid/path?token=value#fragment",
        context={"token": "value", "nested": [Unprintable()], "path": "/private/file"},
    )

    assert "secret" not in str(error)
    assert "user:pass@" not in str(error)
    assert "value" not in error.context["token"]
    assert error.context["nested"] == ("<unsupported>",)
    assert error.context["path"] == "<local-path>"
    with pytest.raises(TypeError):
        error.context["new"] = "value"


def test_report_enforces_aggregate_projected_diagnostic_capacity():
    """Repeated safe source associations still obey the report byte budget."""

    source = RequirementSource("l" * 256, module="m" * 512, qualname="q" * 512)
    issue = RequirementIssue("example.conflict", "conflict", sources=(source,) * 3276)

    assert RequirementReport((issue,)).ok is False
    with pytest.raises(RequirementError):
        RequirementReport((RequirementIssue("example.conflict", "conflict", sources=(source,) * 3277),))


def test_public_surface_is_exact_and_documented():
    """The package exposes only the settled shared API with usable documentation."""

    import dryml.requirements as requirements

    expected = {
        "AdmissionReport",
        "RequirementBarrierError",
        "RequirementCombinationError",
        "RequirementCombiner",
        "RequirementDeclaration",
        "RequirementError",
        "RequirementIssue",
        "RequirementReport",
        "RequirementResult",
        "RequirementSource",
        "combine_requirements",
        "require_admission",
    }
    assert set(requirements.__all__) == expected
    for name in expected:
        assert inspect.getdoc(getattr(requirements, name))
