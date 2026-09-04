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


def test_diagnostics_redact_nested_values_and_neutralize_controls():
    """Every retained diagnostic surface uses the same bounded sanitizer."""

    error = RequirementError(
        'Authorization: Bearer alpha beta at /home/user/My Secret/file.txt\x1b[2J',
        context={
            "nested": {
                "token": '"alpha beta"',
                "file": "file:///home/user/My Secret/file.txt",
                "path": "/home/user/My Secret/file.txt",
                "control": "safe\r\n\t\x1b[2J",
                "unknown": object(),
            },
        },
    )

    rendered = str(error)
    nested = error.context["nested"]
    assert "alpha beta" not in rendered
    assert "/home/user/My Secret/file.txt" not in rendered
    assert all(ord(char) >= 32 and ord(char) != 127 for char in rendered)
    assert nested["token"] == "<redacted>"
    assert nested["file"] == "file://<redacted>"
    assert nested["path"] == "<local-path>"
    assert nested["control"] == "safe????[2J"
    assert nested["unknown"] == "<unsupported>"


@pytest.mark.parametrize(
    "message,path",
    (
        ("x" * 513, None),
        ("message\n", None),
        ("message", "x" * 513),
        ("message", "path\x7f"),
    ),
)
def test_requirement_issue_rejects_malformed_explicit_text(message, path):
    """Direct issue construction cannot bypass text protocol boundaries."""

    with pytest.raises(RequirementError, match="invalid requirement issue text"):
        RequirementIssue("example.conflict", message, path=path)


def test_invalid_issue_text_uses_a_fixed_unchained_error():
    """Rejected caller text cannot escape through the error chain."""

    with pytest.raises(RequirementError) as raised:
        RequirementIssue("example.conflict", "token=secret\n")

    assert str(raised.value) == "invalid requirement issue text"
    assert raised.value.__cause__ is None


def test_requirement_issue_sanitizes_sensitive_text_with_exact_bound():
    """Issues retain the exact accepted bound while redacting sensitive text."""

    issue = RequirementIssue(
        "example.conflict",
        'token="alpha beta" at /home/user/My Secret/file.txt',
        path="file:///home/user/My Secret/file.txt",
    )

    assert issue.message == "token=<redacted> at <local-path>"
    assert issue.path == "file://<redacted>"
    assert RequirementIssue("example.conflict", "x" * 512).message == "x" * 512


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
