import pytest

import dryml.environments as envs
from dryml.environments.compatibility import report_from_issues
from dryml.requirements import RequirementBarrierError, require_admission


def test_compatibility_report_roundtrip_explain_and_raise():
    issue = envs.CompatibilityIssue("package_missing", "error", "torch is missing")
    report = envs.CompatibilityReport("incompatible", (issue,))
    clone = envs.CompatibilityReport.from_data(report.to_data())
    assert clone.to_data() == report.to_data()
    assert not clone.ok
    assert "package_missing" in clone.explain()
    with pytest.raises(envs.EnvironmentCompatibilityError):
        clone.raise_if_incompatible()


def test_compatibility_report_ok_property():
    assert envs.CompatibilityReport("compatible").ok
    assert envs.CompatibilityReport("warning").is_compatible
    assert not envs.CompatibilityReport("unknown").ok
    assert not envs.CompatibilityReport("incompatible").ok


def test_policy_coercion():
    assert envs.coerce_policy("strict") == "strict"
    assert envs.coerce_policy("compat") == "compatible"
    with pytest.raises(envs.EnvironmentCompatibilityError):
        envs.coerce_policy("explode")


def test_admission_rejects_inconsistent_and_manual_reports() -> None:
    """Compatibility status and caller metadata cannot grant admission."""

    issue = envs.CompatibilityIssue("package_missing", "error", "torch is missing")
    inconsistent = envs.CompatibilityReport(
        "compatible", (issue,), details={"policy": "strict"}
    )
    manual = envs.CompatibilityReport(
        "compatible", details={"policy": "strict"}
    )

    assert not inconsistent.admission_ok
    assert not manual.admission_ok
    for report in (inconsistent, manual):
        with pytest.raises(RequirementBarrierError):
            require_admission(report)


def test_report_details_cannot_override_applied_policy_or_admit() -> None:
    """Diagnostic details cannot turn an ignored check into hard admission."""

    report = report_from_issues(
        (), policy="ignore", details={"policy": "strict"}
    )

    assert report.details["policy"] == "ignore"
    assert not report.admission_ok
    with pytest.raises(RequirementBarrierError):
        require_admission(report)


def test_deserialized_compatible_report_cannot_admit() -> None:
    """Serialized fields preserve diagnostics but not evaluation provenance."""

    report = envs.CompatibilityReport(
        "compatible", details={"policy": "strict"}
    )
    clone = envs.CompatibilityReport.from_data(report.to_data())

    assert clone.to_data() == report.to_data()
    assert not clone.admission_ok
