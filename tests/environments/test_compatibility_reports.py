import pytest

import dryml.environments as envs


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
