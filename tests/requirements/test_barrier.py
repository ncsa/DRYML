"""Tests for explicit fail-closed requirement admission."""

import pytest

from dryml.requirements import RequirementBarrierError, RequirementError, require_admission


class Report:
    """Structural report fixture with an independent admission decision."""

    def __init__(self, admission_ok):
        self.admission_ok = admission_ok


def test_admission_accepts_only_an_exact_true_boolean():
    """The barrier consumes the policy-independent structural property only."""

    assert require_admission(Report(True), operation="start") is None
    report = Report(False)
    with pytest.raises(RequirementBarrierError) as raised:
        require_admission(report, operation="run token=secret at /private/file")
    assert raised.value.report is report
    assert "secret" not in raised.value.operation
    assert "/private/file" not in raised.value.operation


@pytest.mark.parametrize("report", (object(), Report(1), Report("true")))
def test_admission_rejects_malformed_structural_reports(report):
    """Missing and non-boolean admissions never inherit an ``ok`` decision."""

    with pytest.raises(RequirementError, match="invalid admission report"):
        require_admission(report)


def test_admission_suppresses_property_failure_details():
    """A failing report property exposes only the fixed shared failure."""

    class RaisingReport:
        @property
        def admission_ok(self):
            raise RuntimeError("token=secret")

    with pytest.raises(RequirementError, match="invalid admission report") as raised:
        require_admission(RaisingReport())
    assert raised.value.__cause__ is None
    assert "secret" not in str(raised.value)
