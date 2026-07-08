import sys
from collections.abc import Mapping

import dryml.environments as envs
import dryml.runtime as runtime


def test_current_environment_probe_returns_structured_result():
    result = envs.probe()

    assert isinstance(result, envs.EnvironmentProbeResult)
    assert result.ok is True
    assert result.record is not None
    assert result.to_data()["ok"] is True


def test_environment_record_contains_python_version_and_executable():
    record = envs.probe_current().require_ok()

    assert record.python.version
    assert record.python.executable == sys.executable


def test_environment_record_contains_platform_and_distribution_data():
    record = envs.probe_current().require_ok()

    assert record.platform.platform
    assert isinstance(record.distributions, Mapping)


def test_probe_does_not_require_world_allocation():
    before = runtime.active_runtime()
    result = envs.probe_current()
    after = runtime.active_runtime()

    assert result.ok
    assert before == after
    assert after.mode is runtime.RuntimeMode.ORCHESTRATOR
    assert after.allocation is runtime.NoAllocation


def test_bad_python_executable_probe_returns_structured_failure():
    result = envs.probe_python("/definitely/not/a/python/executable", timeout=1)

    assert result.ok is False
    assert result.record is None
    assert result.report is not None
    assert result.report.issues[0].code == "probe_failed"
