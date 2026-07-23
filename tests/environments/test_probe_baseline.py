import sys
from collections.abc import Mapping

import dryml.environments as envs
import dryml.runtime as runtime


def test_current_environment_probe_returns_structured_result():
    before = runtime.active_runtime()
    result = envs.probe()
    after = runtime.active_runtime()

    assert isinstance(result, envs.EnvironmentProbeResult)
    assert result.ok is True
    assert result.record is not None
    assert result.to_data()["ok"] is True
    record = result.require_ok()
    assert record.python.version
    assert record.python.executable == sys.executable
    assert record.platform.platform
    assert isinstance(record.distributions, Mapping)
    assert before == after
    assert after.mode is runtime.RuntimeMode.ORCHESTRATOR
    assert after.allocation is runtime.NoAllocation


def test_bad_python_executable_probe_returns_structured_failure():
    result = envs.probe_python("/definitely/not/a/python/executable", timeout=1)

    assert result.ok is False
    assert result.record is None
    assert result.report is not None
    assert result.report.issues[0].code == "probe_failed"
