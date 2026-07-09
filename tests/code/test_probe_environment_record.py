from __future__ import annotations

import sys
from pathlib import Path

import dryml.code as code
from dryml.environments import ContainerEnvironmentSpec, CurrentEnvironmentSpec, EnvironmentRecord, PythonExecutableSpec


TARGET = "probe_targets:plain_function"
FIXTURE_DIR = str(Path(__file__).parents[1] / "fixtures")


def test_include_environment_record_true_returns_record():
    result = code.probe_target(TARGET, environment=CurrentEnvironmentSpec(), include_environment_record=True)

    assert result.ok
    assert isinstance(result.environment_record, EnvironmentRecord)


def test_include_environment_record_false_omits_record():
    result = code.probe_target(TARGET, include_environment_record=False)

    assert result.ok
    assert result.environment_record is None


def test_environment_record_serializes_and_round_trips():
    result = code.probe_target(TARGET, include_environment_record=True)
    restored = code.CodeProbeResult.from_data(result.to_data())

    assert restored.ok
    assert restored.environment_record is not None
    assert restored.environment_record.python.executable == result.environment_record.python.executable


def test_environment_record_failure_produces_diagnostic(monkeypatch):
    import dryml.code.probe as probe

    def fail():
        raise RuntimeError("environment failure")

    monkeypatch.setattr(probe.environments, "inspect_current", fail)
    result = code.probe_target(TARGET, include_environment_record=True)

    assert not result.ok
    assert "code_probe.environment_record_error" in {item.code for item in result.diagnostics}


def test_python_executable_spec_probe_path():
    spec = PythonExecutableSpec(
        executable=sys.executable,
        pythonpath_policy="inherit",
        extra_pythonpath=(FIXTURE_DIR,),
    )
    result = code.probe_target(TARGET, environment=spec, include_environment_record=False, timeout=10)

    assert result.ok
    assert result.analysis is not None
    assert result.analysis.facts_of_kind("callable")


def test_subprocess_probe_rejects_non_serializable_local_function():
    def local_target():
        return None

    spec = PythonExecutableSpec(executable=sys.executable, pythonpath_policy="dryml-source")
    result = code.probe_target(local_target, environment=spec, include_environment_record=False)

    assert not result.ok
    assert "code_probe.non_serializable_target" in {item.code for item in result.diagnostics}


def test_current_import_path_timeout_routes_through_subprocess(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", FIXTURE_DIR)
    result = code.probe_target(
        "probe_slow_import:slow_target",
        include_environment_record=False,
        timeout=0.2,
    )

    assert not result.ok
    assert result.diagnostics[0].code == "code_probe.timeout"


def test_unsupported_environment_spec_returns_diagnostic():
    result = code.probe_target(TARGET, environment=ContainerEnvironmentSpec(image="example:latest"))
    assert not result.ok
    assert result.diagnostics[0].code == "code_probe.unsupported_environment"


def test_subprocess_timeout_returns_diagnostic():
    spec = PythonExecutableSpec(
        executable=sys.executable,
        pythonpath_policy="inherit",
        extra_pythonpath=(FIXTURE_DIR,),
    )
    result = code.probe_target(
        "probe_slow_import:slow_target",
        environment=spec,
        include_environment_record=False,
        timeout=0.2,
    )

    assert not result.ok
    assert result.diagnostics[0].code == "code_probe.timeout"
