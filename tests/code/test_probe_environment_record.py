from __future__ import annotations

import sys
from pathlib import Path

import dryml.code as code
from dryml.environments import CondaEnvironmentSpec, ContainerEnvironmentSpec, CurrentEnvironmentSpec, EnvironmentRecord, PythonExecutableSpec


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


def test_conda_spec_routes_import_path_to_worker(monkeypatch):
    import dryml.code.probe as probe

    captured = {}

    def fake_worker(request, command, *, timeout, env):
        captured.update(request=request, command=command, timeout=timeout, env=env)
        return code.CodeProbeResult(
            ok=True,
            analysis=code.CodeAnalysisResult(target=request.target),
            environment_record=None,
        )

    monkeypatch.setattr(probe, "probe_target_in_subprocess", fake_worker)
    environment = CondaEnvironmentSpec(prefix="/opt/test-conda", pythonpath_policy="none")
    result = code.probe_target(TARGET, environment=environment, include_environment_record=False, timeout=10)

    assert result.ok
    assert captured["command"] == ["/opt/test-conda/bin/python", "-m", "dryml.code.probe_worker", "--json"]
    assert captured["timeout"] == 10


def test_empty_source_spec_and_malformed_import_path_are_not_worker_eligible():
    empty_source = code.CodeTargetSpec("source_spec", source_spec={})
    malformed_path = code.CodeTargetSpec.from_import_path("malformed-path")
    executable = PythonExecutableSpec(executable=sys.executable, pythonpath_policy="dryml-source")

    source_result = code.probe_target(empty_source, environment=executable, include_environment_record=False)
    malformed_result = code.probe_target(malformed_path, environment=executable, include_environment_record=False)

    assert source_result.diagnostics[0].code == "code_probe.source_spec_reconstruction_unavailable"
    assert malformed_result.diagnostics[0].code == "code_probe.non_serializable_target"


def test_source_spec_with_unusable_import_path_reports_reconstruction_limit():
    executable = PythonExecutableSpec(executable=sys.executable, pythonpath_policy="dryml-source")
    target = code.CodeTargetSpec(
        "source_spec",
        import_path="__main__:target",
        source_spec={"kind": "function", "source": "lambda: None"},
    )

    result = code.probe_target(target, environment=executable, include_environment_record=False)

    assert result.diagnostics[0].code == "code_probe.source_spec_reconstruction_unavailable"


def test_subprocess_probe_rejects_non_serializable_local_function():
    def local_target():
        return None

    spec = PythonExecutableSpec(executable=sys.executable, pythonpath_policy="dryml-source")
    result = code.probe_target(local_target, environment=spec, include_environment_record=False)

    assert not result.ok
    assert "code_probe.non_serializable_target" in {item.code for item in result.diagnostics}


def test_subprocess_probe_rejects_source_spec_without_reconstruction():
    spec = PythonExecutableSpec(executable=sys.executable, pythonpath_policy="dryml-source")
    target = code.CodeTargetSpec("source_spec", source_spec={"kind": "function", "source": "lambda x: x"})

    result = code.probe_target(target, environment=spec, include_environment_record=False)

    assert not result.ok
    assert result.diagnostics[0].code == "code_probe.source_spec_reconstruction_unavailable"


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
