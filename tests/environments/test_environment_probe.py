import json
import os
import shlex
import stat
import sys
from types import SimpleNamespace

import pytest

import dryml.environments as envs
from dryml.environments.probe_worker import main as probe_worker_main


def make_fake_executable(tmp_path, payload, *, sleep=False):
    script = tmp_path / "fake-python"
    text = "#!/bin/sh\n"
    if sleep:
        text += "sleep 2\n"
    text += f"printf '%s\\n' {shlex.quote(json.dumps(payload))}\n"
    script.write_text(text)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def sample_payload():
    record = envs.EnvironmentRecord(
        python=envs.PythonRecord("3.11.8", "CPython"),
        platform=envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
    )
    return {
        "kind": "dryml.environment_probe_result",
        "schema_version": envs.ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION,
        "ok": True,
        "record": record.to_data(),
    }


def test_probe_current_spec():
    result = envs.probe(envs.CurrentEnvironmentSpec())
    assert result.ok
    assert result.record.python.version


def test_probe_python_current_executable():
    result = envs.probe_python(sys.executable, timeout=30)
    assert result.ok
    assert result.record.python.version


def test_probe_python_fake_executable_success(tmp_path):
    exe = make_fake_executable(tmp_path, sample_payload())
    result = envs.probe(envs.PythonExecutableSpec(exe))
    assert result.ok
    assert result.stdout
    assert result.require_ok().python.version == "3.11.8"
    assert envs.EnvironmentProbeResult.from_data(result.to_data()).to_data() == result.to_data()


def test_probe_python_missing_executable():
    result = envs.probe_python("/definitely/missing/python")
    assert not result.ok
    assert result.report.issues[0].code == "probe_failed"
    with pytest.raises(envs.EnvironmentProbeError):
        result.require_ok()


def test_probe_python_timeout_and_malformed_output(tmp_path):
    timeout_exe = make_fake_executable(tmp_path, sample_payload(), sleep=True)
    timeout = envs.probe(envs.PythonExecutableSpec(timeout_exe), timeout=0.05)
    assert timeout.report.issues[0].code == "probe_timeout"

    malformed = tmp_path / "malformed-python"
    malformed.write_text("#!/bin/sh\nprintf '%s\\n' 'not json'\n")
    malformed.chmod(malformed.stat().st_mode | stat.S_IXUSR)
    result = envs.probe(envs.PythonExecutableSpec(str(malformed)))
    assert result.report.issues[0].code == "probe_failed"


def test_probe_worker_json_schema(capsys):
    assert probe_worker_main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "dryml.environment_probe_result"
    assert payload["schema_version"] == envs.ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION
    assert payload["ok"] is True
    assert "record" in payload


def test_probe_worker_returns_nonzero_on_internal_failure(monkeypatch, capsys):
    from dryml.environments import probe_worker

    def fail():
        raise RuntimeError("forced failure")

    monkeypatch.setattr(probe_worker, "inspect_current", fail)
    assert probe_worker.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "forced failure" in payload["error"]


def test_probe_container_unsupported_and_conda_absent_clear_failure():
    container = envs.probe(envs.ContainerEnvironmentSpec("example/image"))
    assert container.report.issues[0].code == "unsupported_environment_spec"
    conda = envs.probe(envs.CondaEnvironmentSpec(prefix="/missing", conda_executable="/missing/conda", launch_mode="conda-run"))
    assert conda.report.issues[0].code == "probe_failed"


def test_probe_pythonpath_policy_none_removes_parent_pythonpath(monkeypatch):
    import importlib

    probe_module = importlib.import_module("dryml.environments.probe")
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=json.dumps(sample_payload()), stderr="")

    monkeypatch.setenv("PYTHONPATH", "/parent")
    monkeypatch.setattr(probe_module.subprocess, "run", fake_run)

    result = envs.probe(envs.PythonExecutableSpec("python", env={"PYTHONPATH": "/override"}))

    assert result.ok
    assert "PYTHONPATH" not in captured["env"]


def test_probe_pythonpath_policy_explicit_uses_only_extra_paths(monkeypatch):
    import importlib

    probe_module = importlib.import_module("dryml.environments.probe")
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=json.dumps(sample_payload()), stderr="")

    monkeypatch.setenv("PYTHONPATH", "/parent")
    monkeypatch.setattr(probe_module.subprocess, "run", fake_run)

    result = envs.probe(
        envs.PythonExecutableSpec(
            "python",
            env={"PYTHONPATH": "/override"},
            pythonpath_policy="explicit",
            extra_pythonpath=("/explicit-a", "/explicit-b"),
        )
    )

    assert result.ok
    assert captured["env"]["PYTHONPATH"] == os.pathsep.join(("/explicit-a", "/explicit-b"))


def test_probe_pythonpath_policy_inherit_preserves_parent_pythonpath(monkeypatch):
    env = envs.build_probe_env(
        base={"PYTHONPATH": "/parent"},
        overrides={"PYTHONPATH": "/override"},
        pythonpath_policy="inherit",
        extra_pythonpath=("/extra",),
    )

    assert env["PYTHONPATH"] == os.pathsep.join(("/parent", "/extra"))


def test_conda_probe_uses_pythonpath_policy(monkeypatch):
    import importlib

    probe_module = importlib.import_module("dryml.environments.probe")
    captured = {}

    def fake_run(command, **kwargs):
        captured.update({"command": command, **kwargs})
        return SimpleNamespace(returncode=0, stdout=json.dumps(sample_payload()), stderr="")

    monkeypatch.setenv("PYTHONPATH", "/parent")
    monkeypatch.setattr(probe_module.subprocess, "run", fake_run)

    result = envs.probe(
        envs.CondaEnvironmentSpec(
            prefix="/conda/env",
            pythonpath_policy="explicit",
            extra_pythonpath=("/only",),
        )
    )

    assert result.ok
    assert captured["command"][0] == "/conda/env/bin/python"
    assert captured["env"]["PYTHONPATH"] == "/only"
