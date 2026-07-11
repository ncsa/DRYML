import json
import math
import os
import shlex
import stat
import sys
import time

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


def test_probe_decodes_a_bounded_large_protocol_payload(tmp_path):
    payload = sample_payload()
    payload["record"]["details"] = {"padding": "x" * (70 * 1024)}

    result = envs.probe(envs.PythonExecutableSpec(make_fake_executable(tmp_path, payload)))

    assert result.ok
    assert result.record is not None
    assert len(result.record.details["padding"]) == 70 * 1024
    assert len(result.stdout) == 64 * 1024


@pytest.mark.parametrize("timeout", (0, -1, math.nan, math.inf))
def test_probe_rejects_invalid_timeout_as_a_structured_failure(timeout):
    result = envs.probe(envs.CurrentEnvironmentSpec(), timeout=timeout)

    assert not result.ok
    assert result.report.issues[0].code == "invalid_probe_timeout"


def test_probe_spec_rejects_invalid_pythonpath_policy():
    with pytest.raises(envs.EnvironmentSpecError, match="unknown Python path probe policy"):
        envs.PythonExecutableSpec("python", pythonpath_policy="unsupported")


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


@pytest.mark.skipif(os.name != "posix", reason="process-group timeout behavior is POSIX-specific")
def test_probe_timeout_kills_descendants_holding_capture_pipes(tmp_path):
    script = tmp_path / "forking-python"
    script.write_text("#!/bin/sh\nsleep 2 &\nwait\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    started = time.monotonic()
    result = envs.probe(envs.PythonExecutableSpec(str(script)), timeout=0.05)

    assert not result.ok
    assert result.report.issues[0].code == "probe_timeout"
    assert time.monotonic() - started < 1.0


@pytest.mark.skipif(os.name != "posix", reason="inherited pipe behavior is POSIX-specific")
def test_probe_does_not_wait_for_descendant_that_escapes_capture_group(tmp_path):
    script = tmp_path / "detached-python"
    script.write_text("#!/bin/sh\nsetsid sleep 2 &\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    started = time.monotonic()
    result = envs.probe(envs.PythonExecutableSpec(str(script)), timeout=0.05)

    assert not result.ok
    assert result.report.issues[0].code == "probe_timeout"
    assert time.monotonic() - started < 1.0

    invalid_utf8 = tmp_path / "invalid-utf8-python"
    invalid_utf8.write_bytes(b"#!/bin/sh\nprintf '\\377'\n")
    invalid_utf8.chmod(invalid_utf8.stat().st_mode | stat.S_IXUSR)
    result = envs.probe(envs.PythonExecutableSpec(str(invalid_utf8)))
    assert not result.ok
    assert result.report.issues[0].code == "probe_failed"


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {"kind": "other", "schema_version": envs.ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION, "ok": True},
        {"kind": "dryml.environment_probe_result", "schema_version": 999, "ok": True},
        {"kind": "dryml.environment_probe_result", "schema_version": envs.ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION, "ok": "true"},
    ),
)
def test_probe_rejects_invalid_worker_protocol_payload(tmp_path, payload):
    exe = make_fake_executable(tmp_path, payload)

    result = envs.probe(envs.PythonExecutableSpec(exe))

    assert not result.ok
    assert result.report.issues[0].code == "probe_failed"


def test_probe_rejects_malformed_nested_record_data(tmp_path):
    payload = sample_payload()
    payload["record"]["distributions"] = []
    exe = make_fake_executable(tmp_path, payload)

    result = envs.probe(envs.PythonExecutableSpec(exe))

    assert not result.ok
    assert result.report.issues[0].code == "probe_failed"
    with pytest.raises(envs.EnvironmentProbeError, match="could not be decoded"):
        envs.EnvironmentProbeResult.from_data({
            "spec": envs.CurrentEnvironmentSpec().to_data(),
            "ok": True,
            "record": payload["record"],
        })


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

    def fake_run(command, *, timeout, env):
        captured.update({"command": command, "timeout": timeout, "env": env})
        return 0, json.dumps(sample_payload()).encode(), False, b"", False, False

    monkeypatch.setenv("PYTHONPATH", "/parent")
    monkeypatch.setattr(probe_module, "_run_bounded_command", fake_run)

    result = envs.probe(envs.PythonExecutableSpec("python", env={"PYTHONPATH": "/override"}))

    assert result.ok
    assert "PYTHONPATH" not in captured["env"]


def test_probe_pythonpath_policy_explicit_uses_only_extra_paths(monkeypatch):
    import importlib

    probe_module = importlib.import_module("dryml.environments.probe")
    captured = {}

    def fake_run(command, *, timeout, env):
        captured.update({"command": command, "timeout": timeout, "env": env})
        return 0, json.dumps(sample_payload()).encode(), False, b"", False, False

    monkeypatch.setenv("PYTHONPATH", "/parent")
    monkeypatch.setattr(probe_module, "_run_bounded_command", fake_run)

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

    def fake_run(command, *, timeout, env):
        captured.update({"command": command, "timeout": timeout, "env": env})
        return 0, json.dumps(sample_payload()).encode(), False, b"", False, False

    monkeypatch.setenv("PYTHONPATH", "/parent")
    monkeypatch.setattr(probe_module, "_run_bounded_command", fake_run)

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


def test_current_environment_probe_uses_bounded_worker_path(monkeypatch):
    import importlib

    probe_module = importlib.import_module("dryml.environments.probe")
    observed = []
    expected = envs.EnvironmentProbeResult(envs.CurrentEnvironmentSpec(), True, record=envs.inspect_current())
    monkeypatch.setattr(probe_module, "_probe_command", lambda spec, command, *, timeout, env=None: observed.append((spec, command, timeout)) or expected)

    result = envs.probe(envs.CurrentEnvironmentSpec(), timeout=0.25)

    assert result is expected
    assert observed[0][1][0] == sys.executable
    assert observed[0][2] == 0.25
