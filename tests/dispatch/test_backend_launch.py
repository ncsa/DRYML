import os
import sys
from pathlib import Path

from dryml.dispatch.backends import build_worker_command
from dryml.environments import CondaEnvironmentSpec, CurrentEnvironmentSpec, PythonExecutableSpec


def test_current_environment_command_uses_parent_python():
    cmd, env = build_worker_command(CurrentEnvironmentSpec().to_data())

    assert cmd == [sys.executable]
    assert "PYTHONPATH" in env


def test_python_executable_env_and_pythonpath_policies(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "parent-path")
    spec = PythonExecutableSpec(sys.executable, env={"DRYML_TEST_ENV": "1"}, pythonpath_policy="explicit", extra_pythonpath=(str(tmp_path),))
    cmd, env = build_worker_command(spec.to_data())

    assert cmd == [sys.executable]
    assert env["DRYML_TEST_ENV"] == "1"
    assert env["PYTHONPATH"] == str(tmp_path)

    none_spec = PythonExecutableSpec(sys.executable, pythonpath_policy="none")
    _, none_env = build_worker_command(none_spec.to_data())
    assert none_env.get("PYTHONPATH") == "parent-path"


def test_conda_command_construction():
    spec = CondaEnvironmentSpec(name="dryml-test", launch_mode="conda-run", conda_executable="conda")
    cmd, _ = build_worker_command(spec.to_data())

    assert cmd[:4] == ["conda", "run", "-n", "dryml-test"]


def test_dryml_source_prefers_module_checkout_over_cwd(tmp_path, monkeypatch):
    unrelated = tmp_path / "src" / "dryml"
    unrelated.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    _, env = build_worker_command(PythonExecutableSpec(sys.executable, pythonpath_policy="dryml-source").to_data())
    source_root = Path(env["PYTHONPATH"])

    assert (source_root / "dryml").is_dir()
    assert source_root != tmp_path / "src"
