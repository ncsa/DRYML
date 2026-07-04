import sys
from pathlib import Path

from dryml.dispatch.backends import build_worker_command
from dryml.environments import PythonExecutableSpec


def test_pythonpath_policy_none_inherit_explicit_and_dryml_source(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "parent")

    _, none_env = build_worker_command(PythonExecutableSpec(sys.executable, pythonpath_policy="none").to_data())
    _, inherit_env = build_worker_command(PythonExecutableSpec(sys.executable, pythonpath_policy="inherit").to_data())
    _, explicit_env = build_worker_command(PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(tmp_path),)).to_data())
    _, source_env = build_worker_command(PythonExecutableSpec(sys.executable, pythonpath_policy="dryml-source").to_data())

    assert none_env["PYTHONPATH"] == "parent"
    assert inherit_env["PYTHONPATH"] == "parent"
    assert explicit_env["PYTHONPATH"] == str(tmp_path)
    source_root = Path(source_env["PYTHONPATH"])
    assert (source_root / "dryml").is_dir()
