"""Explicit opt-in real existing interpreter proof for the subprocess backend."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dryml.execute.executor import Executor
from dryml.execute.subprocess import SubProcessConfig
from dryml.environments import EnvironmentRequirement
from dryml.environments.specs import CondaEnvironmentSpec


def _runtime_details() -> tuple[str, str, int, str | None, str]:
    """Return worker runtime identity without relying on coordinator imports."""
    from importlib.metadata import version
    import os
    import sys

    return sys.prefix, sys.executable, os.getpid(), os.environ.get("PYTHONPATH"), version("dryml")


def _enabled_path(name: str) -> Path:
    """Require an explicit existing integration target when integration is enabled."""
    if os.environ.get("DRYML_EXECUTE_INTEGRATION") != "1":
        pytest.skip("set DRYML_EXECUTE_INTEGRATION=1 to run existing-environment proof")
    value = os.environ.get(name)
    assert value, f"{name} is required when DRYML_EXECUTE_INTEGRATION=1"
    path = Path(value)
    if name == "DRYML_TEST_CONDA_PREFIX":
        assert path.is_dir(), f"{name} must identify an existing Conda prefix"
    else:
        assert path.is_file(), f"{name} must identify an existing Python executable"
    return path


@pytest.mark.parametrize("variable", ["DRYML_TEST_CONDA_PREFIX", "DRYML_TEST_VENV_PYTHON"])
def test_existing_supplied_interpreters_execute_real_workloads(tmp_path: Path, variable: str):
    """No-provisioning integration runs a real callable in each supplied target."""
    target = _enabled_path(variable)
    if variable == "DRYML_TEST_CONDA_PREFIX":
        prefix = target
        executable = prefix / ("python.exe" if os.name == "nt" else "bin/python")
        assert executable.is_file(), "DRYML_TEST_CONDA_PREFIX must contain an existing Python"
    else:
        executable = target
        prefix = executable.parent.parent
    def runtime_identity() -> tuple[str, str]:
        """Return worker identity without importing this pytest module."""
        import sys

        return sys.prefix, sys.executable

    executor = Executor(SubProcessConfig(spool_directory=tmp_path, python_executable=executable))
    try:
        observed_prefix, observed_executable = executor.run(runtime_identity)
        assert observed_prefix == str(prefix)
        assert observed_executable == str(executable)
    finally:
        executor.close(cancel=True, timeout=10)


def test_existing_conda_run_selector_uses_wrapper_and_installed_runtime(tmp_path: Path):
    """An explicit Conda selector proves wrapper PID and worker runtime association."""
    prefix = _enabled_path("DRYML_TEST_CONDA_PREFIX")
    conda_value = os.environ.get("CONDA_EXE")
    assert conda_value, "activate the supplied Conda installation so CONDA_EXE supplies an absolute launcher"
    conda = Path(conda_value)
    assert conda.is_absolute() and conda.is_file(), "CONDA_EXE must identify an existing absolute Conda executable"
    spec = CondaEnvironmentSpec(prefix=str(prefix), conda_executable=str(conda), launch_mode="conda-run")
    executor = Executor(
        SubProcessConfig(
            spool_directory=tmp_path,
            automatic_environment_discovery=False,
            environment_candidates=(spec,),
        )
    )
    try:
        future = executor.submit(_runtime_details, environment=EnvironmentRequirement())
        worker_prefix, worker_executable, worker_pid, pythonpath, dryml_version = future.result(timeout=30)
        expected = prefix / ("python.exe" if os.name == "nt" else "bin/python")
        assert worker_prefix == str(prefix)
        assert worker_executable == str(expected)
        assert future.process is not None
        assert future.process.pid != worker_pid
        assert future.pid == worker_pid
        assert pythonpath is None
        assert dryml_version
        future.cleanup(timeout=10)
    finally:
        executor.close(cancel=True, timeout=10)
