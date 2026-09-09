"""Explicit existing-runtime integration proof for Ray environment options."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dryml.environments import EnvironmentRequirement
from dryml.environments.specs import CondaEnvironmentSpec, PythonExecutableSpec
from dryml.execute.executor import Executor
from dryml.execute.ray import RayBackendConfig
from .conftest import require_ray_integration


def _enabled(name: str) -> Path:
    """Require real caller-provided runtime targets when integration is enabled."""
    require_ray_integration()
    value = os.environ.get(name)
    assert value, f"{name} is required when integration is enabled"
    path = Path(value)
    assert path.exists(), f"{name} must identify an existing target"
    return path


def _address() -> str:
    """Require the explicit existing-server endpoint without filesystem probing."""
    return require_ray_integration()


@pytest.mark.parametrize("kind", ["conda", "venv"])
def test_existing_ray_runtime_forms_execute_actual_workloads(tmp_path: Path, kind: str):
    """Use each supplied existing runtime rather than only asserting options maps."""
    def runtime_identity() -> tuple[str, str]:
        """Return worker runtime identity without importing this test module."""
        import sys

        return sys.prefix, sys.executable

    address = _address()
    if kind == "conda":
        prefix = _enabled("DRYML_TEST_CONDA_PREFIX")
        spec = CondaEnvironmentSpec(prefix=str(prefix))
        expected = prefix / ("python.exe" if os.name == "nt" else "bin/python")
    else:
        expected = _enabled("DRYML_TEST_VENV_PYTHON")
        prefix = expected.parent.parent
        spec = PythonExecutableSpec(str(expected))
    assert expected.is_file(), "supplied runtime must contain an existing Python interpreter"
    executor = Executor(RayBackendConfig(address=address, spool_directory=tmp_path, admission_timeout=90, connect_timeout=60, automatic_environment_discovery=False, environment_candidates=(spec,)))
    try:
        observed_prefix, observed_executable = executor.run(runtime_identity, environment=EnvironmentRequirement())
        assert observed_prefix == str(prefix)
        assert observed_executable == str(expected)
    finally:
        executor.close(cancel=True, timeout=30)
