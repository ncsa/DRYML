"""Focused failure conformance for external opt-ins and one-shot worker loss."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dryml.execute.errors import ExecutionUncertainError
from dryml.execute.executor import Executor
from dryml.execute.subprocess import SubProcessConfig


def _exit_after_marker(marker: str) -> None:
    """Record exactly one invocation and end the worker before it can reply."""
    from pathlib import Path
    import os

    path = Path(marker)
    count = int(path.read_text(encoding="ascii")) if path.exists() else 0
    path.write_text(str(count + 1), encoding="ascii")
    os._exit(7)


def test_worker_loss_after_go_is_uncertain_and_never_replays_the_workload(tmp_path: Path):
    """A real worker loss after payload transfer preserves one invocation and uncertainty."""
    spool = tmp_path / "spool"
    spool.mkdir()
    marker = tmp_path / "invocations"
    executor = Executor(SubProcessConfig(spool_directory=spool))
    try:
        future = executor.submit(_exit_after_marker, str(marker))
        with pytest.raises(ExecutionUncertainError):
            future.result(timeout=10)
        assert marker.read_text(encoding="ascii") == "1"
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_missing_enabled_environment_opt_in_fails_during_isolated_collection_without_network():
    """The test-category gate rejects missing targets before any Ray connection can start."""
    environment = dict(os.environ)
    environment["DRYML_EXECUTE_INTEGRATION"] = "1"
    environment.pop("DRYML_TEST_CONDA_PREFIX", None)
    environment.pop("DRYML_TEST_VENV_PYTHON", None)
    environment.pop("DRYML_TEST_RAY_ADDRESS", None)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "tests/execute/test_existing_environments.py"],
        cwd=Path(__file__).parents[2],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "DRYML_TEST_CONDA_PREFIX is required" in completed.stderr


def _valid_external_target_environment(tmp_path: Path) -> dict[str, str]:
    """Build collection-only target paths without provisioning a runtime or contacting Ray."""
    prefix = tmp_path / "conda-prefix"
    executable = prefix / ("python.exe" if os.name == "nt" else "bin/python")
    executable.parent.mkdir(parents=True)
    executable.touch()
    venv = tmp_path / "venv-python"
    venv.touch()
    environment = dict(os.environ)
    environment.update(
        DRYML_EXECUTE_INTEGRATION="1",
        DRYML_TEST_CONDA_PREFIX=str(prefix),
        DRYML_TEST_VENV_PYTHON=str(venv),
        CONDA_EXE=sys.executable,
    )
    return environment


@pytest.mark.parametrize(
    ("address", "ray_source", "expected"),
    [
        ("not-an-endpoint", None, "DRYML_TEST_RAY_ADDRESS must be an existing host:port endpoint"),
        ("127.0.0.1:1", "__version__ = '0.0.0'\n", "requires Ray 2.56.0, found 0.0.0"),
        ("127.0.0.1:1", "raise ImportError('missing test Ray')\n", "Ray 2.56.0 is required"),
    ],
)
def test_invalid_enabled_ray_opt_ins_fail_during_isolated_collection_without_connecting(
    tmp_path: Path, address: str, ray_source: str | None, expected: str,
):
    """Malformed endpoints and unavailable/wrong SDKs fail before a native connection attempt."""
    environment = _valid_external_target_environment(tmp_path)
    environment["DRYML_TEST_RAY_ADDRESS"] = address
    if ray_source is not None:
        fake_modules = tmp_path / "fake-modules"
        fake_modules.mkdir()
        (fake_modules / "ray.py").write_text(ray_source, encoding="ascii")
        environment["PYTHONPATH"] = str(fake_modules) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "tests/ray/test_execute_ray_backend.py"],
        cwd=Path(__file__).parents[2],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert expected in completed.stderr
