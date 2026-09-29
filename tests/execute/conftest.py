"""Execute integration opt-in validation shared by external conformance cases."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


_INTEGRATION = "DRYML_EXECUTE_INTEGRATION"
_CONDA = "DRYML_TEST_CONDA_PREFIX"
_VENV = "DRYML_TEST_VENV_PYTHON"


def integration_enabled() -> bool:
    """Return whether explicit existing-environment integration was requested."""
    return os.environ.get(_INTEGRATION) == "1"


def require_integration() -> None:
    """Skip an external case unless the caller explicitly enabled integration."""
    if not integration_enabled():
        pytest.skip(f"set {_INTEGRATION}=1 to run existing-environment integration")


def _required_path(name: str, *, directory: bool) -> Path:
    """Return one explicitly supplied existing target without resolving symlinks."""
    value = os.environ.get(name)
    if not value:
        raise pytest.UsageError(f"{name} is required when {_INTEGRATION}=1")
    path = Path(value)
    valid = path.is_dir() if directory else path.is_file()
    kind = "directory" if directory else "file"
    if not valid:
        raise pytest.UsageError(f"{name} must identify an existing {kind}")
    return path


def validate_existing_environment_opt_in() -> None:
    """Fail enabled collection when a supplied Conda or venv target is unusable."""
    if not integration_enabled():
        return
    prefix = _required_path(_CONDA, directory=True)
    executable = prefix / ("python.exe" if os.name == "nt" else "bin/python")
    if not executable.is_file():
        raise pytest.UsageError(f"{_CONDA} must contain an existing Python interpreter")
    _required_path(_VENV, directory=False)
    conda = os.environ.get("CONDA_EXE")
    if not conda or not Path(conda).is_absolute() or not Path(conda).is_file():
        raise pytest.UsageError("CONDA_EXE must identify the active Conda launcher when integration is enabled")


def require_ray_integration() -> str:
    """Return an enabled existing Ray endpoint after checking its pinned SDK identity."""
    require_integration()
    address = os.environ.get("DRYML_TEST_RAY_ADDRESS")
    if not address:
        raise pytest.UsageError("DRYML_TEST_RAY_ADDRESS is required when DRYML_EXECUTE_INTEGRATION=1")
    host, separator, port = address.rpartition(":")
    if not host or not separator or not port.isdecimal() or not 0 < int(port) <= 65535 or "://" in address or any(character.isspace() for character in address):
        raise pytest.UsageError("DRYML_TEST_RAY_ADDRESS must be an existing host:port endpoint")
    try:
        import ray
    except ImportError as exc:
        raise pytest.UsageError("Ray 2.56.0 is required for enabled existing-Ray integration") from exc
    if ray.__version__ != "2.56.0":
        raise pytest.UsageError(f"enabled existing-Ray integration requires Ray 2.56.0, found {ray.__version__}")
    return address


def pytest_configure(config: pytest.Config) -> None:
    """Validate explicitly enabled environment targets before external cases collect."""
    del config
    validate_existing_environment_opt_in()
