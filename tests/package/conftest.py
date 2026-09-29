"""Fixtures for installed DRYML release-artifact verification."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import uuid
import venv

import pytest

ROOT = Path(__file__).resolve().parents[2]
_OFFLINE_PIP_ENV = {
    "PIP_NO_INDEX": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
}


def _run_package_subprocess(args: list[str], *, cwd: Path) -> None:
    """Run one package build or installation command with network access disabled."""

    subprocess.run(
        args,
        cwd=cwd,
        check=True,
        env={**os.environ, **_OFFLINE_PIP_ENV, "PYTHONPATH": ""},
    )


@pytest.fixture(scope="session")
def release_artifacts() -> tuple[Path, Path]:
    """Build and return one sdist and wheel beneath the workspace temp root."""

    output = Path("/tmp/dryml/package-tests") / uuid.uuid4().hex
    output.mkdir(parents=True)
    _run_package_subprocess(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output)],
        cwd=ROOT,
    )
    sdists = tuple(
        path for path in output.iterdir() if path.is_file() and tarfile.is_tarfile(path)
    )
    wheels = tuple(output.glob("*.whl"))
    assert len(sdists) == len(wheels) == 1
    return sdists[0], wheels[0]


@pytest.fixture(scope="session")
def installed_python(release_artifacts: tuple[Path, Path]) -> Path:
    """Build a wheel from the sdist, install it, and return its interpreter."""

    sdist, _ = release_artifacts
    root = Path("/tmp/dryml/package-installs") / uuid.uuid4().hex
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(root)
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    source = root / "source"
    with tarfile.open(sdist, mode="r:*") as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(source, filter="data")
        else:
            archive.extractall(source)
    project = next(source.iterdir())
    wheel_dir = root / "wheel"
    wheel_dir.mkdir()
    _run_package_subprocess(
        [str(python), "-m", "build", "--no-isolation", "--wheel", "--outdir", str(wheel_dir)],
        cwd=project,
    )
    (wheel,) = wheel_dir.glob("*.whl")
    _run_package_subprocess(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        cwd=root,
    )
    yield python
    shutil.rmtree(root, ignore_errors=True)
