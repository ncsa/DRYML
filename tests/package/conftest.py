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


@pytest.fixture(scope="session")
def release_artifacts() -> tuple[Path, Path]:
    """Build and return one sdist and wheel beneath the workspace temp root."""

    output = Path("/tmp/dryml/package-tests") / uuid.uuid4().hex
    output.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output)],
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONPATH": ""},
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
    subprocess.run(
        [str(python), "-m", "build", "--wheel", "--outdir", str(wheel_dir)],
        cwd=project,
        check=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    (wheel,) = wheel_dir.glob("*.whl")
    subprocess.run(
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
        check=True,
    )
    yield python
    shutil.rmtree(root, ignore_errors=True)
