"""Clean-distribution contract for the unsupported graph prototype package."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def _run(command, *, cwd: Path) -> None:
    """Run one deterministic packaging command or expose its bounded failure."""

    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, (
        f"packaging command failed: {command!r}\n"
        f"stdout:\n{completed.stdout[-4_096:]}\n"
        f"stderr:\n{completed.stderr[-4_096:]}"
    )


def _assert_no_graph_members(archive: Path) -> None:
    """Assert that a wheel or tar archive contains no graph prototype package."""

    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as contents:
            members = contents.namelist()
    else:
        with tarfile.open(archive) as contents:
            members = contents.getnames()
    assert not any("/dryml/graph/" in member or member.startswith("dryml/graph/") for member in members)


def _extract_trusted_archive(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract a locally generated archive across supported Python versions."""

    if sys.version_info >= (3, 12):
        archive.extractall(destination, filter="data")
    else:  # pragma: no cover - exercised on the supported Python 3.10/3.11 CI.
        archive.extractall(destination)


def _assert_isolated_import_contract(wheel: Path, *, work_dir: Path) -> None:
    """Install *wheel* into an isolated target and verify supported imports."""

    work_dir.mkdir()
    target = work_dir / "installed"
    _run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
        cwd=work_dir,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util\n"
            "import dryml\n"
            "import dryml.code\n"
            "assert importlib.util.find_spec('dryml.graph') is None\n"
            "assert not hasattr(dryml, 'graph')\n"
            "assert not hasattr(dryml.code, 'graph')\n"
            "try:\n"
            "    import dryml.graph\n"
            "except ModuleNotFoundError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('dryml.graph unexpectedly imports')\n",
        ],
        cwd=work_dir,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, (
        f"isolated import contract failed\nstdout:\n{completed.stdout[-4_096:]}\n"
        f"stderr:\n{completed.stderr[-4_096:]}"
    )


def test_clean_tracked_distribution_excludes_graph_prototype(tmp_path):
    """Build wheel/sdist from tracked content without touching local prototypes."""

    repository = Path(__file__).resolve().parents[2]
    archive_tar = tmp_path / "tracked-source.tar"
    with archive_tar.open("wb") as output:
        subprocess.run(
            ["git", "archive", "--format=tar", "HEAD"],
            cwd=repository,
            stdout=output,
            check=True,
        )
    source_tree = tmp_path / "source"
    source_tree.mkdir()
    with tarfile.open(archive_tar) as archive:
        _extract_trusted_archive(archive, source_tree)

    direct_dist = tmp_path / "direct-dist"
    direct_dist.mkdir()
    _run(
        [
            sys.executable,
            "setup.py",
            "sdist",
            "--dist-dir",
            str(direct_dist),
            "bdist_wheel",
            "--dist-dir",
            str(direct_dist),
        ],
        cwd=source_tree,
    )
    direct_wheel = next(direct_dist.glob("*.whl"))
    direct_sdist = next(direct_dist.glob("*.tar.gz"))
    _assert_no_graph_members(direct_wheel)
    _assert_no_graph_members(direct_sdist)
    _assert_isolated_import_contract(direct_wheel, work_dir=tmp_path / "wheel-import")

    sdist_tree = tmp_path / "sdist-source"
    sdist_tree.mkdir()
    with tarfile.open(direct_sdist) as archive:
        _extract_trusted_archive(archive, sdist_tree)
    unpacked = next(path for path in sdist_tree.iterdir() if path.is_dir())
    sdist_dist = tmp_path / "sdist-dist"
    sdist_dist.mkdir()
    _run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(sdist_dist)], cwd=unpacked)
    sdist_wheel = next(sdist_dist.glob("*.whl"))
    _assert_no_graph_members(sdist_wheel)
    _assert_isolated_import_contract(sdist_wheel, work_dir=tmp_path / "sdist-import")

    # The test's tracked-source archive must be the only input. In particular,
    # no cleanup or inspection of a local untracked src/dryml/graph is needed.
    assert not (source_tree / "src" / "dryml" / "graph").exists()
