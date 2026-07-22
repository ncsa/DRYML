"""Offline PEP 517 release-artifact contracts."""

from __future__ import annotations

import email.parser
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 20_000
BUILD_TOOLS = ("build", "setuptools", "wheel")
RELEASE_CONTRACT_PATH = "tests/package/test_release_artifacts.py"
EXAMPLES = {
    "examples/requirements/requirements_and_explain.py",
    "examples/dispatch/python_shaped_dispatch.py",
    "examples/code_analysis/static_and_dynamic_analysis.py",
    "examples/notebooks/objects_definitions_and_repos.ipynb",
    "examples/notebooks/datasets_and_transforms.ipynb",
    "examples/notebooks/local_defaults_and_plain_mode.ipynb",
    "examples/notebooks/models_experiments_and_metrics.ipynb",
    "examples/notebooks/definition_driven_experiments.ipynb",
    "examples/notebooks/local_hyperparameter_search.ipynb",
}
EXTRAS = {
    "test": {("pytest", "", 'extra == "test"'), ("pytest-cov", "", 'extra == "test"'), ("flake8", "", 'extra == "test"'), ("pyarrow", "", 'python_version < "3.14" and extra == "test"')},
    "parquet": {("pyarrow", "", 'python_version < "3.14" and extra == "parquet"')},
    "tf": {("tensorflow", "", 'python_version < "3.14" and extra == "tf"')},
    "jax": {("jax", "<0.6,>=0.5", 'python_version < "3.14" and extra == "jax"'), ("jaxlib", "<0.6,>=0.5", 'python_version < "3.14" and extra == "jax"')},
    "torch": {("torch", ">=2.7.0", 'python_version < "3.14" and extra == "torch"')},
    "sklearn": {("scikit-learn", "", 'python_version < "3.14" and extra == "sklearn"')},
    "xgboost": {("scikit-learn", "", 'python_version < "3.14" and extra == "xgboost"'), ("xgboost", "", 'python_version < "3.14" and extra == "xgboost"')},
    "ray": {("ray", "", 'python_version < "3.14" and extra == "ray"')},
}
FORBIDDEN_PARTS = {
    ".cache", ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".svn", "__pycache__", "cache", "caches", "checkpoint", "checkpoints",
    "htmlcov", "pytest-temp", "ray_results", "tuning", "tuning-results",
    "tuning_results",
}
FORBIDDEN_ROOTS = (
    "build/", "dist/", "objects/", "products/", "records/", "stores/",
)
FORBIDDEN_SUFFIXES = {
    ".cache", ".checkpoint", ".ckpt", ".db", ".dill", ".dry", ".pickle",
    ".pkl", ".pyc", ".pyo", ".sqlite", ".sqlite3",
}
FORBIDDEN_NAMES = {
    ".coverage", "ci-metadata.json", "coverage.xml",
    "credentials", "credentials.json", "secrets.json", "timing.json",
    "timings.json",
}
REQUIRED_CANDIDATE_FILES = {"MANIFEST.in", RELEASE_CONTRACT_PATH}
RETIRED_CORE_PACKAGE = "core" + "2"


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, (
        f"command failed: {command!r}\n"
        f"stdout:\n{completed.stdout[-4096:]}\n"
        f"stderr:\n{completed.stderr[-4096:]}"
    )


def _offline_environment(network_guard: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not (
            key.upper().startswith(("PIP_", "UV_", "TWINE_"))
            or key.lower() in {  # noqa: W503
                "http_proxy", "https_proxy", "all_proxy", "no_proxy",
            }
        )
    }
    environment.update({
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(network_guard),
    })
    return environment


def _head_sha(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()


def _tracked_files(repository: Path, revision: str = "HEAD") -> set[str]:
    return set(subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision],
        cwd=repository,
        text=True,
    ).splitlines())


def _worktree_blob_oid(repository: Path, relative_path: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "hash-object",
            f"--path={relative_path}",
            str(repository / relative_path),
        ],
        cwd=repository,
        text=True,
    ).strip()


def _archive_candidate(repository: Path, destination: Path, archive_path: Path, environment: dict[str, str]) -> tuple[set[str], str]:
    candidate_sha = _head_sha(repository)
    tracked = _tracked_files(repository, candidate_sha)
    assert REQUIRED_CANDIDATE_FILES <= tracked, (
        "release manifest and artifact contract must be committed before the "
        f"candidate is built: {sorted(REQUIRED_CANDIDATE_FILES - tracked)}"
    )
    committed_contract_oid = subprocess.check_output(
        ["git", "rev-parse", f"{candidate_sha}:{RELEASE_CONTRACT_PATH}"],
        cwd=repository,
        text=True,
    ).strip()
    assert _worktree_blob_oid(repository, RELEASE_CONTRACT_PATH) == committed_contract_oid, (
        "the executing release-artifact contract must match committed HEAD"
    )
    _run(
        ["git", "archive", "--format=tar", f"--output={archive_path}", candidate_sha],
        cwd=repository,
        environment=environment,
    )
    _extract_tar(archive_path, destination)
    return tracked, candidate_sha


def _network_guard(root: Path) -> Path:
    guard = root / "network-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        """import socket

class _OfflineSocket(socket.socket):
    def connect(self, *args, **kwargs):
        raise RuntimeError("network access is disabled for release artifact tests")

    def connect_ex(self, *args, **kwargs):
        raise RuntimeError("network access is disabled for release artifact tests")

def _offline(*args, **kwargs):
    raise RuntimeError("network access is disabled for release artifact tests")

socket.socket = _OfflineSocket
socket.create_connection = _offline
""",
        encoding="utf-8",
    )
    return guard


def _extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path) as archive:
        if sys.version_info >= (3, 12):
            archive.extractall(destination, filter="data")
        else:  # pragma: no cover - supported Python 3.10/3.11 CI path.
            archive.extractall(destination)


def _assert_no_forbidden_payload(paths, *, forbidden_roots=()) -> None:
    for path in paths:
        relative = Path(path)
        parts = {part.casefold() for part in relative.parts}
        name = relative.name.casefold()
        assert not path.startswith((*FORBIDDEN_ROOTS, *forbidden_roots)), relative
        assert not FORBIDDEN_PARTS.intersection(parts), relative
        assert relative.suffix.lower() not in FORBIDDEN_SUFFIXES, relative
        assert name not in FORBIDDEN_NAMES, relative
        assert not name.startswith(("timing-", ".test-timings-")), relative


def test_manifest_uses_exact_example_allowlist():
    repository = Path(__file__).resolve().parents[2]
    lines = {
        line.strip()
        for line in (repository / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    included_examples = {
        line.removeprefix("include ")
        for line in lines
        if line.startswith("include examples/")
    }

    assert included_examples == EXAMPLES
    assert not any(
        line.startswith(("include tutorials/", "recursive-include examples", "recursive-include tutorials"))
        for line in lines
    )


def test_worktree_blob_oid_honors_git_line_ending_conversion(tmp_path):
    repository = tmp_path / "repository"
    contract_relative = RELEASE_CONTRACT_PATH
    contract = repository / contract_relative
    contract.parent.mkdir(parents=True)
    contract.write_bytes(b"first line\nsecond line\n")

    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    subprocess.run(
        ["git", "config", "core.autocrlf", "true"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", contract_relative], cwd=repository, check=True)
    committed_oid = subprocess.check_output(
        ["git", "rev-parse", f":{contract_relative}"],
        cwd=repository,
        text=True,
    ).strip()

    contract.write_bytes(b"first line\r\nsecond line\r\n")
    assert _worktree_blob_oid(repository, contract_relative) == committed_oid

    contract.write_bytes(b"first line\r\nchanged line\r\n")
    assert _worktree_blob_oid(repository, contract_relative) != committed_oid


@pytest.mark.parametrize(
    "path",
    (
        "examples/notebooks/state.dry",
        "examples/notebooks/state.dill",
        "examples/notebooks/state.pickle",
        "examples/notebooks/state.db",
        "examples/notebooks/checkpoint/model.ckpt",
        "examples/notebooks/tuning/results.json",
        "examples/notebooks/cache/results.json",
        "examples/notebooks/timing.json",
    ),
)
def test_generated_tutorial_payloads_are_forbidden(path):
    with pytest.raises(AssertionError):
        _assert_no_forbidden_payload((path,))


def test_torch_dependency_floor_matches_package_and_ci_configuration():
    repository = Path(__file__).resolve().parents[2]
    setup = (repository / "setup.cfg").read_text(encoding="utf-8")
    workflow = (repository / ".github/workflows/tests.yaml").read_text(
        encoding="utf-8"
    )

    assert "torch>=2.7.0" in setup
    assert "torch>=2.8.0" not in setup
    assert workflow.count('"torch>=2.7.0"') == 2
    assert "torch>=2.8.0" not in workflow
    assert EXTRAS["torch"] == {
        ("torch", ">=2.7.0", 'python_version < "3.14" and extra == "torch"')
    }


def test_worktree_wheel_metadata_emits_torch_2_7_floor(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    source.mkdir()
    for relative in _tracked_files(repository):
        origin = repository / relative
        if not origin.is_file():
            continue
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
    guard = _network_guard(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()

    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(dist),
            str(source),
        ],
        cwd=tmp_path,
        environment=_offline_environment(guard),
    )

    metadata = _metadata_from_wheel(next(dist.glob("*.whl")))
    torch_requirements = {
        (
            requirement.name,
            str(requirement.specifier),
            str(requirement.marker or ""),
        )
        for requirement in map(Requirement, metadata.get_all("Requires-Dist"))
        if 'extra == "torch"' in str(requirement.marker or "")
    }
    assert torch_requirements == EXTRAS["torch"]


@pytest.fixture(scope="module")
def release_artifacts(tmp_path_factory):
    missing_tools = [
        package for package in BUILD_TOOLS
        if importlib.util.find_spec(package) is None
    ]
    if missing_tools:
        pytest.fail(
            "the offline PEP 517 artifact contract requires pre-provisioned "
            f"build tools: {missing_tools}"
        )

    repository = Path(__file__).resolve().parents[2]
    root = tmp_path_factory.mktemp("release-artifacts")
    source = root / "source"
    source.mkdir()
    guard = _network_guard(root)
    environment = _offline_environment(guard)
    tracked, candidate_sha = _archive_candidate(
        repository,
        source,
        root / "candidate.tar",
        environment,
    )
    dist = root / "dist"
    dist.mkdir()
    _run(
        [sys.executable, "-m", "build", "--no-isolation", "--sdist", "--outdir", str(dist), str(source)],
        cwd=root,
        environment=environment,
    )
    sdist = next(dist.glob("*.tar.gz"))
    unpack_root = root / "unpacked"
    unpack_root.mkdir()
    _extract_tar(sdist, unpack_root)
    unpacked = next(path for path in unpack_root.iterdir() if path.is_dir())
    _run(
        [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", str(dist), str(unpacked)],
        cwd=root,
        environment=environment,
    )
    wheel = next(dist.glob("*.whl"))
    assert _head_sha(repository) == candidate_sha, (
        "repository HEAD changed while release artifacts were being built"
    )
    artifacts = tracked, candidate_sha, root, source, guard, sdist, wheel
    try:
        yield artifacts
    finally:
        assert _head_sha(repository) == candidate_sha, (
            "repository HEAD changed while release artifacts were being tested"
        )


def _metadata_from_wheel(wheel: Path):
    with zipfile.ZipFile(wheel) as archive:
        name = next(member for member in archive.namelist() if member.endswith(".dist-info/METADATA"))
        return email.parser.BytesParser().parsebytes(archive.read(name))


def _metadata_from_sdist(sdist: Path):
    with tarfile.open(sdist) as archive:
        member = next(member for member in archive.getmembers() if member.name.endswith("/PKG-INFO"))
        extracted = archive.extractfile(member)
        assert extracted is not None
        return email.parser.BytesParser().parsebytes(extracted.read())


def test_artifact_metadata_and_optional_extras_are_exact(release_artifacts):
    _, _, _, source, _, sdist, wheel = release_artifacts
    wheel_metadata = _metadata_from_wheel(wheel)
    sdist_metadata = _metadata_from_sdist(sdist)

    for field in ("Name", "Version", "Requires-Python", "License"):
        assert wheel_metadata[field] == sdist_metadata[field]
    assert set(wheel_metadata.get_all("Requires-Dist")) == set(
        sdist_metadata.get_all("Requires-Dist")
    )
    assert set(wheel_metadata.get_all("Provides-Extra")) == set(
        sdist_metadata.get_all("Provides-Extra")
    )
    assert wheel_metadata["Name"] == "dryml"
    assert wheel_metadata["Version"] == "0.3.0.dev0"
    assert wheel_metadata["Requires-Python"] == ">=3.10"
    source_version = next(
        line.split('"')[1]
        for line in (source / "src/dryml/__init__.py").read_text().splitlines()
        if line.startswith("__version__")
    )
    assert Version(source_version) == Version(wheel_metadata["Version"])
    readme = (source / "README.md").read_text().strip()
    assert wheel_metadata.get_payload().splitlines() == readme.splitlines()
    assert sdist_metadata.get_payload().splitlines() == readme.splitlines()
    assert "?token=" not in wheel_metadata.get_payload().lower()
    assert set(wheel_metadata.get_all("Provides-Extra")) == set(EXTRAS)

    parsed = [Requirement(value) for value in wheel_metadata.get_all("Requires-Dist")]
    core = {(requirement.name, str(requirement.specifier)) for requirement in parsed if requirement.marker is None}
    assert core == {("dill", ""), ("tqdm", ""), ("numpy", ""), ("gputil", ""), ("packaging", "")}
    for extra, expected in EXTRAS.items():
        actual = set()
        for requirement in parsed:
            marker = str(requirement.marker or "")
            if f'extra == "{extra}"' not in marker:
                continue
            actual.add((
                requirement.name,
                str(requirement.specifier),
                marker,
            ))
        assert actual == expected


def test_sdist_and_wheel_content_and_bounds(release_artifacts):
    tracked, _, _, _, _, sdist, wheel = release_artifacts
    assert sdist.stat().st_size <= MAX_ARTIFACT_BYTES
    assert wheel.stat().st_size <= MAX_ARTIFACT_BYTES

    with tarfile.open(sdist) as archive:
        members = archive.getmembers()
        sdist_members = {member.name for member in members}
        sdist_files = {member.name for member in members if member.isfile()}
    with zipfile.ZipFile(wheel) as archive:
        wheel_members = set(archive.namelist())
    assert len(sdist_members) <= MAX_ARCHIVE_MEMBERS
    assert len(wheel_members) <= MAX_ARCHIVE_MEMBERS

    prefix = next(iter(sdist_members)).split("/", 1)[0]
    relative_sdist = {
        member.removeprefix(f"{prefix}/")
        for member in sdist_members
        if member != prefix
    }
    relative_sdist_files = {
        member.removeprefix(f"{prefix}/") for member in sdist_files
    }
    tracked_docs = {path for path in tracked if path.startswith("docs/")}
    assert tracked_docs <= relative_sdist
    assert {path for path in relative_sdist_files if path.startswith("examples/")} == EXAMPLES
    assert "README.md" in relative_sdist
    assert "LICENSE" in relative_sdist
    assert not any(path.startswith("src/dryml/graph/") for path in relative_sdist)

    package_dirs = {
        Path(path).parent
        for path in tracked
        if path.startswith("src/dryml/") and path.endswith("/__init__.py")
    }
    tracked_modules = set()
    for path in tracked:
        module = Path(path)
        if not path.startswith("src/dryml/") or module.suffix != ".py":
            continue
        parent = module.parent
        while parent != Path("src") and parent in package_dirs:
            parent = parent.parent
        if parent == Path("src"):
            tracked_modules.add(path)
    assert tracked_modules <= relative_sdist_files
    expected_wheel_modules = {
        path.removeprefix("src/") for path in tracked_modules
    }
    assert expected_wheel_modules <= wheel_members
    obsolete_package = f"/dryml/{RETIRED_CORE_PACKAGE}/"
    assert not any(obsolete_package in f"/{path}" for path in relative_sdist)
    assert not any(obsolete_package in f"/{path}" for path in wheel_members)

    _assert_no_forbidden_payload(
        relative_sdist_files,
        forbidden_roots=("tests/",),
    )

    forbidden_roots = ("tests/", "examples/", "docs/")
    assert not any(path.startswith(forbidden_roots) for path in wheel_members)
    assert not any("/dryml/graph/" in path or path.startswith("dryml/graph/") for path in wheel_members)
    _assert_no_forbidden_payload(wheel_members, forbidden_roots=forbidden_roots)


def test_isolated_wheel_install_uses_artifact_and_stays_lightweight(release_artifacts):
    _, _, root, _, guard, _, wheel = release_artifacts
    target = root / "target"
    work = root / "smoke"
    work.mkdir()
    environment = _offline_environment(guard)
    _run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
        cwd=work,
        environment=environment,
    )
    (target / "sitecustomize.py").write_text(
        (guard / "sitecustomize.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    environment["PYTHONPATH"] = str(target)
    script = f"""
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
import sys

import dryml
import dryml.code
import dryml.core.store.store
import dryml.core.utils.general

target = Path({str(target)!r}).resolve()
obsolete = {RETIRED_CORE_PACKAGE!r}
obsolete_path = "dryml." + obsolete
assert Path(dryml.__file__).resolve().is_relative_to(target)
assert dryml.__version__ == importlib.metadata.version("dryml") == "0.3.0.dev0"
for name in ("annotations", "core", "dispatch", "artifacts", "env", "environments", "formats", "operations", "providers", "reporting", "records", "runtime", "world", "worlds"):
    importlib.import_module(f"dryml.{{name}}")
assert importlib.util.find_spec("dryml.data.transforms") is None
assert importlib.util.find_spec("dryml.graph") is None
assert importlib.util.find_spec(obsolete_path) is None
assert obsolete not in dryml.__dict__
try:
    importlib.import_module(obsolete_path)
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("obsolete core package remained importable")
assert not {{"jax", "ray", "tensorflow", "torch"}} & {{name.split(".", 1)[0] for name in sys.modules}}
spec = dryml.operations.make_function_call_spec("builtins:sum", args=[[1, 2]])
assert spec["kind"] == "function_call"
"""
    _run([sys.executable, "-c", script], cwd=work, environment=environment)
