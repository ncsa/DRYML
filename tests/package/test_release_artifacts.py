"""Verify that source and wheel artifacts contain the intended package graph."""

from __future__ import annotations

import tarfile
from pathlib import Path
import zipfile

_REQUIRED_MODULES = {
    "dryml/core/__init__.py",
    "dryml/core/cdef_codec.py",
    "dryml/core/cdef_identity.py",
    "dryml/core/materialization.py",
    "dryml/core/reference_values.py",
    "dryml/core/repo.py",
    "dryml/core/repo_plan.py",
    "dryml/core/query/reference.py",
    "dryml/core/store/records.py",
    "dryml/formats/__init__.py",
    "dryml/environments/__init__.py",
    "dryml/worlds/__init__.py",
    "dryml/runtime/__init__.py",
    "dryml/session/__init__.py",
    "dryml/tf/runtime.py",
    "dryml/torch/runtime.py",
    "dryml/jax/runtime.py",
    "dryml/ray/__init__.py",
}

_RETAINED_ANNOTATION_MODULES = {
    "dryml/annotations/__init__.py",
    "dryml/annotations/model.py",
    "dryml/annotations/attachment.py",
    "dryml/annotations/collect.py",
    "dryml/annotations/errors.py",
}

_RETIRED_ANNOTATION_MODULES = {
    "dryml/annotations/storage.py",
    "dryml/annotations/decorators.py",
    "dryml/annotations/env.py",
    "dryml/annotations/world.py",
    "dryml/annotations/runtime.py",
    "dryml/annotations/merge.py",
    "dryml/annotations/namespaces.py",
}


def test_wheel_contains_port_modules_without_retired_core(
    release_artifacts: tuple[Path, Path],
) -> None:
    """Check installed-package paths directly in the built wheel."""

    _, wheel = release_artifacts
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert _REQUIRED_MODULES <= names
    annotation_modules = {
        name
        for name in names
        if name.startswith("dryml/annotations/") and name.endswith(".py")
    }
    assert annotation_modules == _RETAINED_ANNOTATION_MODULES
    assert not _RETIRED_ANNOTATION_MODULES & names
    assert not any(name.startswith("dryml/core2/") for name in names)
    assert "dryml/core/repo_graph.py" not in names


def test_sdist_contains_port_modules_without_retired_core(
    release_artifacts: tuple[Path, Path],
) -> None:
    """Check source-package paths directly in the built sdist."""

    sdist, _ = release_artifacts
    with tarfile.open(sdist, "r:gz") as archive:
        names = {"/".join(name.split("/")[1:]) for name in archive.getnames()}
    required = {f"src/{name}" for name in _REQUIRED_MODULES}
    assert required <= names
    annotation_modules = {
        name.removeprefix("src/")
        for name in names
        if name.startswith("src/dryml/annotations/") and name.endswith(".py")
    }
    assert annotation_modules == _RETAINED_ANNOTATION_MODULES
    assert not {f"src/{name}" for name in _RETIRED_ANNOTATION_MODULES} & names
    assert not any(name.startswith("src/dryml/core2/") for name in names)
    assert "src/dryml/core/repo_graph.py" not in names
    assert not any(name.startswith("tutorials/") for name in names)
