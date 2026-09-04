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
    "dryml/requirements/__init__.py",
    "dryml/requirements/barrier.py",
    "dryml/requirements/collection.py",
    "dryml/requirements/combination.py",
    "dryml/requirements/errors.py",
    "dryml/requirements/model.py",
    "dryml/environments/__init__.py",
    "dryml/worlds/__init__.py",
    "dryml/runtime/__init__.py",
    "dryml/session/__init__.py",
    "dryml/tf/runtime.py",
    "dryml/torch/runtime.py",
    "dryml/jax/runtime.py",
    "dryml/ray/__init__.py",
    "dryml/methods/__init__.py",
    "dryml/methods/errors.py",
    "dryml/methods/implementation.py",
    "dryml/methods/method.py",
    "dryml/methods/signature.py",
    "dryml/methods/traits.py",
    "dryml/code/__init__.py",
    "dryml/code/analysis.py",
    "dryml/code/ast_tools.py",
    "dryml/code/callable_info.py",
    "dryml/code/errors.py",
    "dryml/code/facts.py",
    "dryml/code/graph.py",
    "dryml/code/kernels.py",
    "dryml/code/probe.py",
    "dryml/code/source.py",
    "dryml/code/targets.py",
    "dryml/code/trace.py",
    "dryml/code/algorithms/__init__.py",
    "dryml/code/algorithms/lexical_dependencies.py",
}

_RETIRED_CODE_MODULES = {
    "dryml/code/compiler_info.py",
    "dryml/code/method.py",
    "dryml/code/traits.py",
    "dryml/code/probe_worker.py",
    "dryml/code/transformation.py",
    "dryml/code/algorithms/direct_annotations.py",
    "dryml/code/algorithms/method_contracts.py",
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

_RETIRED_ENVIRONMENT_MODULES = {
    "dryml/environments/fragment.py",
    "dryml/environments/fragments.py",
}

_RETIRED_ENVIRONMENT_SYMBOLS = {
    "ENVIRONMENT_FRAGMENT_SCHEMA_VERSION",
    "FRAGMENT_ATTR",
    "RequirementFragment",
    "__dryml_environment_fragments__",
    "add_req",
    "compose_fragments",
    "fragments_for_class",
    "override_req",
    "requirements_for_class",
}


def test_wheel_contains_port_modules_without_retired_core(
    release_artifacts: tuple[Path, Path],
) -> None:
    """Check installed-package paths directly in the built wheel."""

    _, wheel = release_artifacts
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        environment_sources = {
            name: archive.read(name).decode("utf-8")
            for name in names
            if name.startswith("dryml/environments/") and name.endswith(".py")
        }
    assert _REQUIRED_MODULES <= names
    code_modules = {
        name for name in names if name.startswith("dryml/code/") and name.endswith(".py")
    }
    assert code_modules == {
        name for name in _REQUIRED_MODULES if name.startswith("dryml/code/")
    }
    annotation_modules = {
        name
        for name in names
        if name.startswith("dryml/annotations/") and name.endswith(".py")
    }
    assert annotation_modules == _RETAINED_ANNOTATION_MODULES
    assert not _RETIRED_ANNOTATION_MODULES & names
    assert not _RETIRED_ENVIRONMENT_MODULES & names
    assert not {
        symbol
        for symbol in _RETIRED_ENVIRONMENT_SYMBOLS
        if any(symbol in source for source in environment_sources.values())
    }
    assert not _RETIRED_CODE_MODULES & names
    assert not any(name.startswith("dryml/core2/") for name in names)
    assert "dryml/core/repo_graph.py" not in names


def test_sdist_contains_port_modules_without_retired_core(
    release_artifacts: tuple[Path, Path],
) -> None:
    """Check source-package paths directly in the built sdist."""

    sdist, _ = release_artifacts
    with tarfile.open(sdist, "r:gz") as archive:
        archive_names = archive.getnames()
        names = {"/".join(name.split("/")[1:]) for name in archive_names}
        environment_sources = {
            name: archive.extractfile(name).read().decode("utf-8")
            for name in archive_names
            if name.startswith("dryml-0.3.0.dev2/src/dryml/environments/") and name.endswith(".py")
        }
    required = {f"src/{name}" for name in _REQUIRED_MODULES}
    assert required <= names
    code_modules = {
        name.removeprefix("src/")
        for name in names
        if name.startswith("src/dryml/code/") and name.endswith(".py")
    }
    assert code_modules == {
        name for name in _REQUIRED_MODULES if name.startswith("dryml/code/")
    }
    annotation_modules = {
        name.removeprefix("src/")
        for name in names
        if name.startswith("src/dryml/annotations/") and name.endswith(".py")
    }
    assert annotation_modules == _RETAINED_ANNOTATION_MODULES
    assert not {f"src/{name}" for name in _RETIRED_ANNOTATION_MODULES} & names
    assert not {f"src/{name}" for name in _RETIRED_ENVIRONMENT_MODULES} & names
    assert not {
        symbol
        for symbol in _RETIRED_ENVIRONMENT_SYMBOLS
        if any(symbol in source for source in environment_sources.values())
    }
    assert not {f"src/{name}" for name in _RETIRED_CODE_MODULES} & names
    assert not any(name.startswith("src/dryml/core2/") for name in names)
    assert "src/dryml/core/repo_graph.py" not in names
    assert not any(name.startswith("tutorials/") for name in names)
