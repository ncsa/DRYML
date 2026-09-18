"""Verify that source and wheel artifacts contain the intended package graph."""

from __future__ import annotations

import tarfile
from pathlib import Path
from pathlib import PurePosixPath
import zipfile

from tests.tools.native_lock_audit import native_advisory_lock_offenders

_REQUIRED_MODULES = {
    "dryml/locking.py",
    "dryml/core/state.py",
    "dryml/core/__init__.py",
    "dryml/core/cdef_codec.py",
    "dryml/core/cdef_identity.py",
    "dryml/core/execute.py",
    "dryml/core/execute_codec.py",
    "dryml/core/_callable_inspection.py",
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
    "dryml/environments/kernel.py",
    "dryml/environments/selection.py",
    "dryml/worlds/__init__.py",
    "dryml/worlds/kernel.py",
    "dryml/runtime/__init__.py",
    "dryml/runtime/activation.py",
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
    "dryml/code/inspection.py",
    "dryml/code/kernels.py",
    "dryml/code/probe.py",
    "dryml/code/source.py",
    "dryml/code/static_dependencies.py",
    "dryml/code/targets.py",
    "dryml/code/trace.py",
    "dryml/code/algorithms/__init__.py",
    "dryml/code/algorithms/lexical_dependencies.py",
    "dryml/managed/__init__.py",
    "dryml/managed/config.py",
    "dryml/managed/context.py",
    "dryml/managed/control.py",
    "dryml/managed/descriptor.py",
    "dryml/managed/errors.py",
    "dryml/managed/identity.py",
    "dryml/managed/model.py",
    "dryml/managed/runtime.py",
    "dryml/managed/storage.py",
    "dryml/execute/__init__.py",
    "dryml/execute/_process.py",
    "dryml/execute/_protocol.py",
    "dryml/execute/_spooling.py",
    "dryml/execute/_worker.py",
    "dryml/execute/accounting.py",
    "dryml/execute/admission.py",
    "dryml/execute/backend.py",
    "dryml/execute/config.py",
    "dryml/execute/discovery.py",
    "dryml/execute/errors.py",
    "dryml/execute/executor.py",
    "dryml/execute/future.py",
    "dryml/execute/models.py",
    "dryml/execute/output.py",
    "dryml/execute/ray.py",
    "dryml/execute/subprocess.py",
    "dryml/dispatch/__init__.py",
    "dryml/dispatch/_admission.py",
    "dryml/dispatch/_preflight.py",
    "dryml/dispatch/_probe.py",
    "dryml/dispatch/_probe_protocol.py",
    "dryml/dispatch/_state.py",
    "dryml/dispatch/api.py",
    "dryml/dispatch/errors.py",
    "dryml/dispatch/models.py",
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

_RETIRED_EXECUTE_MODULES = {
    "dryml/execute/orchestrator.py",
    "dryml/execute/protocol.py",
    "dryml/execute/transfer.py",
    "dryml/execute/worker.py",
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


def _sdist_member_path(name: str) -> str:
    """Return an sdist member path relative to its generated root directory."""

    return "/".join(PurePosixPath(name).parts[1:])


def test_wheel_contains_port_modules_without_retired_core(
    release_artifacts: tuple[Path, Path],
) -> None:
    """Check installed-package paths directly in the built wheel."""

    _, wheel = release_artifacts
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        package_sources = {
            name: archive.read(name)
            for name in names
            if name.startswith("dryml/") and name.endswith(".py")
        }
        environment_sources = {
            name: source.decode("utf-8")
            for name, source in package_sources.items()
            if name.startswith("dryml/environments/")
        }
        native_lock_sources = {
            name
            for name, source in package_sources.items()
            if native_advisory_lock_offenders(source, filename=name)
        }
    assert _REQUIRED_MODULES <= names
    code_modules = {
        name for name in names if name.startswith("dryml/code/") and name.endswith(".py")
    }
    assert code_modules == {
        name for name in _REQUIRED_MODULES if name.startswith("dryml/code/")
    }
    dispatch_modules = {
        name
        for name in names
        if name.startswith("dryml/dispatch/") and name.endswith(".py")
    }
    assert dispatch_modules == {
        name
        for name in _REQUIRED_MODULES if name.startswith("dryml/dispatch/")
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
    assert not _RETIRED_EXECUTE_MODULES & names
    assert not any(name.startswith("dryml/core2/") for name in names)
    assert "dryml/core/store/locking.py" not in names
    assert "dryml/managed/locking.py" not in names
    assert native_lock_sources == {"dryml/locking.py"}
    assert "dryml/core/repo_graph.py" not in names


def test_sdist_contains_port_modules_without_retired_core(
    release_artifacts: tuple[Path, Path],
) -> None:
    """Check source-package paths directly in the built sdist."""

    sdist, _ = release_artifacts
    with tarfile.open(sdist, mode="r:*") as archive:
        archive_names = archive.getnames()
        names = {_sdist_member_path(name) for name in archive_names}
        package_sources = {}
        for member in archive.getmembers():
            name = _sdist_member_path(member.name)
            if not (member.isfile() and name.startswith("src/dryml/") and name.endswith(".py")):
                continue
            source = archive.extractfile(member)
            assert source is not None
            package_sources[name] = source.read()
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
    dispatch_modules = {
        name.removeprefix("src/")
        for name in names
        if name.startswith("src/dryml/dispatch/") and name.endswith(".py")
    }
    assert dispatch_modules == {
        name
        for name in _REQUIRED_MODULES if name.startswith("dryml/dispatch/")
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
        if any(symbol in source.decode("utf-8") for source in package_sources.values())
    }
    assert not {f"src/{name}" for name in _RETIRED_CODE_MODULES} & names
    assert not {f"src/{name}" for name in _RETIRED_EXECUTE_MODULES} & names
    assert not any(name.startswith("src/dryml/core2/") for name in names)
    assert "src/dryml/core/store/locking.py" not in names
    assert "src/dryml/managed/locking.py" not in names
    assert {
        name
        for name, source in package_sources.items()
        if native_advisory_lock_offenders(source, filename=name)
    } == {"src/dryml/locking.py"}
    assert "src/dryml/core/repo_graph.py" not in names
    assert not any(name.startswith("tutorials/") for name in names)
