"""Opt-in existing-target Ray integration coverage for Dispatch ownership."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import dryml.dispatch as dispatch
from dryml.core import Repo, Serializable
from dryml.core.execute import CoreOptions
from dryml.core.store.dir import DirStore
from dryml.dispatch import InProcess, ProbeOptions
from dryml.environments.specs import CondaEnvironmentSpec, PythonExecutableSpec
from dryml.execute.ray import RayBackendConfig
from dryml.execute.errors import AdmissionError
from dryml.execute.subprocess import SubProcessConfig
from dryml.managed import managed_operation
from dryml.worlds import (CountConstraint, ResourceRequirement,
                          RoleRequirement, WorldRequirement)
from .conftest import require_ray_integration


def _scalar(value: int = 1) -> int:
    """Return a transport-safe scalar without importing a test package."""

    return value + 1


def _worker_identity() -> tuple[str, str]:
    """Return the worker prefix and executable used by an exact pin."""

    import sys

    return sys.prefix, sys.executable


class _RayCounter(Serializable):
    """Persist a mutable receiver through caller-supplied Store authority."""

    def __init__(self, value: int = 0) -> None:
        """Initialize the scalar persisted by the state hooks."""

        self.value = value

    @managed_operation()
    def add(self, value: int, *, managed) -> int:
        """Mutate the receiver once and return its updated scalar."""

        del managed
        self.value += value
        return self.value

    def save_state_to_dir_imp(self, dest_dir, *, codec) -> None:
        """Store the scalar through DRYML's selected state codec."""

        del codec
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec) -> None:
        """Restore the scalar through DRYML's selected state codec."""

        del codec
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


@pytest.fixture(autouse=True)
def _existing_ray_target() -> None:
    """Skip unless the caller supplied and enabled an existing target."""

    require_ray_integration()
    dispatch._state._reset_for_testing()
    yield
    dispatch._state._reset_for_testing()


def _ray_config(
    tmp_path: Path, *, address: str | None = None,
) -> RayBackendConfig:
    """Create an inert configuration for the supplied existing Ray endpoint."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    return RayBackendConfig(
        address=require_ray_integration() if address is None else address,
        spool_directory=tmp_path,
        admission_timeout=90,
        connect_timeout=60,
        termination_timeout=20,
        automatic_environment_discovery=False,
    )


def _repo(tmp_path: Path) -> Repo:
    """Open test-owned Store authority for core and bound-object recovery."""

    return Repo(DirStore(tmp_path / "store", query_index="none"))


def _one_cpu_world() -> WorldRequirement:
    """Request the portable logical Ray CPU evidence used by this suite."""

    return WorldRequirement({
        "main": RoleRequirement(
            resources=ResourceRequirement(cpus=CountConstraint(1, None)),
        ),
    })


@pytest.mark.parametrize("kind", ("conda", "venv"))
def test_dispatch_ray_workload_honors_exact_existing_environment_pin(
    tmp_path: Path, kind: str,
) -> None:
    """Use the Dispatch environment_spec handoff for both supplied runtimes."""

    if kind == "conda":
        prefix = Path(os.environ["DRYML_TEST_CONDA_PREFIX"])
        expected = prefix / ("python.exe" if os.name == "nt" else "bin/python")
        spec = CondaEnvironmentSpec(prefix=str(prefix))
    else:
        expected = Path(os.environ["DRYML_TEST_VENV_PYTHON"])
        prefix = expected.parent.parent
        spec = PythonExecutableSpec(str(expected))
    assert expected.is_file(), (
        "the caller must supply an existing Python target"
    )
    repo = _repo(tmp_path)
    try:
        view = dispatch.with_options(
            backend=_ray_config(tmp_path / "ray"),
            core=CoreOptions(repo=repo, return_objects=False),
            python=spec,
            world=_one_cpu_world(),
        )
        report = view.explain(_worker_identity)
        assert report.probe_placement == "in_process"
        assert report.workload_placement == "execute"
        try:
            identity = view.run(_worker_identity)
        except AdmissionError as error:
            issues = tuple(
                (issue.code, issue.message)
                for issue in getattr(error.report, "issues", ())
            )
            pytest.fail(f"Exact worker admission failed: {issues}")
        assert identity == (str(prefix), str(expected))
    finally:
        repo.close(flush=False)


def test_dispatch_uses_ray_probe_for_local_subprocess_workload(
    tmp_path: Path,
) -> None:
    """Keep an explicit Ray probe independent from a local workload backend."""

    repo = _repo(tmp_path)
    local_spool = tmp_path / "local"
    local_spool.mkdir()
    try:
        view = dispatch.with_options(
            backend=SubProcessConfig(spool_directory=local_spool),
            core=CoreOptions(repo=repo, return_objects=False),
            probe=ProbeOptions(
                placement="execute", backend=_ray_config(tmp_path / "probe"),
            ),
        )
        report = view.explain(_scalar)
        assert report.probe_placement == "execute"
        assert report.probe_backend == "RayBackendConfig"
        assert report.workload_backend == "SubProcessConfig"
        assert view.run(_scalar, 4) == 5
    finally:
        repo.close(flush=False)


def test_dispatch_uses_ray_probe_before_direct_in_process_workload(
    tmp_path: Path,
) -> None:
    """Keep a selected Ray probe from moving direct local invocation."""

    view = dispatch.with_options(
        backend=InProcess(),
        probe=ProbeOptions(
            placement="execute", backend=_ray_config(tmp_path / "probe"),
        ),
    )
    report = view.explain(_scalar)
    assert report.probe_placement == "execute"
    assert report.workload_placement == "in_process"
    assert view.run(_scalar, 8) == 9


def test_same_ray_configuration_keeps_probe_and_workload_owners_separate(
    tmp_path: Path,
) -> None:
    """Run a probe then workload without sharing lifecycle ownership."""

    repo = _repo(tmp_path)
    try:
        ray = _ray_config(tmp_path / "ray")
        view = dispatch.with_options(
            backend=ray,
            core=CoreOptions(repo=repo, return_objects=False),
            probe=ProbeOptions(placement="execute", backend=ray),
        )
        report = view.explain(_scalar)
        assert (
            report.probe_backend
            == report.workload_backend
            == "RayBackendConfig"
        )
        assert view.run(_scalar, 2) == 3
    finally:
        repo.close(flush=False)


def test_dispatch_ray_workload_recovers_a_bound_object_from_shared_store(
    tmp_path: Path,
) -> None:
    """Reuse explicit Store authority for a Ray-bound receiver update."""

    repo = _repo(tmp_path)
    try:
        counter = _RayCounter(3, repo=repo)
        repo.save(counter, deep_capture=True)
        view = dispatch.with_options(
            backend=_ray_config(tmp_path / "ray"),
            core=CoreOptions(
                repo=repo, return_objects=False, update_args=True,
            ),
        )
        assert view.run(counter.add, 4) == 7
    finally:
        repo.close(flush=False)


def test_dispatch_unavailable_selected_ray_probe_never_falls_back(
    tmp_path: Path,
) -> None:
    """Reject unavailable probing without leaking native SDK initialization."""

    local_spool = tmp_path / "local"
    local_spool.mkdir()
    (tmp_path / "missing").mkdir()
    # Native Ray initialization may outlive a caller's bounded wait. Keep this
    # deliberately unreachable endpoint outside the shared test interpreter.
    script = """
from pathlib import Path
import sys
import dryml.dispatch as dispatch
from dryml.execute.errors import ExecutionError
from dryml.execute.ray import RayBackendConfig
from dryml.execute.subprocess import SubProcessConfig

def must_not_run():
    raise AssertionError("unavailable probe must not fall back")

root = Path(sys.argv[1])
view = dispatch.with_options(
    backend=SubProcessConfig(spool_directory=root / "local"),
    probe=dispatch.ProbeOptions(
        placement="execute",
        backend=RayBackendConfig(
            address="127.0.0.1:1", spool_directory=root / "missing",
            connect_timeout=1, admission_timeout=3, termination_timeout=1,
            automatic_environment_discovery=False,
        ),
    ),
)
try:
    view.run(must_not_run)
except ExecutionError:
    print("unavailable-probe-rejected")
else:
    raise AssertionError("unavailable probe was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "unavailable-probe-rejected" in result.stdout.splitlines()

    usable = dispatch.with_options(
        backend=InProcess(),
        probe=ProbeOptions(
            placement="execute",
            backend=_ray_config(tmp_path / "usable"),
        ),
    )
    assert usable.run(_scalar, 3) == 4
