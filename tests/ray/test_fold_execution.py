"""Explicit existing-target Ray evidence for result-only Fold publication."""

from __future__ import annotations

import base64
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

from dryml.core import Executor as CoreExecutor
from dryml.core import Repo, StateRef
from dryml.core.execute import CoreOptions
from dryml.core.store.dir import DirStore
from dryml.execute.ray import RayBackendConfig
from dryml.managed import ManagedConfig
from dryml.worlds import CountConstraint, ResourceRequirement, RoleRequirement, WorldRequirement
from tests.artifacts.test_fold_execution import FileDataset, _fold
from .conftest import require_ray_integration


def _one_cpu_world() -> WorldRequirement:
    """Request the one logical CPU used by this representative Ray case."""

    return WorldRequirement({
        "main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1))),
    })


def _assert_fresh_result(state_dir: Path, state: StateRef) -> None:
    """Read a result from exact state without providing source/control authority."""

    script = """
import base64
import pickle
import sys
from dryml.core import Repo
from dryml.core.store.dir import DirStore
state = pickle.loads(base64.b64decode(sys.argv[2]))
loaded = Repo(DirStore.open_existing(sys.argv[1])).load_state_ref(state, reuse_live='never')
assert loaded.value() == 6.0
assert not {'tensorflow', 'torch', 'jax', 'jaxlib', 'ray'} & set(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(state_dir), base64.b64encode(pickle.dumps(state)).decode("ascii")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_existing_ray_fold_publishes_and_restores_its_result_without_source(tmp_path: Path):
    """Use only the caller-supplied Ray target for one Fold publish/restore flow."""

    source_path = tmp_path / "source.txt"
    source_path.write_text("1\n2\n3\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    control_dir = tmp_path / "control"
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    repo = Repo(DirStore(state_dir, query_index="none"))
    control_store = DirStore(control_dir, query_index="none")
    fold = _fold(FileDataset(str(source_path), repo=repo), repo=repo)
    repo.save_object(fold, deep_capture=True)
    executor = CoreExecutor(
        RayBackendConfig(
            address=require_ray_integration(), spool_directory=spool_dir,
            admission_timeout=90, connect_timeout=60, termination_timeout=10,
        ),
        core=CoreOptions(repo=repo, control_store=control_store, return_objects=False),
    )
    try:
        future = executor.submit(
            fold.compute,
            kwargs={"managed": ManagedConfig(state_repo=repo, control_store=control_store)},
            world=_one_cpu_world(),
        )
        assert future.result(timeout=30) is None
        future.cleanup(timeout=30)
    finally:
        executor.close(cancel=True, timeout=30)

    state = fold.compute.status(
        state_repo=repo, control_store=control_store,
    ).final_state_ref
    assert isinstance(state, StateRef)
    source_path.unlink()
    control_store.close()
    shutil.rmtree(control_dir)
    _assert_fresh_result(state_dir, state)
