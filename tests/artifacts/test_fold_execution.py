"""U8 execution evidence for result-only Value and Fold Artifacts."""

from __future__ import annotations

import base64
import pickle
import shutil
import subprocess
import sys
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pytest

from dryml.artifacts import Fold, Value
from dryml.core import Executor as CoreExecutor
from dryml.core import Repo, StateRef
from dryml.core.execute import CoreOptions
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Dataset
from dryml.execute.subprocess import SubProcessConfig
from dryml.managed import ManagedConfig, managed_operation
from dryml.methods import Accumulator, Method, traits


_VALUE_FORMAT = "dryml.artifacts.value"
_SOURCE_SPEC = TensorSpec("float64", shape=(), batch=Dynamic, backend="numpy")


class FileValue(Value):
    """Compute one text result from a caller-owned input file."""

    def __init__(self, source_path: str) -> None:
        """Retain the input path without reading it during declaration."""

        self.source_path = source_path

    @property
    def ready(self) -> bool:
        """Return whether a complete result envelope is installed."""

        return self._value_is_present()

    @managed_operation()
    def compute(self, *, managed) -> None:
        """Read the input file and install its complete text result.

        Args:
            managed: Framework-managed lifecycle context.

        Raises:
            FileNotFoundError: If the caller-owned input file is unavailable.
        """

        result = Path(self.source_path).read_text(encoding="utf-8")
        self._install_value_payload({
            "format": _VALUE_FORMAT,
            "version": 1,
            "present": True,
            "result": result,
        })


class FileDataset(Dataset):
    """Re-iterable numeric Dataset backed by a caller-owned text file."""

    def __init__(self, source_path: str) -> None:
        """Retain the source path and publish its explicit batch specification."""

        self.source_path = source_path
        super().__init__(spec=_SOURCE_SPEC)

    def __iter__(self):
        """Yield one batched NumPy scalar for every line in the source file.

        Raises:
            FileNotFoundError: If the source no longer exists when computation starts.
        """

        for line in Path(self.source_path).read_text(encoding="utf-8").splitlines():
            yield np.array([float(line)], dtype=np.float64)


class Zero(Method):
    """Create a float64 scalar carry for the representative Fold."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        """Return a fresh zero carry without consuming the observation."""

        return np.array(0.0, dtype=np.float64)

    def infer_output_spec(self, observation_spec):
        """Declare the scalar float64 carry contract."""

        return TensorSpec("float64", shape=(), backend="numpy")


class Sum(Accumulator):
    """Add each numeric batch to an invocation-owned scalar carry."""

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, state):
        """Return the next carry after one batch transition."""

        return state + observation.sum(dtype=np.float64)

    def infer_output_spec(self, observation_spec, state_spec):
        """Preserve the declared scalar carry specification."""

        return state_spec


def _fold(source: FileDataset, *, repo: Repo | None = None) -> Fold:
    """Build the representative deferred Fold without traversing its source."""

    return Fold(source, initial_state=Zero(), accumulator=Sum(), repo=repo)


def _run_managed(artifact, *, placement: str, repo: Repo, control_store: DirStore, spool: Path):
    """Compute one Artifact directly or through one real core subprocess.

    Args:
        artifact: Concrete Value or Fold Artifact whose managed method is invoked.
        placement: ``"direct"`` or ``"subprocess"`` execution placement.
        repo: State authority shared with the worker when applicable.
        control_store: Separate managed-control authority.
        spool: Existing subprocess spool directory.

    Returns:
        The final exact StateRef published by the managed lifecycle.
    """

    config = ManagedConfig(state_repo=repo, control_store=control_store)
    if placement == "direct":
        assert artifact.compute(managed=config) is None
    else:
        spool.mkdir()
        repo.save_object(artifact, deep_capture=True)
        executor = CoreExecutor(
            SubProcessConfig(spool_directory=spool),
            core=CoreOptions(
                repo=repo, control_store=control_store, return_objects=False,
            ),
        )
        try:
            future = executor.submit(artifact.compute, kwargs={"managed": config})
            assert future.result(timeout=15) is None
            future.cleanup(timeout=5)
        finally:
            executor.close(cancel=True, timeout=10)
    status = artifact.compute.status(state_repo=repo, control_store=control_store)
    assert status.state == "completed"
    assert isinstance(status.final_state_ref, StateRef)
    return status.final_state_ref


def _assert_fresh_result(state_dir: Path, state: StateRef, expected) -> None:
    """Restore one exact result in a process that receives no input/control authority."""

    script = """
import base64
import pickle
import sys

from dryml.core import Repo
from dryml.core.store.dir import DirStore

state = pickle.loads(base64.b64decode(sys.argv[2]))
loaded = Repo(DirStore.open_existing(sys.argv[1])).load_state_ref(
    state, reuse_live='never',
)
assert loaded.value() == pickle.loads(base64.b64decode(sys.argv[3]))
assert not {'tensorflow', 'torch', 'jax', 'jaxlib', 'ray'} & set(sys.modules)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(state_dir),
            base64.b64encode(pickle.dumps(state)).decode("ascii"),
            base64.b64encode(pickle.dumps(expected)).decode("ascii"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
@pytest.mark.parametrize("placement", ("direct", "subprocess"))
def test_value_execution_publishes_a_result_only_state_for_a_fresh_reader(tmp_path, placement):
    """A custom Value survives direct and worker computation without its source file."""

    source = tmp_path / "value-input.txt"
    source.write_text("worker value", encoding="utf-8")
    state_dir = tmp_path / "state"
    control_dir = tmp_path / "control"
    repo = Repo(DirStore(state_dir, query_index="none"))
    control_store = DirStore(control_dir, query_index="none")
    value = FileValue(str(source), repo=repo)

    state = _run_managed(
        value,
        placement=placement,
        repo=repo,
        control_store=control_store,
        spool=tmp_path / "spool",
    )
    source.unlink()
    with pytest.raises(FileNotFoundError):
        value.compute(managed=ManagedConfig(
            state_repo=repo, control_store=control_store, rerun=True,
        ))
    control_store.close()
    shutil.rmtree(control_dir)

    _assert_fresh_result(state_dir, state, "worker value")


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
@pytest.mark.parametrize("placement", ("direct", "subprocess"))
def test_fold_execution_publishes_a_result_only_state_for_a_fresh_reader(tmp_path, placement):
    """A representative Fold restores its exact result after source removal."""

    source_path = tmp_path / "fold-input.txt"
    source_path.write_text("1\n2\n3\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    control_dir = tmp_path / "control"
    repo = Repo(DirStore(state_dir, query_index="none"))
    control_store = DirStore(control_dir, query_index="none")
    fold = _fold(FileDataset(str(source_path), repo=repo), repo=repo)

    state = _run_managed(
        fold,
        placement=placement,
        repo=repo,
        control_store=control_store,
        spool=tmp_path / "spool",
    )
    source_path.unlink()
    with pytest.raises(FileNotFoundError):
        fold.compute(managed=ManagedConfig(
            state_repo=repo, control_store=control_store, rerun=True,
        ))
    control_store.close()
    shutil.rmtree(control_dir)

    _assert_fresh_result(state_dir, state, 6.0)
