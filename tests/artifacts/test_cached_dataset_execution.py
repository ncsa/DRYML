"""Representative local and subprocess CachedDataset managed integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dryml.artifacts import CachedDataset
from dryml.core import Executor as CoreExecutor, Node, Repo
from dryml.core.execute import CoreOptions
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Dataset
from dryml.execute.subprocess import SubProcessConfig
from dryml.managed import InvocationResult, ManagedConfig
from dryml.artifacts.dataset import _saved_digest


class ObservedDataset(Dataset):
    """Finite source that records each real traversal in a same-host file."""

    def __init__(self, marker: str) -> None:
        """Retain the caller-selected observation path and fixed values."""

        self.marker = marker
        super().__init__(TensorSpec("int32", shape=(1,), backend="numpy"))

    def __iter__(self):
        """Record one source open and return deterministic values."""

        with Path(self.marker).open("a", encoding="ascii") as output:
            output.write("opened\n")
        return iter((np.asarray([2], dtype=np.int32), np.asarray([5], dtype=np.int32)))

    def __len__(self) -> int:
        """Return the exact finite yield count."""

        return 2


def test_local_compute_and_subprocess_republication_return_exact_state_refs(
        tmp_path):
    """A direct worker republishes completed content to one explicit Store."""

    state_store = DirStore(tmp_path / "state", query_index="none")
    control_store = DirStore(tmp_path / "control", query_index="none")
    repo = Repo(state_store)
    marker = tmp_path / "source-opens.txt"
    cache = CachedDataset(ObservedDataset(str(marker)))

    invocation = cache.compute(
        codec="numpy",
        managed=ManagedConfig(state_repo=repo, control_store=control_store),
    )
    assert type(invocation) is InvocationResult
    assert invocation.value is None
    initial = invocation.final_state_ref
    assert isinstance(initial, Node)
    assert cache.last_state_ref.graph_equal(initial)
    assert state_store.read_node(_saved_digest(initial), kind="saved").graph_equal(initial)
    execution_state = DirStore.open_existing(tmp_path / "state", query_index="none")
    execution_override = DirStore(tmp_path / "override", query_index="none")
    execution_control = DirStore(tmp_path / "republication-control", query_index="none")
    execution_repo = Repo(execution_state)
    completed = execution_repo.load_state_ref(initial, reuse_live="never")
    assert completed.last_state_ref.graph_equal(initial)
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(
            repo=execution_repo,
            control_store=execution_control,
            return_objects=False,
        ),
    )
    try:
        future = executor.submit(
            completed.compute,
            kwargs={
                "codec": "numpy",
                "store": execution_override,
                "managed": ManagedConfig(
                    state_repo=execution_repo,
                    control_store=execution_control,
                    rerun=True,
                ),
            },
        )
        worker_result = future.result(timeout=60)
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=10)

    assert type(worker_result) is InvocationResult
    assert worker_result.value is None
    republished = worker_result.final_state_ref
    assert isinstance(republished, Node)
    reader_store = DirStore.open_existing(tmp_path / "override", query_index="none")
    assert reader_store.read_node(_saved_digest(republished), kind="saved").graph_equal(republished)
    assert marker.read_text(encoding="ascii").splitlines() == ["opened"]
    restored = Repo(reader_store).load_state_ref(republished, reuse_live="never")
    assert [value.item() for value in restored] == [2, 5]
