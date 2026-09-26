"""Representative local and subprocess CachedDataset managed integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dryml.artifacts import CachedDataset
from dryml.core import Executor as CoreExecutor, Repo, StateRef
from dryml.core.execute import CoreOptions
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Dataset
from dryml.execute.subprocess import SubProcessConfig
from dryml.managed import ManagedConfig


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

    initial = cache.compute(
        codec="numpy",
        managed=ManagedConfig(state_repo=repo, control_store=control_store),
    )
    assert type(initial) is StateRef
    assert cache.last_state_ref == initial
    assert state_store.read_state_ref_record(initial.digest()).state_ref == initial
    execution_state = DirStore.open_existing(tmp_path / "state", query_index="none")
    execution_override = DirStore(tmp_path / "override", query_index="none")
    execution_control = DirStore(tmp_path / "republication-control", query_index="none")
    execution_repo = Repo(execution_state)
    completed = execution_repo.load_state_ref(initial, reuse_live="never")
    assert completed.last_state_ref == initial
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
        republished = future.result(timeout=20)
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=10)

    assert type(republished) is StateRef
    reader_store = DirStore.open_existing(tmp_path / "override", query_index="none")
    assert reader_store.read_state_ref_record(republished.digest()).state_ref == republished
    assert marker.read_text(encoding="ascii").splitlines() == ["opened"]
    restored = Repo(reader_store).load_state_ref(republished, reuse_live="never")
    assert [value.item() for value in restored] == [2, 5]
