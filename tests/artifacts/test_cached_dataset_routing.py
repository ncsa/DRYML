"""CachedDataset completed-state integrity and import-boundary coverage."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from dryml.artifacts import CacheIntegrityError, CachedDataset
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Dataset
from dryml.managed import ManagedConfig


class OneArray(Dataset):
    """Minimal typed source for persisted cache corruption checks."""

    def __init__(self):
        super().__init__(TensorSpec("int32", shape=(1,), backend="numpy"))

    def __iter__(self):
        return iter((np.asarray([1], dtype=np.int32),))

    def __len__(self):
        return Cardinality.finite(1)


class TwoArrays(Dataset):
    """Small source used to prove retired reader generations retain their lease."""

    def __init__(self):
        self.opens = 0
        super().__init__(TensorSpec("int32", shape=(1,), backend="numpy"))

    def __iter__(self):
        self.opens += 1
        return iter((np.asarray([1], dtype=np.int32), np.asarray([2], dtype=np.int32)))

    def __len__(self):
        return Cardinality.finite(2)


def test_late_chunk_corruption_fails_before_affected_yield(tmp_path):
    """Restore reads only manifest metadata and iteration authenticates chunks."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    state = CachedDataset(OneArray()).compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    restored = repo.load_state_ref(state, reuse_live="never")
    chunk = next((tmp_path / "store").rglob("*.npz"))
    chunk.write_bytes(b"not an archive")

    with pytest.raises(CacheIntegrityError):
        list(restored)


def test_completed_cache_uses_snapshot_after_retained_work_is_removed(tmp_path):
    """Completed live and restored readers do not depend on private work files."""
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    cache = CachedDataset(OneArray())
    state = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))

    # Work is private scratch, while the associated StateRef owns the completed bytes.
    for work in (tmp_path / "store" / ".staging").iterdir():
        store.discard_local_state_staging(work)

    assert [item.item() for item in cache] == [1]
    restored = repo.load_state_ref(state, reuse_live="never")
    assert [item.item() for item in restored] == [1]


def test_republication_retires_generation_only_after_cursor_lease_closes(tmp_path):
    """Existing cursors survive a same-codec publication and cross-thread close."""
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    source = TwoArrays()
    cache = CachedDataset(source)
    repo = Repo(DirStore(tmp_path / "store"))
    cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    cursor = cache.iterator()
    assert next(cursor).item() == 1
    old = cache._cache_generation

    cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo, rerun=True))

    assert cache._cache_generation is not old
    assert old.retired and old.leases == 1
    assert next(cursor).item() == 2
    errors = []
    thread = threading.Thread(target=lambda: _close_cursor(cursor, errors))
    thread.start()
    thread.join()
    assert errors == []
    assert old.leases == 0
    assert source.opens == 0


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_current_generation_corruption_preserves_pinned_old_generation_and_other_cache(tmp_path):
    """A lazy corruption invalidates only its current cache generation."""

    from dryml.artifacts.dataset import _CacheGeneration
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    class MutableArray(Dataset):
        """Re-iterable source whose next rerun deliberately changes payload bytes."""

        def __init__(self, value):
            self.value = value
            super().__init__(TensorSpec("int32", shape=(1,), backend="numpy"))

        def __iter__(self):
            return iter((np.asarray([self.value], dtype=np.int32),))

        def __len__(self):
            return Cardinality.finite(1)

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    source = MutableArray(1)
    cache = CachedDataset(source)
    cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    old_cursor = cache.iterator()
    replacement = CachedDataset(MutableArray(2))
    replacement.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    old = cache._cache_generation
    old.retired = True
    cache._retired_cache_generations = (old,)
    cache._cache_payload = replacement._cache_payload
    cache._cache_generation = _CacheGeneration(replacement._cache_payload)
    other = CachedDataset(MutableArray(9))
    other.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    next((Path(cache._cache_payload["source_dir"]) / "chunks").glob("*.npz")).write_bytes(b"corrupt")

    with pytest.raises(CacheIntegrityError):
        list(cache)
    assert [item.item() for item in old_cursor] == [1]
    assert [item.item() for item in other] == [9]


def _close_cursor(cursor, errors):
    """Close a Dataset cursor from a foreign thread for lease-affinity coverage."""

    try:
        cursor.close()
    except Exception as error:  # pragma: no cover - asserted by the caller.
        errors.append(error)


def test_artifacts_import_does_not_import_optional_tensor_frameworks():
    """The NumPy codec keeps optional framework imports lazy."""

    result = subprocess.run(
        [sys.executable, "-c", "import sys; import dryml.artifacts; assert 'tensorflow' not in sys.modules; assert 'torch' not in sys.modules; assert 'jax' not in sys.modules"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_routed_scratch_uses_first_local_store_and_final_replicates_all(tmp_path):
    """Scratch allocation is local while completed snapshots retain every replica."""

    from dryml.core import Repo, SaveRouting, Selector
    from dryml.core.store.dir import DirStore

    scratch = DirStore(tmp_path / "scratch")
    replica = DirStore(tmp_path / "replica")
    repo = Repo(
        [scratch, replica],
        save_routing=SaveRouting(
            ((Selector(CachedDataset), scratch), (Selector(CachedDataset), replica)),
            match_mode="all",
        ),
    )
    cache = CachedDataset(OneArray())

    def stop_after_checkpoint(_obj, _context):
        raise RuntimeError("inspect scratch placement")

    with pytest.raises(RuntimeError, match="scratch placement"):
        cache.compute(codec="numpy", managed=ManagedConfig(
            state_repo=repo, callbacks=[stop_after_checkpoint],
        ))
    token = cache._cache_payload["work_token"]
    assert scratch._resolve_local_state_staging_id(token) is not None
    assert replica._resolve_local_state_staging_id(token) is None

    state = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))

    assert scratch.read_state_ref_record(state.digest()).state_ref == state
    assert replica.read_state_ref_record(state.digest()).state_ref == state
    assert [item.item() for item in Repo(scratch).load_state_ref(state, reuse_live="never")] == [1]
    assert [item.item() for item in Repo(replica).load_state_ref(state, reuse_live="never")] == [1]


def test_explicit_store_override_controls_work_and_completed_placement(tmp_path):
    """An override bypasses routes for both checkpoint work and final publication."""

    from dryml.core import Repo, SaveRouting, Selector
    from dryml.core.store.dir import DirStore

    routed = DirStore(tmp_path / "routed")
    override = DirStore(tmp_path / "override")
    repo = Repo(routed, save_routing=SaveRouting(((Selector(CachedDataset), routed),)))
    state = CachedDataset(OneArray()).compute(
        codec="numpy", store=override, managed=ManagedConfig(state_repo=repo),
    )

    assert override.read_state_ref_record(state.digest()).state_ref == state
    assert routed.read_state_ref_record(state.digest()) is None
