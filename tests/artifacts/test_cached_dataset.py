"""CachedDataset lifecycle and finite-admission coverage."""

from __future__ import annotations

import numpy as np
import pytest

from dryml.artifacts import CachedDataset
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Dataset
from dryml.managed import ManagedConfig, ManagedConflictError


class CountingDataset(Dataset):
    """Finite source that records cursor opening for cache admission tests."""

    def __init__(self, values, cardinality):
        self.values = list(values)
        self.cardinality = cardinality
        self.opens = 0
        super().__init__(TensorSpec("int32", shape=(1,), backend="numpy"))

    def __iter__(self):
        self.opens += 1
        return iter(self.values)

    def __len__(self):
        return self.cardinality


def test_cache_rejects_nonfinite_and_mismatched_cardinality_before_completion(tmp_path):
    """Unknown and dishonest cardinality cannot publish partial completed state."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    unknown = CountingDataset([np.asarray([1], dtype=np.int32)], Cardinality.UNKNOWN)
    with pytest.raises(ValueError, match="known finite"):
        CachedDataset(unknown).compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    assert unknown.opens == 0

    short = CountingDataset([np.asarray([1], dtype=np.int32)], Cardinality.finite(2))
    cache = CachedDataset(short)
    with pytest.raises(ValueError, match="exhausted"):
        cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    assert not cache.ready

    long = CountingDataset([np.asarray([1], dtype=np.int32), np.asarray([2], dtype=np.int32)], Cardinality.finite(1))
    with pytest.raises(ValueError, match="more"):
        CachedDataset(long).compute(codec="numpy", managed=ManagedConfig(state_repo=repo))


def test_cache_accepts_exact_integer_cardinality_before_opening_cursor(tmp_path):
    """An ArrayDataset-style exact integer length is a finite cache admission."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    source = CountingDataset([np.asarray([1], dtype=np.int32)], 1)

    cache = CachedDataset(source)
    cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))

    assert cache.ready
    assert [item.item() for item in cache] == [1]


def test_cache_construction_keeps_live_and_saved_sources_inert(tmp_path):
    """Live and saved source references neither traverse nor implicitly publish data."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    source = CountingDataset([np.asarray([1], dtype=np.int32)], Cardinality.finite(1))
    before = tuple(store.iter_state_ref_records())
    live = CachedDataset(source)

    assert source.opens == 0
    assert tuple(store.iter_state_ref_records()) == before
    saved = repo.save_object(source, store=store, deep_capture=True)
    saved_before = tuple(store.iter_state_ref_records())
    referenced = CachedDataset(saved)

    assert source.opens == 0
    assert tuple(store.iter_state_ref_records()) == saved_before
    assert not live.ready and not referenced.ready


def test_same_codec_rerun_republishes_without_source_traversal(tmp_path):
    """Managed rerun keeps a completed matching codec source-independent."""

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    source = CountingDataset([np.asarray([1], dtype=np.int32)], Cardinality.finite(1))
    cache = CachedDataset(source)
    first = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    source.opens = 0

    second = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo, rerun=True))

    assert source.opens == 0
    assert second == cache.last_state_ref
    assert list(cache)[0].item() == 1
    assert first == second


def test_post_completion_cleanup_failure_retries_exact_state_without_source_traversal(tmp_path, monkeypatch):
    """A completed-but-undelivered cache retries publication without reopening source."""

    from dryml.artifacts import dataset as cache_module
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    source = CountingDataset([np.asarray([1], dtype=np.int32)], Cardinality.finite(1))
    cache = CachedDataset(source)
    original = cache_module._mark_work_reclaimable
    calls = 0

    def fail_first_cleanup(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("cleanup delivery failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(cache_module, "_mark_work_reclaimable", fail_first_cleanup)
    with pytest.raises(RuntimeError, match="cleanup delivery failed"):
        cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))

    failed_status = cache.compute.status(state_repo=repo)
    assert cache.ready
    assert failed_status.state == "completed"
    assert failed_status.final_state_ref == cache.last_state_ref
    source.opens = 0

    returned = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo, rerun=True))

    status = cache.compute.status(state_repo=repo)
    assert source.opens == 0
    assert returned == status.final_state_ref == cache.last_state_ref
    assert status.generation > failed_status.generation


def test_same_instance_overlap_is_a_managed_conflict(tmp_path):
    """An admitted cache build fences a second same-instance compute request."""

    import threading

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    source = CountingDataset(
        [np.asarray([1], dtype=np.int32), np.asarray([2], dtype=np.int32)],
        Cardinality.finite(2),
    )
    cache = CachedDataset(source)
    entered = threading.Event()
    release = threading.Event()
    failures = []
    paused = False

    def pause_after_checkpoint(_obj, _context):
        nonlocal paused
        if not paused:
            paused = True
            entered.set()
            assert release.wait(10)

    def first_compute():
        try:
            cache.compute(codec="numpy", target_chunk_bytes=1, managed=ManagedConfig(
                state_repo=repo, callbacks=[pause_after_checkpoint],
            ))
        except BaseException as error:  # pragma: no cover - asserted by caller.
            failures.append(error)

    thread = threading.Thread(target=first_compute)
    thread.start()
    assert entered.wait(10)
    with pytest.raises(ManagedConflictError):
        cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    release.set()
    thread.join(10)
    assert not thread.is_alive()
    assert failures == []
