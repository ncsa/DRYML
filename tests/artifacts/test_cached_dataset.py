"""CachedDataset lifecycle and finite-admission coverage."""

from __future__ import annotations

import numpy as np
import pytest

from dryml.artifacts import CachedDataset
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Dataset
from dryml.managed import ManagedConfig, ManagedConflictError

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

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
    assert first.value is second.value is None
    assert second.final_state_ref.graph_equal(cache.last_state_ref)
    assert list(cache)[0].item() == 1
    assert first.final_state_ref.graph_equal(second.final_state_ref)
    assert first.execution.run_id != second.execution.run_id


def test_live_rerun_keeps_borrowed_final_chunks_until_republication(tmp_path, monkeypatch):
    """Retire the old final association without reclaiming the executing live input."""
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.managed.control import ManagedControlStore
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    cache = CachedDataset(CountingDataset([np.asarray([1], dtype=np.int32)], 1))
    config = ManagedConfig(state_repo=repo)
    first = cache.compute(codec="numpy", managed=config)
    original = CachedDataset.save_state_to_dir_imp
    observed = []

    def capture(self, directory, *, codec):
        control = ManagedControlStore(store, repo)
        roots = control.list_checkpoints(first.execution.operation_id)
        old = [root for root in roots if root.run_id == first.execution.run_id and root.kind == "final"]
        assert len(old) == 1 and old[0].disposition == "release_pending"
        assert store.delete_state(first.final_state_ref.state.key).status == "retained"
        observed.append(True)
        return original(self, directory, codec=codec)

    monkeypatch.setattr(CachedDataset, "save_state_to_dir_imp", capture)
    second = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo, rerun=True))
    assert observed == [True]
    assert second.value is None
    assert list(cache)[0].item() == 1
    roots = ManagedControlStore(store, repo).list_checkpoints(first.execution.operation_id)
    assert not any(root.run_id == first.execution.run_id and root.kind == "final" for root in roots)


@pytest.mark.parametrize("cleanup_fails, notes_available", ((False, True), (True, True), (True, False)))
def test_failed_live_rerun_retires_its_superseded_final_grant(tmp_path, monkeypatch, cleanup_fails, notes_available):
    """Preserve capture failure and grant debt with or without exception-note support."""
    from dryml.core import Repo
    from dryml.core.repo import RepoSaveError, RepoLoadError
    from dryml.core.store.dir import DirStore
    from dryml.managed.control import ManagedControlStore
    from dryml.core.store.retention import RetentionAuthority
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    cache = CachedDataset(CountingDataset([np.asarray([1], dtype=np.int32)], 1))
    first = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    release = RetentionAuthority.release
    if cleanup_fails:
        if not notes_available:
            monkeypatch.setattr(RepoSaveError, "add_note", None, raising=False)
        old_final, = (root for root in ManagedControlStore(store, repo).list_checkpoints(first.execution.operation_id)
                      if root.kind == "final")
        def fail_old_final(self, token, stores):
            if token.replace("-", "") == old_final.token:
                raise OSError("grant retirement failed")
            return release(self, token, stores)
        monkeypatch.setattr(RetentionAuthority, "release", fail_old_final)
    monkeypatch.setattr(CachedDataset, "save_state_to_dir_imp",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("rerun capture failed")))
    with pytest.raises(RepoSaveError) as caught:
        cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo, rerun=True))
    assert cache.compute.status(state_repo=repo).state != "completed"
    assert first.value is None
    roots = ManagedControlStore(store, repo).list_checkpoints(first.execution.operation_id)
    if cleanup_fails:
        assert isinstance(caught.value.cleanup_error, OSError)
        assert any(root.disposition == "release_pending" for root in roots)
        assert store.delete_state(first.final_state_ref.state.key).status == "retained"
        monkeypatch.setattr(RetentionAuthority, "release", release)
        monkeypatch.setattr(CachedDataset, "save_state_to_dir_imp",
                            lambda *a, **k: pytest.fail("grant recovery captured payload"))
        recovered = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
        assert recovered.disposition == "projection_reconciled"
        roots = ManagedControlStore(store, repo).list_checkpoints(first.execution.operation_id)
    assert not any(root.run_id == first.execution.run_id and root.kind == "final" for root in roots)
    with pytest.raises(RepoLoadError):
        repo.load_state_ref(first.final_state_ref, reuse_live="never")


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
    assert failed_status.final.graph_equal(cache.last_state_ref)
    source.opens = 0

    returned = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo, rerun=True))

    status = cache.compute.status(state_repo=repo)
    assert source.opens == 0
    assert returned.final_state_ref.graph_equal(status.final)
    assert status.final.graph_equal(cache.last_state_ref)
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
            assert release.wait(60)

    def first_compute():
        try:
            cache.compute(codec="numpy", target_chunk_bytes=1, managed=ManagedConfig(
                state_repo=repo, callbacks=[pause_after_checkpoint],
            ))
        except BaseException as error:  # pragma: no cover - asserted by caller.
            failures.append(error)

    thread = threading.Thread(target=first_compute)
    thread.start()
    try:
        assert entered.wait(60)
        with pytest.raises(ManagedConflictError):
            cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    finally:
        release.set()
        thread.join(180)
    assert not thread.is_alive()
    assert failures == []


def test_completed_cache_restores_example_metadata_without_reading_chunks(tmp_path, monkeypatch):
    """Completed cache count metadata remains usable before a lazy chunk read."""
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    cache = CachedDataset(CountingDataset([np.asarray([1], dtype=np.int32)], Cardinality.finite(1)))
    state = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo)).final_state_ref
    restored = repo.load_state_ref(state, reuse_live="never")
    monkeypatch.setattr(restored, "_iter_payload", lambda payload: (_ for _ in ()).throw(AssertionError("no chunk read")))

    assert restored.example_cardinality() == Cardinality.finite(1)


def _cache_cleanup_process(store, result, *, cleanup=False):
    """Probe real native exclusion and optionally retire roots in a fresh process."""
    import json
    import subprocess
    import sys

    key = result.final_state_ref.state.key
    completed = subprocess.run([
        sys.executable, "-c", """
import json, sys
from dryml.core import Repo, StateKey
from dryml.core.store.dir import DirStore
from dryml.managed.control import ManagedControlStore
store = DirStore.open_existing(sys.argv[1], query_index="none")
repo = Repo(store)
key = StateKey(sys.argv[2], sys.argv[3])
with store.state_leases((key,), exclusive=True, blocking=False) as acquired:
    native_available = acquired
outcomes = (ManagedControlStore(store, repo).cleanup_checkpoints(sys.argv[4])
            if sys.argv[5] == "True" else ())
print(json.dumps({"native_available": native_available,
                  "cleanup": [o.status for o in outcomes if o.key == key],
                  "delete": store.delete_state(key).status}))
repo.close(flush=False)
""", str(store.base_dir), key.cdef_digest, key.codec_hash,
        result.execution.operation_id, str(cleanup),
    ], check=True, text=True, capture_output=True, timeout=60)
    return json.loads(completed.stdout)


@pytest.mark.parametrize("finish", ("exhaustion", "close", "error", "cross-thread", "unstarted", "abandoned"))
def test_cache_cursor_holds_native_lease_until_termination(tmp_path, monkeypatch, finish):
    """Second-process cleanup cannot reclaim chunks until this cursor releases use."""
    from concurrent.futures import ThreadPoolExecutor
    from dryml.artifacts import CacheIntegrityError
    from dryml.artifacts import dataset as cache_module
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    cache = CachedDataset(CountingDataset([np.asarray([i], dtype=np.int32) for i in range(3)], 3))
    result = cache.compute(codec="numpy", target_chunk_bytes=1, managed=ManagedConfig(state_repo=repo))
    assert len(cache._cache_payload["chunks"]) == 3
    cursor = cache.iterator()
    try:
        if finish != "unstarted":
            assert next(cursor).item() == 0
        protected = _cache_cleanup_process(store, result, cleanup=True)
        assert protected["native_available"] is False
        assert "in-use" in protected["cleanup"]
        assert protected["delete"] == "in-use"
        if finish == "exhaustion":
            assert [value.item() for value in cursor] == [1, 2]
            assert cursor.observed_eof
        elif finish == "error":
            def fail_chunk(*args):
                raise CacheIntegrityError("injected chunk decode failure")
            monkeypatch.setattr(cache_module, "_read_chunk", fail_chunk)
            with pytest.raises(CacheIntegrityError, match="injected"):
                next(cursor)
        elif finish == "cross-thread":
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(cursor.close).result(timeout=30)
        elif finish == "abandoned":
            import gc
            import weakref
            reference = weakref.ref(cursor)
            cursor = None
            gc.collect()
            assert reference() is None
        else:
            cursor.close()
        released = _cache_cleanup_process(store, result)
        assert released["native_available"] is True
        assert released["delete"] == "removed"
    finally:
        if cursor is not None:
            cursor.close()
        repo.close(flush=False)


def test_cache_concurrent_cursors_own_independent_native_leases(tmp_path):
    """Non-LIFO close and replacement cannot release another cursor's native use."""
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    cache = CachedDataset(CountingDataset([np.asarray([i], dtype=np.int32) for i in range(2)], 2))
    first = cache.compute(codec="numpy", target_chunk_bytes=1, managed=ManagedConfig(state_repo=repo))
    with cache.iterator() as older, cache.iterator() as newer:
        assert next(older).item() == next(newer).item() == 0
        old = cache._cache_generation
        current = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo, rerun=True))
        assert old.retired and old.leases == 2
        with cache.iterator() as replacement:
            older.close()
            assert old.leases == 1
            protected = _cache_cleanup_process(store, current, cleanup=True)
            assert protected["native_available"] is False
            assert protected["delete"] == "in-use"
            assert [value.item() for value in newer] == [1]
            assert old.leases == 0 and not old.payload
            assert _cache_cleanup_process(store, first)["delete"] == "in-use"
            assert [value.item() for value in replacement] == [0, 1]
    assert _cache_cleanup_process(store, current)["delete"] == "removed"
    repo.close(flush=False)


@pytest.mark.parametrize("restore", ("load", "into"))
def test_restored_cache_reacquires_native_full_key_use(tmp_path, restore):
    """Both Core restoration paths detach exact source evidence for later cursors."""
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    cache = CachedDataset(CountingDataset([np.asarray([i], dtype=np.int32) for i in range(2)], 2))
    result = cache.compute(codec="numpy", target_chunk_bytes=1, managed=ManagedConfig(state_repo=repo))
    if restore == "load":
        cache = repo.load_state_ref(result.final_state_ref, reuse_live="never")
    else:
        repo.restore_state_ref_into(cache, result.final_state_ref)
    generation = cache._cache_generation
    assert generation.leases == 0
    with cache.iterator() as cursor:
        assert next(cursor).item() == 0
        protected = _cache_cleanup_process(store, result, cleanup=True)
        assert protected["native_available"] is False
        assert protected["delete"] == "in-use"
        assert [value.item() for value in cursor] == [1]
    assert _cache_cleanup_process(store, result)["delete"] == "removed"
    repo.close(flush=False)


@pytest.mark.parametrize("stale", ("unavailable", "replaced", "closed"))
def test_cache_cursor_revalidates_stale_source_before_decoding(tmp_path, monkeypatch, stale):
    """Retained metadata/path evidence cannot authorize stale Store-backed reads."""
    from pathlib import Path
    from dryml.artifacts import CacheIntegrityError
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    cache = CachedDataset(CountingDataset([np.asarray([1], dtype=np.int32)], 1))
    result = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    restored = repo.load_state_ref(result.final_state_ref, reuse_live="never")
    assert [value.item() for value in cache] == [1]
    assert [value.item() for value in restored] == [1]
    if stale == "unavailable":
        assert _cache_cleanup_process(store, result, cleanup=True)["delete"] in {"removed", "absent"}
    elif stale == "replaced":
        Path(store.base_dir).rename(tmp_path / "original-store")
        replacement = DirStore(tmp_path / "store", query_index="none")
        assert replacement.store_uuid != store.store_uuid
        replacement.close()
    else:
        store.close()
    for reader in (cache, restored):
        monkeypatch.setattr(reader, "_iter_payload", lambda payload: pytest.fail("stale source reached decoder"))
        generation = reader._cache_generation
        with pytest.raises(CacheIntegrityError, match="unavailable|domain"):
            reader.iterator()
        assert generation.leases == 0
    repo.close(flush=False)


@pytest.mark.parametrize("copy_error", (False, True))
def test_cache_capture_holds_fresh_native_source_lease(tmp_path, monkeypatch, copy_error):
    """Deferred copying protects the full source key and refuses unavailable bytes."""
    from dryml.artifacts import CacheIntegrityError
    from dryml.artifacts import dataset as cache_module
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    cache = CachedDataset(CountingDataset([np.asarray([1], dtype=np.int32)], 1))
    result = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    restored = repo.load_state_ref(result.final_state_ref, reuse_live="never")
    copy = cache_module.shutil.copyfile
    observed = []

    def protected_copy(source, target):
        outcomes = _cache_cleanup_process(store, result, cleanup=True)
        assert outcomes["native_available"] is False
        assert outcomes["delete"] == "in-use"
        observed.append(True)
        if copy_error:
            raise OSError("injected copy failure")
        return copy(source, target)

    monkeypatch.setattr(cache_module.shutil, "copyfile", protected_copy)
    destination = tmp_path / "capture"
    destination.mkdir()
    if copy_error:
        with pytest.raises(OSError, match="injected copy failure"):
            restored.save_state_to_dir(str(destination), codec=restored.state_codec)
    else:
        restored.save_state_to_dir(str(destination), codec=restored.state_codec)
    assert observed == [True]
    assert _cache_cleanup_process(store, result)["delete"] == "removed"
    with pytest.raises(CacheIntegrityError, match="unavailable"):
        restored.save_state_to_dir(str(destination), codec=restored.state_codec)
    repo.close(flush=False)
