"""Private retained-cache work-token coverage."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.data import Dataset
from dryml.managed import ManagedConfig


_reject_source_open = False


class EOFCheckpointDataset(Dataset):
    """One-yield source that proves a full checkpoint resumes without reopening."""

    def __init__(self):
        import numpy as np

        self.value = np.asarray([7], dtype=np.int32)
        super().__init__(TensorSpec("int32", shape=(1,), backend="numpy"))

    def __iter__(self):
        if _reject_source_open:
            raise AssertionError("resume reopened an EOF-complete source")
        return iter((self.value,))

    def __len__(self):
        return Cardinality.finite(1)


def test_work_tokens_only_resolve_allocated_direct_staging_children(tmp_path):
    """Private staging IDs reject malformed values without allocation or deletion."""

    store = DirStore(tmp_path / "store")
    path = store.create_local_state_staging()
    token = store._local_state_staging_id(path)

    assert store._resolve_local_state_staging_id(token) == path
    for invalid in ("", ".", "..", "/tmp/elsewhere", token + "00", "g" * 32):
        with pytest.raises(StoreAuthorityError):
            store._resolve_local_state_staging_id(invalid)


def test_eof_checkpoint_resumes_without_reopening_source(tmp_path):
    """An associated post-EOF checkpoint can publish using its retained prefix."""
    from dryml.artifacts import CachedDataset
    from dryml.core import Repo

    global _reject_source_open
    _reject_source_open = False
    repo = Repo(DirStore(tmp_path / "store"))
    cache = CachedDataset(EOFCheckpointDataset())

    def stop_after_association(_obj, _context):
        raise RuntimeError("stop after checkpoint association")

    with pytest.raises(RuntimeError, match="checkpoint association"):
        cache.compute(
            codec="numpy", managed=ManagedConfig(state_repo=repo, callbacks=[stop_after_association]),
        )

    _reject_source_open = True
    try:
        state = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    finally:
        _reject_source_open = False
    restored = repo.load_state_ref(state, reuse_live="never")
    assert [item.item() for item in restored] == [7]


def test_working_snapshot_records_only_the_prior_ready_state_reference(tmp_path):
    """A replacement checkpoint retains prior ready access without copying bytes."""

    import json

    from dryml.artifacts import ArtifactNotReadyError, CachedDataset
    from dryml.artifacts.dataset import _WORK_FORMAT, _WORK_MARKER, _VERSION, _prior_ready_state_ref
    from dryml.core import Repo

    store = DirStore(tmp_path / "store")
    other = DirStore(tmp_path / "other")
    repo = Repo((store, other))
    cache = CachedDataset(EOFCheckpointDataset())
    prior = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    workdir = store.create_local_state_staging()
    token = store._local_state_staging_id(workdir)
    marker = {
        "format": _WORK_FORMAT, "version": _VERSION, "token": token,
        "object_ref_digest": cache.object_ref.digest(), "operation_id": "replace",
        "attempt_id": "attempt", "state": "retained",
    }
    (tmp_path / "store" / ".staging" / token / _WORK_MARKER).write_text(json.dumps(marker))
    cache._cache_payload = {
        "mode": "working", "codec": "numpy", "spec": cache.spec, "expected": 1,
        "count": 0, "work_token": token, "chunks": [],
        "object_ref_digest": cache.object_ref.digest(), "operation_id": "replace",
        "attempt_id": "attempt", "prior_ready_state_digest": prior.digest(),
    }

    checkpoint = repo.save_object(cache, store=store, deep_capture=True)
    restored = repo.load_state_ref(checkpoint, reuse_live="never")

    assert not restored.ready
    assert restored.processed_count == 0
    with pytest.raises(ArtifactNotReadyError):
        _ = restored.spec
    assert restored._cache_payload["prior_ready_state_digest"] == prior.digest()
    assert _prior_ready_state_ref(repo, restored._cache_payload) == prior
    assert [item.item() for item in repo.load_state_ref(_prior_ready_state_ref(repo, restored._cache_payload), reuse_live="never")] == [7]
    restored._cache_payload["prior_ready_state_digest"] = "0" * 64
    with pytest.raises(Exception, match="prior ready state"):
        _prior_ready_state_ref(repo, restored._cache_payload)
    manifest = next((tmp_path / "store").rglob("cache-manifest.json")).read_text()
    assert workdir not in manifest
    assert "source_dir" not in manifest
    assert "_workdir" not in manifest


def test_resume_discards_only_contiguous_unassociated_chunk_tail(tmp_path):
    """A pre-association sealed tail cannot become resumed checkpoint content."""

    from dryml.artifacts.dataset import _discard_unassociated_tail

    store = DirStore(tmp_path / "store")
    workdir = store.create_local_state_staging()
    chunks = tmp_path / "store" / ".staging" / store._local_state_staging_id(workdir) / "data" / "chunks"
    chunks.mkdir()
    (chunks / "chunk-00000000.npz").write_bytes(b"associated")
    (chunks / "chunk-00000001.npz").write_bytes(b"unassociated")
    (chunks / "unrelated.bin").write_bytes(b"preserve")

    _discard_unassociated_tail(workdir, [{"file": "chunk-00000000.npz"}])

    assert (chunks / "chunk-00000000.npz").exists()
    assert not (chunks / "chunk-00000001.npz").exists()
    assert (chunks / "unrelated.bin").exists()


def test_sweep_reclaims_only_explicitly_reclaimable_direct_children(tmp_path):
    """Retained, malformed, and unrelated staging entries survive a bounded sweep."""

    import json

    from dryml.artifacts.dataset import _WORK_FORMAT, _WORK_MARKER, _VERSION, _sweep_reclaimable_work
    from dryml.core import Repo

    store = DirStore(tmp_path / "store")
    other = DirStore(tmp_path / "other")
    repo = Repo((store, other))

    def allocate(state):
        workdir = store.create_local_state_staging()
        token = store._local_state_staging_id(workdir)
        marker = {
            "format": _WORK_FORMAT, "version": _VERSION, "token": token,
            "object_ref_digest": "object", "operation_id": "operation",
            "attempt_id": "attempt", "state": state,
        }
        (tmp_path / "store" / ".staging" / token / _WORK_MARKER).write_text(json.dumps(marker))
        return workdir

    reclaimable = allocate("reclaimable")
    retained = allocate("retained")
    malformed = store.create_local_state_staging()
    (tmp_path / "store" / ".staging" / store._local_state_staging_id(malformed) / _WORK_MARKER).write_text("not json")
    unrelated = tmp_path / "store" / ".staging" / "notes.txt"
    unrelated.write_text("preserve")

    _sweep_reclaimable_work(repo)

    assert not (tmp_path / reclaimable).exists()
    assert (tmp_path / retained).exists()
    assert (tmp_path / malformed).exists()
    assert unrelated.exists()


def test_explicit_rerun_uses_new_token_and_marks_only_superseded_work(tmp_path):
    """A fresh managed attempt cannot append the prior attempt's retained prefix."""

    import json
    import numpy as np

    from dryml.artifacts import CachedDataset
    from dryml.artifacts.dataset import _WORK_MARKER
    from dryml.core import Repo

    class ThreeArrays(Dataset):
        """Source shaped to force a checkpoint after the first sealed chunk."""

        def __init__(self):
            super().__init__(TensorSpec("int32", shape=(1,), backend="numpy"))

        def __iter__(self):
            return iter(tuple(np.asarray([value], dtype=np.int32) for value in range(3)))

        def __len__(self):
            return Cardinality.finite(3)

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    cache = CachedDataset(ThreeArrays())

    def stop_at_checkpoint(_obj, _context):
        raise RuntimeError("leave retained work")

    with pytest.raises(RuntimeError, match="retained work"):
        cache.compute(codec="numpy", target_chunk_bytes=1, managed=ManagedConfig(
            state_repo=repo, callbacks=[stop_at_checkpoint],
        ))
    old_token = cache._cache_payload["work_token"]

    with pytest.raises(RuntimeError, match="retained work"):
        cache.compute(codec="numpy", target_chunk_bytes=1, managed=ManagedConfig(
            state_repo=repo, rerun=True, callbacks=[stop_at_checkpoint],
        ))

    new_token = cache._cache_payload["work_token"]
    assert new_token != old_token
    old_marker = json.loads((tmp_path / "store" / ".staging" / old_token / _WORK_MARKER).read_text())
    new_marker = json.loads((tmp_path / "store" / ".staging" / new_token / _WORK_MARKER).read_text())
    assert old_marker["state"] == "reclaimable"
    assert new_marker["state"] == "retained"


@pytest.mark.parametrize("failure", ("missing", "ambiguous", "wrong-marker"))
def test_resume_rejects_invalid_work_before_source_open(tmp_path, monkeypatch, failure):
    """Missing, ambiguous, or foreign work metadata never opens a resume source."""

    from dryml.artifacts import CacheIntegrityError, CachedDataset
    from dryml.artifacts.dataset import _WORK_MARKER
    from dryml.core import Repo

    class TwoArrays(Dataset):
        """Small source that creates one associated retained prefix."""

        def __init__(self):
            super().__init__(TensorSpec("int32", shape=(1,), backend="numpy"))

        def __iter__(self):
            return iter((np.asarray([1], dtype=np.int32), np.asarray([2], dtype=np.int32)))

        def __len__(self):
            return Cardinality.finite(2)

    store = DirStore(tmp_path / "store")
    other = DirStore(tmp_path / "other")
    repo = Repo((store, other))
    cache = CachedDataset(TwoArrays())

    def stop_after_checkpoint(_obj, _context):
        raise RuntimeError("leave checkpoint")

    with pytest.raises(RuntimeError, match="leave checkpoint"):
        cache.compute(codec="numpy", target_chunk_bytes=1, managed=ManagedConfig(
            state_repo=repo, callbacks=[stop_after_checkpoint],
        ))
    checkpoint = cache.compute.status(state_repo=repo).checkpoint_state_ref
    restored = repo.load_state_ref(checkpoint, reuse_live="never")
    token = restored._cache_payload["work_token"]
    recovered_stores = [store, other]
    if failure == "missing":
        store.discard_local_state_staging(store._resolve_local_state_staging_id(token))
    elif failure == "ambiguous":
        (tmp_path / "other" / ".staging" / token).mkdir(parents=True)
    else:
        marker = tmp_path / "store" / ".staging" / token / _WORK_MARKER
        record = json.loads(marker.read_text(encoding="utf-8"))
        record["attempt_id"] = "another-attempt"
        marker.write_text(json.dumps(record), encoding="utf-8")
    recovered = Repo._for_state_io(tuple(recovered_stores))
    monkeypatch.setattr(CachedDataset, "_load_source", lambda *_: (_ for _ in ()).throw(AssertionError("source opened")))

    assert not restored.ready
    assert restored.processed_count == 1
    with pytest.raises(CacheIntegrityError):
        restored.compute(codec="numpy", target_chunk_bytes=1, managed=ManagedConfig(state_repo=recovered))
    assert not restored.ready
    assert restored.processed_count == 1


def test_completed_cache_restores_in_a_fresh_process_without_source_or_work(tmp_path):
    """A completed StateRef exposes spec and repeated reads in a new interpreter."""

    from dryml.artifacts import CachedDataset
    from dryml.core import Repo
    from dryml.data import ArrayDataset

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    cache = CachedDataset(ArrayDataset(np.asarray([[3], [4]], dtype=np.int32)))
    state = cache.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
    for work in (tmp_path / "store" / ".staging").iterdir():
        store.discard_local_state_staging(work)
    program = """
import sys
from dryml.core import Repo
from dryml.core.store.dir import DirStore
store = DirStore.open_existing(sys.argv[1])
record = store.read_state_ref_record(sys.argv[2])
repo = Repo._for_state_io((store,))
cache = repo.load_state_ref(record.state_ref, reuse_live='never')
assert cache.spec is not None
assert len(cache) == 2
assert [[item.item() for item in cache] for _ in range(2)] == [[3, 4], [3, 4]]
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "store"), state.digest()],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
