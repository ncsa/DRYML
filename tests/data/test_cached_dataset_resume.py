from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dryml.artifacts import CachedDataset
from dryml.core import TensorSpec
from dryml.core.cardinality import Cardinality
from dryml.core.store.dir import DirStore
from dryml.data import ArrayDataset, Dataset, GeneratorDataset, Shuffle
from dryml.data.resume import (
    DatasetResumeCapability,
    ResumableDatasetIterator,
    ResumeMode,
    dataset_resume_capability,
    open_resumable_dataset,
)
from dryml.managed import (
    ControlRequest,
    ManagedCallback,
    ManagedCapabilityError,
    ManagedInterruptedError,
    ManagedRerunRequiredError,
)


def _rows(dataset):
    return [int(np.asarray(item)[0]) for item in dataset]


def _generator():
    return iter(np.arange(8, dtype=np.int32)[:, None])


class CheckpointPrefetch(Dataset):
    """Synthetic one-element prefetch stage with an explicit checkpoint contract."""

    __dryml_dataset_resume_capability__ = DatasetResumeCapability(
        ResumeMode.EXACT,
        ("test.CheckpointPrefetch",),
        "source cursor and prefetched queue are checkpointed",
        "dryml.dataset-pipeline.v1",
    )

    def __init__(self, src):
        self.src = src
        super().__init__(spec=src.spec)

    def __iter__(self):
        iterator = iter(self.src)
        queued = []
        try:
            queued.append(next(iterator))
        except StopIteration:
            return
        while queued:
            value = queued.pop(0)
            try:
                queued.append(next(iterator))
            except StopIteration:
                pass
            yield value

    def __len__(self):
        return self.src.__len__()

    def __dryml_open_resumable__(self, state):
        return CheckpointPrefetchIterator(self, state)


class CheckpointPrefetchIterator(ResumableDatasetIterator):
    def __init__(self, dataset, state):
        if state is None:
            self.source = open_resumable_dataset(dataset.src)
            self.queued = []
            try:
                self.queued.append(next(self.source))
            except StopIteration:
                pass
        else:
            assert set(state) == {"kind", "source", "queued"}
            assert state["kind"] == "checkpoint-prefetch"
            self.source = open_resumable_dataset(dataset.src, state["source"])
            self.queued = list(state["queued"])

    def __next__(self):
        if not self.queued:
            raise StopIteration
        value = self.queued.pop(0)
        try:
            self.queued.append(next(self.source))
        except StopIteration:
            pass
        return value

    def checkpoint(self):
        return {
            "kind": "checkpoint-prefetch",
            "source": self.source.checkpoint(),
            "queued": list(self.queued),
        }


class OpaquePrefetch(Dataset):
    """Synthetic stateful stage that deliberately has no checkpoint contract."""

    def __init__(self, src):
        self.src = src
        super().__init__(spec=src.spec)

    def __iter__(self):
        yield from self.src

    def __len__(self):
        return self.src.__len__()


def test_interrupted_shuffled_cache_resumes_exact_order_without_duplicates(tmp_path):
    store = DirStore(tmp_path / "store")
    shuffled = Shuffle(
        ArrayDataset(np.arange(40, dtype=np.int32)[:, None]),
        buffer_size=7,
        seed=23,
    )
    source = CheckpointPrefetch(shuffled)
    expected = _rows(source)
    cached = CachedDataset(
        source,
        spec=source.spec,
        cardinality=Cardinality.finite(40),
    )
    interrupted = {"done": False}

    def stop_after_first_shard(event):
        if event.kind == "progress" and event.progress_snapshot.current >= 5 and not interrupted["done"]:
            interrupted["done"] = True
            return ControlRequest.INTERRUPT

    callback = ManagedCallback(
        stop_after_first_shard,
        controls={ControlRequest.INTERRUPT},
        fail_soft=True,
    )
    with pytest.raises(ManagedInterruptedError):
        cached.compute(
            store=store,
            representation="numpy-sequence",
            shard_rows=5,
            callbacks=(callback,),
        )

    status = cached.compute.status(store=store)
    assert status.status == "interrupted"
    assert status.checkpoint_head is not None
    with pytest.raises(RuntimeError, match="completed.*active"):
        list(cached.view(store=store))

    with pytest.raises(ManagedCapabilityError, match="shard_rows"):
        cached.compute(store=store, shard_rows=6)
    assert len(cached.compute.history(store=store)[0].attempt_ids) == 1

    result = cached.compute(store=store)

    assert result.action == "resume"
    assert _rows(cached.view(store=store)) == expected
    history = cached.compute.history(store=store)
    assert len(history) == 1
    assert len(history[0].attempt_ids) == 2


def test_replay_only_pipeline_reports_capability_and_requires_rerun(tmp_path):
    source = GeneratorDataset(_generator, spec=TensorSpec("int32", shape=(1,)))
    cached = CachedDataset(source)
    capability = dataset_resume_capability(source.definition)

    assert capability.mode is ResumeMode.REPLAY
    assert "replay" in capability.diagnostic

    class FailOnce:
        def __init__(self):
            self.failed = False

        def __call__(self, event):
            if event.kind == "progress" and not self.failed:
                self.failed = True
                raise RuntimeError("observer failure")

    callback = ManagedCallback(FailOnce(), fail_soft=True)
    store = DirStore(tmp_path / "store")
    # A fail-soft observer demonstrates that replay-only computation can still
    # complete, but exact-resume guarantees and strict callbacks are unavailable.
    cached.compute(
        store=store,
        representation="numpy-sequence",
        callbacks=(callback,),
    )
    assert _rows(cached.view(store=store)) == list(range(8))


def test_non_resumable_pipeline_rejects_strict_callback_before_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    source = GeneratorDataset(_generator, spec=TensorSpec("int32", shape=(1,)))
    cached = CachedDataset(source)
    callback = ManagedCallback(lambda event: None)

    with pytest.raises(ManagedCapabilityError, match="strict callback"):
        cached.compute(
            store=store,
            representation="numpy-sequence",
            callbacks=(callback,),
        )

    assert not Path(store.managed_control_root()).exists()


def test_unsupported_stateful_stage_rejects_exact_guarantee_before_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    source = OpaquePrefetch(ArrayDataset(np.arange(5, dtype=np.int32)[:, None]))
    cached = CachedDataset(
        source,
        spec=source.spec,
        cardinality=Cardinality.finite(5),
    )

    capability = dataset_resume_capability(source.definition)
    assert capability.mode is ResumeMode.NONE
    assert "no checkpoint contract" in capability.diagnostic

    with pytest.raises(ManagedCapabilityError, match="strict callback"):
        cached.compute(
            store=store,
            representation="numpy-sequence",
            callbacks=(ManagedCallback(lambda event: None),),
        )
    assert not Path(store.managed_control_root()).exists()


def test_graceful_stop_is_rejected_before_cache_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    cached = CachedDataset(ArrayDataset(np.arange(6, dtype=np.int32)[:, None]))
    callback = ManagedCallback(
        lambda event: ControlRequest.GRACEFUL_STOP,
        controls={ControlRequest.GRACEFUL_STOP},
        fail_soft=True,
    )

    with pytest.raises(ManagedCapabilityError, match="early completion"):
        cached.compute(
            store=store,
            representation="numpy-sequence",
            callbacks=(callback,),
        )

    assert not Path(store.managed_control_root()).exists()


def test_non_resumable_failure_requires_explicit_rerun_and_keeps_old_active(tmp_path):
    store = DirStore(tmp_path / "store")
    source = GeneratorDataset(_generator, spec=TensorSpec("int32", shape=(1,)))
    cached = CachedDataset(source)
    first = cached.compute(store=store, representation="numpy-sequence")

    interrupted = {"done": False}

    def interrupt_rerun(event):
        if event.kind == "progress" and not interrupted["done"]:
            interrupted["done"] = True
            return ControlRequest.INTERRUPT

    callback = ManagedCallback(
        interrupt_rerun,
        controls={ControlRequest.INTERRUPT},
        fail_soft=True,
    )
    with pytest.raises(ManagedInterruptedError):
        cached.compute.rerun(
            store=store,
            representation="numpy-sequence",
            shard_rows=2,
            callbacks=(callback,),
        )

    assert cached.compute.results(store=store)["data"].record_id == first.outputs["data"].record_id
    assert _rows(cached.view(store=store)) == list(range(8))
    with pytest.raises(ManagedRerunRequiredError):
        cached.compute(store=store)

    second = cached.compute.rerun(store=store, representation="numpy-sequence")
    assert second.realization_id != first.realization_id
