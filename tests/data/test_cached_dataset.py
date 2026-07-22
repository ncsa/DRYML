from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import dryml.artifacts
from dryml.artifacts import CachedDataset
from dryml.core import Repo, TensorSpec
from dryml.core.cardinality import Cardinality
from dryml.core.repo import default_repo
from dryml.core.store.dir import DirStore
from dryml.data import ArrayDataset
from dryml.managed import AmbiguousManagedStoreError, ManagedCapabilityError
from dryml.managed.dispatch import ManagedDispatchRequest
from dryml.records import DataRecord, RealizationRecord


class CountingArrayDataset(ArrayDataset):
    builds = 0

    def __init__(self, arrays, *, spec=None, batched=True, validate_lengths=True):
        type(self).builds += 1
        super().__init__(
            arrays,
            spec=spec,
            batched=batched,
            validate_lengths=validate_lengths,
        )


def _values(dataset):
    return [np.asarray(item).tolist() for item in dataset]


def test_cached_dataset_is_canonical_lightweight_dataset_definition(tmp_path):
    source = CountingArrayDataset(np.arange(6, dtype=np.int32).reshape(3, 2))
    cached = CachedDataset(source)

    assert isinstance(cached, dryml.data.Dataset)
    assert CachedDataset is dryml.artifacts.CachedDataset
    assert CachedDataset.__module__ == "dryml.artifacts"
    assert cached.src == source.definition
    assert cached.spec == TensorSpec("int32", shape=(2,), backend="numpy")
    assert cached.__len__() == Cardinality.finite(3)

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    repo.save_definition(cached.definition, main=True)
    CountingArrayDataset.builds = 0

    loaded = repo.load(cached.definition)

    assert isinstance(loaded, CachedDataset)
    assert loaded.src == source.definition
    assert CountingArrayDataset.builds == 0


def test_absent_cache_iteration_raises_without_materializing_source(tmp_path):
    source = CountingArrayDataset(np.arange(4, dtype=np.int32).reshape(2, 2))
    cached = CachedDataset(source)
    CountingArrayDataset.builds = 0

    with pytest.raises(RuntimeError, match="completed.*active"):
        list(cached.view(store=DirStore(tmp_path / "store")))

    assert CountingArrayDataset.builds == 0


def test_first_realization_requires_explicit_compatible_representation(tmp_path):
    store = DirStore(tmp_path / "store")
    cached = CachedDataset(ArrayDataset(np.arange(4, dtype=np.int32).reshape(2, 2)))

    with pytest.raises(ManagedCapabilityError, match="representation"):
        cached.compute(store=store)
    assert not (tmp_path / "store" / ".dryml" / "managed-v1").exists()
    with pytest.raises(ManagedCapabilityError, match="representation"):
        cached.compute(store=store, representation="parquet")
    assert not (tmp_path / "store" / ".dryml" / "managed-v1").exists()

    result = cached.compute("numpy-sequence", store=store)
    assert result.action == "start"
    assert _values(cached.view(store=store)) == [[0, 1], [2, 3]]
    assert cached.compute(store=store).action == "reuse"


def test_dispatched_first_realization_validates_before_control_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    cached = CachedDataset(ArrayDataset(np.arange(4, dtype=np.int32).reshape(2, 2)))
    request = ManagedDispatchRequest(cached.compute, (), {})
    session = None

    try:
        with pytest.raises(ManagedCapabilityError, match="representation"):
            session = request._prepare(SimpleNamespace(store=store))
    finally:
        if session is not None:
            session.lease.release()

    assert not (tmp_path / "store" / ".dryml" / "managed-v1").exists()


def test_empty_cache_with_declared_metadata_completes_and_iterates(tmp_path):
    store = DirStore(tmp_path / "store")
    source = ArrayDataset(
        np.empty((0, 2), dtype=np.float32),
        spec=TensorSpec("float32", shape=(2,), backend="numpy"),
    )
    cached = CachedDataset(
        source,
        spec=source.spec,
        cardinality=Cardinality.finite(0),
    )

    cached.compute(store=store, representation="numpy-sequence")

    assert list(cached.view(store=store)) == []
    assert cached.__len__() == Cardinality.finite(0)


def test_default_and_explicit_repo_views_resolve_one_authority(tmp_path):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    cached = CachedDataset(ArrayDataset(np.array([[1], [2]], dtype=np.int32)))
    cached.compute(store=first, representation="numpy-sequence")

    assert _values(cached.view(store=first)) == [[1], [2]]
    with default_repo(Repo(first)):
        assert _values(cached) == [[1], [2]]
    with default_repo(Repo((first, second))):
        with pytest.raises(AmbiguousManagedStoreError):
            list(cached)


def test_completed_cache_read_never_materializes_source(tmp_path):
    store = DirStore(tmp_path / "store")
    source = CountingArrayDataset(np.array([[7], [8]], dtype=np.int32))
    cached = CachedDataset(source)
    cached.compute(store=store, representation="numpy-sequence")
    Repo(store).save_definition(cached.definition, main=True)
    CountingArrayDataset.builds = 0

    reopened_store = DirStore(store.base_dir)
    loaded = Repo(reopened_store).load(cached.definition)

    assert _values(loaded.view(store=reopened_store)) == [[7], [8]]
    assert CountingArrayDataset.builds == 0


def test_definition_only_recipe_can_compute_in_another_store(tmp_path):
    source = ArrayDataset(np.array([[3], [4]], dtype=np.int32))
    definition = CachedDataset(source).definition
    cached = Repo().load_or_build(definition)
    store = DirStore(tmp_path / "collaborator")

    result = cached.compute(store=store, representation="numpy-sequence")

    assert _values(cached.view(store=store)) == [[3], [4]]
    data = DataRecord.from_envelope(store.records.read_record(result.outputs["data"].record_id))
    realization = RealizationRecord.from_envelope(
        store.records.read_record(result.realization_record_id)
    )
    assert data.realization_id == result.realization_id
    assert data.output_slot == "data"
    assert realization.outputs[0].record_id == result.outputs["data"].record_id


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_active_cache_rejects_missing_or_corrupt_shard(tmp_path, damage):
    store = DirStore(tmp_path / "store")
    cached = CachedDataset(ArrayDataset(np.arange(8, dtype=np.int32)[:, None]))
    cached.compute(
        store=store,
        representation="numpy-sequence",
        shard_rows=3,
    )
    _located, _record, root = cached.active_record(store=store)
    shard = sorted(root.glob("shards/*.npz"))[0]
    if damage == "missing":
        shard.unlink()
    else:
        shard.write_bytes(b"corrupt")

    with pytest.raises(Exception, match="integrity"):
        list(cached.view(store=store))
