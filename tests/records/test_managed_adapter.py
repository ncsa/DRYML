from __future__ import annotations

import numpy as np
import pytest

from dryml.artifacts import CachedDataset
from dryml.artifacts.representations import (
    NUMPY_SEQUENCE_KIND,
    PARQUET_KIND,
    PARQUET_REPRESENTATION,
    iter_parquet_sequence,
)
from dryml.core2 import Repo
from dryml.core2.store.dir import DirStore
from dryml.data import ArrayDataset
from dryml.records import (
    AdapterDescriptor,
    AdapterRecord,
    AdapterRegistry,
    AdapterSearchLimits,
    DataRecord,
    LocatedTypedRecord,
    RepresentationRequirement,
    resolve_data_record,
)


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


def _completed_cache(tmp_path):
    store = DirStore(tmp_path / "store")
    source = CountingArrayDataset(np.arange(18, dtype=np.int32).reshape(9, 2))
    cached = CachedDataset(source)
    invocation = cached.compute(
        store=store,
        representation="numpy-sequence",
        shard_rows=3,
    )
    CountingArrayDataset.builds = 0
    return store, cached, invocation


def test_completed_realization_adapts_to_parquet_without_recomputation(tmp_path):
    pytest.importorskip("pyarrow")
    store, cached, invocation = _completed_cache(tmp_path)
    active_before = cached.compute.status(store=store)

    converted = cached.request_representation("parquet", store=store)

    assert converted.status == "ok"
    assert CountingArrayDataset.builds == 0
    target_ref = converted.target_records[-1]
    target = DataRecord.from_envelope(store.records.read_record(target_ref.record_id))
    source = DataRecord.from_envelope(
        store.records.read_record(invocation.outputs["data"].record_id)
    )
    adapter = AdapterRecord.from_envelope(
        store.records.read_record(converted.adapter_records[-1].record_id)
    )
    assert target.realization_id == source.realization_id == invocation.realization_id
    assert target.output_slot == source.output_slot == "data"
    assert target.derived_from == (invocation.outputs["data"].record_id,)
    assert adapter.source_record_id == invocation.outputs["data"].record_id
    assert adapter.target_record_id == target_ref.record_id
    assert cached.compute.results(store=store)["data"] == invocation.outputs["data"]
    assert cached.compute.status(store=store) == active_before
    root = store.records.resolve_storage_ref(target.storage[0], record_id=target_ref.record_id)
    assert [row.tolist() for row in iter_parquet_sequence(root)] == [
        row.tolist() for row in np.arange(18, dtype=np.int32).reshape(9, 2)
    ]

    reused = cached.request_representation(PARQUET_REPRESENTATION, store=store)
    assert reused.status == "ok"
    assert reused.target_records == (target_ref,)
    assert reused.adapter_records == ()
    assert CountingArrayDataset.builds == 0


def test_adapter_failure_and_no_path_leave_active_unchanged(tmp_path):
    store, cached, invocation = _completed_cache(tmp_path)
    active_before = cached.compute.status(store=store)
    failing = AdapterRegistry()

    def fail(_context):
        raise RuntimeError("conversion failed")

    failing.register(
        AdapterDescriptor(
            "test.fail",
            RepresentationRequirement(kind=NUMPY_SEQUENCE_KIND),
            RepresentationRequirement(
                kind=PARQUET_KIND,
                representation_id=PARQUET_REPRESENTATION.id,
            ),
            streaming=True,
            materializes_source=False,
        ),
        runner=fail,
    )

    failed = cached.request_representation("parquet", store=store, adapters=failing)
    missing = cached.request_representation(
        "parquet", store=store, adapters=AdapterRegistry()
    )

    assert failed.status == "failed"
    assert failed.issues[0].code == "adapter_failed"
    assert missing.status == "unsupported"
    assert missing.issues[0].code == "unsupported"
    assert cached.compute.results(store=store)["data"] == invocation.outputs["data"]
    assert cached.compute.status(store=store) == active_before
    assert CountingArrayDataset.builds == 0


def test_adapter_runner_type_error_is_not_masked_by_legacy_signature_retry(tmp_path):
    store, cached, invocation = _completed_cache(tmp_path)
    failing = AdapterRegistry()
    calls = []

    def fail(context, optional=None):
        calls.append(context)
        raise TypeError("runner implementation type error")

    failing.register(
        AdapterDescriptor(
            "test.type-error",
            RepresentationRequirement(kind=NUMPY_SEQUENCE_KIND),
            RepresentationRequirement(
                kind=PARQUET_KIND,
                representation_id=PARQUET_REPRESENTATION.id,
            ),
            streaming=True,
            materializes_source=False,
        ),
        runner=fail,
    )

    result = cached.request_representation("parquet", store=store, adapters=failing)

    assert result.status == "failed"
    assert result.issues[0].message == "runner implementation type error"
    assert len(calls) == 1
    assert cached.compute.results(store=store)["data"] == invocation.outputs["data"]


def test_huge_products_reject_materializing_adapter_but_allow_streaming(tmp_path):
    pytest.importorskip("pyarrow")
    store, cached, _invocation = _completed_cache(tmp_path)
    materializing = AdapterRegistry()
    materializing.register(
        AdapterDescriptor(
            "test.materialize",
            RepresentationRequirement(kind=NUMPY_SEQUENCE_KIND),
            RepresentationRequirement(
                kind=PARQUET_KIND,
                representation_id=PARQUET_REPRESENTATION.id,
            ),
        ),
        runner=lambda context: None,
    )

    rejected = cached.request_representation(
        "parquet",
        store=store,
        adapters=materializing,
        limits=AdapterSearchLimits(max_materialize_bytes=1),
    )

    assert rejected.status == "unsupported"
    assert rejected.issues[0].code == "materializing_adapter_rejected"
    assert cached.request_representation(
        "parquet",
        store=store,
        limits=AdapterSearchLimits(max_materialize_bytes=1),
    ).status == "ok"


def test_managed_data_resolution_selects_bounded_best_cost_path(tmp_path):
    store, cached, _invocation = _completed_cache(tmp_path)
    located, record, _root = cached.active_record(store=store)
    registry = AdapterRegistry()
    target = RepresentationRequirement(
        kind=PARQUET_KIND,
        representation_id=PARQUET_REPRESENTATION.id,
    )
    registry.register(
        AdapterDescriptor(
            "test.direct",
            RepresentationRequirement(kind=NUMPY_SEQUENCE_KIND),
            target,
            cost=10,
            streaming=True,
            materializes_source=False,
        )
    )
    registry.register(
        AdapterDescriptor(
            "test.to_mid",
            RepresentationRequirement(kind=NUMPY_SEQUENCE_KIND),
            RepresentationRequirement(kind="test.mid"),
            cost=1,
            streaming=True,
            materializes_source=False,
        )
    )
    registry.register(
        AdapterDescriptor(
            "test.mid_to_target",
            RepresentationRequirement(kind="test.mid"),
            target,
            cost=1,
            streaming=True,
            materializes_source=False,
        )
    )

    resolved = resolve_data_record(
        Repo(store),
        LocatedTypedRecord(located, record),
        target,
        adapters=registry,
        limits=AdapterSearchLimits(max_steps=3, max_expansions=8),
    )

    assert resolved.status == "requires_adapter"
    assert [step.descriptor.name for step in resolved.adapter_plan.steps] == [
        "test.to_mid",
        "test.mid_to_target",
    ]
    assert resolved.adapter_plan.total_cost == 2

    bounded = AdapterRegistry()
    for name in ("test.to_mid", "test.mid_to_target"):
        descriptor = next(
            item for item in registry.descriptors() if item.name == name
        )
        bounded.register(descriptor)
    rejected = resolve_data_record(
        Repo(store),
        LocatedTypedRecord(located, record),
        target,
        adapters=bounded,
        limits=AdapterSearchLimits(max_steps=1, max_expansions=8),
    )
    assert rejected.status == "unsupported"
    assert rejected.report.issues[0].code == "search_bound_exceeded"


def test_representation_request_does_not_conflate_explicit_rerun(tmp_path):
    pytest.importorskip("pyarrow")
    store, cached, first = _completed_cache(tmp_path)
    converted = cached.request_representation("parquet", store=store)
    target = DataRecord.from_envelope(
        store.records.read_record(converted.target_records[-1].record_id)
    )

    second = cached.compute.rerun(
        store=store,
        representation="numpy-sequence",
        shard_rows=3,
    )

    assert target.realization_id == first.realization_id
    assert second.realization_id != first.realization_id


def test_missing_pyarrow_is_a_structured_unsupported_outcome(tmp_path, monkeypatch):
    store, cached, invocation = _completed_cache(tmp_path)
    from dryml.artifacts.representations import parquet

    def unavailable():
        raise parquet.ParquetUnavailableError("install dryml[parquet]")

    monkeypatch.setattr(parquet, "_load_pyarrow", unavailable)
    result = cached.request_representation("parquet", store=store)

    assert result.status == "unsupported"
    assert result.issues[0].code == "optional_dependency_missing"
    assert cached.compute.results(store=store)["data"] == invocation.outputs["data"]
