from __future__ import annotations

import multiprocessing
import os
import tracemalloc

import pytest

from dryml.core.store.dir import DirStore
from dryml.formats.ids import content_id
from dryml.formats.refs import format_cdef_id
from dryml.managed import (
    ManagedOperationStore,
    ManagedStateError,
    OperationKey,
    StaleManagedLeaseError,
)
from dryml.records import (
    DataRecord,
    DurableProductWriter,
    ExecutionRecord,
    make_representation_spec,
    RecordIOError,
    StorageRef,
    require_product_integrity,
)


KEY = OperationKey(format_cdef_id("a" * 64), "compute")
FP = "managed-declaration-v1-" + "b" * 64
REPRESENTATION = make_representation_spec(
    "fake.bytes", version="1", storage_kinds=("product-dir",)
)
REPR = REPRESENTATION["id"]
OP = content_id("op", 1, {"method": "compute"})


def _operation(path):
    return ManagedOperationStore(DirStore(path)).operation(KEY, FP)


def _output(realization_id, slot="result"):
    return DataRecord(
        representation_id=REPR,
        storage=(StorageRef.self_product(role=slot),),
        realization_id=realization_id,
        output_slot=slot,
    )


def _execution(realization_id):
    return ExecutionRecord(
        execution_kind="python",
        operation_id=OP,
        backend={"name": "dryml.fake"},
        status="ok",
        realization_id=realization_id,
    )


def _publish_representation(operation):
    operation.managed_store.store.records.write_spec(
        REPRESENTATION, family="representation"
    )


def _crash_after_checkpoint(path):
    operation = _operation(path)
    lease = operation.acquire()
    decision = lease.prepare(resumable=True)
    writer = DurableProductWriter(operation.managed_store.store.records, lease, decision.realization.realization_id)
    writer.write_stream("result", "partial.bin", (b"one", b"two"))
    writer.write_checkpoint_stream("cursor.bin", (b"cursor-", b"2"))
    writer.commit_checkpoint("cursor-v1", metadata={"position": 2})
    os._exit(0)


def test_process_death_preserves_partial_and_checkpoint_for_resume(tmp_path):
    store_path = tmp_path / "death"
    process = multiprocessing.Process(target=_crash_after_checkpoint, args=(store_path,))
    process.start()
    process.join(10)
    assert process.exitcode == 0

    operation = _operation(store_path)
    with operation.acquire() as lease:
        decision = lease.prepare(resumable=True)
        writer = DurableProductWriter(operation.managed_store.store.records, lease, decision.realization.realization_id)
        checkpoint = writer.checkpoint_path(decision.realization.checkpoint_head)
        partials = writer.retained_output_paths("result")
        lease.interrupt(decision.realization.realization_id)

    assert checkpoint.joinpath("cursor.bin").read_bytes() == b"cursor-2"
    assert any(path.name == "partial.bin" for path in partials)


def test_durable_product_capability_is_explicit_on_live_writable_store(tmp_path):
    store = DirStore(tmp_path / "store")

    assert store.supports_store_capability("managed-durable-products-v1")


def test_public_completion_and_activation_require_immutable_publication(tmp_path):
    operation = _operation(tmp_path / "publication-required")
    with operation.acquire() as lease:
        decision = lease.prepare(resumable=False)
        with pytest.raises(TypeError):
            lease.complete(decision.realization.realization_id)
        lease._complete_control_only(decision.realization.realization_id)
        with pytest.raises(ManagedStateError, match="immutable realization record"):
            lease.activate(decision.realization.realization_id)


def test_streaming_writer_has_bounded_python_memory_and_hashes_while_writing(tmp_path):
    operation = _operation(tmp_path / "bounded")
    chunk = b"x" * (64 * 1024)
    with operation.acquire() as lease:
        decision = lease.prepare(resumable=True)
        writer = DurableProductWriter(operation.managed_store.store.records, lease, decision.realization.realization_id)
        tracemalloc.start()
        entry = writer.write_stream("result", "large.bin", (chunk for _ in range(128)))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        lease.interrupt(decision.realization.realization_id)

    assert entry.size == 8 * 1024 * 1024
    assert peak < 2 * 1024 * 1024


def test_multi_output_record_failure_never_activates_and_recovery_adopts_orphans(tmp_path, monkeypatch):
    operation = _operation(tmp_path / "recover")
    _publish_representation(operation)
    with operation.acquire() as lease:
        old = lease.prepare(resumable=False)
        lease._complete_control_only(old.realization.realization_id)
        lease._activate_control_only(old.realization.realization_id)
    old_active = operation.active().realization_id

    lease = operation.acquire()
    decision = lease.prepare(resumable=True, rerun=True)
    realization_id = decision.realization.realization_id
    writer = DurableProductWriter(operation.managed_store.store.records, lease, realization_id)
    writer.write_stream("result", "data.bin", (b"result",))
    writer.write_stream("metrics", "metrics.bin", (b"metrics",))
    original = writer.record_io.write_record
    calls = 0

    def fail_second(record, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RecordIOError("simulated record failure")
        return original(record, **kwargs)

    monkeypatch.setattr(writer.record_io, "write_record", fail_second)
    with pytest.raises(RecordIOError, match="simulated"):
        writer.finalize(
            {"result": _output(realization_id), "metrics": _output(realization_id, "metrics")},
            _execution(realization_id),
            primary_output_slot="result",
            required_output_slots=("result", "metrics"),
            activate=True,
        )
    assert operation.active().realization_id == old_active
    monkeypatch.setattr(writer.record_io, "write_record", original)
    lease.release()

    with operation.acquire() as recovered_lease:
        recovered = DurableProductWriter.recover_finalization(
            operation.managed_store.store.records,
            recovered_lease,
            realization_id,
            activate=False,
        )
        assert operation.active().realization_id == old_active
        recovered_lease.activate(realization_id)

    assert operation.active().realization_id == realization_id
    assert set(recovered.output_records) == {"result", "metrics"}
    for ref in recovered.output_records.values():
        require_product_integrity(
            operation.managed_store.store.records,
            operation.managed_store.store.records.read_record(ref.record_id),
        )


def test_completion_before_activation_keeps_old_active_and_is_idempotently_promotable(tmp_path):
    operation = _operation(tmp_path / "promote")
    _publish_representation(operation)
    with operation.acquire() as lease:
        old = lease.prepare(resumable=False)
        lease._complete_control_only(old.realization.realization_id)
        lease._activate_control_only(old.realization.realization_id)
    old_id = operation.active().realization_id

    with operation.acquire() as lease:
        decision = lease.prepare(resumable=False, rerun=True)
        realization_id = decision.realization.realization_id
        writer = DurableProductWriter(operation.managed_store.store.records, lease, realization_id)
        writer.write_stream("result", "data.bin", (b"complete",))
        result = writer.finalize(
            {"result": _output(realization_id)},
            _execution(realization_id),
            primary_output_slot="result",
            required_output_slots=("result",),
            activate=False,
        )
        assert operation.active().realization_id == old_id
        lease.activate(realization_id)

    assert operation.active().realization_id == realization_id
    assert result.realization_record.record_id


def test_stale_fence_is_checked_before_workspace_mutation(tmp_path):
    operation = _operation(tmp_path / "stale")
    lease = operation.acquire()
    decision = lease.prepare(resumable=True)
    writer = DurableProductWriter(operation.managed_store.store.records, lease, decision.realization.realization_id)
    lease.release()

    with pytest.raises(StaleManagedLeaseError):
        writer.write_stream("result", "forbidden.bin", (b"no",))
    assert not writer.workspace.joinpath("outputs", "result", "forbidden.bin").exists()


def test_attempt_files_are_immutable_and_corrupt_checkpoints_are_rejected(tmp_path):
    operation = _operation(tmp_path / "immutable")
    with operation.acquire() as lease:
        decision = lease.prepare(resumable=True)
        writer = DurableProductWriter(
            operation.managed_store.store.records,
            lease,
            decision.realization.realization_id,
        )
        path = writer.workspace / "outputs" / "result" / "data.bin"
        writer.write_stream("result", "data.bin", (b"original",))
        with pytest.raises(RecordIOError, match="different bytes"):
            writer.write_stream("result", "data.bin", (b"replacement",))
        assert path.read_bytes() == b"original"

        writer.write_checkpoint_stream("cursor.bin", (b"cursor",))
        checkpoint = writer.commit_checkpoint("cursor-v1")
        checkpoint.product_root.joinpath("cursor.bin").write_bytes(b"corrupt")
        with pytest.raises(RecordIOError, match="integrity"):
            writer.checkpoint_path(checkpoint.checkpoint_id)
        lease.interrupt(decision.realization.realization_id)


def test_failed_stream_fragment_is_retained_outside_published_output(tmp_path):
    operation = _operation(tmp_path / "failed-stream")
    _publish_representation(operation)
    with operation.acquire() as lease:
        decision = lease.prepare(resumable=True)
        realization_id = decision.realization.realization_id
        writer = DurableProductWriter(
            operation.managed_store.store.records,
            lease,
            realization_id,
        )

        def failing_chunks():
            yield b"partial"
            raise RuntimeError("stream failed")

        with pytest.raises(RuntimeError, match="stream failed"):
            writer.write_stream("result", "data.bin", failing_chunks())

        assert not writer.workspace.joinpath("outputs", "result", "data.bin").exists()
        assert tuple(writer.workspace.joinpath("partials").rglob("*data.bin*"))

        writer.write_stream("result", "data.bin", (b"complete",))
        result = writer.finalize(
            {"result": _output(realization_id)},
            _execution(realization_id),
            primary_output_slot="result",
            required_output_slots=("result",),
        )

    require_product_integrity(
        operation.managed_store.store.records,
        operation.managed_store.store.records.read_record(
            result.output_records["result"].record_id
        ),
    )
