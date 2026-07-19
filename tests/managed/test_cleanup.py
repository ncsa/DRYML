from __future__ import annotations

from dataclasses import replace

import pytest

from dryml.core2 import Object
from dryml.core2.store.dir import DirStore
from dryml.managed import (
    ManagedCleanupRefusedError,
    ManagedOperationStore,
    ManagedOutput,
    ManagedStateError,
    OperationKey,
    declaration_fingerprint,
    execute_cleanup,
    managed,
    plan_cleanup,
    resume_cleanup,
)
from dryml.records import make_record


class CleanupProducer(Object):
    @managed(
        outputs=(ManagedOutput("result", primary=True, kind="data"),),
        resumable=True,
        checkpoint_schema="u8-v1",
    )
    def compute(self):
        from dryml.managed import current_operation_context
        from dryml.records import make_representation_spec

        representation = make_representation_spec(
            "u8.cleanup", version="1", storage_kinds=("product-dir",)
        )
        current_operation_context().write_output(
            "result", "value.bin", (b"cleanup",), representation=representation
        )


class CleanupConsumer(Object):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def __dryml_managed_inputs__(self, method, args, kwargs):
        return (self.source,)

    @managed(outputs=(ManagedOutput("result", primary=True, kind="data"),))
    def compute(self):
        from dryml.managed import current_operation_context
        from dryml.records import make_representation_spec

        representation = make_representation_spec(
            "u8.cleanup-consumer", version="1", storage_kinds=("product-dir",)
        )
        current_operation_context().write_output(
            "result", "value.bin", (b"consumer",), representation=representation
        )


def _operation(store, producer):
    descriptor = type(producer).__dict__["compute"]
    key = OperationKey.from_producer(producer, "compute")
    fingerprint = declaration_fingerprint(
        "compute", descriptor.declaration, producer=producer.definition
    )
    return ManagedOperationStore(store).operation(key, fingerprint)


def test_cleanup_refuses_active_leased_checkpoint_and_consumed_state(tmp_path):
    active_store = DirStore(tmp_path / "active")
    active_producer = CleanupProducer()
    active = active_producer.compute(store=active_store)
    with pytest.raises(ManagedCleanupRefusedError, match="active"):
        plan_cleanup(
            active_store,
            active_producer.compute.result,
            realization_ids=(active.realization_id,),
        )

    leased_store = DirStore(tmp_path / "leased")
    leased_producer = CleanupProducer()
    operation = _operation(leased_store, leased_producer)
    lease = operation.acquire()
    pending = lease.prepare(resumable=True)
    try:
        with pytest.raises(ManagedCleanupRefusedError, match="leased"):
            plan_cleanup(
                leased_store,
                leased_producer.compute.result,
                realization_ids=(pending.realization.realization_id,),
            )
    finally:
        lease.release()

    with operation.acquire() as resumed:
        resumed.interrupt(
            pending.realization.realization_id,
            checkpoint_head="checkpoint-v1-retained",
        )
    with pytest.raises(ManagedCleanupRefusedError, match="checkpoint"):
        plan_cleanup(
            leased_store,
            leased_producer.compute.result,
            realization_ids=(pending.realization.realization_id,),
        )

    consumed_store = DirStore(tmp_path / "consumed")
    consumed_producer = CleanupProducer()
    consumed = consumed_producer.compute(store=consumed_store)
    CleanupConsumer(consumed_producer.compute.result).compute(store=consumed_store)
    consumed_producer.compute.rerun(store=consumed_store)
    with pytest.raises(ManagedCleanupRefusedError, match="referenced"):
        plan_cleanup(
            consumed_store,
            consumed_producer.compute.result,
            realization_ids=(consumed.realization_id,),
        )


def test_cleanup_crash_resume_deletes_only_declared_abandoned_state(
    tmp_path, monkeypatch
):
    import dryml.managed.cleanup as cleanup_module

    store = DirStore(tmp_path / "store")
    producer = CleanupProducer()
    operation = _operation(store, producer)
    with operation.acquire() as lease:
        first = lease.prepare(resumable=True)
        lease.interrupt(first.realization.realization_id)
        second = lease.prepare(resumable=True, rerun=True)
        lease.interrupt(second.realization.realization_id)

    plan = plan_cleanup(
        store,
        producer.compute.result,
        realization_ids=(first.realization.realization_id,),
    )
    with pytest.raises(ManagedStateError, match="path"):
        replace(plan, paths=("../outside",))
    concurrent = operation.acquire()
    try:
        with pytest.raises(ManagedCleanupRefusedError, match="leased"):
            execute_cleanup(store, plan)
    finally:
        concurrent.release()
    original = cleanup_module._delete_declared_path
    calls = 0

    def crash_once(root, relative_path):
        nonlocal calls
        calls += 1
        original(root, relative_path)
        if calls == 1:
            raise OSError("simulated cleanup crash")

    monkeypatch.setattr(cleanup_module, "_delete_declared_path", crash_once)
    with pytest.raises(OSError, match="simulated"):
        execute_cleanup(store, plan)

    monkeypatch.setattr(cleanup_module, "_delete_declared_path", original)
    report = resume_cleanup(store, plan.cleanup_id)
    repeated = resume_cleanup(store, plan.cleanup_id)

    assert report == repeated
    assert first.realization.realization_id not in {
        state.realization_id for state in operation.history()
    }
    assert second.realization.realization_id in {
        state.realization_id for state in operation.history()
    }


def test_cleanup_deletes_inactive_completed_records_and_products_only_explicitly(
    tmp_path,
):
    store = DirStore(tmp_path / "store")
    producer = CleanupProducer()
    inactive = producer.compute(store=store)
    active = producer.compute.rerun(store=store)

    assert store.records.has_record(inactive.realization_record_id)
    plan = plan_cleanup(
        store,
        producer.compute.result,
        realization_ids=(inactive.realization_id,),
    )
    execute_cleanup(store, plan)

    operation = _operation(store, producer)
    assert operation.active().realization_id == active.realization_id
    assert inactive.realization_id not in {
        state.realization_id for state in operation.history()
    }
    assert not store.records.has_record(inactive.realization_record_id)
    assert not store.records.product_root(inactive.outputs["result"].record_id).exists()


def test_cleanup_resume_refuses_realization_activated_after_partial_deletion(
    tmp_path, monkeypatch
):
    import dryml.managed.cleanup as cleanup_module

    store = DirStore(tmp_path / "store")
    producer = CleanupProducer()
    selected = producer.compute(store=store)
    producer.compute.rerun(store=store)
    operation = _operation(store, producer)
    plan = plan_cleanup(
        store,
        producer.compute.result,
        realization_ids=(selected.realization_id,),
    )
    original = cleanup_module._delete_declared_path
    deleted = []

    def crash_after_activation_history(root, relative_path):
        if "/activations/" not in relative_path:
            return
        original(root, relative_path)
        deleted.append(relative_path)
        raise OSError("simulated partial cleanup crash")

    monkeypatch.setattr(
        cleanup_module,
        "_delete_declared_path",
        crash_after_activation_history,
    )
    with pytest.raises(OSError, match="simulated partial"):
        execute_cleanup(store, plan)
    assert len(deleted) == 1

    monkeypatch.setattr(cleanup_module, "_delete_declared_path", original)
    with operation.acquire() as lease:
        lease.activate(selected.realization_id)
    assert operation.active().realization_id == selected.realization_id

    with pytest.raises(ManagedCleanupRefusedError, match="active"):
        resume_cleanup(store, plan.cleanup_id)
    assert operation.active().realization_id == selected.realization_id
    assert store.records.has_record(selected.realization_record_id)


def test_cleanup_resume_refuses_reference_added_after_partial_deletion(
    tmp_path, monkeypatch
):
    import dryml.managed.cleanup as cleanup_module

    store = DirStore(tmp_path / "store")
    producer = CleanupProducer()
    selected = producer.compute(store=store)
    producer.compute.rerun(store=store)
    plan = plan_cleanup(
        store,
        producer.compute.result,
        realization_ids=(selected.realization_id,),
    )
    original = cleanup_module._delete_declared_path

    def crash_after_activation_history(root, relative_path):
        if "/activations/" not in relative_path:
            return
        original(root, relative_path)
        raise OSError("simulated partial cleanup crash")

    monkeypatch.setattr(
        cleanup_module,
        "_delete_declared_path",
        crash_after_activation_history,
    )
    with pytest.raises(OSError, match="simulated partial"):
        execute_cleanup(store, plan)
    monkeypatch.setattr(cleanup_module, "_delete_declared_path", original)

    output_id = selected.outputs["result"].record_id
    external = store.records.write_record(
        make_record(kind="adapter", payload={"record_id": output_id})
    )
    with pytest.raises(ManagedCleanupRefusedError, match="referenced"):
        resume_cleanup(store, plan.cleanup_id)
    assert store.records.has_record(external.record_id)
    assert store.records.has_record(selected.realization_record_id)
