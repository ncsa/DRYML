from __future__ import annotations

from dataclasses import replace

import pytest

from dryml.core import Object
from dryml.core.store.dir import DirStore
from dryml.managed import (
    ConcurrentManagedActivationError,
    ManagedOutput,
    MissingManagedOutputError,
    resolve_inputs,
    managed,
)
from dryml.records import (
    DataRecord,
    DurableProductWriter,
    ExecutionRecord,
    StorageRef,
    make_representation_spec,
)
from dryml.operations import attach_operation_id, make_method_call_spec


REPRESENTATION = make_representation_spec("fake.bytes", version="1", storage_kinds=("product-dir",))


class Producer(Object):
    def __init__(self, name="producer"):
        super().__init__()
        self.name = name

    @managed(outputs=(ManagedOutput("result", primary=True, kind="data"),))
    def compute(self):
        raise AssertionError("resolution must not compute a missing dependency")


def _publish(ref, store, value: bytes):
    from dryml.managed import ManagedOperationStore, OperationKey, declaration_fingerprint

    descriptor = Producer.__dict__["compute"]
    key = OperationKey.from_producer(ref.producer, ref.method)
    fingerprint = declaration_fingerprint(ref.method, descriptor.declaration, producer=ref.producer)
    operation = ManagedOperationStore(store).operation(key, fingerprint)
    store.records.write_spec(REPRESENTATION, family="representation")
    operation_spec = attach_operation_id(make_method_call_spec(key.producer_cdef_id, key.method))
    store.records.write_spec(operation_spec, family="operation")
    with operation.acquire() as lease:
        decision = lease.prepare(resumable=False)
        realization_id = decision.realization.realization_id
        writer = DurableProductWriter(store.records, lease, realization_id)
        writer.write_stream("result", "value.bin", (value,))
        result = writer.finalize(
            {
                "result": DataRecord(
                    representation_id=REPRESENTATION["id"],
                    storage=(StorageRef.self_product(role="result"),),
                    realization_id=realization_id,
                    output_slot="result",
                )
            },
            ExecutionRecord(
                execution_kind="python",
                operation_id=operation_spec["id"],
                backend={"name": "dryml.fake"},
                status="ok",
                realization_id=realization_id,
            ),
            primary_output_slot="result",
            required_output_slots=("result",),
            activate=True,
        )
    return result.output_records["result"].record_id


def test_resolves_stable_multi_input_vector_in_declared_order(tmp_path):
    store = DirStore(tmp_path / "store")
    left = Producer("left").compute.result
    right = Producer("right").compute.result
    left_id = _publish(left, store, b"left")
    right_id = _publish(right, store, b"right")

    resolved = resolve_inputs((right, left), store=store)

    assert tuple(item.record_id for item in resolved) == (right_id, left_id)
    assert tuple(item.output_slot for item in resolved) == ("result", "result")


def test_missing_output_fails_without_materializing_or_computing_producer(tmp_path):
    store = DirStore(tmp_path / "store")

    with pytest.raises(MissingManagedOutputError, match="active"):
        resolve_inputs((Producer().compute.result,), store=store)


def test_double_collect_retries_then_succeeds_without_mixing_vectors(tmp_path, monkeypatch):
    import dryml.managed.resolution as resolution

    store = DirStore(tmp_path / "store")
    ref = Producer().compute.result
    _publish(ref, store, b"value")
    stable = resolution._collect_input_vector((ref,), store)
    changed = (replace(stable[0], activation_generation=stable[0].activation_generation + 1),)
    values = iter((stable, changed, stable, stable))
    monkeypatch.setattr(resolution, "_collect_input_vector", lambda refs, selected: next(values))

    assert resolve_inputs((ref,), store=store, max_attempts=2) == stable


def test_double_collect_has_bounded_explicit_conflict(tmp_path, monkeypatch):
    import dryml.managed.resolution as resolution

    store = DirStore(tmp_path / "store")
    ref = Producer().compute.result
    _publish(ref, store, b"value")
    stable = resolution._collect_input_vector((ref,), store)
    calls = 0

    def changing(_refs, _store):
        nonlocal calls
        calls += 1
        return (replace(stable[0], activation_generation=stable[0].activation_generation + calls),)

    monkeypatch.setattr(resolution, "_collect_input_vector", changing)
    with pytest.raises(ConcurrentManagedActivationError, match="stable"):
        resolve_inputs((ref,), store=store, max_attempts=3)
    assert calls == 6
