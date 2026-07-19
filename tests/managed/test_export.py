from __future__ import annotations

from pathlib import Path

import pytest

from dryml.core2 import Object, RefCDef, Repo
from dryml.core2.store.dir import DirStore
from dryml.managed import (
    ManagedOutput,
    MissingManagedOutputError,
    StaleManagedResultError,
    export_recipe,
    resolve_output,
    transfer_realizations,
    managed,
)
from dryml.records import RealizationRecord, require_checkpoint_integrity


class CountingSource(Object):
    builds = 0
    runs = 0

    def __init__(self, value="source"):
        super().__init__()
        type(self).builds += 1
        self.value = value

    @managed(outputs=(ManagedOutput("result", primary=True, kind="data"),))
    def compute(self):
        from dryml.managed import current_operation_context
        from dryml.records import make_representation_spec

        type(self).runs += 1
        representation = make_representation_spec(
            "u8.bytes", version="1", storage_kinds=("product-dir",)
        )
        current_operation_context().write_output(
            "result", "value.bin", (self.value.encode(),), representation=representation
        )


class Recipe(Object):
    def __init__(self, source: RefCDef):
        super().__init__()
        self.source = source


class Consumer(Object):
    runs = 0

    def __init__(self, source):
        super().__init__()
        self.source = source

    def __dryml_managed_inputs__(self, method, args, kwargs):
        return (self.source,)

    @managed(outputs=(ManagedOutput("result", primary=True, kind="data"),))
    def compute(self):
        from dryml.managed import current_operation_context
        from dryml.records import make_representation_spec

        type(self).runs += 1
        representation = make_representation_spec(
            "u8.consumer-bytes", version="1", storage_kinds=("product-dir",)
        )
        current_operation_context().write_output(
            "result", "value.bin", (b"consumed",), representation=representation
        )


class CheckpointProducer(Object):
    @managed(
        outputs=(ManagedOutput("result", primary=True, kind="data"),),
        resumable=True,
        checkpoint_schema="u8-v1",
    )
    def compute(self):
        from dryml.managed import current_operation_context
        from dryml.records import make_representation_spec

        context = current_operation_context()
        context.write_checkpoint("cursor.bin", (b"cursor",))
        context.commit_checkpoint()
        representation = make_representation_spec(
            "u8.checkpoint", version="1", storage_kinds=("product-dir",)
        )
        context.write_output(
            "result", "value.bin", (b"complete",), representation=representation
        )


def _product_bytes(store, record_id):
    record = store.records.read_record(record_id)
    root = store.records.resolve_storage_ref(
        record["payload"]["storage"][0], record_id=record_id
    )
    return root.joinpath("value.bin").read_bytes()


def test_recipe_export_traverses_ref_cdefs_without_materializing(tmp_path):
    source = CountingSource("recipe")
    recipe = Recipe(source)
    CountingSource.builds = 0
    destination = DirStore(tmp_path / "destination")

    report = export_recipe(recipe.definition, destination, main=True)

    assert CountingSource.builds == 0
    assert destination.has(recipe.definition)
    assert destination.has(source.definition)
    assert report.root == recipe.definition
    assert Repo(destination).load_definition() == recipe.definition


def test_exact_active_transfer_recurses_consumed_lineage_and_reopens(tmp_path):
    source = DirStore(tmp_path / "source")
    destination_path = tmp_path / "destination"
    destination = DirStore(destination_path)
    producer = CountingSource("exact")
    produced = producer.compute(store=source)
    consumer = Consumer(producer.compute.result)
    consumed = consumer.compute(store=source)

    report = transfer_realizations(source, destination, consumer.compute.result)

    resolved = resolve_output(consumer.compute.result, store=destination)
    assert resolved.record_id == consumed.outputs["result"].record_id
    assert _product_bytes(destination, resolved.record_id) == b"consumed"
    assert produced.outputs["result"].record_id in report.records

    source_realization = RealizationRecord.from_envelope(
        source.records.read_record(consumed.realization_record_id)
    )
    reopened = DirStore(destination_path)
    imported = RealizationRecord.from_envelope(
        reopened.records.read_record(consumed.realization_record_id)
    )
    assert imported.consumed_records == source_realization.consumed_records
    assert _product_bytes(
        reopened, imported.consumed_records[0].record_id
    ) == b"exact"
    assert not tuple(Path(reopened.managed_control_root()).glob("**/attempts/*"))
    assert not tuple(Path(reopened.managed_control_root()).glob("**/owner.json"))


def test_exact_transfer_normal_reuse_resolves_transferred_consumed_vector(
    tmp_path,
):
    source = DirStore(tmp_path / "source")
    destination = DirStore(tmp_path / "destination")
    producer = CountingSource("exact-reuse")
    produced = producer.compute(store=source)
    consumer = Consumer(producer.compute.result)
    completed = consumer.compute(store=source)
    transfer_realizations(source, destination, consumer.compute.result)
    CountingSource.runs = 0
    Consumer.runs = 0

    reused = consumer.compute(store=destination)

    assert reused.action == "reuse"
    assert reused.realization_id == completed.realization_id
    assert reused.consumed_records[0].record_id == produced.outputs["result"].record_id
    assert CountingSource.runs == 0
    assert Consumer.runs == 0


def test_inactive_root_transfer_still_activates_exact_consumed_dependencies(
    tmp_path,
):
    source = DirStore(tmp_path / "source")
    destination = DirStore(tmp_path / "destination")
    producer = CountingSource("inactive-root")
    produced = producer.compute(store=source)
    consumer = Consumer(producer.compute.result)
    consumer.compute(store=source)

    report = transfer_realizations(
        source,
        destination,
        consumer.compute.result,
        activate="inactive",
    )

    assert report.activated_realization_id is None
    assert resolve_output(producer.compute.result, store=destination).record_id == (
        produced.outputs["result"].record_id
    )
    with pytest.raises(MissingManagedOutputError, match="active"):
        resolve_output(consumer.compute.result, store=destination)


def test_transfer_does_not_overwrite_conflicting_dependency_selection(tmp_path):
    source = DirStore(tmp_path / "source")
    destination = DirStore(tmp_path / "destination")
    producer = CountingSource("selection")
    producer.compute(store=source)
    consumer = Consumer(producer.compute.result)
    consumer.compute(store=source)
    destination_result = producer.compute(store=destination)

    transfer_realizations(source, destination, consumer.compute.result)
    CountingSource.runs = 0
    Consumer.runs = 0

    assert resolve_output(producer.compute.result, store=destination).record_id == (
        destination_result.outputs["result"].record_id
    )
    with pytest.raises(StaleManagedResultError):
        consumer.compute(store=destination)
    assert CountingSource.runs == 0
    assert Consumer.runs == 0


def test_all_history_transfer_is_inactive_when_requested(tmp_path):
    source = DirStore(tmp_path / "source")
    destination = DirStore(tmp_path / "destination")
    producer = CountingSource("first")
    first = producer.compute(store=source)
    second = producer.compute.rerun(store=source)

    report = transfer_realizations(
        source,
        destination,
        producer.compute.result,
        history="all",
        activate="inactive",
    )

    assert report.realization_ids == (first.realization_id, second.realization_id)
    assert set(destination.records.find_records(kind="realization"))
    assert report.activated_realization_id is None


def test_all_history_transfer_selects_only_source_active_dependency_chain(tmp_path):
    source = DirStore(tmp_path / "source")
    producer = CountingSource("history")
    produced_a = producer.compute(store=source)
    consumer = Consumer(producer.compute.result)
    consumed_a = consumer.compute(store=source)
    produced_b = producer.compute.rerun(store=source)
    consumed_b = consumer.compute.rerun(store=source)

    destination = DirStore(tmp_path / "destination")
    report = transfer_realizations(
        source,
        destination,
        consumer.compute.result,
        history="all",
    )
    CountingSource.runs = 0
    Consumer.runs = 0

    reused = consumer.compute(store=destination)

    assert report.realization_ids == (consumed_a.realization_id, consumed_b.realization_id)
    assert set(report.dependency_realization_ids) == {
        produced_a.realization_id,
        produced_b.realization_id,
    }
    assert reused.action == "reuse"
    assert reused.realization_id == consumed_b.realization_id
    assert reused.consumed_records[0].realization_id == produced_b.realization_id
    assert resolve_output(producer.compute.result, store=destination).realization_id == (
        produced_b.realization_id
    )
    assert {state.realization_id for state in producer.compute.history(store=destination)} == {
        produced_a.realization_id,
        produced_b.realization_id,
    }
    for completed in (produced_a, produced_b, consumed_a, consumed_b):
        assert destination.records.has_record(completed.realization_record_id)
    assert CountingSource.runs == 0
    assert Consumer.runs == 0

    inactive_destination = DirStore(tmp_path / "inactive-destination")
    inactive_report = transfer_realizations(
        source,
        inactive_destination,
        consumer.compute.result,
        history="all",
        activate="inactive",
    )

    assert inactive_report.realization_ids == report.realization_ids
    assert set(inactive_report.dependency_realization_ids) == set(
        report.dependency_realization_ids
    )
    assert inactive_report.activated_realization_id is None
    assert resolve_output(
        producer.compute.result, store=inactive_destination
    ).realization_id == produced_b.realization_id
    with pytest.raises(MissingManagedOutputError, match="active"):
        resolve_output(consumer.compute.result, store=inactive_destination)
    assert {state.realization_id for state in producer.compute.history(
        store=inactive_destination
    )} == {produced_a.realization_id, produced_b.realization_id}


def test_completed_transfer_preserves_exact_checkpoint_payload(tmp_path):
    source = DirStore(tmp_path / "source")
    destination = DirStore(tmp_path / "destination")
    producer = CheckpointProducer()
    completed = producer.compute(store=source)

    transfer_realizations(source, destination, producer.compute.result)

    realization = RealizationRecord.from_envelope(
        destination.records.read_record(completed.realization_record_id)
    )
    matches = tuple(
        Path(destination.managed_control_root()).glob(
            f"**/checkpoints/{realization.checkpoint_head}"
        )
    )
    assert len(matches) == 1
    require_checkpoint_integrity(matches[0], realization.checkpoint_head)
