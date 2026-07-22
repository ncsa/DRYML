from __future__ import annotations

import pytest

from dryml.core2 import Object
from dryml.core2.store.dir import DirStore
from dryml.managed import (
    ManagedOutput,
    ManagedStateError,
    resolve_output,
    transfer_realizations,
    managed,
)
from dryml.records import RecordExportError


class ImportProducer(Object):
    @managed(outputs=(ManagedOutput("result", primary=True, kind="data"),))
    def compute(self):
        from dryml.managed import current_operation_context
        from dryml.records import make_representation_spec

        representation = make_representation_spec(
            "u8.import", version="1", storage_kinds=("product-dir",)
        )
        current_operation_context().write_output(
            "result", "value.bin", (b"import",), representation=representation
        )


def test_transfer_retry_adopts_identical_products_after_control_phase_crash(
    tmp_path, monkeypatch
):
    import dryml.managed.export as export_module

    source = DirStore(tmp_path / "source")
    destination = DirStore(tmp_path / "destination")
    producer = ImportProducer()
    completed = producer.compute(store=source)
    original = export_module._install_control_closure

    def crash_after_content(*args, **kwargs):
        raise OSError("simulated control install crash")

    monkeypatch.setattr(export_module, "_install_control_closure", crash_after_content)
    with pytest.raises(OSError, match="simulated"):
        transfer_realizations(source, destination, producer.compute.result)
    assert destination.records.has_record(completed.outputs["result"].record_id)

    monkeypatch.setattr(export_module, "_install_control_closure", original)
    retried = transfer_realizations(source, destination, producer.compute.result)
    repeated = transfer_realizations(source, destination, producer.compute.result)

    assert retried == repeated
    assert resolve_output(producer.compute.result, store=destination).record_id == completed.outputs["result"].record_id


def test_transfer_refuses_conflicting_destination_product(tmp_path):
    source = DirStore(tmp_path / "source")
    destination = DirStore(tmp_path / "destination")
    producer = ImportProducer()
    completed = producer.compute(store=source)
    destination.records.product_root(
        completed.outputs["result"].record_id, create=True
    ).joinpath("value.bin").write_bytes(b"conflict")

    with pytest.raises(RecordExportError, match="different bytes"):
        transfer_realizations(source, destination, producer.compute.result)


def test_transfer_refuses_missing_required_product_path(tmp_path):
    source = DirStore(tmp_path / "source")
    destination = DirStore(tmp_path / "destination")
    producer = ImportProducer()
    completed = producer.compute(store=source)
    source.records.product_root(completed.outputs["result"].record_id).joinpath(
        "value.bin"
    ).unlink()

    with pytest.raises(ManagedStateError, match="integrity"):
        transfer_realizations(source, destination, producer.compute.result)


def test_set_active_if_absent_preserves_existing_destination_selection(tmp_path):
    source = DirStore(tmp_path / "source")
    destination = DirStore(tmp_path / "destination")
    producer = ImportProducer()
    imported = producer.compute(store=source)
    existing = producer.compute(store=destination)

    report = transfer_realizations(source, destination, producer.compute.result)

    assert report.activated_realization_id is None
    assert resolve_output(producer.compute.result, store=destination).realization_id == existing.realization_id
    assert destination.records.has_record(imported.realization_record_id)
