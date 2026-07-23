from io import BytesIO
import zipfile

import pytest

from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipExportStore, ZipStore
from dryml.records import RecordStoreIO, StorageRef, make_record, make_spec


def test_zip_store_persists_record_and_spec_sidecars():
    buf = BytesIO()
    store = ZipStore(buf)
    io = RecordStoreIO(store)
    record_ref = io.write_record(make_record(kind="stored_state", payload={"x": 1}))
    spec_ref = io.write_spec(make_spec(family="representation", kind="repr", payload={"x": 1}))
    store.commit()
    store.close()

    reopened = ZipStore(buf)
    reopened_io = RecordStoreIO(reopened)
    assert reopened_io.read_record(record_ref.record_id)["payload"] == {"x": 1}
    assert reopened_io.read_spec(spec_ref.spec_id, family="representation")["payload"] == {"x": 1}
    reopened.close()


def test_zip_export_includes_records_and_products_only_when_requested(tmp_path):
    store = DirStore(tmp_path / "store")
    io = RecordStoreIO(store)
    record_ref = io.write_record(make_record(kind="stored_state", payload={"x": 1}))
    product_path = io.resolve_storage_ref(StorageRef.product_dir(record_ref.record_id, path="artifact"), create=True)
    (product_path / "data.txt").write_text("product", encoding="utf-8")

    with_records = BytesIO()
    ZipExportStore(with_records, store.base_dir, include_paths={"records", "products"}).commit()
    with zipfile.ZipFile(with_records, "r") as zf:
        names = set(zf.namelist())
    assert f"records/items/{record_ref.record_id}.json" in names
    assert any(name.startswith(f"products/{record_ref.record_id}/") for name in names)

    without_records = BytesIO()
    ZipExportStore(without_records, store.base_dir, include_paths={"objects"}).commit()
    with zipfile.ZipFile(without_records, "r") as zf:
        names = set(zf.namelist())
    assert all(not name.startswith("records/") for name in names)
    assert all(not name.startswith("products/") for name in names)


def test_zip_export_missing_path_fails_before_replacing_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "export.zip"
    destination.write_bytes(b"existing")

    with pytest.raises(FileNotFoundError, match="missing"):
        ZipExportStore(destination, source, include_paths={"missing"}).commit()

    assert destination.read_bytes() == b"existing"


def test_zip_path_commit_is_atomic_but_file_like_destination_is_documented_non_atomic(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    source.joinpath("value.txt").write_text("value", encoding="utf-8")
    destination = tmp_path / "export.zip"
    destination.write_bytes(b"existing")
    original = zipfile.ZipFile.write

    def fail_write(self, *args, **kwargs):
        raise OSError("simulated zip failure")

    monkeypatch.setattr(zipfile.ZipFile, "write", fail_write)
    with pytest.raises(OSError, match="simulated"):
        ZipExportStore(destination, source, include_paths={"value.txt"}).commit()
    assert destination.read_bytes() == b"existing"

    monkeypatch.setattr(zipfile.ZipFile, "write", original)
    assert ZipExportStore.file_like_commit_is_atomic is False
