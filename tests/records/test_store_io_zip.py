from io import BytesIO
import zipfile

from dryml.core2.store.dir import DirStore
from dryml.core2.store.zip import ZipExportStore, ZipStore
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
