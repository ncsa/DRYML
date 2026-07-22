import pytest

from dryml.core.store.dir import DirStore
from dryml.formats.ids import content_id
from dryml.records import RecordStoreIO, StorageRef, StorageRefError


def test_self_and_cross_product_refs_resolve(tmp_path):
    store = DirStore(tmp_path / "store")
    io = RecordStoreIO(store)
    record_id = content_id("record", 1, {"self": 1})
    other_id = content_id("record", 1, {"other": 1})

    self_ref = StorageRef.self_product(path=".", role="target-state")
    cross_ref = StorageRef.product_dir(other_id, path="artifact", role="source-product")

    assert "record_id" not in self_ref.to_json()
    assert cross_ref.to_json()["record_id"] == other_id
    assert io.resolve_storage_ref(self_ref, record_id=record_id, create=True) == io.products_dir / record_id
    assert io.resolve_storage_ref(cross_ref, create=True) == io.products_dir / other_id / "artifact"
    with pytest.raises(StorageRefError):
        io.resolve_storage_ref(self_ref)


@pytest.mark.parametrize("path", ["/abs", "x/../y", "x//y", r"x\\y"])
def test_self_product_ref_rejects_invalid_paths(path):
    with pytest.raises(StorageRefError):
        StorageRef.self_product(path=path)
