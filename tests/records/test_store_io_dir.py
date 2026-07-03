import os
import shutil
from pathlib import Path

import pytest

from dryml.core2.store.dir import DirStore
from dryml.formats.canonical import canonical_json_bytes
from dryml.formats.refs import format_cdef_id
from dryml.records import RecordIOError, RecordStoreIO, StorageRef, StorageRefError, attach_record_id, attach_spec_id, make_record, make_spec


class HookedDirStore(DirStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolved_cdef_ids = []

    def object_dir_for_cdef_id(self, cdef_id):
        self.resolved_cdef_ids.append(cdef_id)
        return super().object_dir_for_cdef_id(cdef_id)


def test_fresh_dirstore_without_records_lists_empty(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))

    assert not io.records_dir.exists()
    assert list(io.iter_record_ids()) == []
    assert list(io.iter_spec_ids()) == []


def test_write_read_list_record_and_spec_create_dirs_lazily(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    record_ref = io.write_record(make_record(kind="stored_state", payload={"x": 1}))
    spec_ref = io.write_spec(make_spec(family="representation", kind="repr", payload={"x": 1}))

    assert io.items_dir.exists()
    assert io.spec_family_dir("representation").exists()
    assert list(io.iter_record_ids()) == [record_ref.record_id]
    assert list(io.iter_spec_ids(family="representation")) == [spec_ref.spec_id]
    assert io.read_record(record_ref.record_id)["payload"] == {"x": 1}
    assert io.read_spec(spec_ref.spec_id, family="representation")["payload"] == {"x": 1}


def test_writes_are_canonical_idempotent_and_conflict_on_same_id_different_bytes(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    record = attach_record_id(make_record(kind="stored_state", payload={"b": 2, "a": 1}, metadata={"writer": "a"}))
    ref = io.write_record(record)
    path = io.items_dir / f"{ref.record_id}.json"

    assert path.read_bytes() == canonical_json_bytes(record)
    assert io.write_record(record).record_id == ref.record_id

    same_id_different_metadata = dict(record, metadata={"writer": "b"})
    with pytest.raises(RecordIOError, match="different canonical bytes"):
        io.write_record(same_id_different_metadata)

    spec = attach_spec_id(make_spec(family="representation", kind="repr", payload={"x": 1}, metadata={"writer": "a"}))
    spec_ref = io.write_spec(spec)
    assert io.write_spec(spec).spec_id == spec_ref.spec_id
    with pytest.raises(RecordIOError):
        io.write_spec(dict(spec, metadata={"writer": "b"}))


def test_indexes_are_optional_for_reads_and_lists(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    record_ref = io.write_record(make_record(kind="stored_state", payload={"x": 1}))
    spec_ref = io.write_spec(make_spec(family="operation", kind="placeholder", payload={"x": 1}))
    io.ensure_dirs()
    shutil.rmtree(io.indexes_dir)

    assert io.read_record(record_ref.record_id)["id"] == record_ref.record_id
    assert io.read_spec(spec_ref.spec_id, family="operation")["id"] == spec_ref.spec_id
    assert list(io.iter_record_ids()) == [record_ref.record_id]
    assert list(io.iter_spec_ids(family="operation")) == [spec_ref.spec_id]


def test_product_dir_resolution_can_create_directories(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    record_ref = io.write_record(make_record(kind="stored_state", payload={"x": 1}))
    path = io.resolve_storage_ref(StorageRef.product_dir(record_ref.record_id, path="derived/output"), create=True)

    assert path == io.products_dir / record_ref.record_id / "derived" / "output"
    assert path.is_dir()


def test_object_dir_resolution_requires_existing_full_digest(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    missing_ref = StorageRef.object_dir(format_cdef_id("a" * 64))

    with pytest.raises(StorageRefError, match="does not exist"):
        io.resolve_storage_ref(missing_ref)
    with pytest.raises(StorageRefError, match="full CDef digest"):
        io.resolve_storage_ref(StorageRef.object_dir(format_cdef_id("a" * 16)))

    assert not os.path.exists(io.records_dir / "indexes")


def test_object_dir_resolution_uses_store_resolver_hook(tmp_path):
    store = HookedDirStore(tmp_path / "store")
    cdef_id = format_cdef_id("b" * 64)
    object_dir = store.object_dir_for_cdef_id(cdef_id)
    os.makedirs(object_dir)

    path = RecordStoreIO(store).resolve_storage_ref(StorageRef.object_dir(cdef_id))

    assert path == Path(object_dir)
    assert store.resolved_cdef_ids == [cdef_id, cdef_id]
