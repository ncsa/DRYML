import pytest

from dryml.core2.repo import Repo
from dryml.core2.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.records import RecordExportError, RecordStoreIO, make_record, make_spec


def _cdef(char="a"):
    return format_cdef_id(char * 64)


def test_repo_records_find_read_and_copy_across_stores(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    dest = DirStore(tmp_path / "dest")
    io1 = RecordStoreIO(store1)
    io2 = RecordStoreIO(store2)
    record_ref = io1.write_record(make_record(kind="stored_state", payload={"subject_cdef_id": _cdef()}))
    spec_ref = io2.write_spec(make_spec(family="operation", kind="function_call", payload={"args": [_cdef()], "kwargs": {}}))
    repo = Repo(stores=[store1, store2])

    found_records = repo.records.find_record(record_ref.record_id)
    found_specs = repo.records.find_spec(spec_ref.spec_id, family="operation")
    assert found_records == (record_ref,)
    assert found_specs == (spec_ref,)
    assert repo.records.read_record(found_records[0])["id"] == record_ref.record_id
    assert repo.records.read_spec(found_specs[0])["id"] == spec_ref.spec_id
    assert repo.records.find_records_mentioning_cdef(_cdef()) == (record_ref,)
    assert repo.records.find_specs_mentioning_cdef(_cdef(), family="operation") == (spec_ref,)
    assert repo.records.find_operation_specs_for_cdef(_cdef()) == (spec_ref,)

    report = repo.records.copy_closure(dest, seed_records=[record_ref.record_id])
    assert report.records_written[0].store_ref == RecordStoreIO(dest)._store_ref()
    assert RecordStoreIO(dest).read_record(record_ref.record_id)["id"] == record_ref.record_id


def test_repo_records_ambiguous_source_requires_source_store(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    dest = DirStore(tmp_path / "dest")
    record = make_record(kind="stored_state", payload={"subject_cdef_id": _cdef()})
    ref = RecordStoreIO(store1).write_record(record)
    RecordStoreIO(store2).write_record(record)
    repo = Repo(stores=[store1, store2])

    with pytest.raises(RecordExportError):
        repo.records.copy_closure(dest, seed_records=[ref.record_id])

    report = repo.records.copy_closure(dest, source_store=store1, seed_records=[ref.record_id])
    assert report.records_written
