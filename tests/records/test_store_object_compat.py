import os
import shutil
from pathlib import Path

from dryml.core2 import Object, Repo, load_object
from dryml.core2.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.operations import make_function_call_spec
from dryml.records import RecordStoreIO, StorageRef, make_record, make_spec


class RecordCompatLeaf(Object):
    def __init__(self, value):
        super().__init__()
        self.value = value


def test_repo_save_does_not_create_records_by_default(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = RecordCompatLeaf("x", repo=repo)

    repo.save(obj)

    assert not (tmp_path / "store" / "records").exists()


def test_load_main_works_without_records(tmp_path):
    store = DirStore(tmp_path / "store")
    obj = RecordCompatLeaf("x")
    obj.save(repo=store)

    reopened = DirStore(tmp_path / "store")
    assert not (tmp_path / "store" / "records").exists()

    loaded = load_object(repo=reopened)
    assert loaded.definition == obj.definition


def test_object_save_load_and_membership_unaffected_after_records_exist(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = RecordCompatLeaf("x", repo=repo)
    repo.save_object(obj, main=True)
    repo.flush()
    cdef_id = format_cdef_id(obj.definition.stable_hash())
    io = RecordStoreIO(store)
    io.write_record(make_record(kind="stored_state", payload={"storage": [StorageRef.object_dir(cdef_id).to_json()]}))
    io.write_spec(make_spec(family="representation", kind="repr", payload={"format": "pickle"}))

    assert store.has(obj.definition)
    assert [cdef.stable_hash() for cdef in store.hydrate_index()] == [obj.definition.stable_hash()]
    assert io.resolve_storage_ref(StorageRef.object_dir(cdef_id)) == Path(os.fspath(store.object_dir(obj.definition)))

    loaded = load_object(repo=DirStore(tmp_path / "store"))
    assert loaded.definition == obj.definition


def test_sidecars_do_not_change_cdef_hashes_or_hydrate_index(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = RecordCompatLeaf("x", repo=repo)
    repo.save_object(obj, main=True)
    before = obj.definition.stable_hash()
    io = RecordStoreIO(store)
    record_ref = io.write_record(make_record(kind="stored_state", payload={"x": 1}))
    io.write_spec(make_spec(family="operation", kind="placeholder", payload={"x": 1}))
    io.ensure_dirs()
    shutil.rmtree(io.indexes_dir)

    assert obj.definition.stable_hash() == before
    assert [cdef.stable_hash() for cdef in store.hydrate_index()] == [before]
    assert io.read_record(record_ref.record_id)["payload"] == {"x": 1}


def test_operation_specs_and_ref_index_do_not_affect_object_save_load(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = RecordCompatLeaf("x", repo=repo)
    repo.save_object(obj, main=True)
    repo.flush()
    before = obj.definition.stable_hash()
    cdef_id = format_cdef_id(before)
    io = RecordStoreIO(store)

    op_ref = io.write_spec(make_function_call_spec("pkg.mod:run", args=[cdef_id]), family="operation")
    io.rebuild_ref_index()

    assert obj.definition.stable_hash() == before
    assert [cdef.stable_hash() for cdef in store.hydrate_index()] == [before]
    assert io.find_operation_specs_for_cdef(cdef_id, refresh=False)[0].spec_id == op_ref.spec_id
    assert load_object(repo=DirStore(tmp_path / "store")).definition == obj.definition


def test_repo_save_does_not_create_operation_specs_or_ref_index(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = RecordCompatLeaf("x", repo=repo)

    repo.save(obj)

    assert not (tmp_path / "store" / "records" / "specs" / "operation").exists()
    assert not (tmp_path / "store" / "records" / "indexes" / "ref-index-v1.json").exists()
