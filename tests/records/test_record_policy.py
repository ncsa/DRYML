import os

import pytest

import core2_objects as objects
from dryml.core2.policies import RepoSaveOptions
from dryml.core2.repo import Repo
from dryml.core2.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.records import (
    RecordPolicyError,
    RecordPolicyOptions,
    RecordStoreIO,
    StorageRef,
    default_object_state_representation_spec,
    normalize_record_policy,
)


def test_policy_validation_and_representation_spec_are_stable():
    assert normalize_record_policy(None) == "none"
    assert normalize_record_policy("all") == "all"
    with pytest.raises(RecordPolicyError):
        normalize_record_policy("everything")

    left = default_object_state_representation_spec()
    right = default_object_state_representation_spec()
    assert left == right
    assert left["id"].startswith("repr-v1-")
    assert left["schema"] == "dryml.representation.v1"
    assert left["kind"] == "dryml.object_state"
    assert left["payload"]["storage_kind"] == "object-dir"


def test_repo_save_default_none_and_options_create_no_records(tmp_path):
    for kwargs in ({}, {"record_policy": "none"}, {"options": RepoSaveOptions(record_policy="none")}):
        store = DirStore(tmp_path / f"store_{len(list(tmp_path.iterdir()))}")
        repo = Repo(stores=[store])
        repo.save(objects.HelloStr(msg="test"), **kwargs)
        assert not RecordStoreIO(store).records_dir.exists()


def test_invalid_policy_fails_before_object_write(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=[store])
    with pytest.raises(RecordPolicyError):
        repo.save(objects.HelloStr(msg="test"), record_policy="bad")
    assert list(store.hydrate_index()) == []
    assert not RecordStoreIO(store).records_dir.exists()


def test_descriptive_save_creates_stored_state_and_representation(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=[store])
    obj = objects.TestClassC2(10)
    obj.set_val(20)
    cdef_hash = obj.definition.stable_hash()

    repo.save_object(obj, revision="A", record_policy="descriptive")
    io = RecordStoreIO(store)
    record_ids = list(io.iter_record_ids())
    spec_ids = list(io.iter_spec_ids(family="representation"))

    assert len(record_ids) == 1
    assert len(spec_ids) == 1
    record = io.read_record(record_ids[0])
    payload = record["payload"]
    assert payload["subject_cdef_id"] == format_cdef_id(cdef_hash)
    assert payload["representation_id"] == spec_ids[0]
    assert payload["save"] == {
        "minimum_root_depth": 0,
        "reason": "serializable",
        "revision": "A",
    }
    storage = payload["storage"][0]
    assert storage == StorageRef.object_dir(format_cdef_id(cdef_hash)).to_json()
    assert io.resolve_storage_ref(storage).is_dir()
    assert not any(os.fspath(tmp_path) in str(value) for value in payload.values())
    assert record["metadata"]["writer"] == "dryml.records.policy"
    assert not io.ref_index_path.exists()
    assert list(store.hydrate_index()) == [obj.definition]


def test_descriptive_save_is_idempotent_and_rebuild_index_option(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=[store])
    obj = objects.HelloStr(msg="test")

    repo.save(obj, record_policy="descriptive")
    io = RecordStoreIO(store)
    record_ids = list(io.iter_record_ids())
    spec_ids = list(io.iter_spec_ids(family="representation"))
    repo.save(obj, record_policy="descriptive")

    assert list(io.iter_record_ids()) == record_ids
    assert list(io.iter_spec_ids(family="representation")) == spec_ids
    assert not io.ref_index_path.exists()

    repo.save(
        obj,
        record_policy="descriptive",
        record_options=RecordPolicyOptions(rebuild_index=True),
    )
    assert io.ref_index_path.exists()
    assert not io.ref_index_is_dirty()


def test_descriptive_save_marks_existing_index_dirty_only_on_changed_sidecar(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=[store])
    first = objects.HelloStr(msg="first")
    second = objects.HelloStr(msg="second")

    repo.save(first, record_policy="descriptive", record_options=RecordPolicyOptions(rebuild_index=True))
    io = RecordStoreIO(store)
    assert not io.ref_index_is_dirty()

    repo.save(first, record_policy="descriptive")
    assert not io.ref_index_is_dirty()

    repo.save(second, record_policy="descriptive")
    assert io.ref_index_is_dirty()


def test_object_convenience_save_passes_record_policy(tmp_path):
    store = DirStore(tmp_path / "store")
    obj = objects.HelloStr(msg="test")
    obj.save(repo=[store], record_policy="descriptive")
    assert len(list(RecordStoreIO(store).iter_record_ids())) == 1


def test_cdef_hash_unchanged_by_descriptive_policy(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=[store])
    obj = objects.HelloStr(msg="test")
    before = obj.definition.stable_hash()
    repo.save(obj, record_policy="descriptive")
    assert obj.definition.stable_hash() == before
