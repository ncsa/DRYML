import pytest

from dryml.formats.ids import content_id
from dryml.formats.refs import format_cdef_id
from dryml.records import StorageRef, StorageRefError


def test_storage_ref_json_round_trips():
    cdef_id = format_cdef_id("a" * 64)
    record_id = content_id("record", 1, {"record": 1})
    blob_id = content_id("blob", 1, {"blob": 1})

    refs = [
        StorageRef.object_dir(cdef_id, path=".", role="default-state"),
        StorageRef.product_dir(record_id, path="derived/output", role="artifact"),
        StorageRef.blob(blob_id, role="weights"),
    ]

    for ref in refs:
        assert StorageRef.from_json(ref.to_json()) == ref


@pytest.mark.parametrize("path", ["", "/abs", "../x", "x/../y", "x//y", r"x\\y", "C:temp"])
def test_storage_ref_rejects_invalid_paths(path):
    with pytest.raises(StorageRefError):
        StorageRef.object_dir(format_cdef_id("a" * 64), path=path)


def test_storage_ref_normalizes_root_and_dot_components():
    ref = StorageRef.product_dir(content_id("record", 1, {}), path="./a/./b")

    assert ref.path == "a/b"


def test_storage_ref_rejects_missing_and_wrong_fields():
    cdef_id = format_cdef_id("a" * 64)
    record_id = content_id("record", 1, {})
    blob_id = content_id("blob", 1, {})

    with pytest.raises(StorageRefError):
        StorageRef("object-dir")
    with pytest.raises(StorageRefError):
        StorageRef("object-dir", subject_cdef_id=cdef_id, record_id=record_id)
    with pytest.raises(StorageRefError):
        StorageRef("product-dir", record_id=blob_id)
    with pytest.raises(StorageRefError):
        StorageRef("blob", blob_id=record_id)


def test_storage_ref_rejects_empty_role_and_malformed_cdef():
    with pytest.raises(StorageRefError):
        StorageRef.object_dir(format_cdef_id("a" * 64), role="")
    with pytest.raises(StorageRefError):
        StorageRef.object_dir("cdef-v4-nothex")
