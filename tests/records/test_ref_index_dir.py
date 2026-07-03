import pytest

from dryml.core2.store.dir import DirStore
from dryml.formats.canonical import canonical_json_bytes
from dryml.formats.refs import format_cdef_id
from dryml.records import RecordRefIndexDirty, RecordRefIndexMissing, RecordRefIndexValidationError, RecordStoreIO, make_record, make_spec


def cdef(char="a"):
    return format_cdef_id(char * 64)


def test_ref_index_rebuild_empty_and_canonical(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))

    assert list(io.iter_records()) == []
    assert list(io.iter_specs()) == []
    report = io.rebuild_ref_index()
    first = io.ref_index_path.read_bytes()

    assert report.source_count == 0
    assert report.mention_count == 0
    assert io.read_ref_index().to_json()["mentions"] == []
    assert first == canonical_json_bytes(io.read_ref_index().to_json())
    assert io.rebuild_ref_index().changed is False
    assert io.ref_index_path.read_bytes() == first


def test_ref_index_dirty_missing_corrupt_and_queries(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    record_ref = io.write_record(make_record(kind="stored_state", payload={"subject_cdef_id": cdef()}))
    spec_ref = io.write_spec(make_spec(family="operation", kind="function_call", payload={"function": "pkg.mod:run", "args": [cdef(), "ref(" + cdef("b") + ")"], "kwargs": {}}), family="operation")

    with pytest.raises(RecordRefIndexMissing):
        io.find_mentions(refresh=False)
    assert io.find_mentions(target_id=cdef(), refresh="auto")
    assert not io.ref_index_is_dirty()
    first_mentions = [mention.to_json() for mention in io.read_ref_index().mentions]

    io.ref_index_path.unlink()
    assert [mention.to_json() for mention in io.find_mentions(refresh="auto")] == first_mentions

    io.write_record(make_record(kind="stored_state", payload={"subject_cdef_id": cdef("c")}))
    assert io.ref_index_is_dirty()
    with pytest.raises(RecordRefIndexDirty):
        io.find_mentions(refresh=False)
    assert io.find_mentions(target_id=cdef("c"), refresh="auto")
    assert not io.ref_index_is_dirty()

    io.ref_index_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(RecordRefIndexValidationError):
        io.find_mentions(refresh=False)
    assert io.find_mentions(target_id=cdef(), refresh="auto")

    record_refs = io.find_records_mentioning_cdef(cdef(), refresh=False)
    spec_refs = io.find_operation_specs_for_cdef(cdef(), cdef_semantics="materialize", refresh=False)
    ref_specs = io.find_operation_specs_for_cdef(cdef("b"), cdef_semantics="reference", refresh=False)

    assert [ref.record_id for ref in record_refs] == [record_ref.record_id]
    assert [ref.spec_id for ref in spec_refs] == [spec_ref.spec_id]
    assert spec_refs[0].kind == "operation"
    assert [ref.spec_id for ref in ref_specs] == [spec_ref.spec_id]
    assert io.find_records_mentioning_cdef(cdef("d"), refresh=False) == ()
    assert io.find_specs_mentioning_cdef(cdef("d"), refresh=False) == ()


def test_idempotent_writes_do_not_mark_dirty(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    record = make_record(kind="stored_state", payload={"subject_cdef_id": cdef()})
    spec = make_spec(family="representation", kind="repr", payload={"subject_cdef_id": cdef()})
    io.write_record(record)
    io.write_spec(spec, family="representation")
    io.rebuild_ref_index()

    io.write_record(record)
    io.write_spec(spec, family="representation")

    assert not io.ref_index_is_dirty()
