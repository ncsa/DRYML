import pytest

from dryml.core.store.dir import DirStore
from dryml.formats.canonical import canonical_json_bytes
from dryml.formats.ids import content_id
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


def test_ref_index_store_ref_mismatch_rebuilds_on_auto_and_fails_when_refresh_false(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    spec_ref = io.write_spec(make_spec(family="operation", kind="function_call", payload={"function": "pkg.mod:run", "args": [cdef()], "kwargs": {}}), family="operation")
    io.rebuild_ref_index()
    data = io.read_ref_index().to_json()
    obsolete = "core" + chr(50)
    data["store_ref"] = f"dryml.{obsolete}.store.dir.DirStore:/old/location"
    io.ref_index_path.write_bytes(canonical_json_bytes(data))

    with pytest.raises(RecordRefIndexValidationError, match="store_ref"):
        io.find_mentions(refresh=False)

    mentions = io.find_mentions(target_id=cdef(), refresh="auto")

    assert mentions
    assert io.read_ref_index().store_ref == io._store_ref()
    assert io.find_operation_specs_for_cdef(cdef(), refresh=False)[0].spec_id == spec_ref.spec_id


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


def test_record_write_marks_index_dirty_before_authoritative_mutation(tmp_path, monkeypatch):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    io.rebuild_ref_index()
    original = io._write_json

    def fail_write(*args, **kwargs):
        assert io.ref_index_is_dirty()
        raise OSError("simulated write failure")

    monkeypatch.setattr(io, "_write_json", fail_write)
    with pytest.raises(OSError, match="simulated"):
        io.write_record(make_record(kind="data", payload={"subject_cdef_id": cdef()}))
    monkeypatch.setattr(io, "_write_json", original)

    assert io.ref_index_is_dirty()
    assert list(io.iter_records()) == []


def test_rebuild_and_publication_overlap_cannot_clear_a_new_dirty_marker(tmp_path, monkeypatch):
    import threading

    import dryml.records.index as index_module

    io = RecordStoreIO(DirStore(tmp_path / "store"))
    io.rebuild_ref_index()
    entered = threading.Event()
    proceed = threading.Event()
    original = index_module.build_record_ref_index

    def paused_build(record_io):
        entered.set()
        assert proceed.wait(5)
        return original(record_io)

    monkeypatch.setattr(index_module, "build_record_ref_index", paused_build)
    rebuild = threading.Thread(target=io.rebuild_ref_index)
    rebuild.start()
    assert entered.wait(5)
    write = threading.Thread(
        target=lambda: io.write_record(
            make_record(kind="data", payload={"subject_cdef_id": cdef("d")})
        )
    )
    write.start()
    assert write.is_alive()
    proceed.set()
    rebuild.join(5)
    write.join(5)

    assert not rebuild.is_alive() and not write.is_alive()
    assert io.ref_index_is_dirty()
    assert io.find_records(kind="data")


def test_deleted_reference_index_rebuilds_from_realization_records(tmp_path):
    from dryml.records import RealizationOutput, RealizationRecord

    io = RecordStoreIO(DirStore(tmp_path / "store"))
    output_id = io.write_record(
        make_record(kind="data", payload={"subject_cdef_id": cdef("e")})
    ).record_id
    representation_id = content_id("repr", 1, {"kind": "fake"})
    realization = RealizationRecord(
        realization_id="realization-v1-" + "1" * 32,
        producer_cdef_id=cdef(),
        method="compute",
        declaration_fingerprint="managed-declaration-v1-" + "2" * 64,
        attempt_ids=("attempt-v1-" + "3" * 32,),
        outputs=(
            RealizationOutput(
                "result",
                output_id,
                "data",
                representation_id,
            ),
        ),
        primary_output_slot="result",
        primary_representation_id=representation_id,
        execution_record_id=output_id,
        completed_attempt_id="attempt-v1-" + "3" * 32,
        completion_fence=1,
    )
    realization_ref = io.write_record(realization.to_envelope())
    assert io.find_records(
        kind="realization",
        realization_id=realization.realization_id,
        producer_cdef_id=realization.producer_cdef_id,
        method=realization.method,
    ) == (realization_ref,)
    io.rebuild_ref_index()
    io.ref_index_path.unlink()

    mentions = io.find_mentions(target_id=output_id, refresh="auto")

    assert any(item.source_id == realization_ref.record_id for item in mentions)
