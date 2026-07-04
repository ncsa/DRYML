import pytest

from dryml.core2.repo import Repo
from dryml.core2.store.dir import DirStore
from dryml.formats.ids import content_id
from dryml.formats.refs import format_cdef_id
from dryml.records import (
    AdapterRecord,
    DataRecord,
    ExecutionCancellationInfo,
    ExecutionErrorInfo,
    ExecutionLogRef,
    ExecutionRecord,
    ExecutionRecordLink,
    ProductWriteSession,
    RecordPolicyOptions,
    RecordStoreIO,
    RecordValidationError,
    StorageRef,
    StoredStateRecord,
    copy_record_closure,
    default_object_state_representation_spec,
    execution_record_for_adapter_result,
    execution_record_for_probe_report,
    find_execution_records,
    typed_record_from_envelope,
    unsupported_compiler_execution_record,
)


def _id(prefix, char="a"):
    return content_id(prefix, 1, {prefix: char})


def _cdef(char="a"):
    return format_cdef_id(char * 64)


def _execution(**kwargs):
    base = {"execution_kind": "python", "operation_id": _id("op"), "backend": {"name": "dryml.fake", "kind": "fake"}, "status": "ok"}
    base.update(kwargs)
    return ExecutionRecord(**base)


def test_execution_record_round_trips_and_dispatches_typed_wrapper():
    record = _execution(
        dispatch_id=_id("dispatch"),
        recipe_id=_id("recipe"),
        consumed_records=(ExecutionRecordLink(_id("record", "c"), role="model-state"),),
        produced_records=(_id("record", "p"),),
        input_cdef_ids=(_cdef(),),
        output_cdef_ids=(_cdef("b"),),
        logs=(ExecutionLogRef("stdout", StorageRef.self_product(path="stdout.txt", role="stdout"), "text/plain"),),
    )
    envelope = record.to_envelope()
    wrapped = ExecutionRecord.from_envelope(envelope)

    assert wrapped.to_payload() == record.to_payload()
    assert wrapped.consumed_record_ids == (_id("record", "c"),)
    assert wrapped.produced_record_ids == (_id("record", "p"),)
    assert isinstance(typed_record_from_envelope(envelope), ExecutionRecord)


@pytest.mark.parametrize("status", ["failed", "cancelled", "timeout", "unsupported", "skipped", "degraded"])
def test_execution_status_variants_validate(status):
    kwargs = {"status": status}
    if status == "failed":
        kwargs["error"] = ExecutionErrorInfo(type="ValueError", message="bad")
    if status == "cancelled":
        kwargs["cancellation"] = ExecutionCancellationInfo(method="SIGTERM", reason="test")
    record = _execution(**kwargs)

    assert ExecutionRecord.from_envelope(record.to_envelope()).status == status


def test_probe_adapter_and_compiler_helpers():
    class Probe:
        status = "ok"
        operation_id = _id("op")
        report_id = _id("record", "probe")

    probe = execution_record_for_probe_report(Probe())
    assert probe.execution_kind == "probe"
    assert probe.probe_report_ids == (_id("record", "probe"),)

    adapter = AdapterRecord(adapter={"name": "fake"}, source_record_id=_id("record", "s"), source_representation_id=_id("repr", "s"), target_record_id=_id("record", "t"), target_representation_id=_id("repr", "t"), produced_records=(_id("record", "t"),), derived_from=(_id("record", "s"),))
    adapter_result = type("Result", (), {"status": "ok", "adapter_records": (_id("record", "adapter"),), "target_records": (_id("record", "t"),)})()
    adapter_exec = execution_record_for_adapter_result(adapter_result, operation_id=_id("op"), consumed_records=(_id("record", "s"),))
    compiler = unsupported_compiler_execution_record(operation_id=_id("op"), program_record_ids=(_id("record", "program"),))

    assert adapter.to_envelope()["kind"] == "adapter"
    assert adapter_exec.execution_kind == "adapter"
    assert adapter_exec.produced_record_ids == (_id("record", "t"),)
    assert compiler.status == "unsupported"
    assert compiler.execution_kind == "compiler"


def test_execution_record_invalid_shapes_fail():
    with pytest.raises(RecordValidationError, match="status"):
        _execution(status="running")
    with pytest.raises(RecordValidationError, match="execution_kind"):
        _execution(execution_kind="worker")
    with pytest.raises(RecordValidationError, match="operation_id"):
        _execution(operation_id=_id("repr"))
    with pytest.raises(RecordValidationError, match="JSON array"):
        ExecutionRecord.from_envelope({**_execution().to_envelope(), "payload": {**_execution().to_payload(), "consumed_records": "not-a-list"}})


def test_execution_queries_store_and_repo_with_deleted_index(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    io1 = RecordStoreIO(store1)
    io2 = RecordStoreIO(store2)
    op_id = _id("op")
    consumed = _id("record", "c")
    produced = _id("record", "p")
    first = io1.write_record(_execution(operation_id=op_id, dispatch_id=_id("dispatch"), recipe_id=_id("recipe"), consumed_records=(consumed,), produced_records=(produced,)).to_envelope())
    second = io2.write_record(_execution(operation_id=op_id, status="failed", error={"message": "bad"}).to_envelope())
    repo = Repo(stores=[store1, store2])

    io1.rebuild_ref_index()
    io1.ref_index_path.unlink()
    assert io1.find_execution_records(operation_id=op_id) == (first,)
    assert io1.find_execution_records(consumed_record_id=consumed) == (first,)
    assert io1.find_execution_records(produced_record_id=produced) == (first,)
    assert io1.find_execution_records(dispatch_id=_id("dispatch")) == (first,)
    assert io1.find_execution_records(recipe_id=_id("recipe")) == (first,)
    assert io1.find_execution_records(status="ok") == (first,)
    assert io1.find_execution_records(execution_kind="python") == (first,)
    assert set(repo.records.find_execution_records(operation_id=op_id)) == {first, second}
    assert find_execution_records(repo, status="failed") == (second,)


def test_execution_log_product_export_and_missing_products(tmp_path):
    source = DirStore(tmp_path / "source")
    dest = DirStore(tmp_path / "dest")
    io = RecordStoreIO(source)
    execution = _execution(logs=(ExecutionLogRef("stderr", StorageRef.self_product(path="stderr.txt", role="stderr"), "text/plain"),))
    with ProductWriteSession(io) as session:
        session.write_text("stderr.txt", "log text")
        result = session.commit_record(execution.to_envelope())

    copied = copy_record_closure(source, dest, seed_records=[result.located.record_id], policy="descriptive", options=RecordPolicyOptions(include_products=True))
    dest_io = RecordStoreIO(dest)
    loaded = ExecutionRecord.from_envelope(dest_io.read_record(result.located.record_id))
    copied_log = dest_io.resolve_storage_ref(loaded.logs[0].storage, record_id=result.located.record_id)

    assert copied.products_copied == (result.located.record_id,)
    assert copied_log.read_text(encoding="utf-8") == "log text"


def test_load_and_resolution_without_execution_records(tmp_path):
    store = DirStore(tmp_path / "store")
    io = RecordStoreIO(store)
    spec = default_object_state_representation_spec()
    io.write_spec(spec, family="representation")
    state = io.write_record(StoredStateRecord(_cdef(), spec["id"], (StorageRef.self_product(role="state"),)).to_envelope())
    data = DataRecord(representation_id=_id("repr"), storage=(StorageRef.self_product(role="data"),), derived_from=(state.record_id,))

    assert io.find_records(kind="stored_state") == (state,)
    assert data.to_envelope()["kind"] == "data"
