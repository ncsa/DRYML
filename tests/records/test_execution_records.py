import pytest

from dryml.formats.canonical import canonical_json_bytes
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
    attach_record_id,
    copy_record_closure,
    default_object_state_representation_spec,
    execution_record_for_adapter_result,
    execution_record_for_probe_report,
    find_execution_records,
    make_record,
    typed_record_from_envelope,
    unsupported_compiler_execution_record,
)
from dryml.records.execution import persistence_safe_execution_error


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


def test_persistence_safe_execution_error_excludes_exception_text():
    secret = "projection-secret-sentinel-f632"
    oversized_name = "SecretSentinel" + ("X" * 128)
    oversized_error = type(oversized_name, (RuntimeError,), {})(secret)

    projected = persistence_safe_execution_error(RuntimeError(secret))
    bounded = persistence_safe_execution_error(oversized_error)

    assert projected == {
        "type": "RuntimeError",
        "metadata": {"code": "execution_failed"},
    }
    assert secret not in str(projected)
    assert bounded == {
        "type": "Error",
        "metadata": {"code": "execution_failed"},
    }
    assert secret not in str(bounded)
    assert oversized_name not in str(bounded)


@pytest.mark.parametrize("status", ["failed", "cancelled", "timeout", "unsupported", "skipped", "degraded"])
def test_execution_status_variants_validate(status):
    kwargs = {"status": status}
    if status == "failed":
        kwargs["error"] = ExecutionErrorInfo(type="ValueError", message="bad")
    if status == "cancelled":
        kwargs["cancellation"] = ExecutionCancellationInfo(method="SIGTERM", reason="test")
    if status == "timeout":
        kwargs["error"] = ExecutionErrorInfo(type="TimeoutError", message="timed out")
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
    assert probe.produced_record_ids == (_id("record", "probe"),)

    adapter = AdapterRecord(adapter={"name": "fake"}, source_record_id=_id("record", "s"), source_representation_id=_id("repr", "s"), target_record_id=_id("record", "t"), target_representation_id=_id("repr", "t"), produced_records=(_id("record", "t"),), derived_from=(_id("record", "s"),))
    adapter_result = type("Result", (), {"status": "ok", "adapter_records": (_id("record", "adapter"),), "target_records": (_id("record", "t"),)})()
    adapter_exec = execution_record_for_adapter_result(adapter_result, operation_id=_id("op"), consumed_records=(_id("record", "s"),))
    compiler = unsupported_compiler_execution_record(operation_id=_id("op"), program_record_ids=(_id("record", "program"),))

    assert adapter.to_envelope()["kind"] == "adapter"
    assert adapter_exec.execution_kind == "adapter"
    assert adapter_exec.produced_record_ids == (_id("record", "t"), _id("record", "adapter"))
    assert compiler.status == "unsupported"
    assert compiler.execution_kind == "compiler"
    assert compiler.produced_record_ids == (_id("record", "program"),)


def test_probe_helper_reads_envelope_payload_status_and_diagnostics():
    op_id = _id("op")
    probe_record = attach_record_id(
        make_record(
            kind="probe_report",
            payload={"operation_id": op_id, "status": "failed", "diagnostics": [{"message": "probe failed"}]},
        )
    )
    probe = execution_record_for_probe_report(probe_record)

    assert probe.operation_id == op_id
    assert probe.status == "failed"
    assert probe.diagnostics == ({"message": "probe failed"},)
    assert probe.probe_report_ids == (probe_record["id"],)


def test_adapter_helper_copies_diagnostics_and_error():
    adapter_result = type(
        "Result",
        (),
        {
            "status": "failed",
            "target_records": (_id("record", "t"),),
            "adapter_records": (_id("record", "adapter"),),
            "diagnostics": ({"message": "adapter failed"},),
            "error": {"message": "adapter crashed"},
        },
    )()

    adapter_exec = execution_record_for_adapter_result(adapter_result, operation_id=_id("op"), consumed_records=(_id("record", "s"),))

    assert adapter_exec.status == "failed"
    assert adapter_exec.diagnostics == ({"message": "adapter failed"},)
    assert adapter_exec.error == ExecutionErrorInfo(message="adapter crashed")
    assert adapter_exec.produced_record_ids == (_id("record", "t"), _id("record", "adapter"))


def test_execution_record_invalid_shapes_fail():
    with pytest.raises(RecordValidationError, match="status"):
        _execution(status="running")
    with pytest.raises(RecordValidationError, match="execution_kind"):
        _execution(execution_kind="worker")
    with pytest.raises(RecordValidationError, match="operation_id"):
        _execution(operation_id=_id("repr"))
    with pytest.raises(RecordValidationError, match="JSON array"):
        ExecutionRecord.from_envelope({**_execution().to_envelope(), "payload": {**_execution().to_payload(), "consumed_records": "not-a-list"}})


def test_execution_status_context_invariants_fail():
    with pytest.raises(RecordValidationError, match="ok execution records must not include error"):
        _execution(error={"message": "boom"})
    with pytest.raises(RecordValidationError, match="require error or diagnostics"):
        _execution(status="failed")
    with pytest.raises(RecordValidationError, match="require cancellation"):
        _execution(status="cancelled")
    with pytest.raises(RecordValidationError, match="cancellation is only valid"):
        _execution(status="failed", error={"message": "boom"}, cancellation={"method": "SIGTERM"})
    with pytest.raises(RecordValidationError, match="requires details"):
        _execution(status="failed", error={})

    assert _execution(status="failed", error={}, diagnostics=({"message": "boom"},)).status == "failed"


def test_execution_timing_validation():
    record = _execution(started_at="2026-07-04T12:34:56Z", ended_at="2026-07-04T12:34:56.123Z", duration_ms=1.25)
    assert record.duration_ms == 1.25

    with pytest.raises(RecordValidationError, match="RFC3339 UTC"):
        _execution(started_at="2026-07-04 12:34:56")
    with pytest.raises(RecordValidationError, match="RFC3339 UTC"):
        _execution(started_at="2026-99-99T99:99:99Z")
    with pytest.raises(RecordValidationError, match="finite"):
        _execution(duration_ms=float("nan"))


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
    assert io1.find_execution_records(status="OK") == (first,)
    assert io1.find_execution_records(execution_kind="PYTHON") == (first,)
    with pytest.raises(RecordValidationError, match="status"):
        io1.find_execution_records(status="bogus")
    with pytest.raises(RecordValidationError, match="execution_kind"):
        io1.find_execution_records(execution_kind="worker")
    assert set(repo.records.find_execution_records(operation_id=op_id)) == {first, second}
    assert find_execution_records(repo, status="failed") == (second,)


def test_generic_execution_writes_are_strict_and_queries_identify_bad_records(tmp_path):
    store = DirStore(tmp_path / "store")
    io = RecordStoreIO(store)
    malformed = make_record(kind="execution", payload={"operation_id": _id("op")})

    with pytest.raises(RecordValidationError, match="invalid execution record"):
        io.write_record(malformed)

    attached = attach_record_id(malformed)
    io.items_dir.mkdir(parents=True, exist_ok=True)
    io.items_dir.joinpath(f"{attached['id']}.json").write_bytes(canonical_json_bytes(attached))
    with pytest.raises(RecordValidationError) as excinfo:
        io.find_execution_records(operation_id=_id("op"))
    assert excinfo.value.context["record_id"] == attached["id"]


def test_generic_execution_write_persists_normalized_envelope(tmp_path):
    store = DirStore(tmp_path / "store")
    io = RecordStoreIO(store)
    probe_report_id = _id("record", "p")
    shorthand = make_record(
        kind="execution",
        payload={
            "execution_kind": "probe",
            "operation_id": _id("op"),
            "backend": {"name": "dryml.provider_probe", "kind": "probe"},
            "status": "ok",
            "probe_report_ids": [probe_report_id],
        },
    )
    canonical = attach_record_id(ExecutionRecord.from_envelope(shorthand).to_envelope())

    located = io.write_record(shorthand)
    loaded = io.read_record(located.record_id)

    assert located.record_id == canonical["id"]
    assert loaded == canonical
    assert loaded["payload"]["produced_records"] == [{"record_id": probe_report_id, "role": "probe-report", "required": False}]


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
