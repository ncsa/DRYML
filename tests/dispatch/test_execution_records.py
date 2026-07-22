import sys

import dryml

from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec
from dryml.records import ExecutionRecord, copy_record_closure


def _env(target_module):
    return PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()


def test_success_and_failure_execution_records_query(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    dispatcher = Dispatcher(store=store)
    secret = "dispatch-secret-sentinel-91e6"
    ok = attach_operation_id(make_function_call_spec("dispatch_target:noisy_add", args=[2, 3], metadata={"owner": "user"}))
    fail = attach_operation_id(make_function_call_spec("dispatch_target:fail_secret"))

    plan = dispatcher.plan(ok, environment=_env(target_module))
    assert plan.envelope.operation_spec["payload"] == ok["payload"]
    assert plan.envelope.operation_spec["metadata"]["owner"] == "user"
    assert plan.envelope.operation_spec["metadata"]["dryml.dispatch.transport"] == "operation_spec"

    capture = dryml.reporting.CaptureReporter()
    with dryml.config(reporting={"level": "details", "reporter": capture}):
        ok_result = dispatcher.run(ok, environment=_env(target_module))

    assert ok_result.status == "ok"
    assert ok_result.result_canonical == 5
    assert ok_result.execution_record_id
    assert store.records.read_spec(ok_result.dispatch_id, family="dispatch")["id"] == ok_result.dispatch_id
    assert store.records.read_spec(ok_result.recipe_id, family="execution_recipe")["id"] == ok_result.recipe_id
    assert store.records.read_spec(ok_result.operation_id, family="operation")["id"] == ok_result.operation_id

    record = ExecutionRecord.from_envelope(store.records.read_record(ok_result.execution_record_id))
    stdout = store.records.resolve_storage_ref(record.logs[0].storage, record_id=ok_result.execution_record_id)
    stderr = store.records.resolve_storage_ref(record.logs[1].storage, record_id=ok_result.execution_record_id)
    assert "hello stdout" in stdout.read_text(encoding="utf-8")
    assert "hello stderr" in stderr.read_text(encoding="utf-8")

    names = [event.name for event in capture.events]
    assert "dryml.dispatch.plan.start" in names
    assert "dryml.dispatch.worker.launch" in names
    assert "dryml.dispatch.complete" in names

    fail_result = dispatcher.run(fail, environment=_env(target_module))

    assert fail_result.status == "failed"
    assert fail_result.execution_record_id
    assert fail_result.error == {
        "type": "ValueError",
        "message": secret,
    }
    failure_record_data = store.records.read_record(fail_result.execution_record_id)
    failure_record = ExecutionRecord.from_envelope(failure_record_data)
    assert failure_record.error.to_json() == {
        "type": "ValueError",
        "metadata": {"code": "execution_failed"},
    }
    source_durable = (*store.records.iter_records(), *store.records.iter_specs())
    assert secret not in str(source_durable)

    exported = DirStore(tmp_path / "exported", query_index="none")
    copy_record_closure(
        store,
        exported,
        seed_records=[fail_result.execution_record_id],
        policy="descriptive",
    )
    exported_durable = (
        *exported.records.iter_records(),
        *exported.records.iter_specs(),
    )
    assert secret not in str(exported_durable)
    assert store.records.find_execution_records(operation_id=ok["id"], status="ok")[0].record_id == ok_result.execution_record_id
    assert store.records.find_execution_records(operation_id=fail["id"], status="failed")[0].record_id == fail_result.execution_record_id


def test_result_object_saved_to_store(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = _env(target_module)
    op = attach_operation_id(make_function_call_spec("dispatch_target:make_box", args=[4]))

    result = Dispatcher(store=store).run(op, environment=env)

    assert result.status == "ok"
    assert result.result_canonical.startswith("cdef-v4-")
    assert result.result_cdef_ids == (result.result_canonical,)
    assert result.produced_record_ids == ()
