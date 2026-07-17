import sys

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec


def _env(target_module):
    return PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()


def test_success_and_failure_execution_records_query(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    dispatcher = Dispatcher(store=store)
    ok = attach_operation_id(make_function_call_spec("dispatch_target:add", args=[2, 3]))
    fail = attach_operation_id(make_function_call_spec("dispatch_target:fail"))

    ok_result = dispatcher.run(ok, environment=_env(target_module))

    assert ok_result.status == "ok"
    assert ok_result.result_canonical == 5
    assert ok_result.execution_record_id
    assert store.records.read_spec(ok_result.dispatch_id, family="dispatch")["id"] == ok_result.dispatch_id
    assert store.records.read_spec(ok_result.recipe_id, family="execution_recipe")["id"] == ok_result.recipe_id
    assert store.records.read_spec(ok_result.operation_id, family="operation")["id"] == ok_result.operation_id

    fail_result = dispatcher.run(fail, environment=_env(target_module))

    assert fail_result.status == "failed"
    assert fail_result.execution_record_id
    assert fail_result.error["type"] == "ValueError"
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
