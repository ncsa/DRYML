import sys

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec
from dryml.records import ExecutionRecord


def test_stdout_stderr_captured_as_execution_products(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    op = attach_operation_id(make_function_call_spec("dispatch_target:noisy_add", args=[1, 2]))

    result = Dispatcher(store=store).run(op, environment=env)
    record = ExecutionRecord.from_envelope(store.records.read_record(result.execution_record_id))

    stdout = store.records.resolve_storage_ref(record.logs[0].storage, record_id=result.execution_record_id)
    stderr = store.records.resolve_storage_ref(record.logs[1].storage, record_id=result.execution_record_id)
    assert "hello stdout" in stdout.read_text(encoding="utf-8")
    assert "hello stderr" in stderr.read_text(encoding="utf-8")
