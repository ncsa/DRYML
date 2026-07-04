import sys

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec


def _env(target_module):
    return PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()


def test_subprocess_function_call_success(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    op = attach_operation_id(make_function_call_spec("dispatch_target:add", args=[2, 3]))

    result = Dispatcher(store=store).run(op, environment=_env(target_module))

    assert result.status == "ok"
    assert result.result_canonical == 5
    assert result.execution_record_id


def test_function_call_failure_returns_structured_result(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    op = attach_operation_id(make_function_call_spec("dispatch_target:fail"))

    result = Dispatcher(store=store).run(op, environment=_env(target_module))

    assert result.status == "failed"
    assert result.execution_record_id
