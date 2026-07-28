import sys

from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec


def test_worker_runtime_active_before_target_import(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    op = attach_operation_id(make_function_call_spec("dispatch_target:runtime_status"))

    result = Dispatcher(store=store).run(op, environment=env)

    assert result.result_canonical["mode"] == "worker"
    assert result.result_canonical["bootstrap"] == "1"
    assert result.result_canonical["import_mode"] == "worker"
    assert result.result_canonical["enforcement"] == "strict"


def test_worker_configured_framework_import_succeeds_with_runtime_spec(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    runtime = {"mode": "worker", "frameworks": {"torch": {"num_threads": 3}}, "device_visibility": {"policy": "assigned"}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:configured_torch_import_status"))

    result = Dispatcher(store=store).run(op, environment=env, runtime=runtime)

    assert result.result_canonical["mode"] == "worker"
    assert result.result_canonical["bootstrap"] == "1"
    assert result.result_canonical["marker"] == "fake-dispatch-torch"
    assert result.result_canonical["threads"] == 3
