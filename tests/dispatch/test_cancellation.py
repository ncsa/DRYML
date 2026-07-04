import sys

import pytest

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher, DispatchTimeout
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec


def _plan(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    op = attach_operation_id(make_function_call_spec("dispatch_target:sleep_forever"))
    dispatcher = Dispatcher(store=store)
    return dispatcher, dispatcher.plan(op, environment=env), store


def test_sleeping_worker_cancellation_records_cancelled(tmp_path, target_module):
    dispatcher, plan, store = _plan(tmp_path, target_module)
    future = dispatcher.submit(plan)

    assert future.cancel(grace=0.1, reason="test") is True
    response = future.result(timeout=5)

    assert response.status == "cancelled"
    assert store.records.find_execution_records(status="cancelled")


def test_timeout_records_timeout(tmp_path, target_module):
    dispatcher, plan, store = _plan(tmp_path, target_module)
    future = dispatcher.submit(plan)

    with pytest.raises(DispatchTimeout):
        future.result(timeout=0.1)
    assert store.records.find_execution_records(status="timeout")
