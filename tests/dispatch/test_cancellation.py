import os

import pytest

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher, DispatchTimeout
from dryml.environments import CurrentEnvironmentSpec
from dryml.operations import attach_operation_id, make_function_call_spec


def _plan(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    environment = CurrentEnvironmentSpec().to_data()
    op = attach_operation_id(make_function_call_spec("time:sleep", args=[60]))
    dispatcher = Dispatcher(store=store)
    return dispatcher, dispatcher.plan(op, environment=environment), store


def test_sleeping_worker_cancellation_records_cancelled(tmp_path):
    dispatcher, plan, store = _plan(tmp_path)
    future = dispatcher.submit(plan)

    assert future.cancel(grace=0.1, reason="test") is True
    response = future.result(timeout=5)

    assert response.status == "cancelled"
    assert store.records.find_execution_records(status="cancelled")


def test_cancellation_immediately_removes_launch_artifacts_and_keeps_response(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    environment = CurrentEnvironmentSpec().to_data()

    def sleeping_local_callable():
        import time

        time.sleep(60)

    dispatcher = Dispatcher(store=store)
    plan = dispatcher.plan(sleeping_local_callable, allow_pickle=True, environment=environment)
    future = dispatcher.submit(plan)
    work_dir = future.work_dir
    cleanup_paths = tuple(plan.envelope.launch["cleanup_paths"])

    assert future.cancel(grace=0.1, reason="test") is True
    assert not os.path.exists(work_dir)
    assert all(not os.path.exists(path) for path in cleanup_paths)
    assert future.result(timeout=5).status == "cancelled"


def test_timeout_records_timeout(tmp_path):
    dispatcher, plan, store = _plan(tmp_path)
    future = dispatcher.submit(plan)

    with pytest.raises(DispatchTimeout):
        future.result(timeout=0.1)
    assert store.records.find_execution_records(status="timeout")
    assert store.records.find_execution_records(status="cancelled") == ()
