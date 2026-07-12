from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher, plan, run, submit
from dryml.dispatch.errors import DispatchPlanningError
from dryml.environments import PythonExecutableSpec


def _env(target_module):
    return PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()


def _tests_env():
    return PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(Path(__file__).parents[1]),)).to_data()


def test_plan_submit_and_run_importable_function_without_pickle(tmp_path, target_module):
    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")

    planned = plan(mod.add, store=store, args=(2, 4))
    assert planned.envelope.operation_spec["kind"] == "function_call"
    assert planned.envelope.operation_spec["payload"]["function"] == "dispatch_target:add"
    assert planned.envelope.launch["call_transport"] == "import_ref"

    future = submit(mod.add, store=store, args=(3, 5), environment=_env(target_module))
    assert future.result(timeout=10).status == "ok"

    result = run(mod.add, store=store, args=(6, 7), environment=_env(target_module))
    assert result.status == "ok", result.error
    assert result.result_canonical == 13


def test_non_importable_callable_public_api_pickle_policy(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")

    def local_add(left, right):
        return left + right

    with pytest.raises(DispatchPlanningError, match="allow_pickle=True"):
        Dispatcher(store=store).plan(local_add, args=(1, 2))

    result = Dispatcher(store=store).run(local_add, allow_pickle=True, args=(4, 5), environment=_tests_env())
    assert result.status == "ok", result.error
    assert result.result_canonical == 9

    lambda_result = Dispatcher(store=store).run(lambda value: value + 1, allow_pickle=True, args=(9,), environment=_tests_env())
    assert lambda_result.status == "ok"
    assert lambda_result.result_canonical == 10


def test_dispatcher_submit_accepts_operation_and_forwards_environment_world(tmp_path, target_module):
    mod = importlib.import_module("dispatch_target")
    store = DirStore(tmp_path / "store", query_index="none")
    environment = _env(target_module)
    world = {"roles": {"main": {"replicas": 1, "process": {}}}}

    dispatcher = Dispatcher(store=store)
    planned = dispatcher.plan(mod.add, args=(1, 2), environment=environment, world=world)
    assert planned.envelope.environment_spec == environment
    assert planned.envelope.dispatch_spec["payload"]["environment"]["policy"] == "explicit"
    assert planned.envelope.dispatch_spec["payload"]["world"]["policy"] == "explicit"
    assert planned.envelope.dispatch_spec["payload"]["world"]["spec"]["roles"]["main"]["replicas"] == 1

    future = dispatcher.submit(mod.add, args=(1, 2), environment=environment, world=world)
    assert future.result(timeout=10).status == "ok"
