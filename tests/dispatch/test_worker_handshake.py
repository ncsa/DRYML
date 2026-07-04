import dataclasses
import sys

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher, WorkerStoreRef
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec


def test_handshake_success_and_missing_store_failure(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    op = attach_operation_id(make_function_call_spec("dispatch_target:add", args=[1, 2]))
    dispatcher = Dispatcher(store=store)
    ok = dispatcher.run(op, environment=env)
    assert ok.status == "ok"

    plan = dispatcher.plan(op, environment=env)
    bad_envelope = dataclasses.replace(plan.envelope, store_refs=(WorkerStoreRef("dir_store", "shared", str(tmp_path / "missing")),))
    bad_plan = dataclasses.replace(plan, envelope=bad_envelope)
    response = dispatcher.submit(bad_plan).result(timeout=10)
    assert response.status == "failed"
