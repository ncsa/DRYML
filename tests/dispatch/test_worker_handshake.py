import dataclasses
import sys

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher, LocalSubprocessFuture, WorkerResponse, WorkerStoreRef
from dryml.dispatch.protocol import write_json_file
from dryml.formats.ids import content_id
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec


class _DoneProcess:
    returncode = 0
    pid = 1

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


def test_missing_store_fails_handshake(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    op = attach_operation_id(make_function_call_spec("dispatch_target:add", args=[1, 2]))
    dispatcher = Dispatcher(store=store)
    plan = dispatcher.plan(op, environment=env)
    bad_envelope = dataclasses.replace(plan.envelope, store_refs=(WorkerStoreRef("dir_store", "shared", str(tmp_path / "missing")),))
    bad_plan = dataclasses.replace(plan, envelope=bad_envelope)
    response = dispatcher.submit(bad_plan).result(timeout=10)
    assert response.status == "unsupported"


def test_unsupported_feature_handshake_is_authoritative(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    op = attach_operation_id(make_function_call_spec("dispatch_target:add", args=[1, 2]))
    dispatcher = Dispatcher(store=store)
    plan = dispatcher.plan(op, environment=env)
    bad_envelope = dataclasses.replace(plan.envelope, handshake={"min_protocol": 1, "required_features": ["missing.feature"]})

    response = dispatcher.submit(dataclasses.replace(plan, envelope=bad_envelope)).result(timeout=10)

    assert response.status == "unsupported"
    assert response.error["type"] == "WorkerHandshakeError"


def test_ok_response_without_ok_handshake_is_rejected(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    op = attach_operation_id(make_function_call_spec("dispatch_target:add", args=[1, 2]))
    plan = Dispatcher(store=store).plan(op, environment=env)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    response_path = work_dir / "response.json"
    write_json_file(str(response_path), WorkerResponse(status="ok", operation_id=op["id"]).to_json())
    future = LocalSubprocessFuture(
        _DoneProcess(),
        plan,
        str(work_dir),
        str(work_dir / "request.json"),
        str(work_dir / "missing-handshake.json"),
        str(response_path),
        str(work_dir / "stdout.txt"),
        str(work_dir / "stderr.txt"),
        preserve_work_dir=True,
    )

    future._read_response()

    assert future._response.status == "failed"
    assert future._response.error["type"] == "WorkerHandshakeError"


def test_worker_rejects_inconsistent_envelope_ids(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    op = attach_operation_id(make_function_call_spec("dispatch_target:add", args=[1, 2]))
    dispatcher = Dispatcher(store=store)
    plan = dispatcher.plan(op, environment=env)
    bad_recipe = dict(plan.execution_recipe)
    bad_recipe["payload"] = {**bad_recipe["payload"], "operation_id": content_id("op", 1, {"other": True})}
    bad_envelope = dataclasses.replace(plan.envelope, execution_recipe=bad_recipe)

    response = dispatcher.submit(dataclasses.replace(plan, execution_recipe=bad_recipe, envelope=bad_envelope)).result(timeout=10)

    assert response.status == "failed"
    assert response.error["type"] == "WorkerProtocolError"
