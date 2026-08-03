import dataclasses
import importlib
import os
import sys
from pathlib import Path

import pytest

from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher, ExecutionEnvelope, LocalSubprocessBackend, LocalSubprocessFuture, WorkerHandshakeResponse, WorkerResponse, WorkerStoreRef, attach_recipe_id
from dryml.dispatch.backends import _require_requested_features
from dryml.dispatch.protocol import DISPATCH_WORKER_PROTOCOL_SCHEMA
from dryml.dispatch.protocol import write_json_file
from dryml.formats.ids import content_id
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec
from dryml.worlds import LocalResourceInventory


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
    bad_envelope = dataclasses.replace(plan.envelope, handshake={"min_protocol": 1, "required_features": ["runtime.worker_session.v2", "missing.feature"]})

    response = dispatcher.submit(dataclasses.replace(plan, envelope=bad_envelope)).result(timeout=10)

    assert response.status == "unsupported"
    assert response.error["type"] == "WorkerHandshakeError"


def test_managed_feature_rejection_precedes_control_mutation(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    environment = PythonExecutableSpec(
        sys.executable,
        pythonpath_policy="explicit",
        extra_pythonpath=(str(target_module.parent),),
    ).to_data()
    box = importlib.import_module("dispatch_target").ManagedBox()
    dispatcher = Dispatcher(store=store)
    plan = dispatcher.plan(box.compute, environment=environment)
    bad_envelope = dataclasses.replace(
        plan.envelope,
        handshake={"min_protocol": 1, "required_features": ["runtime.worker_session.v2", "managed.operation.v2"]},
    )

    response = dispatcher.submit(
        dataclasses.replace(plan, envelope=bad_envelope)
    ).result(timeout=10)

    assert response.status == "unsupported"
    assert response.error["type"] == "WorkerHandshakeError"
    assert not Path(store.managed_control_root()).exists()


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
    bad_recipe.pop("id")
    bad_recipe["payload"] = {**bad_recipe["payload"], "operation_id": content_id("op", 1, {"other": True})}
    bad_recipe = attach_recipe_id(bad_recipe)
    bad_envelope = dataclasses.replace(plan.envelope, execution_recipe=bad_recipe)

    response = dispatcher.submit(dataclasses.replace(plan, execution_recipe=bad_recipe, envelope=bad_envelope)).result(timeout=10)

    assert response.status == "failed"
    assert response.error["type"] == "WorkerProtocolError"
    assert response.error["message"] == (
        "execution envelope operation/dispatch/recipe IDs are inconsistent"
    )


@pytest.mark.parametrize("spec_name", ("operation", "dispatch", "recipe"))
def test_worker_rejects_self_consistent_links_with_stale_spec_id(
    tmp_path,
    target_module,
    spec_name,
):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(
        sys.executable,
        pythonpath_policy="explicit",
        extra_pythonpath=(str(target_module.parent),),
    ).to_data()
    op = attach_operation_id(
        make_function_call_spec("dispatch_target:add", args=[1, 2])
    )
    dispatcher = Dispatcher(store=store)
    plan = dispatcher.plan(op, environment=env, record_policy="none")
    field = {
        "operation": "operation_spec",
        "dispatch": "dispatch_spec",
        "recipe": "execution_recipe",
    }[spec_name]
    bad_spec = dict(getattr(plan.envelope, field))
    if spec_name == "operation":
        bad_spec["payload"] = {**bad_spec["payload"], "args": [2, 2]}
    elif spec_name == "dispatch":
        bad_spec["payload"] = {
            **bad_spec["payload"],
            "metadata": {
                **bad_spec["payload"].get("metadata", {}),
                "tampered": True,
            },
        }
    else:
        bad_spec["payload"] = {
            **bad_spec["payload"],
            "constraints": {
                **bad_spec["payload"].get("constraints", {}),
                "tampered": True,
            },
        }
    bad_envelope = dataclasses.replace(plan.envelope, **{field: bad_spec})
    plan_updates = {"envelope": bad_envelope}
    if spec_name != "operation":
        plan_updates[field] = bad_spec

    response = dispatcher.submit(
        dataclasses.replace(plan, **plan_updates)
    ).result(timeout=10)

    assert response.status == "failed"
    assert response.error["type"] == "WorkerProtocolError"
    assert response.error["message"] == (
        "execution envelope contains an invalid canonical spec"
    )


def test_worker_rejects_missing_recipe_id_before_execution(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(
        sys.executable,
        pythonpath_policy="explicit",
        extra_pythonpath=(str(target_module.parent),),
    ).to_data()
    op = attach_operation_id(
        make_function_call_spec("dispatch_target:add", args=[1, 2])
    )
    dispatcher = Dispatcher(store=store)
    plan = dispatcher.plan(op, environment=env, record_policy="none")
    bad_recipe = dict(plan.execution_recipe)
    bad_recipe.pop("id")
    bad_envelope = dataclasses.replace(
        plan.envelope,
        execution_recipe=bad_recipe,
    )

    response = dispatcher.submit(
        dataclasses.replace(
            plan,
            execution_recipe=bad_recipe,
            envelope=bad_envelope,
        )
    ).result(timeout=10)

    assert response.status == "failed"
    assert response.error["type"] == "WorkerProtocolError"
    assert response.error["message"] == (
        "execution envelope operation/dispatch/recipe IDs are inconsistent"
    )
    assert response.execution_record_id is None
    assert list(store.records.iter_records()) == []


def test_parent_rejects_ok_handshake_missing_requested_feature(tmp_path):
    envelope = ExecutionEnvelope(
        dispatch_spec={"schema": "dryml.dispatch.v1", "schema_version": 1, "kind": "dispatch", "payload": {"operation_id": "op-v1-test"}},
        execution_recipe={"schema": "dryml.execution_recipe.v1", "schema_version": 1, "kind": "execution_recipe", "payload": {"operation_id": "op-v1-test", "dispatch_id": "dispatch-v1-test", "backend": {"name": "dryml.local_subprocess"}}},
        operation_spec={"schema": "dryml.operation.v1", "schema_version": 1, "kind": "function_call", "payload": {"function": "operator:add", "args": [1, 2]}},
        environment_spec={"kind": "current", "schema_version": 1},
        world_spec={"roles": {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}},
        runtime_spec={"mode": "worker", "device_visibility": {"policy": "assigned"}},
        allocation_view={"world_allocation_id": "worldalloc-v1-test", "role": "worker", "replica": 0, "rank": 0, "local_rank": 0, "cpus": [0], "accelerators": {}, "env": {}, "metadata": {}},
        requirement_policy="strict",
        requirement_axes=["environment", "world", "runtime"],
        handshake={"min_protocol": 1, "required_features": ["runtime.accelerator_memory.v1", "runtime.worker_session.v2"]},
    )
    handshake = WorkerHandshakeResponse(
        status="ok",
        protocol_schema=DISPATCH_WORKER_PROTOCOL_SCHEMA,
        protocol_version=1,
        dryml_version=None,
        python_version="3.x",
        platform="test",
        pid=1,
        features=("runtime.worker",),
        operation_kinds=(),
        call_transports=(),
        store_ref_kinds=(),
        record_schemas={},
        runtime_modes=("worker",),
    )

    with pytest.raises(Exception, match="omitted requested features"):
        _require_requested_features(envelope, handshake)


def test_old_worker_missing_accelerator_memory_feature_never_receives_start(monkeypatch, tmp_path):
    script = tmp_path / "old_worker.py"
    side_effect = tmp_path / "target-imported"
    script.write_text(
        """
import json
import os
import pathlib
import sys
import time

def argument(name):
    return sys.argv[sys.argv.index(name) + 1]

request = json.loads(pathlib.Path(argument("--request")).read_text(encoding="utf-8"))
handshake = {
    "status": "ok",
    "protocol_schema": "dryml.dispatch.worker_protocol.v1",
    "protocol_version": 1,
    "dryml_version": None,
    "python_version": sys.version.split()[0],
    "platform": "fake-old-worker",
    "pid": os.getpid(),
    "features": ["operation.function_call", "store.dir", "runtime.worker"],
    "operation_kinds": ["function_call"],
    "call_transports": ["import_ref"],
    "store_ref_kinds": ["dir_store"],
    "record_schemas": {},
    "runtime_modes": ["worker"],
    "store_status": {},
    "diagnostics": [],
}
pathlib.Path(argument("--handshake")).write_text(json.dumps(handshake), encoding="utf-8")
start_path = pathlib.Path(request["launch"]["coordination"]["start_path"])
deadline = time.monotonic() + 5
while time.monotonic() < deadline and not start_path.exists():
    time.sleep(0.01)
if start_path.exists():
    pathlib.Path(os.environ["DRYML_OLD_WORKER_EFFECT"]).write_text("imported", encoding="utf-8")
""",
        encoding="utf-8",
    )
    child_env = dict(os.environ)
    child_env["DRYML_OLD_WORKER_EFFECT"] = str(side_effect)
    monkeypatch.setattr(
        "dryml.dispatch.backends.build_worker_command",
        lambda environment: ([sys.executable, str(script)], child_env),
    )
    store = DirStore(tmp_path / "store", query_index="none")
    plan = Dispatcher(store=store).plan(
        make_function_call_spec("operator:add", args=[1, 2]),
        world={
            "roles": {
                "main": {
                    "replicas": 1,
                    "process": {
                        "resources": {
                            "accelerators": {"gpu": 1},
                            "accelerator_memory": {"gpu": ["1GiB"]},
                        }
                    },
                }
            }
        },
        inventory=LocalResourceInventory(
            (0,),
            {"gpu": ("gpu-a",)},
            accelerator_memory={"gpu": {"gpu-a": "2GiB"}},
        ),
        requirement_policy="ignore",
    )

    with pytest.raises(Exception, match="omitted requested features"):
        LocalSubprocessBackend().submit(plan)

    assert not side_effect.exists()
