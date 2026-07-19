from __future__ import annotations

import importlib
import multiprocessing
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import pytest

from dryml.core2.store.dir import DirStore
from dryml.core2.store.zip import ZipStore
from dryml.dispatch import Dispatcher
from dryml.dispatch.errors import DispatchPlanningError
from dryml.environments import PythonExecutableSpec
from dryml.managed import ControlRequest, ManagedCallback
from dryml.managed.store import OperationLease
from dryml.operations import attach_operation_id, make_method_call_spec
from dryml.records import ExecutionRecord


def _env(target_module):
    return PythonExecutableSpec(
        sys.executable,
        pythonpath_policy="explicit",
        extra_pythonpath=(str(target_module.parent),),
    ).to_data()


def _managed_box():
    return importlib.import_module("dispatch_target").ManagedBox()


def _coordinator_process(store_path, module_path):
    sys.path.insert(0, module_path)
    try:
        box = importlib.import_module("dispatch_target").ManagedBox()
        environment = PythonExecutableSpec(
            sys.executable,
            pythonpath_policy="explicit",
            extra_pythonpath=(module_path,),
        ).to_data()
        Dispatcher(store=DirStore(store_path, query_index="none")).run(
            box.compute,
            kwargs={"sleep": 2.0},
            environment=environment,
            timeout=20,
        )
    finally:
        sys.path.remove(module_path)


def _read_output(store, result):
    record_id = result.managed_result["outputs"]["result"]["record_id"]
    record = store.records.read_record(record_id)
    root = store.records.resolve_storage_ref(
        record["payload"]["storage"][0],
        record_id=record_id,
    )
    return root.joinpath("value.bin").read_bytes()


def test_managed_plan_uses_ordinary_method_identity_and_launch_only_ticket(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    box = _managed_box()

    plan = Dispatcher(store=store).plan(
        box.compute,
        args=("planned",),
        environment=_env(target_module),
    )
    ordinary = attach_operation_id(
        make_method_call_spec(
            plan.envelope.operation_spec["payload"]["subject"],
            "compute",
            args=("planned",),
        )
    )

    assert plan.envelope.operation_spec["id"] == ordinary["id"]
    assert plan.envelope.operation_spec["payload"] == ordinary["payload"]
    assert "managed" not in plan.envelope.operation_spec
    assert "managed" not in plan.envelope.operation_spec.get("metadata", {})
    assert plan.envelope.launch["managed"]["schema"] == "dryml.managed.operation_launch.v1"
    assert "managed.operation.v1" in plan.envelope.handshake["required_features"]
    assert plan.resolution.requirements.environment_requirement.python == ">=3"
    world_requirement = plan.resolution.requirements.world_requirement.to_data()
    assert world_requirement["roles"]["main"]["resources"]["cpus"]["exact"] == 1


def test_managed_worker_context_effects_callbacks_and_publication_round_trip(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    box = _managed_box()
    events = []
    callback = ManagedCallback(lambda event: events.append(event.kind), fail_soft=True)

    result = Dispatcher(store=store).run(
        box.compute,
        args=("hello",),
        callbacks=(callback,),
        environment=_env(target_module),
        timeout=10,
    )

    assert result.status == "ok", result.error
    assert result.managed_result["action"] == "start"
    assert result.managed_result["realization_id"].startswith("realization-v1-")
    assert result.managed_result["consumed_records"] == []
    assert result.produced_record_ids == (
        result.managed_result["outputs"]["result"]["record_id"],
    )
    assert _read_output(store, result) == b"hello:worker:0"
    assert "progress" in events
    assert "safe_point" in events
    assert "completed" in events
    assert box.compute.status(store=store).status == "completed"


def test_managed_submit_services_worker_without_waiting_in_result(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    box = _managed_box()
    dispatcher = Dispatcher(store=store)
    future = dispatcher.submit(
        box.compute,
        environment=_env(target_module),
    )
    deadline = time.monotonic() + 10
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert future.done()
    assert box.compute.status(store=store).status == "completed"
    assert future.result(timeout=1).status == "ok"


def test_exact_consumed_and_produced_records_round_trip(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    module = importlib.import_module("dispatch_target")
    producer = module.ManagedBox()
    produced = Dispatcher(store=store).run(
        producer.compute,
        environment=_env(target_module),
        timeout=10,
    )
    consumer = module.ManagedConsumer(producer.compute.result)

    consumed = Dispatcher(store=store).run(
        consumer.compute,
        environment=_env(target_module),
        timeout=10,
    )

    assert consumed.status == "ok", consumed.error
    vector = consumed.managed_result["consumed_records"]
    assert len(vector) == 1
    assert vector[0]["record_id"] == produced.produced_record_ids[0]
    assert vector[0]["realization_id"] == produced.managed_result["realization_id"]
    assert vector[0]["method"] == "compute"
    assert vector[0]["output_slot"] == "result"
    assert vector[0]["activation_generation"] == 1
    execution = store.records.read_record(consumed.execution_record_id)
    assert execution["payload"]["consumed_records"][0]["record_id"] == produced.produced_record_ids[0]


def test_callback_interrupt_crosses_safe_point_and_preserves_checkpoint(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    box = _managed_box()
    requested = False

    def interrupt(event):
        nonlocal requested
        if event.kind == "progress" and not requested:
            requested = True
            return ControlRequest.INTERRUPT
        return None

    result = Dispatcher(store=store).run(
        box.compute,
        callbacks=(ManagedCallback(interrupt, controls={ControlRequest.INTERRUPT}),),
        environment=_env(target_module),
        timeout=10,
    )

    assert result.status == "cancelled"
    assert result.cancellation["reason"] == "managed_interrupt"
    assert result.managed_result["checkpoint_head"].startswith("checkpoint-v1-")
    assert box.compute.status(store=store).status == "interrupted"


def test_worker_failure_retains_structured_effects_and_prior_active(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    box = _managed_box()
    secret = "managed-worker-secret-sentinel-b15c"
    first = Dispatcher(store=store).run(
        box.compute,
        args=("old",),
        environment=_env(target_module),
        timeout=10,
    )

    failed = Dispatcher(store=store).run(
        box.compute,
        args=("new",),
        kwargs={"fail": secret},
        rerun=True,
        environment=_env(target_module),
        timeout=10,
    )

    assert failed.status == "failed"
    assert failed.error == {
        "type": "RuntimeError",
        "metadata": {"code": "execution_failed"},
    }
    assert failed.managed_result["status"] == "failed"
    assert failed.managed_result["effects"]["result"]["slot"] == "result"
    failure_record = ExecutionRecord.from_envelope(
        store.records.read_record(failed.execution_record_id)
    )
    assert failure_record.error.to_json() == failed.error
    assert secret not in str(failure_record.to_envelope())
    assert box.compute.history(store=store)[-1].diagnostics == (
        "RuntimeError: execution_failed",
    )
    assert box.compute.results(store=store)["result"].record_id == first.produced_record_ids[0]


def test_activation_failure_keeps_verified_rerun_inactive_and_old_active(
    tmp_path, target_module, monkeypatch
):
    store = DirStore(tmp_path / "store", query_index="none")
    box = _managed_box()
    secret = "managed-finalization-secret-sentinel-640a"
    first = Dispatcher(store=store).run(
        box.compute,
        environment=_env(target_module),
        timeout=10,
    )

    def fail_activation(self, realization_id):
        raise RuntimeError(secret)

    monkeypatch.setattr(OperationLease, "activate", fail_activation)
    failed = Dispatcher(store=store).run(
        box.compute,
        rerun=True,
        environment=_env(target_module),
        timeout=10,
    )

    assert failed.status == "failed"
    assert failed.error == {
        "type": "RuntimeError",
        "metadata": {"code": "execution_failed"},
    }
    failure_record = ExecutionRecord.from_envelope(
        store.records.read_record(failed.execution_record_id)
    )
    assert failure_record.error.to_json() == failed.error
    assert secret not in str(failure_record.to_envelope())
    assert box.compute.results(store=store)["result"].record_id == first.produced_record_ids[0]
    history = box.compute.history(store=store)
    assert [item.status for item in history] == ["completed", "completed"]
    assert history[1].realization_record_id is not None


def test_worker_death_without_response_is_failed_and_cannot_activate(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    box = _managed_box()

    result = Dispatcher(store=store).run(
        box.compute,
        kwargs={"hard_exit": True},
        environment=_env(target_module),
        timeout=10,
    )

    assert result.status == "failed"
    assert result.error["type"] == "WorkerProtocolError"
    assert result.managed_result["status"] == "failed"
    assert box.compute.status(store=store).status == "failed"
    assert box.compute.results(store=store) == {}


@pytest.mark.skipif(os.name != "posix", reason="process-death lock proof uses fork")
def test_coordinator_death_fences_orphan_worker_and_rerun_publishes(tmp_path, target_module):
    store_path = str(tmp_path / "store")
    process = multiprocessing.get_context("fork").Process(
        target=_coordinator_process,
        args=(store_path, str(target_module.parent)),
    )
    process.start()
    store = DirStore(store_path, query_index="none")
    box = _managed_box()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if box.compute.status(store=store).status == "running":
                break
        except Exception:
            pass
        time.sleep(0.05)
    else:
        process.terminate()
        process.join(timeout=5)
        pytest.fail("managed coordinator did not create a running attempt")

    process.terminate()
    process.join(timeout=5)
    assert process.exitcode is not None

    resumed = Dispatcher(store=store).run(
        box.compute,
        rerun=True,
        environment=_env(target_module),
        timeout=10,
    )

    assert resumed.status == "ok", resumed.error
    assert resumed.managed_result["action"] == "rerun"
    history = box.compute.history(store=store)
    assert len(history) == 2
    assert history[0].status == "abandoned"
    assert history[0].realization_record_id is None
    assert box.compute.status(store=store).active_realization_id == history[1].realization_id


def test_managed_local_world_and_zip_reject_before_mutation(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    box = _managed_box()

    with pytest.raises(DispatchPlanningError, match="single local subprocess"):
        Dispatcher(store=store).plan_world(
            box.compute,
            environment=_env(target_module),
            world={"roles": {"main": {"replicas": 1, "process": {}}}},
        )
    assert not Path(store.managed_control_root()).exists()
    assert not store.has(box.definition)

    archive = BytesIO()
    with pytest.raises(DispatchPlanningError, match="DirStore"):
        Dispatcher(store=ZipStore(archive)).plan(box.compute)
    assert archive.getvalue() == b""


def test_managed_hard_world_requirement_cannot_be_overridden(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")

    with pytest.raises(DispatchPlanningError, match="requirement"):
        Dispatcher(store=store).plan(
            _managed_box().compute,
            environment=_env(target_module),
            world={
                "roles": {
                    "main": {
                        "replicas": 1,
                        "process": {"resources": {"cpus": 2}},
                    }
                }
            },
        )
