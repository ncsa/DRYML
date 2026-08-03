import os
import signal
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher, LocalResourceInventory, LocalSubprocessFuture, LocalWorldBackend, LocalWorldFuture, WorkerResponse, WorldWorkerKey, allocate_local_world
from dryml.dispatch.errors import DispatchPlanningError
from dryml.dispatch.protocol import DISPATCH_WORKER_PROTOCOL_SCHEMA, WorkerHandshakeResponse, write_json_file
from dryml.dispatch.worker import _wait_for_start_barrier
from dryml.environments import PythonExecutableSpec
from dryml.formats import json_ready
from dryml.operations import attach_operation_id, make_function_call_spec
from dryml.worlds import WorldSpec


def _env(target_module):
    return PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()


def _inventory():
    return LocalResourceInventory(cpus=(0, 1, 2, 3), accelerators={"gpu": (0, 1)}, memory=2 * 1024**3, metadata={"source": "test"})


class _FakeProcess:
    returncode = 0
    pid = 1

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


class _LeaderExitsOnSignalProcess:
    pid = 123

    def __init__(self):
        self.returncode = None
        self.signals = []
        self.kill_calls = 0

    def poll(self):
        return self.returncode

    def send_signal(self, signal):
        self.signals.append(signal)
        self.returncode = 0

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.kill_calls += 1


class _FakeWorldWorkerFuture:
    def __init__(self, tmp_path, plan, key, *, handshake=None, done=False):
        self.plan = plan.worker_plans[0]
        self.key = key
        self.work_dir = str(tmp_path / key.label())
        (tmp_path / key.label()).mkdir(exist_ok=True)
        self.handshake_path = str(tmp_path / key.label() / "handshake.json")
        self.response_path = str(tmp_path / key.label() / "response.json")
        self.stdout_path = str(tmp_path / key.label() / "stdout.txt")
        self.stderr_path = str(tmp_path / key.label() / "stderr.txt")
        self.process = _FakeProcess()
        self._done = done
        self._response = None
        self._handshake = None
        if handshake is not None:
            write_json_file(self.handshake_path, handshake.to_json())

    def done(self):
        return self._done

    def cancel(self, *, grace=None, reason="user", record=True):
        self._done = True
        if record and self._response is None:
            self._response = WorkerResponse(status="cancelled", cancellation={"requested": True, "reason": reason}, diagnostics=({"message": "fake cancellation"},))
        return True

    def _read_response(self):
        return None

    def _parent_failure_response(self, status, *, error=None, cancellation=None):
        self._response = WorkerResponse(status=status, operation_id=self.plan.envelope.operation_spec.get("id"), dispatch_id=self.plan.dispatch_spec.get("id"), recipe_id=self.plan.execution_recipe.get("id"), error=error, cancellation=cancellation, diagnostics=({"message": "fake parent failure"},))
        return self._response

    def _persist_logs(self, record_id):
        return None


def _handshake(plan, key, *, status="ok", worker_key=None, world_id=None, world_allocation_id=None, diagnostics=()):
    return WorkerHandshakeResponse(
        status=status,
        protocol_schema=DISPATCH_WORKER_PROTOCOL_SCHEMA,
        protocol_version=1,
        dryml_version=None,
        python_version="3.x",
        platform="test",
        pid=1,
        features=("operation.function_call", "store.dir", "runtime.worker", "runtime.worker_session.v2"),
        operation_kinds=("function_call",),
        call_transports=("import_ref",),
        store_ref_kinds=("dir_store",),
        record_schemas={"record": 1},
        runtime_modes=("worker",),
        world_id=world_id if world_id is not None else plan.world_spec["id"],
        world_allocation_id=world_allocation_id if world_allocation_id is not None else plan.world_allocation_spec["id"],
        worker_key=worker_key if worker_key is not None else key.to_json(),
        diagnostics=diagnostics,
    )


def _fake_future(tmp_path, plan, worker_futures, *, handshake_timeout=0.05):
    return LocalWorldFuture(
        plan=plan,
        group_work_dir=str(tmp_path / "group"),
        start_path=str(tmp_path / "group" / "start.json"),
        cancel_path=str(tmp_path / "group" / "cancel.json"),
        workers=worker_futures,
        preserve_work_dir=True,
        handshake_timeout=handshake_timeout,
        cancel_grace=0.01,
    )


def test_allocate_local_world_two_roles_and_gpu():
    world = {
        "trainer": {"replicas": 1, "process": {"resources": {"cpus": 2, "accelerators": {"gpu": 1}, "memory": "1GiB"}}},
        "data": {"replicas": 1, "process": {"resources": {"cpus": 1}}},
    }

    plan = allocate_local_world(world, inventory=_inventory())

    assert plan.worker_keys == (WorldWorkerKey("data", 0, 0, 0), WorldWorkerKey("trainer", 0, 1, 1))
    data = plan.world_allocation_spec["payload"]["roles"]["data"][0]
    trainer = plan.world_allocation_spec["payload"]["roles"]["trainer"][0]
    assert data["resources"]["cpus"] == [0]
    assert trainer["resources"]["cpus"] == [1, 2]
    assert trainer["resources"]["accelerators"] == {"gpu": [0]}
    assert trainer["resources"]["memory"] == "1GiB"


def test_allocate_local_world_invokes_assignment_kernel_once(monkeypatch):
    import dryml.dispatch.local_world as local_world_module

    assignment_kernel = local_world_module.assign_local_world
    invocations = 0

    def recording_assignment(*args, **kwargs):
        nonlocal invocations
        invocations += 1
        return assignment_kernel(*args, **kwargs)

    monkeypatch.setattr(local_world_module, "assign_local_world", recording_assignment)

    allocate_local_world({"worker": {"replicas": 1, "process": {}}}, inventory=_inventory())

    assert invocations == 1


def test_allocate_local_world_accepts_canonical_world_spec_data():
    world = WorldSpec.from_data({"roles": {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}}).to_data()

    plan = allocate_local_world(world, inventory=_inventory())

    assert plan.worker_keys == (WorldWorkerKey("worker", 0, 0, 0),)


def test_allocate_local_world_rejects_unsupported_canonical_backend():
    world = WorldSpec.from_data(
        {"roles": {"worker": {"replicas": 1, "process": {}}}, "backend": {"kind": "slurm", "parameters": {}}}
    ).to_data()

    with pytest.raises(DispatchPlanningError, match="local-world dispatch supports only"):
        allocate_local_world(world, inventory=_inventory())


def test_allocate_local_world_rejects_unenacted_backend_parameters():
    world = WorldSpec.from_data(
        {"roles": {"worker": {"replicas": 1, "process": {}}}, "backend": {"kind": "local", "parameters": {"workers": 2}}}
    ).to_data()

    with pytest.raises(DispatchPlanningError, match="backend parameters"):
        allocate_local_world(world, inventory=_inventory())


def test_allocate_local_world_rejects_insufficient_cpu():
    world = {"trainer": {"replicas": 2, "process": {"resources": {"cpus": 2}}}}

    try:
        allocate_local_world(world, inventory=LocalResourceInventory(cpus=(0,)))
    except Exception as exc:
        assert "CPU" in str(exc)
    else:
        raise AssertionError("expected CPU planning failure")


def test_allocate_local_world_fake_gpu_success_and_failure():
    world = {"trainer": {"replicas": 1, "process": {"resources": {"accelerators": {"gpu": 1}}}}}

    assert allocate_local_world(world, inventory=_inventory()).world_allocation_spec["id"].startswith("worldalloc-v1-")
    try:
        allocate_local_world(world, inventory=LocalResourceInventory(cpus=(0,), accelerators={}))
    except Exception as exc:
        assert "accelerator" in str(exc)
    else:
        raise AssertionError("expected accelerator planning failure")


def test_world_worker_key_validation_and_collision_resistant_label():
    with pytest.raises(Exception, match="role"):
        WorldWorkerKey.from_json({"replica": 0, "rank": 0, "local_rank": 0})

    left = WorldWorkerKey("a/b", 0, 0, 0)
    right = WorldWorkerKey("a_b", 0, 1, 1)

    assert left.label() == "a_b-0-r0"
    assert right.label() == "a_b-0-r1"
    assert left.label() != right.label()


def test_plan_world_persists_all_specs_before_launch(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("operator:add", args=[1, 2]))

    plan = Dispatcher(store=store).plan_world(op, world=world, inventory=_inventory())

    planning_metadata = plan.dispatch_spec["payload"]["metadata"]
    assert plan.execution_recipe["payload"]["annotation_report"] == planning_metadata
    assert all(
        worker.envelope.to_json()["reporting"]["planning"]
        == json_ready(planning_metadata)
        for worker in plan.worker_plans
    )
    assert store.records.read_spec(plan.operation_spec["id"], family="operation")["id"] == plan.operation_spec["id"]
    assert store.records.read_spec(plan.dispatch_spec["id"], family="dispatch")["id"] == plan.dispatch_spec["id"]
    assert store.records.read_spec(plan.execution_recipe["id"], family="execution_recipe")["id"] == plan.execution_recipe["id"]
    assert store.records.read_spec(plan.world_spec["id"], family="world")["id"] == plan.world_spec["id"]
    assert store.records.read_spec(plan.world_allocation_spec["id"], family="world_allocation")["id"] == plan.world_allocation_spec["id"]


def test_repeated_projected_spec_publication_is_content_addressed(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    op = attach_operation_id(make_function_call_spec("operator:add", args=[1, 2]))
    dispatcher = Dispatcher(store=store)
    default_cpu_world = {"worker": {"replicas": 1, "process": {}}}

    first = dispatcher.plan_world(op, world=default_cpu_world, inventory=_inventory())
    repeated = dispatcher.plan_world(op, world=default_cpu_world, inventory=_inventory())
    explicit_cpu = dispatcher.plan_world(
        op,
        world={"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}},
        inventory=_inventory(),
    )

    assert repeated.world_spec == first.world_spec
    assert repeated.world_allocation_spec == first.world_allocation_spec
    assert repeated.world_allocation_spec["id"] == first.world_allocation_spec["id"]
    assert explicit_cpu.world_spec["id"] != first.world_spec["id"]
    assert explicit_cpu.world_allocation_spec["id"] != first.world_allocation_spec["id"]


def test_plan_world_metadata_summarizes_assigned_memory_and_accelerators(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1, "memory": "1GiB", "accelerators": {"gpu": 1}}}}}
    op = attach_operation_id(make_function_call_spec("operator:add", args=[1, 2]))

    plan = Dispatcher(store=store).plan_world(op, world=world, inventory=_inventory())

    worker = plan.dispatch_spec["payload"]["metadata"]["dryml.world_allocation"]["workers"][0]
    assert worker["memory"] == 1024**3
    assert worker["cpu_count"] == 1
    assert dict(worker["accelerator_counts"]) == {"gpu": 1}


def test_plan_world_metadata_records_explicit_oversubscription(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("operator:add", args=[1, 2]))

    plan = Dispatcher(store=store).plan_world(
        op,
        world=world,
        inventory=LocalResourceInventory((0,)),
        oversubscribe=True,
    )

    metadata = plan.dispatch_spec["payload"]["metadata"]["dryml.world_allocation"]
    assert metadata["allocation_policy"] == "oversubscribed_local"


def test_run_world_two_roles_returns_runtime_and_env_facts(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"trainer": {"replicas": 1, "process": {"resources": {"cpus": 1}}}, "data": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))

    result = Dispatcher(store=store).run_world(op, world=world, environment=_env(target_module), inventory=_inventory(), timeout=10)

    assert result.status == "ok"
    assert {key.role for key in result.workers} == {"data", "trainer"}
    assert result.primary is not None
    assert result.world_allocation_id and result.world_allocation_id.startswith("worldalloc-v1-")
    assert store.records.read_spec(result.world_allocation_id, family="world_allocation")["id"] == result.world_allocation_id
    assert len(result.execution_record_ids) == 2
    assert result.execution_record_ids == tuple(worker.execution_record_id for worker in result.workers.values())
    for key, worker_result in result.workers.items():
        facts = worker_result.result_canonical
        assert facts["mode"] == "worker"
        assert facts["role"] == key.role
        assert facts["replica"] == key.replica
        assert facts["rank"] == key.rank
        assert facts["local_rank"] == key.local_rank
        assert facts["env_role"] == key.role
        assert facts["env_replica"] == str(key.replica)
        assert facts["env_rank"] == str(key.rank)
        assert facts["env_local_rank"] == str(key.local_rank)
        assert facts["world_allocation_id"] == result.world_allocation_id
        assert facts["env_world_allocation_id"] == result.world_allocation_id
        assert facts["is_no_allocation"] is False
        assert facts["import_mode"] == "worker"
        assert worker_result.execution_record_id is not None
        record = store.records.read_record(worker_result.execution_record_id)
        assert record["payload"]["world_allocation_id"] == result.world_allocation_id
        assert record["payload"]["worker_key"] == key.to_json()
        assert record["metadata"]["role"] == key.role
        assert record["metadata"]["replica"] == key.replica
        assert record["metadata"]["rank"] == key.rank
        assert record["metadata"]["local_rank"] == key.local_rank
        assert record["payload"]["logs"]


def test_handshake_allocation_mismatch_reports_failed_not_cancelled(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))
    plan = Dispatcher(store=store).plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    first, second = plan.worker_plans[0].key, plan.worker_plans[1].key
    bad_key = {**first.to_json(), "rank": 99}
    worker_futures = {
        first: _FakeWorldWorkerFuture(tmp_path, plan, first, handshake=_handshake(plan, first, worker_key=bad_key)),
        second: _FakeWorldWorkerFuture(tmp_path, plan, second, handshake=_handshake(plan, second)),
    }

    result = _fake_future(tmp_path, plan, worker_futures).result(timeout=1)

    assert result.status == "failed"
    assert result.cancellation is None
    assert any(item.get("error_type") == "DispatchPlanningError" for item in result.diagnostics)
    assert any("key" in item.get("error_message", "") for item in result.diagnostics)


def test_unsupported_handshake_reports_unsupported_not_cancelled(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))
    plan = Dispatcher(store=store).plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    first, second = plan.worker_plans[0].key, plan.worker_plans[1].key
    worker_futures = {
        first: _FakeWorldWorkerFuture(tmp_path, plan, first, handshake=_handshake(plan, first, status="unsupported", diagnostics=({"message": "missing feature"},))),
        second: _FakeWorldWorkerFuture(tmp_path, plan, second, handshake=_handshake(plan, second)),
    }

    result = _fake_future(tmp_path, plan, worker_futures).result(timeout=1)

    assert result.status == "unsupported"
    assert result.cancellation is None
    assert any(item.get("handshake_status") == "unsupported" for item in result.diagnostics)


def test_missing_handshake_reports_failed_not_cancelled(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))
    plan = Dispatcher(store=store).plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    key = plan.worker_plans[0].key
    worker_futures = {key: _FakeWorldWorkerFuture(tmp_path, plan, key, done=True)}

    result = _fake_future(tmp_path, plan, worker_futures).result(timeout=1)

    assert result.status == "failed"
    assert result.cancellation is None
    assert any(item.get("reason") == "missing_handshake" for item in result.diagnostics)


def test_cancel_before_handshakes_aggregates_cancelled(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))
    plan = Dispatcher(store=store).plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    key = plan.worker_plans[0].key
    future = _fake_future(tmp_path, plan, {key: _FakeWorldWorkerFuture(tmp_path, plan, key, done=False)})

    assert future.cancel(reason="test") is True
    result = future.result(timeout=1)

    assert result.status == "cancelled"
    assert result.cancellation == {"requested": True, "reason": "test"}
    assert not any(item.get("reason") == "missing_handshake" for item in result.diagnostics)


def test_handshake_timeout_reports_timeout_not_cancelled(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))
    plan = Dispatcher(store=store).plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    key = plan.worker_plans[0].key
    worker_futures = {key: _FakeWorldWorkerFuture(tmp_path, plan, key, done=False)}

    result = _fake_future(tmp_path, plan, worker_futures, handshake_timeout=0.0).result(timeout=1)

    assert result.status == "timeout"
    assert result.cancellation is None
    assert all(worker.status == "timeout" for worker in result.workers.values())
    assert any(item.get("reason") == "handshake_timeout" for item in result.diagnostics)


def test_one_worker_failure_cancels_sibling(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"a": {"replicas": 1, "process": {"resources": {"cpus": 1}}}, "b": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:fail_for_role", args=["a"]))

    result = Dispatcher(store=store).run_world(op, world=world, environment=_env(target_module), inventory=_inventory(), timeout=10)

    assert result.status == "failed"
    statuses = {key.role: worker.status for key, worker in result.workers.items()}
    assert statuses["a"] == "failed"
    assert statuses["b"] == "cancelled"
    assert result.error and result.error["type"] == "ValueError"


def test_world_cancel_continues_after_a_worker_cancellation_error(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))
    plan = Dispatcher(store=store).plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    first, second = plan.worker_plans[0].key, plan.worker_plans[1].key
    failed = _FakeWorldWorkerFuture(tmp_path, plan, first)
    completed = _FakeWorldWorkerFuture(tmp_path, plan, second)
    completed_calls = []

    def fail_cancel(**_kwargs):
        raise RuntimeError("record persistence failed")

    def record_cancel(**kwargs):
        completed_calls.append(kwargs)
        return True

    failed.cancel = fail_cancel
    completed.cancel = record_cancel
    future = _fake_future(tmp_path, plan, {first: failed, second: completed})

    assert future.cancel(reason="test") is True
    assert completed_calls == [{"grace": future.cancel_grace, "reason": "test", "record": True}]


def test_world_close_removes_group_directory_without_result(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:sleep_forever"))
    future = Dispatcher(store=store).submit_world(
        Dispatcher(store=store).plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    )
    future.wait_for_handshakes(timeout=5)

    future.close(reason="test")

    assert not Path(future.group_work_dir).exists()


def test_world_close_removes_preserved_group_directory(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:sleep_forever"))
    dispatcher = Dispatcher(backend=LocalWorldBackend(preserve_work_dir=True), store=store)
    future = dispatcher.submit_world(
        dispatcher.plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    )
    future.wait_for_handshakes(timeout=5)

    future.close(reason="test")

    assert not Path(future.group_work_dir).exists()


@pytest.mark.skipif(os.name != "posix", reason="process-group signaling is POSIX-specific")
def test_successful_worker_cleanup_kills_only_backend_owned_process_group(tmp_path, target_module, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))
    plan = Dispatcher(store=store).plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    worker_plan = plan.worker_plans[0]
    process = _FakeProcess()
    future = LocalSubprocessFuture(
        process,
        worker_plan,
        str(tmp_path / "worker"),
        str(tmp_path / "request.json"),
        str(tmp_path / "handshake.json"),
        str(tmp_path / "response.json"),
        str(tmp_path / "stdout.txt"),
        str(tmp_path / "stderr.txt"),
        True,
        process_group=True,
    )
    signals = []
    monkeypatch.setattr("dryml.dispatch.backends.os.killpg", lambda pid, signal: signals.append((pid, signal)))

    future._response = WorkerResponse(status="ok")
    future._cleanup()

    unmanaged = LocalSubprocessFuture(
        process,
        worker_plan,
        str(tmp_path / "unmanaged-worker"),
        str(tmp_path / "unmanaged-request.json"),
        str(tmp_path / "unmanaged-handshake.json"),
        str(tmp_path / "unmanaged-response.json"),
        str(tmp_path / "unmanaged-stdout.txt"),
        str(tmp_path / "unmanaged-stderr.txt"),
        True,
        process_group=False,
    )
    unmanaged._response = WorkerResponse(status="ok")
    unmanaged._cleanup()

    assert len(signals) == 1


@pytest.mark.skipif(os.name != "posix", reason="process-group signaling is POSIX-specific")
def test_cancel_reaps_backend_group_after_signal_exits_leader(tmp_path, monkeypatch):
    process = _LeaderExitsOnSignalProcess()
    future = LocalSubprocessFuture(
        process,
        SimpleNamespace(envelope=SimpleNamespace(operation_id="op")),
        str(tmp_path / "worker"),
        str(tmp_path / "request.json"),
        str(tmp_path / "handshake.json"),
        str(tmp_path / "response.json"),
        str(tmp_path / "stdout.txt"),
        str(tmp_path / "stderr.txt"),
        process_group=True,
    )
    signals = []

    def killpg(pid, sig):
        signals.append((pid, sig))
        if sig == signal.SIGINT:
            process.returncode = 0

    monkeypatch.setattr("dryml.dispatch.backends.os.killpg", killpg)

    assert future.cancel(record=False) is True

    assert process.signals == []
    assert signals == [(process.pid, signal.SIGINT), (process.pid, signal.SIGKILL)]


def test_cancel_uses_taskkill_for_owned_windows_worker_tree(tmp_path, monkeypatch):
    import dryml.dispatch.backends as backends

    host_os_name = os.name
    process = _LeaderExitsOnSignalProcess()
    future = LocalSubprocessFuture(
        process,
        SimpleNamespace(envelope=SimpleNamespace(operation_id="op")),
        str(tmp_path / "worker"),
        str(tmp_path / "request.json"),
        str(tmp_path / "handshake.json"),
        str(tmp_path / "response.json"),
        str(tmp_path / "stdout.txt"),
        str(tmp_path / "stderr.txt"),
        process_tree=True,
    )
    commands = []
    # Do not mutate process-global os.name: pytest/pathlib then attempts to
    # instantiate WindowsPath inside this Linux test process.
    monkeypatch.setattr(backends, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(backends.subprocess, "run", lambda command, **kwargs: commands.append((command, kwargs)))

    assert future.cancel(record=False) is True

    assert commands == [
        (["taskkill", "/PID", str(process.pid), "/T", "/F"], {"check": False, "stdout": backends.subprocess.DEVNULL, "stderr": backends.subprocess.DEVNULL, "timeout": 5})
    ]
    assert process.kill_calls == 1
    assert os.name == host_os_name


def test_world_result_cleans_artifacts_after_aggregate_failure(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}
    op = make_function_call_spec("operator:add", args=[1, 2])
    plan = Dispatcher(store=store).plan_world(op, world=world, inventory=_inventory())
    key = plan.worker_plans[0].key
    worker = _FakeWorldWorkerFuture(tmp_path, plan, key, handshake=_handshake(plan, key), done=True)
    future = _fake_future(tmp_path, plan, {key: worker})
    future.preserve_work_dir = False
    Path(future.group_work_dir).mkdir()
    monkeypatch.setattr(LocalWorldFuture, "_aggregate", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("aggregate failed")))

    with pytest.raises(RuntimeError, match="aggregate failed"):
        future.result(timeout=1)

    assert not Path(future.group_work_dir).exists()


def test_explicit_world_cancel_reaches_all_workers(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:sleep_forever"))
    dispatcher = Dispatcher(store=store)
    plan = dispatcher.plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    future = dispatcher.submit_world(plan)
    future.wait_for_handshakes(timeout=5)

    assert future.cancel(reason="test") is True
    result = future.result(timeout=5)

    assert result.status == "cancelled"
    assert all(worker.status == "cancelled" for worker in result.workers.values())


def test_world_timeout_records_worker_timeouts_not_cancellations(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:sleep_forever"))
    dispatcher = Dispatcher(backend=LocalWorldBackend(cancel_grace=0.05), store=store)

    result = dispatcher.run_world(op, world=world, environment=_env(target_module), inventory=_inventory(), timeout=0.1)

    assert result.status == "timeout"
    assert all(worker.status == "timeout" for worker in result.workers.values())
    records = [store.records.read_record(record_id) for record_id in result.execution_record_ids]
    assert {record["payload"]["status"] for record in records} == {"timeout"}
    assert store.records.find_execution_records(status="cancelled") == ()


def test_worker_start_barrier_timeout_uses_coordination_timeout(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))
    plan = Dispatcher(store=store).plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    worker_plan = plan.worker_plans[0]
    launch = dict(worker_plan.envelope.launch)
    launch["coordination"] = {
        "worker_key": worker_plan.key.to_json(),
        "start_path": str(tmp_path / "missing-start.json"),
        "cancel_path": str(tmp_path / "missing-cancel.json"),
        "start_timeout": 0.01,
    }
    envelope = replace(worker_plan.envelope, launch=launch)

    response = _wait_for_start_barrier(envelope, store)

    assert response.status == "timeout"
    record = store.records.read_record(response.execution_record_id)
    assert record["payload"]["status"] == "timeout"


def test_keyboard_interrupt_cancels_world_workers(tmp_path, target_module, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:sleep_forever"))
    dispatcher = Dispatcher(store=store)
    plan = dispatcher.plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    future = dispatcher.submit_world(plan)

    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(LocalWorldFuture, "wait_for_handshakes", raise_interrupt)
    with pytest.raises(KeyboardInterrupt):
        future.result(timeout=5)

    assert future.done()


def test_record_policy_none_suppresses_world_provenance(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))
    dispatcher = Dispatcher(store=store)
    plan = dispatcher.plan_world(
        op,
        world=world,
        environment=_env(target_module),
        inventory=_inventory(),
        record_policy="none",
    )
    future = dispatcher.submit_world(plan)

    handshakes = future.wait_for_handshakes(timeout=5)
    result = future.result(timeout=10)

    for key, handshake in handshakes.items():
        assert handshake is not None
        assert handshake.worker_key == key.to_json()
        assert handshake.world_id == plan.world_spec["id"]
        assert handshake.world_allocation_id == plan.world_allocation_spec["id"]
    assert result.status == "ok"
    assert sorted(key.replica for key in result.workers) == [0, 1]
    assert all(worker.result_canonical["role"] == "worker" for worker in result.workers.values())
    assert result.execution_record_ids == ()
    assert store.records.find_execution_records() == ()


def test_explicit_local_world_uses_configured_inventory_policy(tmp_path, monkeypatch):
    observed = []
    inventory = LocalResourceInventory((0,))

    def discover(*, policy):
        observed.append(policy)
        return inventory

    monkeypatch.setattr("dryml.dispatch.planner.worlds.local_inventory", discover)
    plan = Dispatcher(store=DirStore(tmp_path / "store", query_index="none"), inventory_policy="external").plan_world(
        make_function_call_spec("operator:add", args=[1, 2]),
        world={"roles": {"main": {"replicas": 1, "process": {}}}},
    )

    assert observed == ["external"]
    assert plan.dispatch_spec["payload"]["metadata"]["dryml.local_inventory"] is None


def test_requirement_free_plan_world_inventory_summary_stays_out_of_dispatch_identity(tmp_path):
    world = {"roles": {"main": {"replicas": 1, "process": {}}}}
    operation = make_function_call_spec("operator:add", args=[1, 2])
    left = Dispatcher(store=DirStore(tmp_path / "left", query_index="none")).plan_world(
        operation,
        world=world,
        inventory=LocalResourceInventory((0,), metadata={"observed_at": "first"}),
    )
    right = Dispatcher(store=DirStore(tmp_path / "right", query_index="none")).plan_world(
        operation,
        world=world,
        inventory=LocalResourceInventory((0,), metadata={"observed_at": "second"}),
    )

    assert left.dispatch_spec["payload"]["metadata"]["dryml.local_inventory"] is None
    assert left.dispatch_spec["id"] == right.dispatch_spec["id"]
