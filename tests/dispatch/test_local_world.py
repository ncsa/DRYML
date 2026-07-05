import sys

import pytest

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher, ExecutionEnvelope, LocalResourceInventory, LocalWorldBackend, LocalWorldFuture, WorldWorkerKey, allocate_local_world
from dryml.dispatch.worker import _wait_for_start_barrier
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec


def _env(target_module):
    return PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()


def _inventory():
    return LocalResourceInventory(cpus=(0, 1, 2, 3), accelerators={"gpu": (0, 1)}, metadata={"source": "test"})


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


def test_plan_world_persists_all_specs_before_launch(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))

    plan = Dispatcher(store=store).plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())

    assert store.records.read_spec(plan.operation_spec["id"], family="operation")["id"] == plan.operation_spec["id"]
    assert store.records.read_spec(plan.dispatch_spec["id"], family="dispatch")["id"] == plan.dispatch_spec["id"]
    assert store.records.read_spec(plan.execution_recipe["id"], family="execution_recipe")["id"] == plan.execution_recipe["id"]
    assert store.records.read_spec(plan.world_spec["id"], family="world")["id"] == plan.world_spec["id"]
    assert store.records.read_spec(plan.world_allocation_spec["id"], family="world_allocation")["id"] == plan.world_allocation_spec["id"]


def test_run_world_two_roles_returns_runtime_and_env_facts(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"trainer": {"replicas": 1, "process": {"resources": {"cpus": 1}}}, "data": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))

    result = Dispatcher(store=store).run_world(op, world=world, environment=_env(target_module), inventory=_inventory(), timeout=10)

    assert result.status == "ok"
    assert {key.role for key in result.workers} == {"data", "trainer"}
    assert result.primary is not None
    assert result.world_allocation_id and result.world_allocation_id.startswith("worldalloc-v1-")
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


def test_world_handshake_reports_allocation_facts(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))
    dispatcher = Dispatcher(store=store)
    plan = dispatcher.plan_world(op, world=world, environment=_env(target_module), inventory=_inventory())
    future = dispatcher.submit_world(plan)

    handshakes = future.wait_for_handshakes(timeout=5)
    future.cancel(reason="test")

    for key, handshake in handshakes.items():
        assert handshake is not None
        assert handshake.worker_key == key.to_json()
        assert handshake.world_id == plan.world_spec["id"]
        assert handshake.world_allocation_id == plan.world_allocation_spec["id"]


def test_run_world_replicated_role(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))

    result = Dispatcher(store=store).run_world(op, world=world, environment=_env(target_module), inventory=_inventory(), timeout=10)

    assert result.status == "ok"
    assert sorted(key.replica for key in result.workers) == [0, 1]
    assert all(worker.result_canonical["role"] == "worker" for worker in result.workers.values())


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
    envelope = ExecutionEnvelope(
        dispatch_spec=worker_plan.envelope.dispatch_spec,
        execution_recipe=worker_plan.envelope.execution_recipe,
        operation_spec=worker_plan.envelope.operation_spec,
        environment_spec=worker_plan.envelope.environment_spec,
        runtime_spec=worker_plan.envelope.runtime_spec,
        allocation_view=worker_plan.envelope.allocation_view,
        store_refs=worker_plan.envelope.store_refs,
        transfer=worker_plan.envelope.transfer,
        record_policy=worker_plan.envelope.record_policy,
        launch=launch,
    )

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


def test_world_provenance_records_actual_allocation(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))

    result = Dispatcher(store=store).run_world(op, world=world, environment=_env(target_module), inventory=_inventory(), timeout=10)

    assert store.records.read_spec(result.world_allocation_id, family="world_allocation")["id"] == result.world_allocation_id
    assert len(result.execution_record_ids) == 2
    for record_id in result.execution_record_ids:
        record = store.records.read_record(record_id)
        assert record["payload"]["world_allocation_id"] == result.world_allocation_id
        assert record["metadata"]["role"] == "worker"
        assert "rank" in record["metadata"]
        assert record["payload"]["logs"]


def test_record_policy_none_suppresses_world_provenance(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    world = {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:allocation_facts"))

    result = Dispatcher(store=store).run_world(op, world=world, environment=_env(target_module), inventory=_inventory(), record_policy="none", timeout=10)

    assert result.status == "ok"
    assert result.execution_record_ids == ()
    assert store.records.find_execution_records() == ()
