import sys

import pytest

from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher, LocalResourceInventory, LocalWorldFuture, WorldWorkerKey, allocate_local_world
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
