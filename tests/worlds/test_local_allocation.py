import pytest

from dryml.worlds import LocalResourceInventory, WorldAllocation, WorldSpec, assign_local_world


def test_assignment_is_disjoint_deterministic_and_rejects_oversubscribe():
    world = WorldSpec.from_payload({"roles": {"a": {"replicas": 2, "process": {"resources": {"cpus": 1, "accelerators": {"gpu": 1}}}}}})
    allocation = assign_local_world(world, inventory=LocalResourceInventory((3, 4), {"gpu": (7, 8)}))
    assert [item.cpus for item in allocation.roles["a"]] == [(3,), (4,)]
    assert [item.accelerators["gpu"] for item in allocation.roles["a"]] == [(7,), (8,)]
    with pytest.raises(Exception, match="oversubscribe"):
        assign_local_world(world, inventory=LocalResourceInventory((3, 4), {"gpu": (7, 8)}), oversubscribe=True)


def test_assignment_reserves_capable_devices_for_harder_requests():
    world = WorldSpec.from_payload({"roles": {
        "low": {"replicas": 1, "process": {"resources": {"cpus": 1, "accelerators": {"gpu": 1}, "accelerator_memory": {"gpu": ["1GiB"]}}}},
        "high": {"replicas": 1, "process": {"resources": {"cpus": 1, "accelerators": {"gpu": 1}, "accelerator_memory": {"gpu": ["4GiB"]}}}},
    }})
    inventory = LocalResourceInventory(
        (0, 1),
        {"gpu": (0, 1)},
        accelerator_memory={"gpu": {0: 1 * 1024**3, 1: 4 * 1024**3}},
    )

    allocation = assign_local_world(world, inventory=inventory)

    assert allocation.roles["low"][0].accelerators["gpu"] == (0,)
    assert allocation.roles["high"][0].accelerators["gpu"] == (1,)


def test_world_allocation_rejects_cross_process_resource_overlap():
    process = {"replica": 0, "rank": 0, "local_rank": 0, "resources": {"cpus": [0], "accelerators": {"gpu": [0]}}}
    other = {"replica": 0, "rank": 1, "local_rank": 0, "resources": {"cpus": [0], "accelerators": {"gpu": [0]}}}
    with pytest.raises(Exception, match="disjoint|assigned more than once"):
        WorldAllocation.from_payload({"roles": {"a": [process], "b": [other]}})
