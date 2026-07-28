import pytest

import dryml.worlds as worlds
from dryml.worlds.errors import WorldSpecValidationError


def allocation_roles(gpus=()):
    return {
        "trainer": [
            {
                "replica": 0,
                "rank": 0,
                "local_rank": 0,
                "resources": {"cpus": [0, 1, 2, 3], "memory": "32GiB", "accelerators": {"gpu": list(gpus)}},
                "environment": "env-v1-example",
                "metadata": {},
            }
        ]
    }


def test_cpu_only_worker_allocation_is_real_allocation():
    spec = worlds.attach_world_allocation_id(worlds.make_world_allocation_spec(allocation_roles()))
    allocation = worlds.WorldAllocation.from_data(spec["payload"])
    view = allocation.runtime_view("trainer", 0, world_allocation_id=spec["id"])

    assert spec["id"].startswith("worldalloc-v1-")
    assert view.cpus == (0, 1, 2, 3)
    assert view.accelerators == {"gpu": ()}
    assert not view.is_no_allocation


def test_single_gpu_worker_allocation_view():
    spec = worlds.attach_world_allocation_id(worlds.make_world_allocation_spec(allocation_roles(gpus=(0,))))
    allocation = worlds.WorldAllocation.from_data(spec["payload"])
    view = allocation.runtime_view("trainer", 0, world_allocation_id=spec["id"])

    assert view.role == "trainer"
    assert view.accelerators["gpu"] == (0,)


def test_allocation_is_separate_from_requested_world_spec():
    world_spec = worlds.attach_world_id(worlds.make_world_spec({"trainer": {"replicas": 1, "process": {"resources": {"accelerators": {"gpu": 1}}}}}))
    allocation_spec = worlds.attach_world_allocation_id(worlds.make_world_allocation_spec(allocation_roles(gpus=(2,))))

    assert world_spec["schema"] == "dryml.world.v1"
    assert allocation_spec["schema"] == "dryml.world_allocation.v1"
    assert world_spec["id"] != allocation_spec["id"]


def test_allocation_rejects_scalar_accelerator_assignment():
    with pytest.raises(WorldSpecValidationError):
        worlds.WorldAllocation.from_data(
            {
                "roles": {
                    "trainer": [
                        {
                            "replica": 0,
                            "rank": 0,
                            "local_rank": 0,
                            "resources": {"accelerators": {"gpu": "cuda0"}},
                        }
                    ]
                }
            }
        )


def test_accelerator_memory_round_trips_without_changing_legacy_payload_identity():
    legacy = worlds.make_world_allocation_spec(allocation_roles(gpus=(0,)))
    extended_roles = allocation_roles(gpus=(0, 2))
    extended_roles["trainer"][0]["resources"]["accelerator_memory"] = {
        "gpu": [{"device": 0, "memory": "1GiB"}, {"device": 2, "memory": "512MiB"}]
    }
    extended = worlds.WorldAllocation.from_data({"roles": extended_roles})

    assert "accelerator_memory" not in legacy["payload"]["roles"]["trainer"][0]["resources"]
    assert extended.to_data()["roles"]["trainer"][0]["resources"]["accelerator_memory"] == {
        "gpu": [{"device": 0, "memory": "1GiB"}, {"device": 2, "memory": "512MiB"}]
    }


def test_accelerator_memory_rejects_unassigned_devices():
    roles = allocation_roles(gpus=(0,))
    roles["trainer"][0]["resources"]["accelerator_memory"] = {
        "gpu": [{"device": 1, "memory": "1GiB"}]
    }

    with pytest.raises(WorldSpecValidationError):
        worlds.WorldAllocation.from_data({"roles": roles})
