import pytest

from dryml.worlds import LocalResourceInventory, WorldSpec, assign_local_world
from dryml.worlds.errors import WorldSpecValidationError


def test_shared_assignment_binds_accelerator_memory_to_selected_devices():
    world = WorldSpec.from_data(
        {
            "roles": {
                "worker": {
                    "replicas": 1,
                    "process": {
                        "resources": {
                            "cpus": 1,
                            "accelerators": {"gpu": 2},
                            "accelerator_memory": {"gpu": ["1GiB", "512MiB"]},
                        }
                    },
                }
            }
        }
    )
    assignment = assign_local_world(
        world,
        inventory=LocalResourceInventory((0,), {"gpu": (2, 4)}),
    )

    assert assignment.roles["worker"][0]["resources"]["accelerator_memory"] == {
        "gpu": [{"device": 2, "memory": "1GiB"}, {"device": 4, "memory": "512MiB"}]
    }


def test_shared_assignment_rejects_known_accelerator_memory_overage():
    world = WorldSpec.from_data(
        {
            "roles": {
                "worker": {
                    "replicas": 1,
                    "process": {
                        "resources": {
                            "accelerators": {"gpu": 1},
                            "accelerator_memory": {"gpu": ["2GiB"]},
                        }
                    },
                }
            }
        }
    )

    with pytest.raises(WorldSpecValidationError):
        assign_local_world(
            world,
            inventory=LocalResourceInventory((0,), {"gpu": (2,)}, accelerator_memory={"gpu": {2: "1GiB"}}),
        )


def test_shared_assignment_ignores_zero_valued_unsupported_resources():
    world = WorldSpec.from_data(
        {
            "roles": {
                "worker": {
                    "replicas": 1,
                    "process": {
                        "resources": {
                            "devices": {"gpu": 0},
                            "named": {"scratch": 0},
                        }
                    },
                }
            }
        }
    )

    assignment = assign_local_world(world, inventory=LocalResourceInventory((0,)))

    assert assignment.roles["worker"][0]["resources"]["cpus"] == [0]


def test_shared_assignment_rejects_positive_unsupported_resources():
    world = WorldSpec.from_data(
        {
            "roles": {
                "worker": {
                    "replicas": 1,
                    "process": {"resources": {"named": {"scratch": 1}}},
                }
            }
        }
    )

    with pytest.raises(WorldSpecValidationError, match="named devices or resources"):
        assign_local_world(world, inventory=LocalResourceInventory((0,)))
