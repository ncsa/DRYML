from __future__ import annotations

from dryml.worlds import LocalResourceInventory, local_inventory


def test_inventory_round_trip_is_deterministic():
    inventory = LocalResourceInventory((3, 1), {"gpu": ("1", 0)}, memory=1024, metadata={"policy": "test"})

    assert inventory.cpus == (1, 3)
    assert LocalResourceInventory.from_data(inventory.to_data()) == inventory
    assert inventory.summary()["accelerator_counts"] == {"gpu": 2}


def test_lightweight_inventory_uses_explicit_accelerator_override_without_mutation():
    environment = {"DRYML_LOCAL_ACCELERATORS": "gpu=2,0;fpga=a"}

    inventory = local_inventory(environ=environment)

    assert inventory.accelerators == {"fpga": ("a",), "gpu": (0, 2)}
    assert environment == {"DRYML_LOCAL_ACCELERATORS": "gpu=2,0;fpga=a"}
