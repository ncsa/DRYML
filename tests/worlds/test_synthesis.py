from dryml.worlds import LocalResourceInventory, WorldRequirement, synthesize


def test_synthesis_selects_minimums_and_assigns_minimum_device_memory():
    requirement = WorldRequirement.from_payload({"roles": {"beta": {"replicas": {"min": 2, "max": None}, "resources": {"cpus": {"min": 1, "max": None}, "accelerators": {"gpu": {"min": 1, "max": 1}}, "accelerator_memory": {"gpu": {"min": "2GiB", "max": None}}}}}})
    inventory = LocalResourceInventory((0, 1), {"gpu": (2, 4)}, memory=8, accelerator_memory={"gpu": {2: 3 * 1024**3, 4: 3 * 1024**3}})
    result = synthesize(requirement, inventory=inventory)
    assert result.ok
    assert result.world.roles["beta"].process.resources.accelerator_memory["gpu"] == (2 * 1024**3,)


def test_synthesis_reports_unknown_memory_and_unsupported_topology():
    memory = synthesize({"roles": {"main": {"resources": {"memory": {"min": 1, "max": None}}}}}, inventory=LocalResourceInventory((0,)))
    topology = synthesize({"roles": {"main": {"topology": {"rack": "a"}}}}, inventory=LocalResourceInventory((0,)))
    assert memory.diagnostics[0].code == "memory_unknown"
    assert topology.diagnostics[0].code == "unsupported_topology"


def test_synthesis_aligns_memory_for_each_accelerator_kind():
    requirement = WorldRequirement.from_payload({"roles": {"main": {"resources": {
        "accelerators": {
            "gpu": {"min": 2, "max": 2},
            "tpu": {"min": 1, "max": 1},
        },
        "accelerator_memory": {
            "gpu": {"min": "2GiB", "max": None},
            "tpu": {"min": "4GiB", "max": None},
        },
    }}}})
    inventory = LocalResourceInventory(
        (0,),
        {"gpu": (0, 1), "tpu": ("a",)},
        accelerator_memory={"gpu": {0: 3 * 1024**3, 1: 3 * 1024**3}, "tpu": {"a": 5 * 1024**3}},
    )

    result = synthesize(requirement, inventory=inventory)

    assert result.ok
    assert result.world.roles["main"].process.resources.accelerator_memory["gpu"] == (2 * 1024**3,) * 2
    assert result.world.roles["main"].process.resources.accelerator_memory["tpu"] == (4 * 1024**3,)
