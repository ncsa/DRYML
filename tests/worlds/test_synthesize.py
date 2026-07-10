from __future__ import annotations

from dryml.worlds import LocalResourceInventory, WorldRequirement, synthesize


def test_synthesize_uses_smallest_aggregate_compatible_local_world():
    requirement = WorldRequirement.from_data(
        {"roles": {"worker": {"replicas": {"exact": 2}, "resources": {"cpus": {"min": 2}, "accelerators": {"gpu": {"exact": 1}}}}}}
    )
    result = synthesize(requirement, inventory=LocalResourceInventory((0, 1, 2, 3), {"gpu": (0, 1)}))

    assert result.ok
    assert result.world is not None
    assert result.world.roles["worker"].replicas == 2
    assert result.world.roles["worker"].process.resources.cpus == 2
    assert result.world.backend["kind"] == "local"


def test_synthesize_reports_insufficient_aggregate_capacity():
    requirement = WorldRequirement.from_data({"roles": {"worker": {"replicas": {"exact": 2}, "resources": {"cpus": {"exact": 2}}}}})

    result = synthesize(requirement, inventory=LocalResourceInventory((0, 1)))

    assert result.status == "insufficient_inventory"
    assert result.diagnostics[0].code == "insufficient_cpus"
