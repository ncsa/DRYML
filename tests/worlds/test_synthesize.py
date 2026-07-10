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


def test_synthesize_default_and_unknown_memory_failure_are_structured():
    default = synthesize(None, inventory=LocalResourceInventory((0,)))
    memory_requirement = WorldRequirement.from_data(
        {"roles": {"worker": {"resources": {"memory": {"min": "1GiB"}}}}}
    )
    insufficient_memory = synthesize(memory_requirement, inventory=LocalResourceInventory((0,)))

    assert default.ok
    assert default.world.roles["main"].process.resources.cpus == 1
    assert insufficient_memory.status == "insufficient_inventory"
    assert insufficient_memory.diagnostics[0].code == "memory_unknown"


def test_synthesize_rejects_unsupported_named_resources():
    requirement = WorldRequirement.from_data(
        {"roles": {"worker": {"resources": {"named": {"fast_disk": {"min": 1}}}}}}
    )

    result = synthesize(requirement, inventory=LocalResourceInventory((0,)))

    assert result.status == "unsupported_requirement"
    assert result.diagnostics[0].code == "unsupported_named"


def test_synthesis_result_serialization_is_bounded():
    result = synthesize({"roles": {"worker": {"resources": {"named": {"x": {"min": 1}}}}}}, inventory=LocalResourceInventory((0,)))
    bounded = result.__class__(
        result.status,
        result.requirement,
        {"deep": {"value": "x" * 5000}},
        result.world,
        result.compatibility,
        result.diagnostics,
        result.policy,
    ).to_data()

    assert len(bounded["inventory"]["deep"]["value"]) == 4096
