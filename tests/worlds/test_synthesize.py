from __future__ import annotations

import pytest

from dryml.worlds import LocalResourceInventory, RoleRequirement, WorldRequirement, synthesize


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
    assert result.diagnostics[0].to_data()["data"] == {"required": 4, "available": 2, "shortfall": 2}


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


def test_synthesis_omits_optional_zero_count_accelerators():
    result = synthesize(
        {"roles": {"worker": {"resources": {"accelerators": {"gpu": {"max": 1}}}}}},
        inventory=LocalResourceInventory((0,)),
    )

    assert result.ok
    assert result.world.roles["worker"].process.resources.accelerators == {}


def test_synthesize_validates_inventory_policy_with_injected_inventory():
    with pytest.raises(Exception, match="unsupported local inventory policy"):
        synthesize(None, inventory=LocalResourceInventory((0,)), inventory_policy="invalid")


def test_synthesize_invalid_injected_inventory_is_structured():
    result = synthesize(None, inventory=object())  # type: ignore[arg-type]

    assert result.status == "error"
    assert result.diagnostics[0].code == "invalid_inventory"


def test_synthesis_rejects_colliding_resource_names_and_invalid_topology():
    with pytest.raises(Exception, match="name"):
        WorldRequirement.from_data({"roles": {"main": {"resources": {"accelerators": {1: {"min": 1}, "1": {"exact": 0}}}}}})
    with pytest.raises(Exception, match="single_process"):
        WorldRequirement.from_data({"roles": {"main": {"topology": {"single_process": "yes"}}}})


def test_synthesis_rejects_oversized_count_before_serialization():
    result = synthesize(
        {"roles": {"main": {"resources": {"cpus": {"exact": 1 << 5000}}}}},
        inventory=LocalResourceInventory((0,)),
    )

    assert result.status == "invalid_requirement"


def test_concrete_world_resources_reject_collisions_and_oversized_values():
    from dryml.worlds import WorldSpec

    with pytest.raises(Exception, match="accelerator name"):
        WorldSpec.from_data({"roles": {"main": {"process": {"resources": {"accelerators": {1: 1, "1": 2}}}}}})
    with pytest.raises(Exception, match="bounded"):
        WorldSpec.from_data({"roles": {"main": {"replicas": 1 << 5000, "process": {}}}})
    with pytest.raises(Exception, match="bounded"):
        WorldSpec.from_data({"roles": {"main": {"process": {"resources": {"memory": 1 << 5000}}}}})


def test_synthesis_cpu_memory_accelerator_and_topology_matrix(monkeypatch):
    inventory = LocalResourceInventory((0, 1, 2), {"gpu": ("a",)}, memory=1024)
    exact = synthesize({"roles": {"main": {"resources": {"cpus": {"exact": 2}, "memory": {"exact": 512}, "accelerators": {"gpu": {"exact": 1}}}}}}, inventory=inventory)
    assert exact.ok
    assert exact.world.roles["main"].process.resources.cpus == 2
    assert exact.world.roles["main"].process.resources.memory == 512
    assert exact.world.roles["main"].process.resources.accelerators == {"gpu": 1}
    assert synthesize({"roles": {"main": {"resources": {"accelerators": {"gpu": {"exact": 2}}}}}}, inventory=inventory).diagnostics[0].code == "insufficient_accelerators"
    assert synthesize({"roles": {"main": {"topology": {"collectives": True}}}}, inventory=inventory).status == "unsupported_requirement"
    assert synthesize({"roles": {"main": {"resources": {"devices": {"gpu": {"min": 1}}}}}}, inventory=inventory).status == "unsupported_requirement"


def test_synthesis_selects_minimums_and_reports_distinct_aggregate_shortfalls():
    inventory = LocalResourceInventory((0, 1), {"gpu": ("a",)}, memory=1024)

    minimums = synthesize(
        {"roles": {"main": {"resources": {"cpus": {"max": 2}, "accelerators": {"gpu": {"min": 1}}}}}},
        inventory=inventory,
    )
    missing_gpu = synthesize(
        {"roles": {"main": {"resources": {"accelerators": {"gpu": {"min": 1}}}}}},
        inventory=LocalResourceInventory((0,)),
    )
    insufficient_memory = synthesize(
        {"roles": {"main": {"resources": {"memory": {"min": 2048}}}}},
        inventory=inventory,
    )
    competing_roles = synthesize(
        {"roles": {"alpha": {"resources": {"cpus": {"exact": 2}}}, "beta": {"resources": {"cpus": {"exact": 1}}}}},
        inventory=inventory,
    )

    assert minimums.ok
    assert minimums.world.roles["main"].process.resources.cpus == 1
    assert minimums.world.roles["main"].process.resources.accelerators == {"gpu": 1}
    assert missing_gpu.diagnostics[0].code == "insufficient_accelerators"
    assert insufficient_memory.diagnostics[0].code == "insufficient_memory"
    assert competing_roles.diagnostics[0].to_data()["data"] == {"required": 3, "available": 2, "shortfall": 1}


def test_synthesis_honors_single_process_and_rejects_shared_filesystem():
    inventory = LocalResourceInventory((0,))

    supported = synthesize(
        {"roles": {"main": {"topology": {"single_process": True}}}},
        inventory=inventory,
    )
    unsupported = synthesize(
        {"roles": {"main": {"topology": {"shared_filesystem": True}}}},
        inventory=inventory,
    )

    assert supported.ok
    assert unsupported.status == "unsupported_requirement"


def test_synthesis_serialization_is_deterministic_across_mapping_order():
    inventory = LocalResourceInventory((0, 1), {"gpu": ("a",)}, memory=1024)
    first = synthesize(
        {"roles": {"beta": {"resources": {"accelerators": {"gpu": {"exact": 1}}}}, "alpha": {"resources": {"cpus": {"exact": 1}}}}},
        inventory=inventory,
    )
    second = synthesize(
        {"roles": {"alpha": {"resources": {"cpus": {"exact": 1}}}, "beta": {"resources": {"accelerators": {"gpu": {"exact": 1}}}}}},
        inventory=inventory,
    )

    assert first.to_data() == second.to_data()


def test_synthesis_rejects_zero_only_and_authoritative_post_check(monkeypatch):
    import dryml.worlds.synthesis as synthesis

    zero = synthesize({"roles": {"main": {"resources": {"cpus": {"max": 0}}}}}, inventory=LocalResourceInventory((0,)))
    assert zero.status == "invalid_requirement"
    monkeypatch.setattr(synthesis, "check_world_spec_satisfies_requirement", lambda *_args: type("Report", (), {"ok": False, "issues": ()})())
    result = synthesis.synthesize({"roles": {"main": {}}}, inventory=LocalResourceInventory((0,)))
    assert result.status == "error"


def test_synthesis_revalidates_direct_requirement_instances():
    malformed = WorldRequirement({"main": RoleRequirement(resources=object())})  # type: ignore[arg-type]

    result = synthesize(malformed, inventory=LocalResourceInventory((0,)))

    assert result.status == "invalid_requirement"
    assert result.diagnostics[0].code == "invalid_requirement"


def test_synthesis_rejects_local_role_and_worker_counts_before_materializing_worlds():
    inventory = LocalResourceInventory((0,))
    too_many_workers = synthesize(
        {"roles": {"main": {"replicas": {"exact": 4097}}}},
        inventory=inventory,
    )
    too_many_roles = synthesize(
        {"roles": {f"role_{index}": {} for index in range(4097)}},
        inventory=inventory,
    )

    assert too_many_workers.status == "invalid_requirement"
    assert too_many_workers.diagnostics[0].code == "worker_count_exceeds_local_limit"
    assert too_many_roles.status == "invalid_requirement"
    assert too_many_roles.diagnostics[0].code == "role_count_exceeds_local_limit"
