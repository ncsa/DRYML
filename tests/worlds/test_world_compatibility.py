from dryml.worlds import WorldAllocation, WorldRequirement, WorldSpec, check_allocation_satisfies_requirement, check_world_spec_satisfies_requirement


def test_compatibility_checks_memory_for_each_assigned_accelerator_and_topology():
    requirement = WorldRequirement.from_payload({"roles": {"worker": {"resources": {"accelerators": {"gpu": {"min": 1, "max": 1}}, "accelerator_memory": {"gpu": {"min": "2GiB", "max": None}}}, "topology": {"rack": "a"}}}})
    world = WorldSpec.from_payload({"roles": {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1, "accelerators": {"gpu": 1}, "accelerator_memory": {"gpu": ["2GiB"]}}}}}})
    allocation = WorldAllocation.from_payload({"roles": {"worker": [{"replica": 0, "rank": 0, "local_rank": 0, "resources": {"cpus": [0], "accelerators": {"gpu": [0]}, "accelerator_memory": {"gpu": [{"device": 0, "memory": "1GiB"}]}}}]}})
    assert not check_world_spec_satisfies_requirement(world, requirement).ok
    assert not check_allocation_satisfies_requirement(allocation, requirement).ok
    assert not check_world_spec_satisfies_requirement(world, requirement).admission_ok
    assert not check_allocation_satisfies_requirement(allocation, requirement).admission_ok


def test_compatibility_paths_escape_dotted_role_and_resource_names():
    """Missing dotted role names and dotted resource names retain their boundaries."""

    requirement = WorldRequirement({
        "main.resources.named": {"resources": {"named": {"license": 1}}},
        "main": {"resources": {"named": {"named.resources.named.license": 1}}},
    })
    world = WorldSpec.from_payload({"roles": {"other": {"process": {"resources": {"cpus": 1}}}}})

    paths = {issue.path for issue in check_world_spec_satisfies_requirement(world, requirement).issues}

    assert paths == {"roles.main", 'roles["main.resources.named"]'}
