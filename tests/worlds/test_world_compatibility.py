import dryml.worlds as worlds


def requirement():
    return worlds.WorldRequirement.from_data(
        {
            "roles": {
                "trainer": {
                    "replicas": {"exact": 1},
                    "resources": {"cpus": {"min": 4}, "memory": {"min": "16GiB"}, "accelerators": {"gpu": {"exact": 1}}},
                    "topology": {"single_process": True, "unknown_hint": True},
                }
            }
        }
    )


def test_world_spec_satisfies_requirement_with_warning():
    world = worlds.WorldSpec.from_data({"roles": {"trainer": {"replicas": 1, "process": {"resources": {"cpus": 4, "memory": "16GiB", "accelerators": {"gpu": 1}}}}}})
    report = worlds.check_world_spec_satisfies_requirement(world, requirement())

    assert report.ok
    assert report.issues[0].severity == "warning"


def test_world_spec_reports_missing_role_and_resource_failures():
    report = worlds.check_world_spec_satisfies_requirement(worlds.WorldSpec.from_data({"roles": {"other": {"replicas": 1}}}), requirement())
    assert not report.ok
    assert report.issues[0].path == "/roles/trainer"

    poor = worlds.WorldSpec.from_data({"roles": {"trainer": {"replicas": 2, "process": {"resources": {"cpus": 1, "memory": "1GiB"}}}}})
    report = worlds.check_world_spec_satisfies_requirement(poor, requirement())
    paths = {issue.path for issue in report.issues if issue.severity == "error"}
    assert "/roles/trainer/replicas" in paths
    assert "/roles/trainer/process/resources/cpus" in paths
    assert "/roles/trainer/process/resources/memory" in paths
    assert "/roles/trainer/process/resources/accelerators/gpu" in paths


def test_allocation_satisfies_requirement_and_reports_gpu_failure():
    allocation = worlds.WorldAllocation.from_data(
        {"roles": {"trainer": [{"replica": 0, "rank": 0, "local_rank": 0, "resources": {"cpus": [0, 1, 2, 3], "memory": "16GiB", "accelerators": {"gpu": [0]}}}]}}
    )
    assert worlds.check_allocation_satisfies_requirement(allocation, requirement()).ok

    bad = worlds.WorldAllocation.from_data({"roles": {"trainer": [{"replica": 0, "rank": 0, "local_rank": 0, "resources": {"cpus": [0], "accelerators": {}}}]}})
    report = worlds.check_allocation_satisfies_requirement(bad, requirement())
    assert not report.ok
    assert any(issue.path.endswith("/accelerators/gpu") for issue in report.issues)
