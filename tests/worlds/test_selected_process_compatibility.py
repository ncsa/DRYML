"""Selected-process world admission coverage."""

from dryml.worlds import (
    ProcessAllocation,
    WorldRequirement,
    check_selected_process_satisfies_requirement as check_selected,
)


def test_selected_process_checker_distinguishes_absence_and_missing_evidence():
    """No requirement is admissible without allocation, a valued one is not."""

    assert check_selected(None, None, None).ok

    requirement = WorldRequirement({
        "main": {"resources": {"cpus": {"min": 1, "max": None}}},
    })
    report = check_selected(None, None, requirement)

    assert not report.ok
    assert report.issues[0].code == "allocation_missing"


def test_selected_process_rejects_wrong_role_and_insufficient_resources():
    """Only selected role/process evidence can satisfy hard requirements."""

    selected = ProcessAllocation(0, 0, 0, cpus=(0,))
    requirement = WorldRequirement(
        {"main": {"resources": {"cpus": {"min": 2, "max": None}}}},
    )

    report = check_selected("other", selected, requirement)

    assert {issue.code for issue in report.issues} == {
        "selected_role_unexpected",
        "missing_role",
    }


def test_selected_process_rejects_replica_topology_and_unknown_resources():
    """One process cannot invent missing topology or named/device evidence."""

    selected = ProcessAllocation(0, 0, 0, cpus=(0,))
    requirement = WorldRequirement({
        "main": {
            "replicas": {"min": 2, "max": 2},
            "resources": {
                "named": {"license": {"min": 1, "max": None}},
                "devices": {"gpu": {"min": None, "max": 1}},
            },
            "topology": {"rack": "a"},
        },
    })

    report = check_selected("main", selected, requirement)

    assert {issue.code for issue in report.issues} == {
        "constraint_unsatisfied",
        "unsupported_resource",
        "unsupported_topology",
    }
    assert sum(issue.code == "unsupported_resource"
               for issue in report.issues) == 2


def test_selected_process_rejects_missing_hard_memory_evidence_even_for_maxima(
):
    """
    Unknown memory facts never become zero-valued evidence for hard bounds.
    """

    selected = ProcessAllocation(0, 0, 0, cpus=(0,))
    requirement = WorldRequirement({
        "main": {"resources": {"memory": {"min": None, "max": 1}}},
    })

    report = check_selected("main", selected, requirement)

    assert [issue.code for issue in report.issues] == ["memory_missing"]


def test_selected_process_rejects_missing_accelerator_memory_for_max_only_bounds(  # noqa: E501
):
    """
    Per-device memory requirements need evidence even with no lower bound.
    """

    selected = ProcessAllocation(0, 0, 0, cpus=(0,))
    requirement = WorldRequirement({
        "main": {
            "resources": {
                "accelerator_memory": {"gpu": {"min": None, "max": "1GiB"}},
            },
        },
    })

    report = check_selected("main", selected, requirement)

    assert [issue.code
            for issue in report.issues] == ["accelerator_memory_missing"]


def test_selected_process_rejects_assigned_accelerator_without_memory_evidence(
):
    """
    An assigned accelerator needs its own memory evidence for every hard bound.
    """

    selected = ProcessAllocation(
        0, 0, 0, cpus=(0,), accelerators={"gpu": ("0",)},
    )
    requirement = WorldRequirement({
        "main": {
            "resources": {
                "accelerator_memory": {"gpu": {"min": None, "max": "1GiB"}},
            },
        },
    })

    report = check_selected("main", selected, requirement)

    assert [issue.code
            for issue in report.issues] == ["accelerator_memory_missing"]
