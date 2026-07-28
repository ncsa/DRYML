from types import MappingProxyType

import pytest

from dryml.environments import EnvironmentRequirement, EnvironmentRequirementError
from dryml.session.configuration import normalize_configuration, select_world_allocation
from dryml.session.errors import SessionConfigurationError
from dryml.worlds import LocalResourceInventory, WorldAllocation


def test_complete_configuration_is_canonical_and_deeply_immutable():
    left = normalize_configuration(
        mode="managed",
        resources={"cpus": 2, "memory": "1GiB", "gpus": 2, "accelerator_memory": ["512MiB", "1GiB"]},
        requested_world={"roles": {"worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}},
        environment={"requirements": ["torch>=2", "dryml>=0.3"], "capabilities": ["b", "a"]},
    )
    right = normalize_configuration(
        mode="managed",
        resources={"gpus": 2, "accelerator_memory": [512 * 1024**2, 1024**3], "memory": 1024**3, "cpus": 2},
        requested_world={"roles": {"worker": {"process": {"resources": {"cpus": 1}}, "replicas": 1}}},
        environment={"capabilities": ["a", "b"], "requirements": ["dryml>=0.3", "torch>=2"]},
    )

    assert left.fingerprint == right.fingerprint
    assert left.resources.to_data()["accelerator_memory"] == {"gpu": ["512MiB", "1GiB"]}
    assert isinstance(left.controls, MappingProxyType)
    with pytest.raises(TypeError):
        left.controls["memory"] = "changed"
    with pytest.raises(TypeError):
        left.environment.schema_versions["new"] = "==1"


@pytest.mark.parametrize(
    "resources",
    (
        {"cpus": True},
        {"cpus": 0},
        {"memory": "1GB"},
        {"gpus": 0, "accelerator_memory": "1GiB"},
        {"gpus": 2, "accelerator_memory": ["1GiB"]},
        {"unknown": 1},
    ),
)
def test_invalid_configuration_resource_section_fails_before_a_candidate(resources):
    with pytest.raises(SessionConfigurationError):
        normalize_configuration(mode="managed", resources=resources)


def test_complete_configuration_requires_mode_and_rejects_conflicting_sections():
    with pytest.raises(SessionConfigurationError):
        normalize_configuration(mode=None)
    with pytest.raises(SessionConfigurationError):
        normalize_configuration(
            mode="managed",
            resources={"cpus": 1},
            allocation={"value": _single_allocation_envelope()},
        )


@pytest.mark.parametrize(
    "value",
    (
        {"resources": {"cpus": 1}, "unknown": {}},
        {"resources": {"cpus": 1, "memory": "x" * 4097}},
        {"resources": {"cpus": 1}, "requested_world": {"nested": {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": 1}}}}}}}}}},
    ),
)
def test_configuration_grammar_is_closed_and_bounded(value):
    with pytest.raises(SessionConfigurationError):
        normalize_configuration(mode="managed", **value)


def test_environment_requirements_merge_semantically_and_conflicts_are_atomic():
    left = EnvironmentRequirement(requirements=("torch>=2",), python=">=3.10")
    right = EnvironmentRequirement(requirements=("torch<3", "dryml>=0.3"), python="<3.13")

    merged = left.merge(right, sources=("session",))

    assert merged.requirements == ("dryml>=0.3", "torch<3,>=2")
    assert merged.python == "<3.13,>=3.10"
    assert merged.details["sources"] == ("session",)
    with pytest.raises(EnvironmentRequirementError):
        merged.merge(EnvironmentRequirement(requirements=("torch<2",)))
    assert merged.requirements == ("dryml>=0.3", "torch<3,>=2")


def test_exact_allocation_selection_requires_unambiguous_process_selector():
    one = WorldAllocation.from_data(_single_allocation_envelope()["payload"])
    selected = select_world_allocation(one)
    assert selected.role == "main"
    assert selected.process.rank == 0

    multi = _multi_allocation_envelope()
    with pytest.raises(SessionConfigurationError):
        select_world_allocation(multi)
    with pytest.raises(SessionConfigurationError):
        select_world_allocation(multi, role="worker")
    selected = select_world_allocation(multi, role="worker", replica=1)
    assert selected.process.cpus == (1,)


def test_exact_allocation_rejects_unknown_envelope_and_invalid_selection():
    invalid = _single_allocation_envelope()
    invalid["extra"] = True
    with pytest.raises(SessionConfigurationError):
        select_world_allocation(invalid)
    with pytest.raises(SessionConfigurationError):
        select_world_allocation(_single_allocation_envelope(), role="main")


def test_exact_allocation_cannot_broaden_an_inherited_inventory():
    allocation = _single_allocation_envelope()
    allocation["payload"]["roles"]["main"][0]["resources"]["cpus"] = [2]

    with pytest.raises(SessionConfigurationError):
        select_world_allocation(allocation, inventory=LocalResourceInventory((0,)))


def _single_allocation_envelope():
    return {
        "schema": "dryml.world_allocation.v1",
        "payload": {
            "roles": {
                "main": [
                    {
                        "replica": 0,
                        "rank": 0,
                        "local_rank": 0,
                        "resources": {"cpus": [0], "accelerators": {}},
                    }
                ]
            }
        },
    }


def _multi_allocation_envelope():
    payload = _single_allocation_envelope()["payload"]
    payload["roles"] = {
        "worker": [
            {"replica": 0, "rank": 0, "local_rank": 0, "resources": {"cpus": [0], "accelerators": {}}},
            {"replica": 1, "rank": 1, "local_rank": 1, "resources": {"cpus": [1], "accelerators": {}}},
        ]
    }
    return {"schema": "dryml.world_allocation.v1", "payload": payload}
