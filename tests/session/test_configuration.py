"""Pure configuration normalization and exact allocation selection contracts."""

from copy import deepcopy

import pytest

from dryml.environments import EnvironmentRequirement
from dryml.session import SessionConfiguration, SessionConfigurationError, normalize_configuration, select_world_allocation
from dryml.worlds import LocalResourceInventory, ProcessAllocation, WorldAllocation


def test_normalization_is_pure_and_retries_are_not_identity():
    """Configuration identity contains semantic state but no operation controls."""

    first = normalize_configuration(mode="orchestrator", environment={"requirements": ["dryml>=0.3"]})
    second = normalize_configuration(mode="orchestrator", environment={"requirements": ["dryml>=0.3"]})

    assert first.fingerprint == second.fingerprint
    assert "restage_retries" not in first.to_data()
    with pytest.raises(SessionConfigurationError, match="source-v1"):
        normalize_configuration(mode="python", source="v1")
    assert normalize_configuration(mode="managed", resources={"cpus": 1}).allocation is None


def test_configuration_uses_a_closed_self_validating_v1_1_envelope():
    """Configuration data uses its own canonical family and rejects drift."""

    configuration = normalize_configuration(mode="orchestrator", environment={"requirements": ["dryml>=0.3"]})
    data = configuration.to_data()

    assert configuration.fingerprint.startswith("sessioncfg-v1.1-")
    assert data["schema"] == "dryml.session_configuration.v1.1"
    assert data["kind"] == "session_configuration"
    assert SessionConfiguration.from_data(data) == configuration

    for mutate in (
        lambda value: value.update({"source": "v1"}),
        lambda value: value.update({"contract_version": "1.2"}),
        lambda value: value["payload"].update({"unknown": True}),
        lambda value: value.update({"id": "sessioncfg-v1.1-" + "0" * 64}),
    ):
        invalid = deepcopy(data)
        mutate(invalid)
        with pytest.raises(Exception):
            SessionConfiguration.from_data(invalid)


def test_configuration_diagnostic_metadata_is_not_identity_bearing():
    """Nested non-identifying diagnostics do not alter the configuration ID."""

    first = SessionConfiguration("python", environment=EnvironmentRequirement(details={"sources": ["one"]}, metadata={"trace": "one"}))
    second = SessionConfiguration("python", environment=EnvironmentRequirement(details={"sources": ["two"]}, metadata={"trace": "two"}))

    assert first.fingerprint == second.fingerprint
    assert SessionConfiguration.from_data(first.to_data()) == first


def test_derived_controls_are_non_identifying_and_environment_is_embedded():
    """Session identity excludes controls and embeds no nested family envelope."""

    first = SessionConfiguration("python", controls={"memory": "undeclared"})
    second = SessionConfiguration("python", controls={"memory": "declarative"})
    data = first.to_data()

    assert first.fingerprint == second.fingerprint
    assert set(data["payload"]["environment"]) == {"python", "requirements", "excludes", "capabilities", "tags", "dryml_protocol", "schema_versions", "details"}
    assert "contract_version" not in data["payload"]["environment"]
    assert SessionConfiguration.from_data(data) == first


def test_resource_memory_accepts_scalar_and_per_gpu_sequences():
    """Managed shorthand expands scalar GPU memory and retains per-device data."""

    scalar = normalize_configuration(mode="managed", resources={"cpus": 1, "gpus": 2, "accelerator_memory": "1GiB"})
    sequence = normalize_configuration(mode="managed", resources={"cpus": 1, "gpus": 2, "accelerator_memory": ["1GiB", "2GiB"]})

    assert scalar.resources.accelerator_memory["gpu"] == (1024**3, 1024**3)
    assert sequence.resources.accelerator_memory["gpu"] == (1024**3, 2 * 1024**3)


def test_selection_rejects_ambiguous_and_broadened_assignments():
    """Exact allocation selection needs complete selectors and inherited bounds."""

    allocation = WorldAllocation({
        "main": (
            ProcessAllocation(0, 0, 0, cpus=(0,)),
            ProcessAllocation(1, 1, 1, cpus=(1,)),
        ),
    })
    with pytest.raises(SessionConfigurationError, match="require role"):
        select_world_allocation(allocation)
    with pytest.raises(SessionConfigurationError, match="appear together"):
        select_world_allocation(allocation, role="main")
    with pytest.raises(SessionConfigurationError, match="broadens"):
        select_world_allocation(
            WorldAllocation({"main": (ProcessAllocation(0, 0, 0, cpus=(9,)),)}),
            inventory=LocalResourceInventory((0,)),
        )


def test_selection_rejects_positive_memory_without_capacity_evidence():
    """Unknown process and device memory never authorize positive allocation."""

    process_memory = WorldAllocation({"main": (ProcessAllocation(0, 0, 0, cpus=(0,), memory="1GiB"),)})
    accelerator_memory = WorldAllocation({"main": (ProcessAllocation(0, 0, 0, cpus=(0,), accelerators={"gpu": (0,)}, accelerator_memory={"gpu": {0: "1GiB"}}),)})

    with pytest.raises(SessionConfigurationError, match="memory capacity is unknown"):
        select_world_allocation(process_memory, inventory=LocalResourceInventory((0,), memory=None))
    with pytest.raises(SessionConfigurationError, match="memory capacity is unknown"):
        select_world_allocation(accelerator_memory, inventory=LocalResourceInventory((0,), {"gpu": (0,)}))
