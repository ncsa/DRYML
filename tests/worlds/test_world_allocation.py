import pytest

from dryml.worlds import WorldAllocation, WorldSpecValidationError


def test_exact_allocation_assigns_identifying_resources_and_nonidentifying_metadata():
    payload = {"roles": {"worker": [{"replica": 0, "rank": 0, "local_rank": 0, "resources": {"cpus": [1], "accelerators": {"gpu": ["a"]}, "accelerator_memory": {"gpu": [{"device": "a", "memory": "1GiB"}]}}, "metadata": {"note": "one"}}]}}
    value = WorldAllocation.from_payload(payload)
    changed = WorldAllocation.from_payload({**payload, "roles": {"worker": [{**payload["roles"]["worker"][0], "metadata": {"note": "two"}}]}})
    assert value.semantic_id == changed.semantic_id
    with pytest.raises(WorldSpecValidationError):
        WorldAllocation.from_payload({"roles": {"worker": [{"replica": 0, "rank": 0, "local_rank": 0, "resources": {"cpus": [1, 1]}}]}})


def test_process_allocation_roundtrips_named_resource_assignments():
    allocation = WorldAllocation.from_payload({"roles": {"main": [{
        "replica": 0,
        "rank": 0,
        "local_rank": 0,
        "resources": {"cpus": [0], "named": {"license": "slot-1"}},
    }]}})

    assert allocation.roles["main"][0].named["license"] == "slot-1"
    assert WorldAllocation.from_data(allocation.to_data()) == allocation
