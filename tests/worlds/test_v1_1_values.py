import pytest

import dryml.worlds as worlds


def test_v1_1_world_values_are_immutable_and_metadata_is_non_identifying():
    first = worlds.WorldRequirement.from_payload({"roles": {"main": {}}}, metadata={"trace": "one"})
    second = worlds.WorldRequirement.from_payload({"roles": {"main": {}}}, metadata={"trace": "two"})
    assert first.semantic_id == second.semantic_id
    assert first.to_data()["schema"] == "dryml.world_requirement.v1.1"
    assert worlds.WorldRequirement.from_data(first.to_data()) == first
    with pytest.raises(TypeError):
        first.roles["other"] = first.roles["main"]


def test_world_envelopes_reject_wrong_family_and_attached_id():
    value = worlds.WorldSpec.from_payload({"roles": {"main": {"replicas": 1, "process": {}}}})
    data = value.to_data()
    data["schema"] = "dryml.world_requirement.v1.1"
    with pytest.raises(Exception):
        worlds.WorldSpec.from_data(data)
