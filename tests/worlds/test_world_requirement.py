import pytest

from dryml.worlds import WorldRequirement, WorldSpecValidationError


def test_requirement_preserves_nonempty_topology_as_declaration():
    requirement = WorldRequirement.from_payload({"roles": {"worker": {"topology": {"rack": "a"}}}})
    assert requirement.roles["worker"].topology == {"rack": "a"}


def test_requirement_role_bound_is_closed():
    with pytest.raises(WorldSpecValidationError):
        WorldRequirement.from_payload({"roles": {f"role_{index}": {} for index in range(4097)}})
