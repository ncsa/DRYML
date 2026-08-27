import pytest

from dryml.worlds import RoleSpec, WorldSpec


def test_requested_world_is_a_launch_shape_not_allocation():
    world = WorldSpec.from_payload({"roles": {"worker": {"replicas": 2, "process": {"resources": {"cpus": 1}}}}})
    assert world.roles["worker"].replicas == 2
    assert "rank" not in world.to_data()["payload"]["roles"]["worker"]["process"]


def test_requested_roles_require_positive_replicas():
    with pytest.raises(Exception, match="positive"):
        RoleSpec(replicas=0)
