import pytest

import dryml.worlds as worlds
from dryml.worlds.errors import WorldSpecValidationError


def req_roles():
    return {
        "trainer": {
            "replicas": {"exact": 1},
            "resources": {"cpus": {"min": 4}, "memory": {"min": "16GiB"}, "accelerators": {"gpu": {"min": 1}}},
            "topology": {"single_process": True},
        }
    }


def test_world_requirement_spec_shape_and_stable_id():
    left = worlds.attach_world_requirement_id(worlds.make_world_requirement_spec(req_roles()))
    right = worlds.attach_world_requirement_id(worlds.make_world_requirement_spec({"trainer": dict(reversed(list(req_roles()["trainer"].items())))}))

    assert left["schema"] == "dryml.world_requirement.v1"
    assert left["id"].startswith("worldreq-v1-")
    assert left["id"] == right["id"]
    assert worlds.validate_world_requirement_spec(left) is left


def test_world_requirement_id_changes_with_semantic_content():
    left = worlds.attach_world_requirement_id(worlds.make_world_requirement_spec(req_roles()))
    changed_roles = req_roles()
    changed_roles["trainer"]["resources"]["cpus"] = {"min": 8}
    changed = worlds.attach_world_requirement_id(worlds.make_world_requirement_spec(changed_roles))

    assert left["id"] != changed["id"]


def test_world_requirement_rejects_malformed_payload():
    spec = worlds.make_world_requirement_spec(req_roles())
    spec["payload"] = {"roles": {"bad role": {}}}
    with pytest.raises(WorldSpecValidationError):
        worlds.validate_world_requirement_spec(spec)


def test_world_requirement_merge_conflict_has_context():
    left = worlds.WorldRequirement.from_data({"roles": req_roles()})
    right = worlds.WorldRequirement.from_data({"roles": {"trainer": {"replicas": {"exact": 2}, "resources": {}, "topology": {}}}})
    with pytest.raises(WorldSpecValidationError) as excinfo:
        left.merge(right)
    assert "roles.trainer" in excinfo.value.context["path"]
