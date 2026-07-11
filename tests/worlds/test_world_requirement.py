import json
import math

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


@pytest.mark.parametrize(
    "topology",
    (
        {"hint": object()},
        {"hint": math.nan},
        {"hint": math.inf},
    ),
)
def test_role_requirement_rejects_non_json_safe_topology_values(topology):
    with pytest.raises(WorldSpecValidationError, match="topology"):
        worlds.RoleRequirement.from_data({"topology": topology})
    with pytest.raises(WorldSpecValidationError, match="topology"):
        worlds.RoleRequirement(topology=topology)


def test_role_requirement_rejects_cyclic_topology_and_preserves_json_safe_direct_data():
    cycle = []
    cycle.append(cycle)

    with pytest.raises(WorldSpecValidationError, match="cycles"):
        worlds.RoleRequirement.from_data({"topology": {"hint": cycle}})
    with pytest.raises(WorldSpecValidationError, match="cycles"):
        worlds.RoleRequirement(topology={"hint": cycle})

    requirement = worlds.WorldRequirement(
        {"main": worlds.RoleRequirement(topology={"hint": {"devices": ["gpu"]}})}
    )

    assert json.loads(json.dumps(requirement.to_data())) == {
        "roles": {
            "main": {
                "replicas": {"exact": 1},
                "resources": {},
                "topology": {"hint": {"devices": ["gpu"]}},
            }
        }
    }


def test_role_requirement_topology_round_trips_direct_construction():
    direct = worlds.RoleRequirement(topology={"hint": {"devices": ["GPU-01234567-89ab-cdef-0123-456789abcdef"]}})

    restored = worlds.RoleRequirement.from_data(direct.to_data())

    assert restored.to_data() == direct.to_data()


@pytest.mark.parametrize(
    ("topology", "message"),
    (
        ({f"hint-{index}": True for index in range(65)}, "mapping exceeds"),
        ({"hint": [True] * 65}, "sequence exceeds"),
        ({"hint": "x" * 4097}, "string exceeds"),
        ({"branches": [[True] * 64 for _ in range(64)]}, "aggregate bounded"),
    ),
)
def test_role_requirement_rejects_bounded_wide_topologies(topology, message):
    for build in (
        lambda: worlds.RoleRequirement(topology=topology),
        lambda: worlds.RoleRequirement.from_data({"topology": topology}),
    ):
        with pytest.raises(WorldSpecValidationError, match=message):
            build()


def test_role_requirement_rejects_deep_topology_without_recursing_unboundedly():
    topology = current = {}
    for _ in range(16):
        next_value = {}
        current["child"] = next_value
        current = next_value

    for build in (
        lambda: worlds.RoleRequirement(topology=topology),
        lambda: worlds.RoleRequirement.from_data({"topology": topology}),
    ):
        with pytest.raises(WorldSpecValidationError, match="nesting exceeds") as excinfo:
            build()
        assert excinfo.value.context["limit"] == 8
