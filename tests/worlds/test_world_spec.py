import pytest
from collections.abc import Mapping

import dryml.worlds as worlds


def test_single_role_world_spec_and_stable_id():
    spec = worlds.attach_world_id(
        worlds.make_world_spec(
            {
                "trainer": {
                    "replicas": 1,
                    "process": {"resources": {"cpus": 8, "memory": "32GiB", "accelerators": {"gpu": 1}}, "environment": "torch-training"},
                }
            },
            backend={"kind": "local_subprocess", "parameters": {}},
        )
    )

    assert spec["schema"] == "dryml.world.v1"
    assert spec["id"].startswith("world-v1-")
    assert worlds.compute_world_id(spec) == spec["id"]
    assert worlds.validate_world_spec(spec) is spec


def test_multi_role_world_spec_is_requested_shape_only():
    spec = worlds.attach_world_id(
        worlds.make_world_spec(
            {
                "trainer": {"replicas": 2, "process": {"resources": {"cpus": 4, "accelerators": {"gpu": 1}}}},
                "evaluator": {"replicas": 1, "process": {"resources": {"cpus": 2}}},
            }
        )
    )

    assert set(spec["payload"]["roles"]) == {"trainer", "evaluator"}
    assert "allocation" not in spec["payload"]


def test_world_spec_bounds_roles_and_resource_maps_before_normalization():
    class Roles(Mapping):
        def __len__(self):
            return 4097

        def __iter__(self):
            return iter(())

        def __getitem__(self, key):
            raise KeyError(key)

        def items(self):
            for index in range(4096):
                yield f"role_{index}", {}
            yield "excess", object()
            raise AssertionError("world spec read past its role bound")

    class Accelerators(Mapping):
        def __len__(self):
            return 257

        def __iter__(self):
            return iter(())

        def __getitem__(self, key):
            raise KeyError(key)

        def items(self):
            for index in range(256):
                yield f"gpu_{index}", 0
            yield "excess", object()
            raise AssertionError("resource parser read past its map bound")

    with pytest.raises(Exception, match="role count"):
        worlds.WorldSpec.from_data({"roles": Roles()})
    with pytest.raises(Exception, match="resource mapping"):
        worlds.WorldSpec.from_data({"roles": {"main": {"process": {"resources": {"accelerators": Accelerators()}}}}})


@pytest.mark.parametrize("env", ({"PATH": "one", "Path": "two"},))
def test_world_spec_rejects_casefold_colliding_process_environment_keys(env):
    with pytest.raises(Exception, match="differ only by case"):
        worlds.WorldSpec.from_data({"roles": {"main": {"process": {"env": env}}}})
