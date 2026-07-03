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
