import dryml.environments as envs
import dryml.runtime as runtime
import dryml.worlds as worlds


def test_environment_world_runtime_boundaries():
    env_req = envs.EnvironmentRequirement(requirements=("torch>=2",), python=">=3.10")
    world_req = worlds.WorldRequirement.from_data({"roles": {"trainer": {"resources": {"accelerators": {"gpu": {"min": 1}}}}}})
    runtime_spec = runtime.RuntimeContextSpec.from_data({"mode": "worker", "device_visibility": {"policy": "assigned"}})

    assert "gpu" not in env_req.to_data()
    assert world_req.roles["trainer"].resources.accelerators["gpu"].to_data() == {"min": 1}
    assert runtime_spec.device_visibility["policy"] == "assigned"


def test_gpu_request_is_not_an_environment_resource_model():
    req = envs.EnvironmentRequirement(requirements=("gpu>=1",))

    assert req.to_data()["requirements"] == ["gpu>=1"]
    assert "accelerators" not in req.to_data()
