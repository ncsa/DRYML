import os

import pytest

import dryml.environments as envs


def roundtrip(spec):
    return envs.spec_from_data(spec.to_data())


def test_environment_spec_roundtrips_and_ids():
    specs = [
        envs.CurrentEnvironmentSpec(),
        envs.PythonExecutableSpec("/usr/bin/python", env={"A": "B"}),
        envs.CondaEnvironmentSpec(prefix="/opt/envs/a"),
        envs.CondaEnvironmentSpec(name="named", launch_mode="conda-run"),
        envs.ContainerEnvironmentSpec("example/image@sha256:abc", runtime="docker"),
    ]
    for spec in specs:
        clone = roundtrip(spec)
        assert clone.to_data() == spec.to_data()
        assert clone.id == spec.id


def test_python_executable_probe_command():
    spec = envs.PythonExecutableSpec("/usr/bin/python")
    assert spec.probe_command() == ["/usr/bin/python", "-m", "dryml.environments.probe_worker", "--json"]


def test_conda_direct_and_conda_run_commands():
    direct = envs.CondaEnvironmentSpec(prefix="/opt/envs/a")
    assert direct.direct_python_executable(os_name="posix") == os.path.join("/opt/envs/a", "bin", "python")
    assert direct.direct_python_executable(os_name="nt") == os.path.join("/opt/envs/a", "python.exe")
    run_prefix = envs.CondaEnvironmentSpec(prefix="/opt/envs/a", launch_mode="conda-run")
    assert run_prefix.probe_command()[:4] == ["conda", "run", "-p", "/opt/envs/a"]
    run_name = envs.CondaEnvironmentSpec(name="named", launch_mode="conda-run")
    assert run_name.probe_command()[:4] == ["conda", "run", "-n", "named"]


def test_conda_spec_validation():
    with pytest.raises(envs.EnvironmentSpecError):
        envs.CondaEnvironmentSpec(prefix="/a", name="b")
    with pytest.raises(envs.EnvironmentSpecError):
        envs.CondaEnvironmentSpec(name="only-name").probe_command()
    with pytest.raises(envs.EnvironmentSpecError):
        envs.spec_from_data({"kind": "unknown"})


def test_environment_spec_rejects_non_string_env_keys():
    with pytest.raises(envs.EnvironmentSerializationError, match="mapping keys must be strings"):
        envs.PythonExecutableSpec("/usr/bin/python", env={"1": "string-key", 1: "integer-key"})


@pytest.mark.parametrize(
    "kwargs",
    (
        {"executable": ""},
        {"executable": 1},
        {"executable": "/python", "env": {"TOKEN": 1}},
        {"executable": "/python", "extra_pythonpath": (1,)},
    ),
)
def test_python_executable_spec_rejects_non_launchable_values(kwargs):
    with pytest.raises(envs.EnvironmentSpecError):
        envs.PythonExecutableSpec(**kwargs)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: envs.PythonExecutableSpec("bad\x00python"),
        lambda: envs.PythonExecutableSpec("/python", env={"BAD=KEY": "value"}),
        lambda: envs.PythonExecutableSpec("/python", env={"KEY": "bad\x00value"}),
        lambda: envs.CondaEnvironmentSpec(prefix="bad\x00prefix"),
        lambda: envs.CondaEnvironmentSpec(name="named", launch_mode="conda-run", conda_executable="bad\x00conda"),
    ),
)
def test_environment_specs_reject_os_invalid_launch_strings(factory):
    with pytest.raises(envs.EnvironmentSpecError):
        factory()


def test_environment_lock_ref_roundtrip_and_id():
    lock = envs.EnvironmentLockRef("conda-lock", "file:///tmp/conda-lock.yml", digest="sha256:abc")
    clone = envs.EnvironmentLockRef.from_data(lock.to_data())
    assert clone.to_data() == lock.to_data()
    assert clone.id == lock.id
