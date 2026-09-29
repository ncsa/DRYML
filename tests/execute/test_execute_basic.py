import pytest

import dryml.execute as execute
from dryml.execute.subprocess import SubProcessConfig


def add(x, y):
    return x + y


def boom():
    raise ValueError("expected failure")


def test_explicit_subprocess_one_off_returns_result(tmp_path):
    assert execute.run(add, 2, 3, backend=SubProcessConfig(spool_directory=tmp_path)) == 5


def test_executor_view_forwards_control_named_workload_keywords(tmp_path):
    def controls(*, backend, environment, world, output, kwargs):
        return backend, environment, world, output, kwargs

    with execute.Executor(SubProcessConfig(spool_directory=tmp_path)) as executor:
        view = executor.with_options()
        assert view.run(
            controls,
            backend="workload-backend",
            environment="workload-environment",
            world="workload-world",
            output="workload-output",
            kwargs="workload-kwargs",
        ) == (
            "workload-backend",
            "workload-environment",
            "workload-world",
            "workload-output",
            "workload-kwargs",
        )


def test_subprocess_execute_propagates_error(tmp_path):
    future = execute.submit(boom, backend=SubProcessConfig(spool_directory=tmp_path))
    err = future.exception(timeout=10)

    assert isinstance(err, execute.RemoteExecutionError)
    assert err.remote_type == "ValueError"
    future.cleanup()


@pytest.mark.parametrize("legacy_backend", ["inline", "process", None])
def test_one_off_requires_a_backend_configuration(legacy_backend):
    with pytest.raises(TypeError, match="backend must be a BackendConfig"):
        execute.run(add, 2, 3, backend=legacy_backend)
