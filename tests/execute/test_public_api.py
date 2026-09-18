"""Public Execute facade contract after the generic cutover."""

from __future__ import annotations

import importlib
import inspect

import pytest

import dryml.execute as execute
from dryml.execute.subprocess import SubProcessConfig


def test_common_public_api_is_exact_and_specializations_remain_separate():
    """Expose the approved generic surface without legacy backend aliases."""
    assert set(execute.__all__) == {
        "ActiveAllocation",
        "AdmissionError",
        "AdmissionReport",
        "Backend",
        "BackendConfig",
        "BackendUnavailableError",
        "CleanupError",
        "DiscoverySnapshot",
        "EnvironmentCandidate",
        "ExecutionDeadlineExceeded",
        "ExecutionError",
        "ExecutionFuture",
        "ExecutionIssue",
        "ExecutionOutput",
        "ExecutionSnapshot",
        "ExecutionUncertainError",
        "Executor",
        "ExecutorView",
        "FeasiblePlan",
        "OutputSnapshot",
        "RemoteExecutionError",
        "ResourceAmounts",
        "ResourceSnapshot",
        "WorkerSetup",
        "WorkerSetupContext",
        "WorkerSetupFactory",
        "run",
        "submit",
    }
    assert hasattr(importlib.import_module("dryml.execute.subprocess"), "SubProcessConfig")


def test_control_bearing_signatures_keep_workload_kwargs_separate():
    """Root helpers require explicit configuration and reserve only controls."""
    expected = {
        execute.run:
        ("fn", "args", "backend", "kwargs", "environment", "environment_spec",
         "world", "execution_timeout", "stream_output", "done_callbacks",
         "output", "worker_setup"),
        execute.submit:
        ("fn", "args", "backend", "kwargs", "environment", "environment_spec",
         "world", "execution_timeout", "stream_output", "done_callbacks",
         "output", "worker_setup"),
        execute.Executor.run:
        ("self", "fn", "args", "kwargs", "environment", "environment_spec",
         "world", "execution_timeout", "stream_output", "done_callbacks",
         "output", "worker_setup"),
        execute.Executor.submit:
        ("self", "fn", "args", "kwargs", "environment", "environment_spec",
         "world", "execution_timeout", "stream_output", "done_callbacks",
         "output", "worker_setup"),
    }
    for function, parameters in expected.items():
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == parameters
        assert signature.parameters["kwargs"].default is None
        if "backend" in signature.parameters:
            assert signature.parameters["backend"].default is inspect.Parameter.empty


@pytest.mark.parametrize("name", ["BackendBase", "InlineBackend", "LocalProcessBackend", "OrchestratedFuture", "ExecutionOrchestrator", "ExecutionRequest", "ExecutionResponse", "StoreRef"])
def test_removed_public_names_have_no_compatibility_alias(name):
    """Retired facade types cannot be recovered through the new root namespace."""
    assert not hasattr(execute, name)


@pytest.mark.parametrize("module", ["dryml.execute.orchestrator", "dryml.execute.transfer", "dryml.execute.protocol", "dryml.execute.worker"])
def test_retired_protocol_modules_are_not_importable(module):
    """The old Store transport is absent rather than retained behind a warning."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize("option", ["repo", "update", "transfer_store", "result_store", "env", "requirements"])
def test_removed_controls_reject_before_execution(option, tmp_path):
    """Legacy orchestration controls cannot be silently interpreted as workloads."""
    called = False

    def workload():
        nonlocal called
        called = True

    with pytest.raises(TypeError):
        execute.run(workload, backend=SubProcessConfig(spool_directory=tmp_path), **{option: object()})
    assert not called


def test_one_off_without_a_backend_configuration_is_rejected():
    """A module-level call cannot select an ambient default backend."""
    with pytest.raises(TypeError):
        execute.run(lambda: None)


@pytest.mark.parametrize(
    "factory",
    ["missing-separator", "module:<locals>", "module:bad-name", ":factory"],
)
def test_worker_setup_is_inert_and_rejects_nonimportable_factory_identifiers(factory):
    """Setup values retain only validated declarative factory identifiers."""
    with pytest.raises((TypeError, ValueError)):
        execute.WorkerSetup(factory=factory, data={})


def test_worker_setup_detaches_json_data_without_repr_fallbacks():
    """Accepted setup data is immutable JSON rather than a live control object."""
    source = {"items": ["original"]}
    setup = execute.WorkerSetup(factory="tests.execute.test_public_api:setup_factory", data=source)
    source["items"].append("changed")

    assert setup.data == {"items": ("original",)}
    with pytest.raises(TypeError):
        setup.data["new"] = "value"
    with pytest.raises(TypeError):
        execute.WorkerSetup(factory="tests.execute.test_public_api:setup_factory", data={"not_json": object()})


def test_worker_setup_uses_one_deep_bounded_format_and_configured_byte_budget(tmp_path):
    """Deep valid JSON transfers while caller oversize rejects before backend launch."""
    value: dict[str, object] = {"leaf": "value"}
    for _ in range(16):
        value = {"next": value}
    setup = execute.WorkerSetup(
        factory="tests.execute.test_public_api:setup_factory", data=value,
    )
    assert setup.to_data(limit_bytes=4096)["data"]
    envelope = setup.to_envelope(
        {"submission_id": "submission", "backend": "fake", "environment": None,
         "allocation": None, "native_grant": {}},
        owner_limit_bytes=4096, admission_limit_bytes=4096,
    )
    assert len(envelope) <= 4096
    with pytest.raises(ValueError, match="configured"):
        setup.to_envelope(
            {"submission_id": "submission", "backend": "fake", "environment": None,
             "allocation": None, "native_grant": {}},
            owner_limit_bytes=4096, admission_limit_bytes=32,
        )

    executor = execute.Executor(SubProcessConfig(
        spool_directory=tmp_path, control_header_limit_bytes=64,
        owner_envelope_limit_bytes=64, admission_message_limit_bytes=64,
    ))
    with pytest.raises(ValueError, match="configured"):
        executor.submit(lambda: None, worker_setup=execute.WorkerSetup(
            factory="tests.execute.test_public_api:setup_factory", data={"secret": "x" * 256},
        ))
    executor.close(timeout=1)


def test_worker_setup_factory_identifier_has_a_finite_safe_length():
    """Factory validation cannot retain an arbitrarily long identifier string."""
    with pytest.raises(ValueError, match="importable"):
        execute.WorkerSetup(factory=f"module:{'a' * 513}", data={})


def setup_factory(_context, _data):
    """Provide an importable inert context-manager fixture for setup validation."""
    from contextlib import nullcontext

    return nullcontext()
