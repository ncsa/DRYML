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
        "run",
        "submit",
    }
    assert hasattr(importlib.import_module("dryml.execute.subprocess"), "SubProcessConfig")


def test_control_bearing_signatures_keep_workload_kwargs_separate():
    """Root helpers require explicit configuration and reserve only controls."""
    expected = {
        execute.run: ("fn", "args", "backend", "kwargs", "environment", "world", "execution_timeout", "stream_output", "done_callbacks", "output"),
        execute.submit: ("fn", "args", "backend", "kwargs", "environment", "world", "execution_timeout", "stream_output", "done_callbacks", "output"),
        execute.Executor.run: ("self", "fn", "args", "kwargs", "environment", "world", "execution_timeout", "stream_output", "done_callbacks", "output"),
        execute.Executor.submit: ("self", "fn", "args", "kwargs", "environment", "world", "execution_timeout", "stream_output", "done_callbacks", "output"),
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
