"""Check Generic Execute documentation against public source and CI contracts."""

from __future__ import annotations

from dataclasses import fields
import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml

from dryml.execute.ray import RayBackendConfig
from dryml.execute.backend import Backend
from dryml.execute.executor import Executor, ExecutorView
from dryml.execute.future import ExecutionFuture
from dryml.execute.ray import RayBackend, RayFuture
from dryml.execute.subprocess import SubProcessBackend, SubProcessConfig, SubProcessFuture
from dryml.core.execute import CoreExecutionFuture, CoreOptions
from dryml.core.execute import Executor as CoreExecutor
from dryml.core.execute import ExecutorView as CoreExecutorView


ROOT = Path(__file__).resolve().parents[2]


def _guide(name: str) -> str:
    """Return one repository documentation file as UTF-8 text."""

    return (ROOT / "docs" / name).read_text(encoding="utf-8")


_COMMON_DEFAULTS: dict[str, tuple[Any, str]] = {
    "admission_timeout": (30.0, "30.0"),
    "discovery_timeout": (30.0, "30.0"),
    "termination_timeout": (5.0, "5.0"),
    "one_off_cleanup_attempts": (2, "2"),
    "one_off_cleanup_retry_interval": (0.1, "0.1"),
    "execution_timeout": (None, "None"),
    "output_final_timeout": (5.0, "5.0"),
    "spool_directory": (None, "None"),
    "spool_limit_bytes": (4_294_967_296, "4,294,967,296"),
    "spool_file_limit": (128, "128"),
    "preflight_limit": (8, "8"),
    "invocation_limit_bytes": (67_108_864, "67,108,864"),
    "result_limit_bytes": (67_108_864, "67,108,864"),
    "control_header_limit_bytes": (1_048_576, "1,048,576"),
    "owner_envelope_limit_bytes": (16_777_216, "16,777,216"),
    "admission_message_limit_bytes": (83_886_080, "83,886,080"),
    "output_frame_limit_bytes": (65_536, "65,536"),
    "output_limit_bytes": (1_048_576, "1,048,576"),
    "live_output_queue_limit_bytes": (262_144, "262,144"),
    "diagnostic_text_limit_bytes": (65_536, "65,536"),
    "diagnostic_issue_limit": (64, "64"),
    "discovery_candidate_limit": (128, "128"),
    "discovery_directory_entry_limit": (1_024, "1,024"),
    "process_read_chunk_bytes": (8_192, "8,192"),
    "process_poll_interval": (0.005, "0.005"),
    "environment_search_depth": (2, "2"),
    "stream_output": (False, "False"),
    "automatic_environment_discovery": (True, "True"),
    "conda_executable": ("conda", '"conda"'),
    "conda_launch_mode": ("direct", '"direct"'),
    "environment_candidates": ((), "()"),
    "environment_search_roots": ((), "()"),
    "working_directory": (None, "None"),
    "env_vars": ({}, "{}"),
}

_CONFIG_DEFAULTS = {
    SubProcessConfig: {**_COMMON_DEFAULTS, "python_executable": (None, "None")},
    RayBackendConfig: {
        **_COMMON_DEFAULTS,
        "address": ("auto", '"auto"'),
        "namespace": (None, "None"),
        "connect_timeout": (30.0, "30.0"),
    },
}


def _row_for_setting(guide: str, name: str) -> str:
    """Return the unique Markdown configuration row for one public setting."""

    rows = [line for line in guide.splitlines() if line.startswith(f"| `{name}` |")]
    assert len(rows) == 1, name
    return rows[0]


def _assert_documented_defaults(actual: dict[str, Any], expected: dict[str, tuple[Any, str]], guide: str) -> None:
    """Assert source defaults and their own rendered guide rows match the contract."""

    assert set(actual) == set(expected)
    for name, (value, rendered) in expected.items():
        assert actual[name] == value, name
        assert rendered in _row_for_setting(guide, name), name


def test_execute_guide_documents_actual_defaults_and_public_names() -> None:
    """Keep every common and specialized configuration default tied to its guide row."""

    guide = _guide("execute.md")
    for config_type, expected in _CONFIG_DEFAULTS.items():
        config = config_type()
        actual = {field.name: getattr(config, field.name) for field in fields(config)}
        _assert_documented_defaults(actual, expected, guide)
    for name in (
        "Executor", "ExecutorView", "ExecutionFuture", "ExecutionOutput",
        "SubProcessConfig", "RayBackendConfig", "SubProcessFuture", "RayFuture",
        "ExecutionSnapshot", "DiscoverySnapshot", "ResourceSnapshot",
        "ExecutionError", "AdmissionError", "CleanupError", "ExecutionUncertainError",
    ):
        assert name in guide
    assert "milliseconds" not in guide


def test_execute_default_contract_rejects_source_or_row_drift() -> None:
    """Prove a changed implementation default or a wrong setting row fails the guard."""

    guide = _guide("execute.md")
    actual = {field.name: getattr(SubProcessConfig(), field.name) for field in fields(SubProcessConfig)}
    changed_default = {**actual, "admission_timeout": 31.0}
    with pytest.raises(AssertionError):
        _assert_documented_defaults(changed_default, _CONFIG_DEFAULTS[SubProcessConfig], guide)
    wrong_row = guide.replace("| `admission_timeout` | `30.0`", "| `admission_timeout` | `31.0`", 1)
    with pytest.raises(AssertionError):
        _assert_documented_defaults(actual, _CONFIG_DEFAULTS[SubProcessConfig], wrong_row)


def test_execute_public_operation_docstrings_state_contract_sections() -> None:
    """Keep affected Execute API references explicit about values, failures, and I/O."""

    operations = (
        (Executor.start, ("Returns:", "Raises:", "Side Effects:")),
        (Executor.discover, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (Executor.resources, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (Executor.with_options, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (Executor.close, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (ExecutorView.submit, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (ExecutorView.run, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (ExecutionFuture.cancel, ("Returns:", "Side Effects:")),
        (ExecutionFuture.output.fget, ("Returns:", "Side Effects:")),
    )
    backend_operations = (
        "start", "capabilities", "create_future", "submit", "discover",
        "resources", "reconcile_cleanup", "close",
    )
    for operation, sections in operations:
        documentation = inspect.getdoc(operation) or ""
        for section in sections:
            assert section in documentation, operation
    for backend_type in (Backend, SubProcessBackend, RayBackend):
        for name in backend_operations:
            documentation = inspect.getdoc(getattr(backend_type, name)) or ""
            for section in ("Returns:", "Side Effects:"):
                assert section in documentation, f"{backend_type.__name__}.{name}"
    for property_ in (
        SubProcessFuture.process, SubProcessFuture.pid, RayFuture.object_ref,
        RayFuture.server_address, RayFuture.worker_pid, RayFuture.worker_id,
        RayFuture.node_id,
    ):
        documentation = inspect.getdoc(property_.fget) or ""
        assert "Returns:" in documentation
        assert "Side Effects:" in documentation
    core_operations = (
        (CoreExecutor.start, ("Returns:", "Raises:", "Side Effects:")),
        (CoreExecutor.submit, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (CoreExecutor.run, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (CoreExecutor.with_options, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (CoreExecutor.close, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (CoreExecutorView.submit, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (CoreExecutorView.run, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (CoreExecutionFuture.result, ("Args:", "Returns:", "Raises:", "Side Effects:")),
        (CoreExecutionFuture.cleanup, ("Raises:", "Side Effects:")),
    )
    for operation, sections in core_operations:
        documentation = inspect.getdoc(operation) or ""
        for section in sections:
            assert section in documentation, operation
    for value in (CoreOptions, CoreExecutionFuture, CoreExecutor, CoreExecutorView):
        documentation = inspect.getdoc(value) or ""
        for section in ("Args:", "Side Effects:"):
            assert section in documentation, value


def test_execute_docs_link_and_preserve_implemented_boundaries() -> None:
    """Reject documentation that overclaims provisioning, isolation, or release."""

    guide = " ".join(_guide("execute.md").split())
    toc = _guide("table_of_content.md")
    release_notes = _guide("release_notes.md")
    testing = _guide("testing.md")
    assert "[Generic Execute](execute.md)" in toc
    for phrase in (
        "trusted", "safe-deserialization boundary", "Spool budget",
        "generation", "caller-owned", "no persistent resource ledger",
        "cross-coordinator", "logical scheduler quantities", "physical isolation",
        "borrowed caller Ray driver", "best-effort SDK", "descendants that escape",
        "same-host", "same-host, single-alive-node", "Python patch versions",
        "no worker disk spool", "output_final_timeout", "one-off",
        "process_read_chunk_bytes", "process_poll_interval",
        "discovery_directory_entry_limit", "conda_launch_mode",
        "developer test-fixture setup", "not a runtime feature",
        "process-global across every active executor lease", "backend type",
        "spool parent", "only after the final lease reaches zero",
        "Per-executor payload limits and spool locations may still vary",
        "lambdas", "nested functions", "closures", "importable unbound builtins",
        "Stateful bound methods", "callable instances", "fixed generic transport error",
        "type-specific core phrases", "generic byte-oriented execution layer",
        "version 2", "SETUP_READY", "execution_timeout", "CoreOptions",
        "pinned index in the frozen Store table", "@function", "managed-operation",
        "AutoRef", "version-local worker/coordinator", "cross-version RPC",
        "maximal roots", "neither rolls back", "nor replays the workload",
        "max_calls=1", "exact `WorldAllocation` grant", "logical scheduler quantities",
    ):
        assert phrase in guide, phrase
    assert "dryml.core.execute` now supplies" in release_notes
    assert "not transactional" in release_notes
    assert "not native Windows Ray or GPU evidence" in testing


def test_ray_ci_job_enables_real_integration_and_uses_job_owned_fixture() -> None:
    """Require CI to run enabled Ray cases, not merely install a Ray dependency."""

    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yaml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["ray-integration"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["defaults"]["run"]["shell"] == "bash -el {0}"
    assert any(step.get("uses") == "conda-incubator/setup-miniconda@v3" for step in job["steps"])
    setup = next(step for step in job["steps"] if step.get("uses") == "conda-incubator/setup-miniconda@v3")
    assert setup["with"]["activate-environment"] == "dryml-ray"
    assert setup["with"]["auto-activate-base"] is False
    ray_requirements = (ROOT / "ray_test_requirements.txt").read_text(encoding="utf-8")
    assert "ray[default]==2.56.0" in ray_requirements
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    for command in (
        "ray_test_requirements.txt", "python -m build", "--force-reinstall dist/*.whl",
        "python -m venv", "ray start --head", "--num-cpus=4", "--num-gpus=0",
        "--disable-usage-stats", "DRYML_EXECUTE_INTEGRATION=1",
        "DRYML_TEST_RAY_ADDRESS=127.0.0.1:6379", "DRYML_TEST_CONDA_PREFIX",
        "DRYML_TEST_VENV_PYTHON", "tests/ray/test_execute_ray_backend.py",
        "tests/ray/test_core_execute_ray.py",
        "tests/core/test_execute_integration.py",
        "tests/ray/test_execute_ray_environments.py",
        "tests/execute/test_backend_conformance.py",
        "tests/execute/test_existing_environments.py",
    ):
        assert command in runs, command
    cleanup = job["steps"][-1]
    assert cleanup["if"] == "always()"
    assert "ray stop --force" in cleanup["run"]
