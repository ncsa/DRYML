from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from notebook_helpers import (
    CANONICAL_NOTEBOOKS,
    NotebookExecutionError,
    _process_exists,
    execute_notebook,
    repository_path,
)

_DETERMINISM_NOTEBOOKS = {
    "definition_driven_experiments.ipynb",
    "local_hyperparameter_search.ipynb",
    "objects_definitions_and_repos.ipynb",
}
_SKLEARN_SPEC = next(item for item in CANONICAL_NOTEBOOKS if item.extras == ("sklearn",))


def _write_notebook(path: Path, source: str) -> Path:
    document = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _assert_clean_result(result, item):
    assert result.returncode == 0
    assert result.state_restored
    assert result.working_directory_restored
    assert result.module_table_restored
    assert result.linecache_restored
    assert result.optional_imports == item.allowed_optional_imports
    assert result.unexpected_writes == ()
    assert result.repository_on_pythonpath is False


def _require_supported_python(item):
    if not item.supports_current_python:
        pytest.skip(f"{item.path.name} requires Python below {item.python_max_exclusive}")


@pytest.mark.parametrize(
    "item",
    [item for item in CANONICAL_NOTEBOOKS if item.path.name not in _DETERMINISM_NOTEBOOKS],
    ids=lambda item: item.path.stem,
)
def test_canonical_notebook_executes_offline_and_cleans_process_state(tmp_path, item):
    _require_supported_python(item)
    result = execute_notebook(repository_path(item.path), item, tmp_path)

    _assert_clean_result(result, item)


def test_objects_notebook_executes_twice_in_independent_processes(tmp_path):
    item = CANONICAL_NOTEBOOKS[0]

    results = [
        execute_notebook(repository_path(item.path), item, tmp_path / f"run-{index}")
        for index in range(2)
    ]

    assert results[0] == results[1]
    for result in results:
        _assert_clean_result(result, item)
    assert all("Traceback" not in result.stdout + result.stderr for result in results)


def test_definition_variants_summary_is_stable_across_independent_processes(tmp_path):
    item = next(
        item
        for item in CANONICAL_NOTEBOOKS
        if item.path.name == "definition_driven_experiments.ipynb"
    )
    _require_supported_python(item)
    results = [
        execute_notebook(repository_path(item.path), item, tmp_path / f"run-{index}")
        for index in range(2)
    ]
    for result in results:
        _assert_clean_result(result, item)

    marker = "DEFINITION_VARIANT_SUMMARY="
    summaries = []
    for result in results:
        lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
        assert len(lines) == 1
        summaries.append(json.loads(lines[0][len(marker):]))

    assert summaries[0] == summaries[1]
    assert len(summaries[0]) == 2
    assert [item["identity"] for item in summaries[0]] == sorted(
        item["identity"] for item in summaries[0]
    )
    assert len({item["identity"] for item in summaries[0]}) == 2
    assert {item["random_state"] for item in summaries[0]} == {11, 29}
    assert all(math.isfinite(item["mse"]) for item in summaries[0])


def test_local_search_summary_is_stable_across_independent_processes(tmp_path):
    item = next(
        item
        for item in CANONICAL_NOTEBOOKS
        if item.path.name == "local_hyperparameter_search.ipynb"
    )
    _require_supported_python(item)
    results = [
        execute_notebook(repository_path(item.path), item, tmp_path / f"run-{index}")
        for index in range(2)
    ]
    for result in results:
        _assert_clean_result(result, item)

    marker = "LOCAL_SEARCH_SUMMARY="
    summaries = []
    for result in results:
        lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
        assert len(lines) == 1
        summaries.append(json.loads(lines[0][len(marker):]))

    assert summaries[0] == summaries[1]
    assert summaries[0]["candidate_count"] == 4
    assert summaries[0]["execution_order"] == sorted(summaries[0]["execution_order"])
    assert len(set(summaries[0]["execution_order"])) == 4
    assert summaries[0]["best_identity"] in summaries[0]["execution_order"]
    assert summaries[0]["sample_identity"] in summaries[0]["execution_order"]
    assert all(math.isfinite(metric) for metric in summaries[0]["metrics"])


def test_network_guard_blocks_socket_access_with_cell_diagnostic(tmp_path):
    notebook = _write_notebook(
        tmp_path / "network.ipynb",
        "import socket\nsocket.socket()",
    )

    with pytest.raises(NotebookExecutionError, match=r"network\.ipynb: cell 1: network access is disabled"):
        execute_notebook(notebook, work_root=tmp_path / "run", validate=False)


def test_network_guard_reaches_dispatch_workers(tmp_path):
    notebook = _write_notebook(
        tmp_path / "worker-network.ipynb",
        """import sys
from tempfile import TemporaryDirectory
from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.operations import make_function_call_spec
from dryml.worlds import LocalResourceInventory

with TemporaryDirectory() as directory:
    store = DirStore(directory, query_index='none')
    try:
        world = {'roles': {'main': {'replicas': 1, 'process': {'env': {'pythonpath': ''}}}}}
        result = Dispatcher(store=store).run(
            make_function_call_spec('socket:getaddrinfo', args=['localhost', 80]),
            environment=PythonExecutableSpec(executable=sys.executable, pythonpath_policy='none'),
            world=world,
            inventory=LocalResourceInventory((0,)),
            requirement_policy='ignore',
        )
        if result.status != 'ok':
            raise RuntimeError(result.error['message'])
    finally:
        store.close()
""",
    )

    with pytest.raises(
        NotebookExecutionError,
        match=r"worker-network\.ipynb: cell 1: network access is disabled",
    ):
        execute_notebook(notebook, work_root=tmp_path / "run")


def test_child_failure_reports_notebook_cell_and_exception(tmp_path):
    notebook = _write_notebook(tmp_path / "failure.ipynb", "raise RuntimeError('fixture failed')")

    with pytest.raises(
        NotebookExecutionError,
        match=r"failure\.ipynb: cell 1: RuntimeError: fixture failed.*child exited with status 1",
    ):
        execute_notebook(notebook, work_root=tmp_path / "run")


def test_timeout_is_hard_and_names_notebook(tmp_path):
    notebook = _write_notebook(tmp_path / "slow.ipynb", "import time\ntime.sleep(10)")

    with pytest.raises(NotebookExecutionError, match=r"slow\.ipynb: cell 1: execution timed out after"):
        execute_notebook(notebook, work_root=tmp_path / "run", timeout=0.2)


def test_timeout_reaps_independently_sessioned_dispatch_worker(tmp_path):
    notebook = _write_notebook(
        tmp_path / "worker-timeout.ipynb",
        """import sys
from tempfile import TemporaryDirectory
from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.operations import make_function_call_spec

with TemporaryDirectory() as directory:
    store = DirStore(directory, query_index='none')
    try:
        Dispatcher(store=store).run(
            make_function_call_spec('time:sleep', args=[60]),
            environment=PythonExecutableSpec(executable=sys.executable, pythonpath_policy='none'),
        )
    finally:
        store.close()
""",
    )
    run_root = tmp_path / "run"

    with pytest.raises(NotebookExecutionError, match=r"worker-timeout\.ipynb: cell 1: execution timed out"):
        execute_notebook(notebook, work_root=run_root, timeout=30.0)

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_root / ".harness" / "workers").glob("worker-*.json")
    ]
    assert records
    for record in records:
        assert not _process_exists(record["pid"])


def test_abrupt_child_exit_reports_active_cell(tmp_path):
    notebook = _write_notebook(tmp_path / "exit.ipynb", "import os\nos._exit(7)")

    with pytest.raises(NotebookExecutionError, match=r"exit\.ipynb: cell 1: child exited with status 7"):
        execute_notebook(notebook, work_root=tmp_path / "run")


def test_zero_exit_without_final_report_is_rejected(tmp_path):
    notebook = _write_notebook(tmp_path / "zero-exit.ipynb", "import os\nos._exit(0)")

    with pytest.raises(NotebookExecutionError, match=r"zero-exit\.ipynb: cell 1: child exited with status 0"):
        execute_notebook(notebook, work_root=tmp_path / "run")


def test_unexpected_write_is_rejected(tmp_path):
    notebook = _write_notebook(
        tmp_path / "write.ipynb",
        "from pathlib import Path\nPath('unexpected.txt').write_text('payload', encoding='utf-8')",
    )

    with pytest.raises(NotebookExecutionError, match=r"write\.ipynb: unexpected writes: work/unexpected\.txt"):
        execute_notebook(notebook, work_root=tmp_path / "run")


def test_undeclared_optional_import_is_rejected(tmp_path):
    notebook = _write_notebook(
        tmp_path / "optional.ipynb",
        "import sys\nimport types\nsys.modules['sklearn'] = types.ModuleType('sklearn')\nimport sklearn",
    )

    with pytest.raises(NotebookExecutionError, match=r"optional\.ipynb: undeclared optional imports: sklearn"):
        execute_notebook(notebook, work_root=tmp_path / "run")


@pytest.mark.skipif(not _SKLEARN_SPEC.supports_current_python, reason="DRYML's sklearn extra excludes this Python")
def test_undeclared_optional_import_in_dispatch_worker_is_rejected(tmp_path):
    notebook = _write_notebook(
        tmp_path / "worker-optional.ipynb",
        """import sys
from tempfile import TemporaryDirectory
from dryml.core2.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.operations import make_function_call_spec
from dryml.worlds import LocalResourceInventory

with TemporaryDirectory() as directory:
    store = DirStore(directory, query_index='none')
    try:
        world = {'roles': {'main': {'replicas': 1, 'process': {'env': {'pythonpath': ''}}}}}
        Dispatcher(store=store).run_world(
            make_function_call_spec('sklearn.linear_model:LinearRegression'),
            environment=PythonExecutableSpec(executable=sys.executable, pythonpath_policy='none'),
            world=world,
            inventory=LocalResourceInventory((0,)),
            timeout=10,
        )
    finally:
        store.close()
""",
    )

    with pytest.raises(
        NotebookExecutionError,
        match=r"worker-optional\.ipynb: undeclared optional imports: sklearn",
    ):
        execute_notebook(notebook, work_root=tmp_path / "run")


def test_child_output_is_bounded_before_returning_to_parent(tmp_path):
    notebook = _write_notebook(tmp_path / "output.ipynb", "print('x' * 1000000)")

    result = execute_notebook(notebook, work_root=tmp_path / "run")

    assert result.stdout == f"{'x' * 8000}\n... output truncated ..."
    assert result.stderr == ""


def test_child_runner_cleans_module_linecache_and_working_directory_after_handled_failure(tmp_path):
    notebook = _write_notebook(
        tmp_path / "cleanup.ipynb",
        """import os
try:
    os.chdir('..')
    raise RuntimeError('handled')
except RuntimeError:
    pass
""",
    )

    result = execute_notebook(notebook, work_root=tmp_path / "run")

    assert result.working_directory_restored
    assert result.module_table_restored
    assert result.linecache_restored


def test_child_cleanup_failure_names_leaked_state(tmp_path):
    notebook = _write_notebook(
        tmp_path / "state-leak.ipynb",
        """import dryml
from dryml.environments import CurrentEnvironmentSpec
dryml.environments.set_current(CurrentEnvironmentSpec())
""",
    )

    with pytest.raises(NotebookExecutionError, match=r"state-leak\.ipynb: child cleanup failed: state_restored"):
        execute_notebook(notebook, work_root=tmp_path / "run")


def test_unreleased_temporary_directory_is_audited(tmp_path):
    notebook = _write_notebook(
        tmp_path / "temporary-leak.ipynb",
        "import tempfile\ntempfile.mkdtemp(prefix='tutorial-leak-')",
    )

    with pytest.raises(NotebookExecutionError, match=r"temporary-leak\.ipynb: unexpected writes: tmp/"):
        execute_notebook(notebook, work_root=tmp_path / "run")
