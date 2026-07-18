from __future__ import annotations

import json
from pathlib import Path

import pytest

from notebook_helpers import (
    CANONICAL_NOTEBOOKS,
    NotebookExecutionError,
    execute_notebook,
    repository_path,
)


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


@pytest.mark.parametrize("item", CANONICAL_NOTEBOOKS, ids=lambda item: item.path.stem)
def test_canonical_notebook_executes_offline_and_cleans_process_state(tmp_path, item):
    result = execute_notebook(repository_path(item.path), item, tmp_path)

    assert result.returncode == 0
    assert result.state_restored
    assert result.working_directory_restored
    assert result.module_table_restored
    assert result.linecache_restored
    assert result.optional_imports == frozenset()
    assert result.unexpected_writes == ()
    assert result.repository_on_pythonpath is False


def test_network_guard_blocks_socket_access_with_cell_diagnostic(tmp_path):
    notebook = _write_notebook(
        tmp_path / "network.ipynb",
        "import socket\nsocket.socket()",
    )

    with pytest.raises(NotebookExecutionError, match=r"network\.ipynb: cell 1: network access is disabled"):
        execute_notebook(notebook, work_root=tmp_path / "run", validate=False)


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


def test_abrupt_child_exit_reports_active_cell(tmp_path):
    notebook = _write_notebook(tmp_path / "exit.ipynb", "import os\nos._exit(7)")

    with pytest.raises(NotebookExecutionError, match=r"exit\.ipynb: cell 1: child exited with status 7"):
        execute_notebook(notebook, work_root=tmp_path / "run")


def test_unexpected_write_is_rejected(tmp_path):
    notebook = _write_notebook(
        tmp_path / "write.ipynb",
        "from pathlib import Path\nPath('unexpected.txt').write_text('payload', encoding='utf-8')",
    )

    with pytest.raises(NotebookExecutionError, match=r"write\.ipynb: unexpected writes: work/unexpected\.txt"):
        execute_notebook(notebook, work_root=tmp_path / "run")


def test_undeclared_optional_import_is_rejected(tmp_path):
    notebook = _write_notebook(tmp_path / "optional.ipynb", "import sklearn")

    with pytest.raises(NotebookExecutionError, match=r"optional\.ipynb: undeclared optional imports: sklearn"):
        execute_notebook(notebook, work_root=tmp_path / "run")


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
