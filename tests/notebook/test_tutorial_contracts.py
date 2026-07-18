from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from notebook_helpers import (
    CANONICAL_NOTEBOOKS,
    NotebookSpec,
    NotebookValidationError,
    repository_path,
    validate_notebook,
)


def _notebook(*sources: str) -> dict[str, object]:
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
            for source in sources
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write_notebook(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_canonical_notebook_registry_orders_objects_data_runtime_and_models():
    assert [item.path.as_posix() for item in CANONICAL_NOTEBOOKS] == [
        "examples/notebooks/objects_definitions_and_repos.ipynb",
        "examples/notebooks/datasets_and_transforms.ipynb",
        "examples/notebooks/local_defaults_and_plain_mode.ipynb",
        "examples/notebooks/models_experiments_and_metrics.ipynb",
    ]
    assert [item.extras for item in CANONICAL_NOTEBOOKS] == [(), (), (), ("sklearn",)]
    assert [item.allowed_optional_imports for item in CANONICAL_NOTEBOOKS] == [
        frozenset(),
        frozenset(),
        frozenset(),
        frozenset({"sklearn"}),
    ]


def test_declared_extras_determine_allowed_optional_imports():
    item = NotebookSpec(Path("model.ipynb"), extras=("sklearn",))

    assert item.allowed_optional_imports == frozenset({"sklearn"})

    with pytest.raises(ValueError, match="unknown notebook extras: undeclared"):
        NotebookSpec(Path("bad.ipynb"), extras=("undeclared",))


@pytest.mark.parametrize("item", CANONICAL_NOTEBOOKS, ids=lambda item: item.path.stem)
def test_canonical_notebooks_satisfy_static_contract(item):
    validate_notebook(repository_path(item.path))


def test_runtime_notebook_teaches_current_execution_distinctions():
    item = next(
        item
        for item in CANONICAL_NOTEBOOKS
        if item.path.name == "local_defaults_and_plain_mode.ipynb"
    )
    document = validate_notebook(repository_path(item.path))
    all_source = "\n".join(
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in document["cells"]
    )
    executable_source = "\n".join(
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    )

    required_executable = {
        "DispatchPlanningError",
        "DirStore",
        "PythonExecutableSpec",
        "TemporaryDirectory",
        "active_runtime().allocation",
        "diagnostic_count",
        "dryml.dispatch.run",
        "operator.add",
        "result.result_canonical",
        "result.status",
    }
    missing = {token for token in required_executable if token not in executable_source}
    assert not missing, f"runtime notebook is missing executable teaching elements: {sorted(missing)}"
    assert "allow_pickle=True" in all_source
    assert "same-Python" in all_source
    assert not {
        token
        for token in (
            "dryml.context",
            "dryml.execute",
            "requirement_policy='ignore'",
            'requirement_policy="ignore"',
            "runtime.disabled",
        )
        if token in executable_source
    }


def test_invalid_json_reports_notebook_path(tmp_path):
    notebook = tmp_path / "invalid.ipynb"
    notebook.write_text("{not json", encoding="utf-8")

    with pytest.raises(NotebookValidationError, match=r"invalid\.ipynb: invalid JSON"):
        validate_notebook(notebook)


def test_wrong_nbformat_is_rejected(tmp_path):
    document = _notebook("value = 1")
    document["nbformat"] = 3
    notebook = _write_notebook(tmp_path / "old.ipynb", document)

    with pytest.raises(NotebookValidationError, match=r"old\.ipynb: nbformat must be 4"):
        validate_notebook(notebook)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.pop("cells"), "cells must be a list"),
        (lambda document: document.__setitem__("cells", {}), "cells must be a list"),
        (lambda document: document["cells"].__setitem__(0, "bad"), "cell 1.*must be an object"),
        (lambda document: document["cells"][0].pop("source"), "cell 1.*source is required"),
        (lambda document: document["cells"][0].__setitem__("source", ["ok\n", 7]), "cell 1.*source"),
        (lambda document: document["cells"][0].__setitem__("cell_type", "heading"), "cell 1.*cell_type"),
    ],
)
def test_malformed_cells_and_sources_are_rejected(tmp_path, mutation, message):
    document = _notebook("value = 1")
    mutation(document)
    notebook = _write_notebook(tmp_path / "malformed.ipynb", document)

    with pytest.raises(NotebookValidationError, match=message):
        validate_notebook(notebook)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document["cells"][0].__setitem__("execution_count", 1), "execution_count must be null"),
        (lambda document: document["cells"][0].__setitem__("outputs", [{"output_type": "stream"}]), "outputs must be empty"),
        (lambda document: document["cells"][0].__setitem__("metadata", {"execution": {"iopub.status.busy": "1"}}), "cell 1.*metadata"),
        (lambda document: document["metadata"].__setitem__("widgets", {}), "notebook metadata key 'widgets'"),
        (lambda document: document["metadata"]["kernelspec"].__setitem__("path", "/host/kernel"), "kernelspec metadata"),
        (lambda document: document["metadata"]["language_info"].__setitem__("version", "3" * 300), "metadata is too large"),
    ],
)
def test_outputs_and_transient_or_unbounded_metadata_are_rejected(tmp_path, mutation, message):
    document = _notebook("value = 1")
    mutation(document)
    notebook = _write_notebook(tmp_path / "metadata.ipynb", document)

    with pytest.raises(NotebookValidationError, match=message):
        validate_notebook(notebook)


def test_syntax_error_reports_notebook_and_cell(tmp_path):
    notebook = _write_notebook(tmp_path / "syntax.ipynb", _notebook("if True print('bad')"))

    with pytest.raises(NotebookValidationError, match=r"syntax\.ipynb: cell 1: invalid Python syntax"):
        validate_notebook(notebook)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("%matplotlib inline", "magic syntax is forbidden"),
        ("!pwd", "shell syntax is forbidden"),
        ("get_ipython().run_line_magic('time', 'value = 1')", "IPython execution is forbidden"),
        ("from pathlib import Path\nPath('/tmp/tutorial')", "absolute path is forbidden"),
        ("import socket\nsocket.socket()", "network module 'socket' is forbidden"),
        ("from urllib.request import urlretrieve\nurlretrieve('https://example.invalid', 'x')", "network module 'urllib.request' is forbidden"),
        ("from dryml.context import Context", "obsolete import 'dryml.context' is forbidden"),
        ("import dryml\ndryml.ObjectDef(int)", "obsolete API 'dryml.ObjectDef' is forbidden"),
        ("value = object()\nvalue.dry_id", "obsolete API 'dry_id' is forbidden"),
    ],
)
def test_executable_source_rejects_nonportable_or_obsolete_constructs(tmp_path, source, message):
    notebook = _write_notebook(tmp_path / "forbidden.ipynb", _notebook(source))

    with pytest.raises(NotebookValidationError, match=message):
        validate_notebook(notebook)


def test_markdown_prose_is_not_scanned_for_executable_source_rules(tmp_path):
    document = _notebook("value = 1")
    document["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "Historical prose: !shell, /absolute/path, dryml.ObjectDef, and https://example.invalid.",
        },
    )
    notebook = _write_notebook(tmp_path / "prose.ipynb", document)

    validated = validate_notebook(notebook)

    assert validated["cells"][0]["cell_type"] == "markdown"


def test_validation_does_not_mutate_document_fixture(tmp_path):
    document = _notebook("value = 1")
    expected = copy.deepcopy(document)
    notebook = _write_notebook(tmp_path / "unchanged.ipynb", document)

    validate_notebook(notebook)

    assert json.loads(notebook.read_text(encoding="utf-8")) == expected
