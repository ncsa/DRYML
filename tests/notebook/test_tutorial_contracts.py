from __future__ import annotations

import copy
import json
import re
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


def _notebook_sources(document):
    cells = [
        (
            cell["cell_type"],
            cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"]),
        )
        for cell in document["cells"]
    ]
    return (
        "\n".join(source for _, source in cells),
        "\n".join(source for cell_type, source in cells if cell_type == "code"),
    )


def test_canonical_notebook_registry_has_final_tutorial_order():
    assert [item.path.as_posix() for item in CANONICAL_NOTEBOOKS] == [
        "examples/notebooks/objects_definitions_and_repos.ipynb",
        "examples/notebooks/datasets_and_transforms.ipynb",
        "examples/notebooks/local_defaults_and_plain_mode.ipynb",
        "examples/notebooks/models_experiments_and_metrics.ipynb",
        "examples/notebooks/definition_driven_experiments.ipynb",
        "examples/notebooks/local_hyperparameter_search.ipynb",
    ]
    assert [item.extras for item in CANONICAL_NOTEBOOKS] == [
        (),
        (),
        (),
        ("sklearn",),
        ("sklearn",),
        ("sklearn",),
    ]
    assert [item.fake_frameworks for item in CANONICAL_NOTEBOOKS] == [
        (),
        (),
        ("tensorflow",),
        (),
        (),
        (),
    ]
    assert [item.python_max_exclusive for item in CANONICAL_NOTEBOOKS] == [
        None,
        None,
        None,
        (3, 14),
        (3, 14),
        (3, 14),
    ]
    assert [item.allowed_optional_imports for item in CANONICAL_NOTEBOOKS] == [
        frozenset(),
        frozenset(),
        frozenset({"tensorflow"}),
        frozenset({"sklearn"}),
        frozenset({"sklearn"}),
        frozenset({"sklearn"}),
    ]


def test_declared_extras_determine_allowed_optional_imports():
    item = NotebookSpec(Path("model.ipynb"), extras=("sklearn",))

    assert item.allowed_optional_imports == frozenset({"sklearn"})

    with pytest.raises(ValueError, match="unknown notebook extras: undeclared"):
        NotebookSpec(Path("bad.ipynb"), extras=("undeclared",))
    with pytest.raises(ValueError, match="python_max_exclusive must be a major/minor tuple"):
        NotebookSpec(Path("bad.ipynb"), python_max_exclusive=(3,))


@pytest.mark.parametrize("item", CANONICAL_NOTEBOOKS, ids=lambda item: item.path.stem)
def test_canonical_notebooks_satisfy_static_contract(item):
    document = validate_notebook(repository_path(item.path))
    all_source, _ = _notebook_sources(document)
    if item.extras == ("sklearn",):
        assert "Python 3.10-3.13" in all_source


def test_runtime_notebook_teaches_session_first_execution_distinctions():
    item = next(
        item
        for item in CANONICAL_NOTEBOOKS
        if item.path.name == "local_defaults_and_plain_mode.ipynb"
    )
    document = validate_notebook(repository_path(item.path))
    all_source, executable_source = _notebook_sources(document)

    required_executable = {
        "AnnotationResolutionError",
        "DirStore",
        "PythonExecutableSpec",
        "TemporaryDirectory",
        "dryml.session.configure",
        "dryml.session.current",
        "dryml.session.manage",
        "dryml.session.worker_env_request",
        "dryml.session.worker_world_request",
        "dryml.session.reset",
        "NOTEBOOK_RESTART_REQUIRED_HANDLED",
        "importlib.import_module('tensorflow')",
        "managed.statuses",
        "diagnostic_count",
        "dryml.dispatch.run",
        "operator.add",
        "result.result_canonical",
        "result.status",
    }
    missing = {token for token in required_executable if token not in executable_source}
    assert not missing, f"runtime notebook is missing executable teaching elements: {sorted(missing)}"
    assert "ordinary unchecked Python" in all_source
    assert "RuntimeMode.NONE" in all_source
    assert "requirement axes" in all_source
    assert "strict orchestration" in all_source
    assert "Orchestration mode prohibits Object materialization" in all_source
    assert "CPU-only" in all_source
    assert "fresh process" in all_source
    assert not {
        token
        for token in (
            "dryml.context",
            "dryml.execute",
            "ExitStack",
            "CurrentEnvironmentSpec",
            "WorldSpec",
            "dryml.environments.use",
            "dryml.worlds.use",
            "requirement_policy='ignore'",
            'requirement_policy="ignore"',
            "runtime.disabled",
        )
        if token in executable_source
    }


def test_session_docs_state_the_shipped_default_and_interception_boundaries():
    repository = Path(__file__).resolve().parents[2]
    session = (repository / "docs/session.md").read_text(encoding="utf-8")
    migration = (repository / "docs/migration/session_runtime_default.md").read_text(encoding="utf-8")

    for token in (
        "ordinary unchecked Python",
        "set_mode(\"python\")",
        "manage()",
        "set_mode(\"orchestrator\")",
        "configure(...)",
        "current-process allowance",
        "requested worker world",
        "generation lease",
        "restart",
        "Mandatory visibility",
        "class-object/custom-metaclass",
        "post-decoration assignment",
        "pre-decoration references",
        "private scoped bypass",
    ):
        assert token in session
    assert "managed session" in session
    assert "managed operation" in session
    assert "strict orchestration" in migration


def test_control_plane_docs_state_the_complete_runtime_boundary():
    repository = Path(__file__).resolve().parents[2]
    documents = {
        path: (repository / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs/session.md",
            "docs/environments.md",
            "docs/world_runtime.md",
            "docs/objects_and_defs.md",
            "docs/repos.md",
            "docs/dispatch.md",
            "docs/migration/session_runtime_default.md",
            "docs/architecture/environment_world_runtime_boundaries.md",
            "docs/architecture/runtime_dispatch_requirements.md",
            "docs/release_notes.md",
        )
    }

    required = {
        "README.md": ("RuntimeMode.NONE", "never auto-dispatch", "strict/all"),
        "docs/session.md": ("worker_env_request", "requirement axes", "object_mode=\"definition\""),
        "docs/environments.md": ("worker_env_request", "hard compatibility requirement"),
        "docs/world_runtime.md": ("RuntimeMode.NONE", "worker session"),
        "docs/objects_and_defs.md": ("Orchestration mode prohibits Object materialization", "definition-only"),
        "docs/repos.md": ("strict orchestration", "load_or_build"),
        "docs/dispatch.md": ("requirement_axes", "runtime.worker_session.v2", "execution-envelope v2"),
        "docs/migration/session_runtime_default.md": ("Serialized `none`", "V1 execution envelopes", "breaking behavior"),
        "docs/architecture/environment_world_runtime_boundaries.md": ("RuntimeMode.NONE", "compatibility only", "trusted-code lifecycle boundary"),
        "docs/architecture/runtime_dispatch_requirements.md": ("all three axes", "before handshake"),
        "docs/release_notes.md": ("serialized `none`", "V1 execution envelopes", "Strict orchestrator"),
    }
    for path, tokens in required.items():
        missing = [token for token in tokens if token not in documents[path]]
        assert not missing, f"{path} is missing runtime-boundary documentation: {missing}"


def test_definition_variants_notebook_teaches_identity_materialization_and_structural_query():
    item = next(
        item
        for item in CANONICAL_NOTEBOOKS
        if item.path.name == "definition_driven_experiments.ipynb"
    )
    document = validate_notebook(repository_path(item.path))
    all_source, executable_source = _notebook_sources(document)

    required_executable = {
        "Definition(ArrayDataset",
        "RandomForestRegressor",
        ".concretize()",
        ".stable_hash()",
        ".stored().defs()",
        "SKIP_ARGS",
        "TemporaryDirectory",
        "cache='none'",
        "instance='new'",
        "mean_squared_error",
        "random_state",
        "with_arg",
        "with_kwarg",
    }
    missing = {token for token in required_executable if token not in executable_source}
    assert not missing, f"definition variants notebook is missing teaching elements: {sorted(missing)}"
    assert "generated IDs" in all_source
    assert "fresh instances are a materialization choice" in all_source.lower()


def test_local_search_notebook_teaches_bounded_deterministic_public_api_workflow():
    item = next(
        item
        for item in CANONICAL_NOTEBOOKS
        if item.path.name == "local_hyperparameter_search.ipynb"
    )
    document = validate_notebook(repository_path(item.path))
    all_source, executable_source = _notebook_sources(document)

    required_executable = {
        ".as_space()",
        ".grid()",
        ".sample(random.Random(FIXED_SEED))",
        ".support_selector()",
        "CachedDataset",
        "Definition(ArrayDataset",
        "DirStore",
        "Repo",
        "experiment.train(store=store)",
        "experiment.trained_model(store=store)",
        "mean_squared_error",
        "np.isfinite",
        "islice(space.grid(), cap + 1)",
        "save_definition",
        "stable_cdef_key",
    }
    missing = {token for token in required_executable if token not in executable_source}
    assert not missing, f"local search notebook is missing teaching elements: {sorted(missing)}"
    assert "cap bounds candidate execution and publication" in all_source.lower()
    assert "arbitrary-range preflight" in all_source.lower()
    assert re.search(r"\bray\b", all_source, flags=re.IGNORECASE) is None
    assert not {
        token
        for token in ("checkpoint", "sys.path", "PYTHONPATH", "tutorials/")
        if token.lower() in all_source.lower()
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
