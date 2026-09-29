"""Focused contracts for detached exact Execute environment selections."""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from dryml.environments import (
    CondaEnvironmentSpec,
    CurrentEnvironmentSpec,
    EnvironmentSpecError,
    PythonExecutableSpec,
)
from dryml.environments.selection import resolve_environment_spec
from dryml.environments.selection import compare_selection, software_digest


def _probe_payload(record) -> str:
    """Return one successful probe-worker response for a captured record."""

    return json.dumps({"ok": True, "record": record.to_data()})


def test_current_selection_captures_once_without_reinterpreting_selector(
        monkeypatch):
    """Current freezes the submitter interpreter rather than a later worker."""
    selection = resolve_environment_spec(CurrentEnvironmentSpec())
    monkeypatch.setattr(sys, "executable", "/later/python")

    assert selection.command == (str(Path(selection.expected_executable)),)
    assert selection.expected_executable != "/later/python"


def test_current_launch_path_uses_the_observed_snapshot_not_a_later_sys_value(
    monkeypatch, synthetic_environment_record,
):
    """Current identity and launch command use one coordinator observation."""
    observed = synthetic_environment_record

    def observe_then_mutate(_spec):
        """Model an ordinary interpreter-global change during resolution."""
        monkeypatch.setattr(sys, "executable", "/later/python")
        return observed

    monkeypatch.setattr(
        "dryml.environments.selection._probe_record", observe_then_mutate
    )
    selection = resolve_environment_spec(CurrentEnvironmentSpec())

    assert selection.command == (
        str(Path(observed.python.executable).absolute()),
    )


def test_explicit_python_keeps_symlink_launch_spelling_and_prefix_evidence(
    monkeypatch, synthetic_environment_record,
):
    """A venv symlink launches as supplied, not as its base Python."""
    observed = synthetic_environment_record
    monkeypatch.setattr(
        "dryml.environments.selection._probe_record",
        lambda _spec: observed,
    )

    selection = resolve_environment_spec(
        PythonExecutableSpec("/venv/bin/python")
    )

    expected = os.path.abspath("/venv/bin/python")
    assert selection.command == (expected,)
    assert selection.expected_executable == expected
    assert selection.expected_prefix == observed.python.prefix


def test_relative_python_path_becomes_absolute_without_symlink_resolution(
        monkeypatch, tmp_path, synthetic_environment_record):
    """Exact launch paths become absolute while retaining the symlink name."""
    observed = synthetic_environment_record
    monkeypatch.setattr(
        "dryml.environments.selection._probe_record", lambda _spec: observed
    )
    monkeypatch.chdir(tmp_path)

    selection = resolve_environment_spec(
        PythonExecutableSpec("venv/bin/python")
    )

    assert selection.command == (str(tmp_path / "venv/bin/python"),)


@pytest.mark.parametrize("inventory", [(), ("/envs/a", "/other/a")])
def test_conda_name_requires_exactly_one_existing_prefix(inventory):
    """A missing or ambiguous Conda name cannot choose another target."""
    with pytest.raises(EnvironmentSpecError, match="Conda name"):
        resolve_environment_spec(
            CondaEnvironmentSpec(name="a"),
            conda_inventory=lambda: inventory,
        )


def test_conda_name_freezes_one_prefix_and_conda_run_form(
    monkeypatch, synthetic_environment_record,
):
    """Name resolution retains the resolved prefix and requested launcher."""
    observed = synthetic_environment_record
    monkeypatch.setattr(
        "dryml.environments.selection._probe_record", lambda _spec: observed
    )

    selection = resolve_environment_spec(
        CondaEnvironmentSpec(
            name="selected",
            conda_executable="/tools/conda",
            launch_mode="conda-run",
        ),
        conda_inventory=lambda: ("/envs/selected",),
    )

    assert selection.resolved_prefix == "/envs/selected"
    assert selection.command[:5] == (
        os.path.abspath("/tools/conda"),
        "run",
        "-p",
        os.path.abspath("/envs/selected"),
        "--no-capture-output",
    )


def test_default_conda_inventory_uses_its_existing_bounded_tuple(
    monkeypatch, synthetic_environment_record,
):
    """Default Conda discovery does not re-copy its bounded inventory."""

    monkeypatch.setattr(
        "dryml.environments.selection._conda_prefixes",
        lambda _executable: ("/envs/selected",),
    )
    monkeypatch.setattr(
        "dryml.environments.selection._bounded_prefixes",
        lambda _prefixes: (_ for _ in ()).throw(
            AssertionError("default inventory is already bounded")
        ),
    )
    monkeypatch.setattr(
        "dryml.environments.selection._probe_record",
        lambda _spec: synthetic_environment_record,
    )

    selection = resolve_environment_spec(CondaEnvironmentSpec(name="selected"))

    assert selection.resolved_prefix == "/envs/selected"


def test_selection_comparison_rejects_identity_but_ignores_advisory_data(
    monkeypatch, synthetic_environment_record,
):
    """Exact identity remains separate from record labels and details."""
    monkeypatch.setattr(
        "dryml.environments.selection._probe_record",
        lambda _spec: synthetic_environment_record,
    )
    selection = resolve_environment_spec(CurrentEnvironmentSpec())
    advisory = replace(
        selection.record,
        tags=("observed-later",),
        details={"timestamp": "later"},
    )
    changed_prefix = replace(
        selection.record,
        python=replace(selection.record.python, prefix="/different/prefix"),
    )

    assert software_digest(advisory) == selection.software_digest
    assert compare_selection(selection, advisory).ok
    assert not compare_selection(selection, changed_prefix).ok


def test_exact_selection_probe_omits_ambient_environment_except_pythonpath(
    monkeypatch, synthetic_environment_record,
):
    """Exact probes pass only launch controls and approved ``PYTHONPATH``."""

    probe_module = importlib.import_module("dryml.environments.probe")
    observed = synthetic_environment_record
    captured = {}

    def fake_run(command, **kwargs):
        """Capture the exact child environment without starting a process."""

        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=_probe_payload(observed),
            stderr="",
        )

    monkeypatch.setenv("DRYML_AMBIENT_SECRET", "exact-probe-secret")
    monkeypatch.setenv("PYTHONPATH", "/coordinator/pythonpath")
    monkeypatch.setattr(probe_module.subprocess, "run", fake_run)

    selection = resolve_environment_spec(
        PythonExecutableSpec(
            "/private/selected/python",
            env={"SELECTOR_CONTROL": "kept"},
            pythonpath_policy="inherit",
        )
    )

    assert "DRYML_AMBIENT_SECRET" not in captured["env"]
    assert captured["env"]["PATH"] == os.defpath
    assert captured["env"]["SELECTOR_CONTROL"] == "kept"
    assert captured["env"]["PYTHONPATH"] == "/coordinator/pythonpath"
    assert selection.launch_env == {
        "SELECTOR_CONTROL": "kept",
        "PYTHONPATH": "/coordinator/pythonpath",
    }


def test_named_conda_inventory_omits_ambient_environment(
    monkeypatch, synthetic_environment_record,
):
    """Named Conda inventory starts from the same minimal environment."""

    selection_module = importlib.import_module("dryml.environments.selection")
    captured = {}

    class FakeProcess:
        """Supply one finite Conda inventory response without a child."""

        def __init__(self):
            self.stdout = io.BytesIO(b'{"envs": ["/envs/selected"]}')

        def wait(self, timeout):
            return 0

    def fake_popen(command, **kwargs):
        """Capture the inventory child environment."""

        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setenv("DRYML_AMBIENT_SECRET", "conda-inventory-secret")
    monkeypatch.setattr(
        selection_module,
        "_probe_record",
        lambda _spec: synthetic_environment_record,
    )
    monkeypatch.setattr(selection_module.subprocess, "Popen", fake_popen)

    selection = resolve_environment_spec(
        CondaEnvironmentSpec(
            name="selected",
            conda_executable="/tools/conda",
            env={"CONDA_CONTROL": "kept"},
        )
    )

    assert selection.resolved_prefix == "/envs/selected"
    assert "DRYML_AMBIENT_SECRET" not in captured["env"]
    assert captured["env"]["PATH"] == os.defpath
    assert captured["env"]["CONDA_CONTROL"] == "kept"


def test_resolved_selection_repr_omits_private_launch_values(
    monkeypatch, synthetic_environment_record,
):
    """Selection carriers remain usable without exposing nested secrets."""

    from dryml.execute.models import PayloadSpool, SubmittedCall

    monkeypatch.setattr(
        "dryml.environments.selection._probe_record",
        lambda _spec: synthetic_environment_record,
    )
    private_path = "/private/selection/python"
    secret = "resolved-selection-secret"
    selection = resolve_environment_spec(
        PythonExecutableSpec(private_path, env={"API_TOKEN": secret})
    )
    call = SubmittedCall(
        submission_id="submission",
        admission_deadline=1.0,
        payload=PayloadSpool(
            Path("/safe/payload"), 0, "0" * 64, "dill", "1", "CPython",
            (3, 11, 0), 5,
        ),
        environment=None,
        environment_spec=selection,
        world=None,
        execution_timeout=None,
        stream_output=False,
        output=object(),
    )

    assert selection.launch_env["API_TOKEN"] == secret
    for rendered in (repr(selection), repr(call), repr((selection,))):
        assert secret not in rendered
        assert private_path not in rendered
        assert selection.record.python.executable not in rendered
