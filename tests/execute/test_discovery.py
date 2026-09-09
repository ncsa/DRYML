"""Proof for bounded Execute environment discovery."""

from __future__ import annotations

import sys
from datetime import datetime
from time import monotonic
from pathlib import Path

from dryml.environments.specs import CondaEnvironmentSpec, CurrentEnvironmentSpec, PythonExecutableSpec
from dryml.execute.config import BackendConfig
from dryml.execute.discovery import CandidateInventory, discover_candidates, identity, probe_candidate
from dryml.execute.models import ExecutionIssue, ResourceAmounts, ResourceSnapshot
from dryml.execute.ray import RayBackendConfig
from dryml.execute.subprocess import SubProcessConfig


class Config(BackendConfig):
    """Inert configuration used to exercise discovery policy."""

    def create_backend(self):
        raise AssertionError


def test_candidate_order_and_automatic_discovery_switch(tmp_path: Path):
    """Built-ins precede explicit/root candidates and disable independently."""
    root = tmp_path / "roots"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "python").write_text("", encoding="utf-8")
    explicit = PythonExecutableSpec(executable="/explicit/python")
    conda = lambda: [CondaEnvironmentSpec(prefix="/conda", launch_mode="direct")]

    automatic = discover_candidates(
        Config(environment_candidates=(explicit,), environment_search_roots=(root,)),
        cwd=tmp_path,
        interpreter=Path(sys.executable),
        conda_inventory=conda,
    )
    assert isinstance(automatic.specs[0], CurrentEnvironmentSpec)
    assert automatic.specs[1] is explicit
    assert isinstance(automatic.specs[2], CondaEnvironmentSpec)
    assert automatic.specs[-1].executable == str(root / "nested" / "python")

    disabled = discover_candidates(
        Config(automatic_environment_discovery=False, environment_candidates=(explicit,), environment_search_roots=(root,)),
        cwd=tmp_path,
        interpreter=Path(sys.executable),
        conda_inventory=conda,
    )
    assert disabled.specs[0] is explicit
    assert disabled.specs[-1].executable == str(root / "nested" / "python")


def test_launch_identity_preserves_venv_and_conda_forms():
    """Identity is prefix plus launch form rather than interpreter realpath."""
    direct = CondaEnvironmentSpec(prefix="/same", launch_mode="direct")
    run = CondaEnvironmentSpec(prefix="/same", launch_mode="conda-run")
    one = PythonExecutableSpec(executable="/one/bin/python")
    two = PythonExecutableSpec(executable="/two/bin/python")
    assert len({identity(value) for value in (direct, run, one, two)}) == 4


def test_identity_includes_all_launch_semantics_without_path_delimiters():
    """Environment overrides remain distinct while snapshot keys stay opaque."""
    one = PythonExecutableSpec(executable="/same/python", env={"TOKEN": "one|two"})
    two = PythonExecutableSpec(executable="/same/python", env={"TOKEN": "one", "OTHER": "two"})
    assert identity(one) != identity(two)


def test_root_errors_and_limits_are_bounded(tmp_path: Path):
    """Inaccessible/missing roots and candidate caps become safe issues."""
    missing = tmp_path / "missing"
    result = discover_candidates(
        Config(environment_search_roots=(missing,), discovery_candidate_limit=1),
        cwd=tmp_path,
        interpreter=Path(sys.executable),
        conda_inventory=lambda: (),
    )
    assert any(issue.code == "discovery_root_unavailable" for issue in result.issues)
    assert len(result.specs) == 1


def test_duplicate_inventory_and_directory_entries_cannot_exceed_examined_bound(tmp_path: Path):
    """Malformed or repeated discovery sources stop at the examination bound."""
    root = tmp_path / "root"
    root.mkdir()
    for index in range(10):
        child = root / f"entry-{index}"
        child.mkdir()
        (child / "python").write_text("", encoding="utf-8")
    result = discover_candidates(
        Config(environment_search_roots=(root,), discovery_candidate_limit=3, discovery_directory_entry_limit=3),
        cwd=tmp_path,
        interpreter=Path(sys.executable),
        conda_inventory=lambda: (CondaEnvironmentSpec(prefix="/same", launch_mode="direct") for _ in range(100)),
    )
    assert len(result.specs) <= 3
    assert not result.complete
    assert any(issue.code == "discovery_candidate_limit" for issue in result.issues)


def test_malformed_probe_evidence_is_not_launchable_or_compatible(tmp_path: Path):
    """A fake executable cannot turn malformed owner evidence into a candidate record."""
    executable = tmp_path / "fake-python"
    executable.write_text("#!/bin/sh\nprintf not-json\n", encoding="utf-8")
    executable.chmod(0o755)
    candidate = probe_candidate(
        PythonExecutableSpec(executable=str(executable)), interpreter=Path(sys.executable), timeout=1, output_limit=1024,
    )
    assert candidate.record is None
    assert candidate.launchable is None
    assert candidate.issues[0].code == "probe_malformed"


def test_probe_applies_spec_environment_and_rejects_truncated_owner_evidence(tmp_path: Path):
    """Probe launch uses owner selector semantics and truncation cannot be success."""
    executable = tmp_path / "fake-python"
    executable.write_text("#!/bin/sh\nprintf '%s!' \"$PROBE_SENTINEL\"\n", encoding="utf-8")
    executable.chmod(0o755)
    candidate = probe_candidate(
        PythonExecutableSpec(executable=str(executable), env={"PROBE_SENTINEL": "x"}),
        interpreter=Path(sys.executable),
        timeout=1,
        output_limit=1,
        deadline=monotonic() + 1,
    )
    assert candidate.launchable is None
    assert candidate.issues[0].code == "probe_output_incomplete"


def _complete_resources() -> ResourceSnapshot:
    """Provide an inert complete resource observation for backend discovery tests."""
    amounts = ResourceAmounts(1.0, 1, {}, {})
    return ResourceSnapshot(datetime.now(), "coordinator-and-backend", None, amounts, ResourceAmounts(0.0, 0, {}, {}), amounts, (), True, ())


def test_subprocess_discovery_propagates_bounded_inventory_issues(tmp_path: Path, monkeypatch):
    """Public local discovery never reports a capped candidate inventory as complete."""
    from dryml.execute import subprocess as subprocess_module

    backend = SubProcessConfig(spool_directory=tmp_path, python_executable=Path(sys.executable), environment_candidates=(CurrentEnvironmentSpec(),)).create_backend()
    monkeypatch.setattr(backend, "_resource_inventory", lambda _deadline: _complete_resources())
    monkeypatch.setattr(
        subprocess_module,
        "discover_candidates",
        lambda *_args, **_kwargs: CandidateInventory((), (ExecutionIssue("discovery_candidate_limit", "candidate cap"), ExecutionIssue("discovery_directory_limit", "directory cap")), False),
    )
    monkeypatch.setattr(subprocess_module, "probe_candidate", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("empty inventory must not probe")))

    snapshot = backend.discover(timeout=1)

    assert not snapshot.complete
    assert {issue.code for issue in snapshot.issues} == {"discovery_candidate_limit", "discovery_directory_limit"}


def test_ray_discovery_propagates_bounded_inventory_issues(tmp_path: Path, monkeypatch):
    """Public Ray discovery carries candidate enumeration limits without reserving."""
    from dryml.execute import ray as ray_module

    backend = RayBackendConfig(address="127.0.0.1:6379", spool_directory=tmp_path, environment_candidates=(CurrentEnvironmentSpec(),)).create_backend()
    monkeypatch.setattr(backend, "_resources_until", lambda _deadline: _complete_resources())
    monkeypatch.setattr(
        ray_module,
        "discover_candidates",
        lambda *_args, **_kwargs: CandidateInventory((), (ExecutionIssue("discovery_candidate_limit", "candidate cap"), ExecutionIssue("discovery_directory_limit", "directory cap")), False),
    )
    monkeypatch.setattr(ray_module, "probe_candidate", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("empty inventory must not probe")))

    snapshot = backend.discover(timeout=1)

    assert not snapshot.complete
    assert {issue.code for issue in snapshot.issues} == {"discovery_candidate_limit", "discovery_directory_limit"}


def test_resource_only_discovery_does_not_scan_environments(tmp_path: Path, monkeypatch):
    """Resource-only discovery returns resource facts without candidate enumeration."""
    from dryml.execute import subprocess as subprocess_module

    backend = SubProcessConfig(spool_directory=tmp_path, python_executable=Path(sys.executable), automatic_environment_discovery=False).create_backend()
    monkeypatch.setattr(backend, "_resource_inventory", lambda _deadline: _complete_resources())
    monkeypatch.setattr(subprocess_module, "discover_candidates", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resource-only discovery must not scan environments")))

    snapshot = backend.discover(timeout=1)

    assert snapshot.complete
    assert snapshot.environments == ()
    assert snapshot.resources.total.cpus == 1.0


def test_ray_resource_only_discovery_does_not_scan_environments(tmp_path: Path, monkeypatch):
    """Resource-only Ray discovery does not enumerate candidate environments."""
    from dryml.execute import ray as ray_module

    backend = RayBackendConfig(address="127.0.0.1:6379", spool_directory=tmp_path, automatic_environment_discovery=False).create_backend()
    monkeypatch.setattr(backend, "_resources_until", lambda _deadline: _complete_resources())
    monkeypatch.setattr(ray_module, "discover_candidates", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resource-only discovery must not scan environments")))

    snapshot = backend.discover(timeout=1)

    assert snapshot.complete
    assert snapshot.environments == ()
    assert snapshot.resources.total.cpus == 1.0
