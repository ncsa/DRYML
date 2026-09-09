"""Focused local subprocess admission and pre-load rejection proof."""

from __future__ import annotations

from pathlib import Path

import pytest

from dryml.environments import EnvironmentRequirement
from dryml.environments.specs import CurrentEnvironmentSpec
from dryml.execute.errors import AdmissionError
from dryml.execute.executor import Executor
from dryml.execute.subprocess import SubProcessConfig
from dryml.worlds import CountConstraint, LocalResourceInventory, ResourceRequirement, RoleRequirement, WorldRequirement


def test_environment_is_checked_against_actual_worker_owner_evidence(tmp_path: Path):
    """A valid existing worker can pass an empty owner requirement before GO."""
    executor = Executor(SubProcessConfig(spool_directory=tmp_path, automatic_environment_discovery=False, environment_candidates=(CurrentEnvironmentSpec(),)))
    try:
        assert executor.run(lambda: 3, environment=EnvironmentRequirement()) == 3
    finally:
        executor.close(cancel=True, timeout=5)


def test_world_cpu_grant_applies_and_reports_exact_worker_affinity(tmp_path: Path):
    """A constrained worker observes only its authority-reserved CPU ID."""
    world = WorldRequirement({"main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1)))})
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    try:
        future = executor.submit(lambda: sorted(__import__("os").sched_getaffinity(0)), world=world)
        observed = future.result(timeout=10)
        allocation = future.snapshot().allocation
        assert allocation is not None
        assert observed == list(allocation.roles["main"][0].cpus)
        assert len(observed) == 1
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_combined_admission_checks_both_axes_and_rejects_unsupported_controls_preload(tmp_path: Path):
    """Environment/world evidence is joint, while memory never fakes enforcement."""
    sentinel = tmp_path / "combined-loaded"
    world = WorldRequirement({"main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1)))})
    executor = Executor(SubProcessConfig(spool_directory=tmp_path, automatic_environment_discovery=False, environment_candidates=(CurrentEnvironmentSpec(),)))
    try:
        admitted = executor.submit(lambda: len(__import__("os").sched_getaffinity(0)), environment=EnvironmentRequirement(), world=world)
        assert admitted.result(timeout=10) == 1
        admitted.cleanup(timeout=5)
        unsupported = WorldRequirement({"main": RoleRequirement(resources=ResourceRequirement(memory=CountConstraint(1, 1)))})
        future = executor.submit(lambda: sentinel.write_text("loaded", encoding="utf-8"), environment=EnvironmentRequirement(), world=unsupported)
        with pytest.raises(AdmissionError):
            future.result(timeout=10)
        assert not sentinel.exists()
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_shared_local_authority_keeps_constrained_executors_disjoint_and_releases_waiters(tmp_path: Path, monkeypatch):
    """Exact IDs cannot overlap across executors, and cleanup wakes a blocked grant."""
    cpu = min(__import__("os").sched_getaffinity(0))
    world = WorldRequirement({"main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1)))})
    first = Executor(SubProcessConfig(spool_directory=tmp_path, admission_timeout=2))
    second = Executor(SubProcessConfig(spool_directory=tmp_path, admission_timeout=0.2))
    reopened = Executor(SubProcessConfig(spool_directory=tmp_path, admission_timeout=2))
    first.start()
    second.start()
    reopened.start()
    # A one-CPU injected semantic observation makes the busy/release proof
    # deterministic without pretending that host-wide capacity is one CPU.
    monkeypatch.setattr(first._backend, "_observe_inventory", lambda deadline: LocalResourceInventory((cpu,)))
    monkeypatch.setattr(second._backend, "_observe_inventory", lambda deadline: LocalResourceInventory((cpu,)))
    monkeypatch.setattr(reopened._backend, "_observe_inventory", lambda deadline: LocalResourceInventory((cpu,)))
    try:
        holder = first.submit(lambda: sorted(__import__("os").sched_getaffinity(0)), world=world)
        assert holder.result(timeout=10) == [cpu]
        blocked = second.submit(lambda: (_ for _ in ()).throw(AssertionError("must not run")), world=world)
        with pytest.raises(AdmissionError):
            blocked.result(timeout=10)
        blocked.cleanup(timeout=5)
        holder.cleanup(timeout=5)
        waiter = reopened.submit(lambda: sorted(__import__("os").sched_getaffinity(0)), world=world)
        assert waiter.result(timeout=10) == [cpu]
        waiter.cleanup(timeout=5)
    finally:
        first.close(cancel=True, timeout=5)
        second.close(cancel=True, timeout=5)
        reopened.close(cancel=True, timeout=5)


def test_injected_gpu_allocation_applies_exact_cuda_visibility(tmp_path: Path, monkeypatch):
    """The accelerator branch verifies a real worker control without hardware claims."""
    cpu = min(__import__("os").sched_getaffinity(0))
    world = WorldRequirement({"main": RoleRequirement(resources=ResourceRequirement(accelerators={"gpu": CountConstraint(1, 1)}))})
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    executor.start()
    monkeypatch.setattr(executor._backend, "_observe_inventory", lambda deadline: LocalResourceInventory((cpu,), {"gpu": (0,)}))
    try:
        future = executor.submit(lambda: __import__("os").environ.get("CUDA_VISIBLE_DEVICES"), world=world)
        assert future.result(timeout=10) == "0"
        assert future.snapshot().allocation is not None
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)
