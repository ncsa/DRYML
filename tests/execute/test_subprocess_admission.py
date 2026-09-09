"""Focused local subprocess admission and pre-load rejection proof."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Barrier, Thread
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from dryml.environments import EnvironmentRequirement
from dryml.environments.specs import CurrentEnvironmentSpec
from dryml.execute import subprocess as subprocess_module
from dryml.execute.errors import AdmissionError
from dryml.execute.accounting import ResourceAuthority
from dryml.execute.executor import Executor
from dryml.execute.subprocess import SubProcessConfig
from dryml.worlds import CountConstraint, LocalResourceInventory, ResourceRequirement, RoleRequirement, WorldRequirement


def _require_cpu_affinity() -> None:
    """Skip only tests whose real worker contract requires native CPU affinity."""
    if not all(callable(getattr(os, name, None)) for name in ("sched_getaffinity", "sched_setaffinity")):
        pytest.skip("native CPU affinity is unavailable on this platform")


def _simulated_affinity_os(monkeypatch) -> SimpleNamespace:
    """Install CPU-affinity shims in an isolated subprocess-module OS proxy."""
    platform_os = SimpleNamespace(**vars(os))
    monkeypatch.setattr(platform_os, "sched_getaffinity", lambda _pid: {0}, raising=False)
    monkeypatch.setattr(platform_os, "sched_setaffinity", lambda _pid, _cpus: None, raising=False)
    monkeypatch.setattr(subprocess_module, "os", platform_os)
    return platform_os


def test_environment_is_checked_against_actual_worker_owner_evidence(tmp_path: Path):
    """A valid existing worker can pass an empty owner requirement before GO."""
    executor = Executor(SubProcessConfig(spool_directory=tmp_path, automatic_environment_discovery=False, environment_candidates=(CurrentEnvironmentSpec(),)))
    try:
        assert executor.run(lambda: 3, environment=EnvironmentRequirement()) == 3
    finally:
        executor.close(cancel=True, timeout=5)


def test_world_cpu_grant_applies_and_reports_exact_worker_affinity(tmp_path: Path):
    """A constrained worker observes only its authority-reserved CPU ID."""
    _require_cpu_affinity()
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
    _require_cpu_affinity()
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
    _require_cpu_affinity()
    cpu = min(os.sched_getaffinity(0))
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
    _simulated_affinity_os(monkeypatch)
    cpu = 0
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


def test_contended_local_admission_reprobes_only_after_authority_transition(tmp_path: Path, monkeypatch):
    """Busy waiters poll cancellation/deadlines without repeatedly spawning probes."""
    _simulated_affinity_os(monkeypatch)
    cpu = 0
    world = WorldRequirement({"main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1)))})
    inventory = LocalResourceInventory((cpu,))
    authority = ResourceAuthority()
    holder = authority.reserve_local("holder", world, inventory, generation="subprocess-v1", attempt="0")
    assert holder is not None
    backends = [SubProcessConfig(spool_directory=tmp_path / str(index), admission_timeout=1).create_backend() for index in range(2)]
    for backend in backends:
        backend._authority = authority
    barrier = Barrier(2)
    probe_count = 0

    def observe(_deadline):
        nonlocal probe_count
        probe_count += 1
        if probe_count <= 2:
            barrier.wait(timeout=1)
        return inventory

    for backend in backends:
        monkeypatch.setattr(backend, "_observe_inventory", observe)
    calls = [SimpleNamespace(submission_id=f"waiter-{index}", world=world, admission_deadline=monotonic() + 1) for index in range(2)]
    futures = [backend.create_future(call.submission_id, None) for backend, call in zip(backends, calls)]
    results: list[tuple[str, object]] = []
    errors: list[BaseException] = []

    def reserve(backend, call, future):
        try:
            results.append((call.submission_id, backend._reserve(call, future)))
        except BaseException as exc:
            errors.append(exc)

    threads = [Thread(target=reserve, args=(backend, call, future)) for backend, call, future in zip(backends, calls, futures)]
    for thread in threads:
        thread.start()
    deadline = monotonic() + 1
    while probe_count < 2 and monotonic() < deadline:
        sleep(0.005)
    assert probe_count == 2
    sleep(0.05)
    assert probe_count == 2

    revision = authority.revision()
    assert not authority.release("holder", generation="subprocess-v1", attempt="0", worker_id=None)
    assert authority.revision() == revision + 1
    assert not authority.release("holder", generation="subprocess-v1", attempt="0", worker_id=None)
    assert authority.revision() == revision + 1
    assert authority.release("holder", generation="subprocess-v1", attempt="0", worker_id=None, qualified_terminal=True)
    deadline = monotonic() + 1
    while not results and monotonic() < deadline:
        sleep(0.005)
    assert len(results) == 1 and results[0][1]
    assert authority.release(results[0][0], generation="subprocess-v1", attempt="0", worker_id=None, qualified_terminal=True)
    for thread in threads:
        thread.join(timeout=2)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert len(results) == 2 and all(reservation for _, reservation in results)
    assert probe_count >= 3
    for submission_id, _reservation in results[1:]:
        assert authority.release(submission_id, generation="subprocess-v1", attempt="0", worker_id=None, qualified_terminal=True)

    blocked = SubProcessConfig(spool_directory=tmp_path / "deadline", admission_timeout=0.05).create_backend()
    blocked._authority = authority
    deadline_probes = 0

    def deadline_observe(_deadline):
        nonlocal deadline_probes
        deadline_probes += 1
        return inventory

    monkeypatch.setattr(blocked, "_observe_inventory", deadline_observe)
    holder = authority.reserve_local("deadline-holder", world, inventory, generation="subprocess-v1", attempt="0")
    assert holder is not None
    with pytest.raises(AdmissionError, match="deadline"):
        blocked._reserve(
            SimpleNamespace(submission_id="deadline-waiter", world=world, admission_deadline=monotonic() + 0.05),
            blocked.create_future("deadline-waiter", None),
        )
    assert deadline_probes == 1
    assert authority.release("deadline-holder", generation="subprocess-v1", attempt="0", worker_id=None, qualified_terminal=True)
