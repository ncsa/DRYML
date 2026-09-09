"""Proof for owner-checked Execute admission."""

from __future__ import annotations

from threading import Event
from time import monotonic

from dryml.environments import EnvironmentRequirement, inspect_current
from dryml.execute.admission import admit, plan_admission
from dryml.worlds import CountConstraint, ResourceRequirement, RoleRequirement, WorldRequirement, local_inventory


def test_optional_axes_do_not_invent_reports_or_world_defaults():
    """Omitted requirement domains bypass their owner checks and barriers."""
    result = admit(record=inspect_current(), inventory=local_inventory(), deadline=monotonic() + 1)
    assert result.ok
    assert result.report.environment is None
    assert result.report.world is None
    assert result.world is None


def test_environment_is_rechecked_from_actual_worker_evidence():
    """Admission derives a fresh owner report rather than trusting transported flags."""
    result = admit(
        environment=EnvironmentRequirement(requirements=("certainly-not-installed>=1",)),
        record=inspect_current(),
        deadline=monotonic() + 1,
    )
    assert not result.ok
    assert result.report.environment is not None
    assert any(issue.code == "environment_incompatible" for issue in result.report.issues)


def test_planning_allows_one_admitting_range_but_never_synthesizes_a_grant():
    """Inventory synthesis is feasibility evidence, not actual-worker admission."""
    one = WorldRequirement({"main": RoleRequirement(replicas=CountConstraint(1, 2))})
    planned = plan_admission(world=one, inventory=local_inventory(), deadline=monotonic() + 1)
    assert planned.feasible
    assert not planned.go
    assert not admit(world=one, inventory=local_inventory(), deadline=monotonic() + 1).ok


def test_multiple_and_unenforceable_controls_reject_without_a_default_cpu_capability():
    """Final admission does not promote inventory facts into resource controls."""
    many = WorldRequirement({"main": RoleRequirement(replicas=CountConstraint(2, 3))})
    assert not admit(world=many, inventory=local_inventory(), deadline=monotonic() + 1).ok
    memory = WorldRequirement({"main": RoleRequirement(resources=ResourceRequirement(memory=CountConstraint(1, 1)))})
    assert not admit(world=memory, inventory=local_inventory(), deadline=monotonic() + 1).ok
    cpu = WorldRequirement({"main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1)))})
    assert not admit(world=cpu, inventory=local_inventory(), deadline=monotonic() + 1).ok


def test_cancelled_or_expired_attempt_cannot_emit_go():
    """Cancellation and deadline races close the admission authorization gate."""
    cancelled = Event()
    cancelled.set()
    assert not admit(record=inspect_current(), deadline=monotonic() + 1, cancelled=cancelled).go
    assert not admit(record=inspect_current(), deadline=monotonic() - 1).go
    assert not admit(record=inspect_current(), deadline=float("nan")).go
