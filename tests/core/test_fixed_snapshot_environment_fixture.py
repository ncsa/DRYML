"""Regression coverage for fixed default snapshot evidence in persistence tests."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

import dryml.environments as envs
import dryml.core.snapshot_capture as snapshot_capture
from dryml.managed import runtime as managed_runtime


@envs.req(requirements=("fixture-dependency>=1",), source="fixture-regression")
class FixtureRequirement:
    """Materializing class proving the fixture retains requirement collection."""


def test_fixed_snapshot_environment_preserves_capture_inputs(
        fixed_snapshot_environment):
    """Default evidence is fixed while observers, failures, clocks, and requirements remain real."""

    default_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
    default = snapshot_capture._capture_snapshot_evidence(
        {}, (FixtureRequirement,), clock=lambda: default_time,
    )

    assert default.environment == fixed_snapshot_environment
    assert default.saved_at == default_time
    assert default.requirements.requirements == ("fixture-dependency>=1",)
    assert (default.requirements_status, default.requirements_coverage) == (
        "value", "complete",
    )

    observed = []
    explicit_environment = replace(fixed_snapshot_environment, kind="explicit")
    explicit_time = datetime(2024, 2, 3, tzinfo=timezone.utc)
    explicit = snapshot_capture._capture_snapshot_evidence(
        {},
        (FixtureRequirement,),
        observer=lambda: observed.append(True) or explicit_environment,
        clock=lambda: explicit_time,
    )

    assert observed == [True]
    assert explicit.lineages == {}
    assert explicit.environment == explicit_environment
    assert explicit.saved_at == explicit_time
    assert explicit.requirements.requirements == ("fixture-dependency>=1",)

    failure_time = datetime(2024, 3, 4, tzinfo=timezone.utc)
    unavailable = snapshot_capture._capture_snapshot_evidence(
        {},
        (FixtureRequirement,),
        observer=lambda: (_ for _ in ()).throw(OSError("synthetic failure")),
        clock=lambda: failure_time,
    )

    assert unavailable.environment is None
    assert unavailable.environment_status == "unavailable"
    assert unavailable.saved_at == failure_time
    assert unavailable.requirements.requirements == ("fixture-dependency>=1",)


def test_fixed_managed_snapshot_environment_bypasses_only_preobservation(
        fixed_managed_snapshot_environment, monkeypatch):
    """Managed opt-in fixes pre-observation without replacing introspection globally."""

    from dryml.environments import introspection

    inspected = []
    monkeypatch.setattr(
        introspection,
        "inspect_current",
        lambda: inspected.append(True) or fixed_managed_snapshot_environment,
    )

    observer = managed_runtime._preobserve_snapshot_environment()

    assert observer() == fixed_managed_snapshot_environment
    assert inspected == []
    assert introspection.inspect_current() == fixed_managed_snapshot_environment
    assert inspected == [True]


def test_managed_preobserver_captures_success_before_deferred_observation(
        fixed_snapshot_environment, monkeypatch):
    """The production seam inspects immediately and defers the captured value."""

    from dryml.environments import introspection

    environment = fixed_snapshot_environment
    observations = []
    monkeypatch.setattr(
        introspection,
        "inspect_current",
        lambda: observations.append("inspect") or environment,
    )

    observer = managed_runtime._preobserve_snapshot_environment()

    assert observations == ["inspect"]
    assert observer() == environment
    assert observations == ["inspect"]


def test_managed_preobserver_defers_observation_failure(monkeypatch):
    """Ordinary inspection failures are replayed by the returned observer."""

    from dryml.environments import introspection

    error = OSError("synthetic failure")

    def fail():
        raise error

    monkeypatch.setattr(introspection, "inspect_current", fail)

    observer = managed_runtime._preobserve_snapshot_environment()

    with pytest.raises(OSError) as raised:
        observer()
    assert raised.value is error


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit()])
def test_managed_preobserver_propagates_interruption(monkeypatch, interruption):
    """Process interruption escapes pre-observation instead of being deferred."""

    from dryml.environments import introspection

    def interrupt():
        raise interruption

    monkeypatch.setattr(introspection, "inspect_current", interrupt)

    with pytest.raises(type(interruption)) as raised:
        managed_runtime._preobserve_snapshot_environment()
    assert raised.value is interruption
