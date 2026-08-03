from __future__ import annotations

import pytest

import dryml

from dryml.dispatch import Dispatcher, PickledCallable, RequirementPolicy, effective_requirement_policy, normalize_user_operation, resolve_dispatch_plan
from dryml.dispatch.errors import DispatchPlanningError
from dryml.core.store.dir import DirStore
from dryml.runtime import RuntimeEnforcement, enter_runtime


@pytest.mark.parametrize(
    "enforcement",
    [RuntimeEnforcement.STRICT, RuntimeEnforcement.WARN, RuntimeEnforcement.OFF],
)
def test_omitted_dispatch_policy_is_strict_independent_of_runtime(enforcement):
    with enter_runtime("orchestrator", enforcement=enforcement):
        assert effective_requirement_policy(None) is RequirementPolicy.STRICT


def test_explicit_policy_overrides_enforcement_and_invalid_value_fails_early():
    with enter_runtime("orchestrator", enforcement=RuntimeEnforcement.OFF):
        assert effective_requirement_policy("strict") is RequirementPolicy.STRICT
    with pytest.raises(DispatchPlanningError, match="invalid requirement_policy"):
        resolve_dispatch_plan(normalize_user_operation(lambda: None, allow_pickle=True), requirement_policy="bad")


def test_pickle_same_environment_restriction_remains_blocking_under_ignore():
    normalized = normalize_user_operation(PickledCallable(lambda: None), allow_pickle=True)
    resolution = resolve_dispatch_plan(
        normalized,
        environment={"kind": "python", "executable": "/not-the-current-python"},
        requirement_policy="ignore",
    )
    assert resolution.launchable is False
    assert any(item.code == "dryml.dispatch.pickle_environment_restriction" for item in resolution.diagnostics)


def test_runtime_requirement_uses_runtime_owned_compatibility_adapter():
    @dryml.annotations.require(namespace="runtime", fragment={"device_visibility": {"policy": "hidden"}})
    def target():
        return None

    resolution = resolve_dispatch_plan(normalize_user_operation(target, allow_pickle=True), requirement_policy="strict")
    assert resolution.runtime_check.status == "incompatible"
    assert resolution.launchable is False


@pytest.mark.parametrize("mode", ("none", "probe", "inline", "orchestrator"))
def test_explicit_non_worker_runtime_is_rejected_by_resolution_explain_and_plan(
    mode, tmp_path
):
    target = lambda: None
    normalized = normalize_user_operation(target, allow_pickle=True)
    dispatcher = Dispatcher(store=DirStore(tmp_path / "store", query_index="none"))

    with pytest.raises(DispatchPlanningError, match="invalid runtime candidate"):
        resolve_dispatch_plan(normalized, runtime_spec={"mode": mode})
    with pytest.raises(DispatchPlanningError, match="invalid runtime candidate"):
        dispatcher.explain(target, runtime={"mode": mode}, allow_pickle=True)
    with pytest.raises(DispatchPlanningError, match="invalid runtime candidate"):
        dispatcher.plan(target, runtime={"mode": mode}, allow_pickle=True)


@pytest.mark.parametrize("policy", ("strict", "warn", "ignore"))
def test_annotation_merge_errors_are_blocking_under_every_policy(policy):
    @dryml.world.req(cpus={"min": 4})
    @dryml.world.req(cpus={"max": 2})
    def target():
        return None

    resolution = resolve_dispatch_plan(normalize_user_operation(target, allow_pickle=True), requirement_policy=policy)
    assert resolution.launchable is False
    assert any(item.severity == "error" for item in resolution.diagnostics)


def test_disabled_requirement_axis_reports_disabled_without_weakening_structural_validation():
    @dryml.world.req(cpus={"min": 2})
    def target():
        return None

    resolution = resolve_dispatch_plan(
        normalize_user_operation(target, allow_pickle=True),
        requirement_axes={"environment": False, "world": False, "runtime": False},
    )

    assert resolution.world_check.status == "disabled"
    assert resolution.world_check.compatible is None


@pytest.mark.parametrize(
    "enabled",
    [
        (),
        ("environment",),
        ("world",),
        ("runtime",),
        ("environment", "world"),
        ("environment", "runtime"),
        ("world", "runtime"),
        ("environment", "world", "runtime"),
    ],
)
def test_dispatch_requirement_axes_control_every_compatibility_stage(enabled):
    @dryml.env.req(requirements=("not-an-installed-dryml-package>=1",))
    @dryml.world.req(cpus={"min": 10_000_000})
    @dryml.annotations.require(
        namespace="runtime",
        fragment={"device_visibility": {"policy": "hidden"}},
    )
    def target():
        return None

    axes = {
        name: name in enabled for name in ("environment", "world", "runtime")
    }
    resolution = resolve_dispatch_plan(
        normalize_user_operation(target, allow_pickle=True),
        requirement_axes=axes,
    )

    reports = {
        "environment": resolution.environment_check,
        "world": resolution.world_check,
        "runtime": resolution.runtime_check,
    }
    for name, report in reports.items():
        assert (report.status == "disabled") is (name not in enabled)
    assert resolution.launchable is (not enabled)
