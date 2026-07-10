from __future__ import annotations

import dryml

from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan
from dryml.operations import make_function_call_spec


def test_live_complete_discovery_skips_code_probe_and_environment_probe_without_requirement():
    @dryml.world.default(cpus=1)
    def target():
        return None

    resolution = resolve_dispatch_plan(normalize_user_operation(target, allow_pickle=True), requirement_policy="strict")
    assert resolution.code_probe is None
    assert resolution.environment_record is None
    assert resolution.environment_check.status == "not_required"


def test_environment_requirement_is_checked_against_selected_candidate_record():
    @dryml.env.req(requirements=("package-that-cannot-exist-for-dryml-test>=1",))
    def target():
        return None

    resolution = resolve_dispatch_plan(normalize_user_operation(target, allow_pickle=True), requirement_policy="strict")
    assert resolution.environment_record is not None
    assert resolution.environment_check.status == "incompatible"
    assert resolution.launchable is False


def test_missing_explicit_final_environment_is_structurally_blocking_without_requirement():
    normalized = normalize_user_operation(make_function_call_spec("operator:add"))
    resolution = resolve_dispatch_plan(
        normalized,
        environment={"kind": "python", "executable": "/definitely/missing/python"},
        requirement_policy="ignore",
    )

    assert resolution.environment_check.status == "error"
    assert resolution.launchable is False
    assert resolution.code_probe is not None
