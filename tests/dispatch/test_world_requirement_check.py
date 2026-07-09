from __future__ import annotations

import dryml

from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan


@dryml.world.req(accelerators={"gpu": {"min": 1}})
def gpu_target():
    return None


def test_strict_world_incompatibility_blocks_selected_cpu_world_without_search():
    cpu_world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}}
    resolution = resolve_dispatch_plan(normalize_user_operation(gpu_target, allow_pickle=True), world=cpu_world, requirement_policy="strict")
    assert resolution.world_selection.source == "explicit"
    assert resolution.world_check.status == "incompatible"
    assert resolution.launchable is False


def test_ignore_skips_world_requirement_but_preserves_selected_world():
    cpu_world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}}}
    resolution = resolve_dispatch_plan(normalize_user_operation(gpu_target, allow_pickle=True), world=cpu_world, requirement_policy="ignore")
    assert resolution.world_check.status == "skipped"
    assert resolution.world_check.details == ({"reason": "requirement_policy_ignore"},)
    assert resolution.launchable is True
