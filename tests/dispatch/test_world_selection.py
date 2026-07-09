from __future__ import annotations

import dryml

from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan
from dryml.worlds import WorldSpec, use


@dryml.world.default(cpus=2)
def target_with_world_default():
    return None


def test_world_selection_precedence_and_single_worker_fallback():
    normalized = normalize_user_operation(target_with_world_default, allow_pickle=True)
    explicit = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 4}}}}}
    resolution = resolve_dispatch_plan(normalized, world=explicit, requirement_policy="ignore")
    assert resolution.world_selection.source == "explicit"
    assert resolution.world_selection.candidate["roles"]["main"]["process"]["resources"]["cpus"] == 4

    def plain_target():
        return None

    fallback = resolve_dispatch_plan(normalize_user_operation(plain_target, allow_pickle=True), requirement_policy="ignore")
    assert fallback.world_selection.source == "fallback"
    assert fallback.world_selection.candidate["roles"]["main"]["replicas"] == 1


def test_world_current_precedes_fallback_without_using_runtime_allocation():
    current = WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 3}}}}})
    with use(current):
        resolution = resolve_dispatch_plan(normalize_user_operation(lambda: None, allow_pickle=True), requirement_policy="ignore")
    assert resolution.world_selection.source == "current"
    assert resolution.world_selection.candidate["roles"]["main"]["process"]["resources"]["cpus"] == 3
