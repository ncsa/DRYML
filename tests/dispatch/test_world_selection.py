from __future__ import annotations

import dryml

from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan
from dryml.worlds import WorldSpec, attach_world_id, make_world_spec, use


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


def test_enveloped_explicit_world_is_preserved():
    world = {"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 2}}}}}

    resolution = resolve_dispatch_plan(
        normalize_user_operation(lambda: None, allow_pickle=True),
        world={"spec": world},
        requirement_policy="ignore",
    )

    assert resolution.world_selection.source == "explicit"
    assert resolution.world_selection.candidate["roles"]["main"]["process"]["resources"]["cpus"] == 2


@dryml.world.req(cpus={"min": 2})
def target_with_hard_world_requirement():
    return None


def test_canonical_explicit_world_envelope_is_checked_and_selected():
    envelope = attach_world_id(
        make_world_spec(
            {"main": {"replicas": 1, "process": {"resources": {"cpus": 2}}}}
        )
    )

    resolution = resolve_dispatch_plan(
        normalize_user_operation(target_with_hard_world_requirement, allow_pickle=True),
        world=envelope,
        requirement_policy="strict",
    )

    assert resolution.world_selection.source == "explicit"
    assert resolution.world_check.status == "satisfied"
    assert not any(item.code == "dryml.dispatch.single_subprocess_world_unsupported" for item in resolution.diagnostics)
