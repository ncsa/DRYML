from __future__ import annotations

import dryml

from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan
from dryml.worlds import LocalResourceInventory


@dryml.world.req(cpus={"min": 2})
def cpu_target():
    return None


def test_no_override_hard_world_requirement_is_synthesized_once():
    inventory = LocalResourceInventory((0, 1, 2, 3))

    resolution = resolve_dispatch_plan(
        normalize_user_operation(cpu_target, allow_pickle=True),
        inventory=inventory,
        requirement_policy="strict",
        single_worker_only=True,
    )

    assert resolution.world_selection.source == "synthesized"
    assert resolution.world_synthesis is not None and resolution.world_synthesis.ok
    assert resolution.inventory_summary == inventory.summary()
    assert resolution.launchable
