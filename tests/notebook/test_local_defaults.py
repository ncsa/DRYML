from __future__ import annotations

import dryml

from dryml.dispatch import Dispatcher
from dryml.environments import CurrentEnvironmentSpec
from dryml.worlds import LocalResourceInventory, WorldSpec


def test_notebook_context_defaults_are_selected_and_restored_without_allocation():
    requested_world = WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 2}}}}})
    previous_environment = dryml.environments.set_current(CurrentEnvironmentSpec())
    previous_world = dryml.worlds.set_current(requested_world)
    try:
        explanation = Dispatcher().explain(lambda: None, allow_pickle=True, requirement_policy="ignore", inventory=LocalResourceInventory((0, 1)))
        assert explanation.resolution.environment_selection.source == "current"
        assert explanation.resolution.world_selection.source == "current"
        assert explanation.resolution.world_allocation_summary["backend"] == "local_subprocess"
        assert "environment_attempts=" in str(explanation)
        with dryml.environments.use(CurrentEnvironmentSpec()), dryml.worlds.use(WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {}}}})):
            assert dryml.worlds.current() != requested_world
        assert dryml.worlds.current() == requested_world
    finally:
        dryml.environments.set_current(previous_environment) if previous_environment is not None else dryml.environments.reset_current()
        dryml.worlds.set_current(previous_world) if previous_world is not None else dryml.worlds.reset_current()
