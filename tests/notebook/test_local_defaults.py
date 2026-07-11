from __future__ import annotations

import dryml

from dryml.dispatch import Dispatcher
from dryml.core2.store.dir import DirStore
from dryml.environments import CurrentEnvironmentSpec, EnvironmentRegistry
from dryml.operations import make_function_call_spec
from dryml.runtime import active_runtime
from dryml.runtime.allocation import is_no_allocation
from dryml.worlds import LocalResourceInventory, WorldSpec


def test_notebook_context_defaults_are_selected_and_restored_without_allocation():
    requested_world = WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 2}}}}})
    previous_environment = dryml.environments.set_current(CurrentEnvironmentSpec())
    previous_world = dryml.worlds.set_current(requested_world)
    try:
        explanation = Dispatcher().explain(lambda: None, allow_pickle=True, requirement_policy="ignore", inventory=LocalResourceInventory((0, 1)))
        assert explanation.resolution.environment_selection.source == "current"
        assert explanation.resolution.world_selection.source == "current"
        assert explanation.resolution.world_allocation_summary is None
        assert "environment_attempts=" in str(explanation)
        with dryml.environments.use(CurrentEnvironmentSpec()), dryml.worlds.use(WorldSpec.from_data({"roles": {"main": {"replicas": 1, "process": {}}}})):
            assert dryml.worlds.current() != requested_world
        assert dryml.worlds.current() == requested_world
    finally:
        dryml.environments.set_current(previous_environment) if previous_environment is not None else dryml.environments.reset_current()
        dryml.worlds.set_current(previous_world) if previous_world is not None else dryml.worlds.reset_current()


def test_notebook_registry_explain_is_explicit_repeatable_and_allocation_free(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    registry = EnvironmentRegistry()
    entry = registry.register("current", CurrentEnvironmentSpec())

    explanation = Dispatcher(store=store, environment_registry=registry).explain(
        make_function_call_spec("operator:add", args=[1, 2]),
    )
    repeated = Dispatcher(store=store, environment_registry=registry).explain(
        make_function_call_spec("operator:add", args=[1, 2]),
    )

    assert explanation.resolution.environment_selection.source == "resolver"
    assert registry.list() == (entry,)
    assert repeated.to_data() == explanation.to_data()
    assert is_no_allocation(active_runtime().allocation)
    assert not store.records.specs_dir.exists()
