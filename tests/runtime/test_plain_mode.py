import pytest

import dryml.environments as environments
import dryml.runtime as runtime
import dryml.worlds as worlds
from dryml.runtime.errors import RuntimeTransitionError


@pytest.fixture(autouse=True)
def reset_state():
    runtime.reset_runtime()
    environments.reset_current()
    worlds.reset_current()
    yield
    runtime.reset_runtime()
    environments.reset_current()
    worlds.reset_current()


def test_plain_mode_sets_inline_off_with_local_allocation():
    with runtime.plain() as state:
        assert state is runtime.active_runtime()
        assert state.mode is runtime.RuntimeMode.INLINE
        assert state.enforcement is runtime.RuntimeEnforcement.OFF
        assert state.allocation is not runtime.NoAllocation
        assert state.allocation.role == "local"
        assert state.allocation.metadata["kind"] == "plain"


def test_plain_mode_restores_previous_state_after_normal_exit():
    allocation = runtime.RuntimeAllocationView(role="worker", cpus=(0,))

    with runtime.enter_runtime(runtime.RuntimeMode.WORKER, allocation, enforcement="warn"):
        before = runtime.active_runtime()
        with runtime.plain():
            assert runtime.active_runtime().mode is runtime.RuntimeMode.INLINE
        assert runtime.active_runtime() is before


def test_plain_mode_restores_previous_state_after_exception():
    before = runtime.active_runtime()

    with pytest.raises(RuntimeError, match="boom"):
        with runtime.plain():
            raise RuntimeError("boom")

    assert runtime.active_runtime() is before


def test_plain_mode_nests_with_disabled_context():
    runtime.set_enforcement("warn")

    with runtime.disabled():
        assert runtime.enforcement() is runtime.RuntimeEnforcement.OFF
        with runtime.plain():
            assert runtime.active_runtime().mode is runtime.RuntimeMode.INLINE
            assert runtime.enforcement() is runtime.RuntimeEnforcement.OFF
        assert runtime.enforcement() is runtime.RuntimeEnforcement.OFF

    assert runtime.enforcement() is runtime.RuntimeEnforcement.WARN


def test_plain_mode_does_not_mutate_current_environment_or_world():
    env = environments.CurrentEnvironmentSpec()
    world = {"policy": "requested"}
    environments.set_current(env)
    worlds.set_current(world)

    with runtime.plain():
        assert environments.current() is env
        assert worlds.current() is world

    assert environments.current() is env
    assert worlds.current() is world


def test_plain_mode_allocation_metadata_mutation_does_not_leak_between_entries():
    with runtime.plain() as first:
        first.allocation.metadata["mutated"] = True

    with runtime.plain() as second:
        assert "mutated" not in second.allocation.metadata
        assert second.allocation.metadata == {"kind": "plain"}


def test_runtime_allocation_mappings_are_mutable_defensive_copies():
    accelerators = {"gpu": [0]}
    accelerator_memory = {"gpu": {0: 1024}}
    env = {"VISIBLE": "0"}
    metadata = {"source": "worker"}

    allocation = runtime.RuntimeAllocationView(
        accelerators=accelerators,
        accelerator_memory=accelerator_memory,
        env=env,
        metadata=metadata,
    )
    allocation.accelerators["gpu"] = (1,)
    allocation.accelerator_memory["gpu"][0] = 2048
    allocation.env["VISIBLE"] = "1"
    allocation.metadata["source"] = "runtime"

    assert accelerators == {"gpu": [0]}
    assert accelerator_memory == {"gpu": {0: 1024}}
    assert env == {"VISIBLE": "0"}
    assert metadata == {"source": "worker"}


def test_runtime_allocation_invariants_remain_valid():
    allocation = runtime.RuntimeAllocationView(cpus=(0,))

    with pytest.raises(RuntimeTransitionError):
        with runtime.enter_runtime(runtime.RuntimeMode.WORKER):
            pass
    with pytest.raises(RuntimeTransitionError):
        with runtime.enter_runtime(runtime.RuntimeMode.ORCHESTRATOR, allocation):
            pass
    with runtime.plain():
        assert runtime.require_allocation().role == "local"
