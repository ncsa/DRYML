import pytest

import dryml.runtime as runtime
import dryml.worlds as worlds


@pytest.fixture(autouse=True)
def reset_current_world():
    worlds.reset_current()
    runtime.reset_runtime()
    yield
    worlds.reset_current()
    runtime.reset_runtime()


def test_discover_current_returns_explicit_current_world_first():
    world = {"policy": "explicit"}
    worlds.set_current(world)

    assert worlds.discover_current(default={"policy": "fallback"}) is world


def test_discover_current_unset_returns_documented_default():
    fallback = {"policy": "fallback"}

    assert worlds.discover_current() is None
    assert worlds.discover_current(default=fallback) is fallback


def test_discover_current_does_not_synthesize_from_runtime_allocation():
    allocation = runtime.RuntimeAllocationView(role="worker", cpus=(0,))
    fallback = {"policy": "fallback"}

    with runtime.enter_runtime(runtime.RuntimeMode.WORKER, allocation):
        assert worlds.discover_current(default=fallback) is fallback


def test_discover_current_does_not_allocate_resources():
    before = runtime.active_runtime()

    assert worlds.discover_current() is None
    assert runtime.active_runtime() is before


def test_discover_current_api_is_exported():
    assert hasattr(worlds, "discover_current")
