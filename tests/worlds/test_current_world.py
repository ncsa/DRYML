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


def test_current_returns_unset_default():
    sentinel = object()

    assert worlds.current() is None
    assert worlds.current(default=sentinel) is sentinel


def test_set_and_reset_current_world():
    world = {"policy": "single_worker"}

    assert worlds.set_current(world) is None
    assert worlds.current() is world
    worlds.reset_current()
    assert worlds.current() is None


def test_use_scopes_and_restores_current_world():
    outer = {"policy": "outer"}
    inner = {"policy": "inner"}
    worlds.set_current(outer)

    with worlds.use(inner) as scoped:
        assert scoped is inner
        assert worlds.current() is inner

    assert worlds.current() is outer


def test_use_restores_after_exception():
    outer = {"policy": "outer"}
    inner = {"policy": "inner"}
    worlds.set_current(outer)

    with pytest.raises(RuntimeError, match="boom"):
        with worlds.use(inner):
            raise RuntimeError("boom")

    assert worlds.current() is outer


def test_nested_use_contexts_restore_correctly():
    first = {"policy": "first"}
    second = {"policy": "second"}

    with worlds.use(first):
        assert worlds.current() is first
        with worlds.use(second):
            assert worlds.current() is second
        assert worlds.current() is first
    assert worlds.current() is None


def test_current_world_is_distinct_from_runtime_allocation():
    world = {"policy": "requested"}
    before = runtime.active_runtime()

    worlds.set_current(world)

    assert worlds.current() is world
    assert runtime.active_runtime() is before
    assert worlds.current() is not runtime.active_runtime().allocation


def test_setting_current_world_does_not_allocate_or_create_requirement():
    world = worlds.make_world_spec({"worker": {"replicas": 1}})
    before = runtime.active_runtime().allocation

    worlds.set_current(world)

    assert runtime.active_runtime().allocation is before
    assert "world_requirement" not in repr(world)
    assert hasattr(worlds, "current")
    assert hasattr(worlds, "set_current")
