import asyncio
import threading

import dryml.environments as envs


def test_current_environment_scope_restores_after_normal_and_exceptional_exit():
    selector = object()
    envs.reset_current()
    assert envs.current() is None
    with envs.use(selector):
        assert envs.current() is selector
    assert envs.current() is None
    try:
        with envs.use(selector):
            raise RuntimeError("expected")
    except RuntimeError:
        pass
    assert envs.current() is None


def test_current_environment_is_context_local_for_tasks_and_threads():
    envs.reset_current()
    parent = object()
    child = object()
    envs.set_current(parent)

    async def observe():
        before = envs.current()
        envs.set_current(child)
        return before, envs.current()

    assert asyncio.run(observe()) == (parent, child)
    assert envs.current() is parent
    observed = []
    thread = threading.Thread(target=lambda: observed.append(envs.current("thread-default")))
    thread.start()
    thread.join()
    assert observed == ["thread-default"]
