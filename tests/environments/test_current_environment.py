import contextvars

import pytest

import dryml.environments as environments
import dryml.runtime as runtime


@pytest.fixture(autouse=True)
def reset_current_environment():
    environments.reset_current()
    runtime.reset_runtime()
    yield
    environments.reset_current()
    runtime.reset_runtime()


def test_current_returns_unset_default():
    sentinel = object()

    assert environments.current() is None
    assert environments.current(default=sentinel) is sentinel


def test_set_and_reset_current_environment():
    spec = environments.CurrentEnvironmentSpec()

    assert environments.set_current(spec) is None
    assert environments.current() is spec
    environments.reset_current()
    assert environments.current() is None


def test_use_scopes_and_restores_current_environment():
    outer = environments.CurrentEnvironmentSpec()
    inner = environments.PythonExecutableSpec("/usr/bin/python")
    environments.set_current(outer)

    with environments.use(inner) as scoped:
        assert scoped is inner
        assert environments.current() is inner

    assert environments.current() is outer


def test_use_restores_after_exception():
    outer = environments.CurrentEnvironmentSpec()
    inner = environments.PythonExecutableSpec("/usr/bin/python")
    environments.set_current(outer)

    with pytest.raises(RuntimeError, match="boom"):
        with environments.use(inner):
            raise RuntimeError("boom")

    assert environments.current() is outer


def test_nested_use_contexts_restore_correctly():
    first = environments.CurrentEnvironmentSpec()
    second = environments.PythonExecutableSpec("/tmp/python")

    with environments.use(first):
        assert environments.current() is first
        with environments.use(second):
            assert environments.current() is second
        assert environments.current() is first
    assert environments.current() is None


def test_contextvars_isolate_copied_contexts():
    outer = environments.CurrentEnvironmentSpec()
    inner = environments.PythonExecutableSpec("/tmp/python")
    environments.set_current(outer)

    copied = contextvars.copy_context()

    def set_inner():
        environments.set_current(inner)
        return environments.current()

    assert copied.run(set_inner) is inner
    assert environments.current() is outer


def test_api_is_exported_and_has_no_runtime_side_effects():
    spec = environments.CurrentEnvironmentSpec()
    before = runtime.active_runtime()

    assert hasattr(environments, "current")
    assert hasattr(environments, "set_current")
    environments.set_current(spec)

    assert runtime.active_runtime() is before
