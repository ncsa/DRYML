"""Inert Dispatch defaults, registry, and view-binding tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import dryml.dispatch as dispatch
from dryml.execute.config import BackendConfig


@dataclass(frozen=True, kw_only=True)
class _Backend(BackendConfig):
    """Name an inert test Execute configuration without creating a backend."""

    def create_backend(self):
        """Fail if inert configuration operations initialize a backend."""

        raise AssertionError("configuration must not initialize a backend")


@pytest.fixture(autouse=True)
def _clear_dispatch_state() -> None:
    """Isolate process-wide defaults without a public reset operation."""

    from dryml.dispatch import _state

    _state._reset_for_testing()
    yield
    _state._reset_for_testing()


def test_registry_is_detached_and_replacement_is_explicit() -> None:
    """Register configurations without giving callers the live registry map."""

    first = _Backend()
    second = _Backend()
    dispatch.register_backend("first", first)
    assert dict(dispatch.backends()) == {"first": first}
    with pytest.raises(TypeError):
        dispatch.backends()["other"] = second
    with pytest.raises(ValueError, match="already registered"):
        dispatch.register_backend("first", second)
    dispatch.register_backend("first", second, replace=True)
    assert dispatch.backends()["first"] is second
    dispatch.unregister_backend("first")
    with pytest.raises(KeyError):
        dispatch.unregister_backend("first")


def test_names_bind_when_a_default_or_view_is_created() -> None:
    """Later registry replacement cannot retarget bound values."""

    first = _Backend()
    second = _Backend()
    dispatch.register_backend("named", first)
    view = dispatch.with_options(backend="named")
    dispatch.set_execute_backend_default("named")
    dispatch.register_backend("named", second, replace=True)

    from dryml.dispatch import _state

    assert _state._effective_options(view).backend is first
    assert _state._effective_options(None).backend is first


def test_inherited_view_reads_defaults_at_call_entry_and_none_clears() -> None:
    """Retain exact inherit versus explicit clearing semantics per field."""

    backend = _Backend()
    view = dispatch.with_options()
    dispatch.set_execute_backend_default(backend)
    assert dispatch._state._effective_options(view).backend is backend
    assert (
        dispatch._state._effective_options(
            dispatch.with_options(backend=None)
        ).backend
        is None
    )
    dispatch.set_worker_environment_default(None)
    dispatch.set_worker_world_default(None)
    dispatch.set_worker_python_default(None)


def test_settings_never_create_backend() -> None:
    """Configuration replacement remains inert even for registered defaults."""

    config = _Backend()
    dispatch.register_backend("backend", config)
    dispatch.set_execute_backend_default("backend")
    dispatch.set_probe_default(dispatch.ProbeOptions(backend="backend"))
    assert dispatch.backends()["backend"] is config
