"""Dispatch call-entry snapshot and preflight ordering tests."""

from __future__ import annotations

from dataclasses import dataclass
import functools

import pytest

import dryml.dispatch as dispatch
from dryml.core._callable_inspection import _FunctionInvocationOwner
from dryml.execute.config import BackendConfig
from dryml.execute.subprocess import SubProcessConfig


@pytest.fixture(autouse=True)
def _clear_dispatch_state() -> None:
    """Keep each process-default snapshot test independent."""

    dispatch._state._reset_for_testing()
    yield
    dispatch._state._reset_for_testing()


@dataclass(frozen=True, kw_only=True)
class _FakeRayConfig(BackendConfig):
    """Model a Ray-like route that must not initialize in this test."""

    def create_backend(self):
        """Fail when rejected modality reaches backend construction."""

        raise AssertionError("modality rejection must not initialize fake Ray")


def _unsupported_root(kind: str, *, wrapped: bool):
    """Build one direct or established-owner unsupported native root."""

    if kind == "coroutine":

        async def target() -> None:
            """Provide a coroutine root."""

    elif kind == "generator":

        def target():
            """Provide a generator root."""

            yield None

    else:

        async def target():
            """Provide an async-generator root."""

            yield None

    if not wrapped:
        return target

    def owner_wrapper():
        """Retain the core-proven raw target as the actual invocation owner."""

        return target()

    owner = _FunctionInvocationOwner(target)
    owner.wrapper = owner_wrapper
    owner_wrapper.__dryml_function_invocation_owner__ = owner
    return owner_wrapper


@pytest.mark.parametrize(
    "operation", [dispatch.explain, dispatch.run, dispatch.submit]
)
@pytest.mark.parametrize("factory", [lambda: (lambda: (yield None))])
def test_unsupported_root_modalities_reject_before_probe(
    monkeypatch: pytest.MonkeyPatch, operation, factory
) -> None:
    """Reject generator roots before capture, probe, or backend startup."""

    dispatch.set_execute_backend_default(SubProcessConfig())
    monkeypatch.setattr(
        dispatch._preflight,
        "prepare_probe",
        lambda *_args, **_kwargs: pytest.fail(
            "probe preparation must not occur"
        ),
    )
    workload = factory()
    with pytest.raises(ValueError, match="generator"):
        operation(workload)


@pytest.mark.parametrize("operation", ("explain", "run", "submit"))
@pytest.mark.parametrize("kind", ("coroutine", "generator", "async_generator"))
@pytest.mark.parametrize("wrapped", (False, True), ids=("direct", "owner"))
@pytest.mark.parametrize(
    "route", ("in_process", "subprocess", "fake_ray")
)
def test_unsupported_modalities_reject_before_capture_or_backend_start(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    kind: str,
    wrapped: bool,
    route: str,
) -> None:
    """Reject all known deferred roots before Dispatch starts later owners."""

    backend = {
        "in_process": dispatch.InProcess(),
        "subprocess": SubProcessConfig(),
        "fake_ray": _FakeRayConfig(),
    }[route]
    dispatch.set_execute_backend_default(backend)
    monkeypatch.setattr(
        dispatch._preflight,
        "prepare_probe",
        lambda *_args, **_kwargs: pytest.fail(
            "modality rejection must not probe"
        ),
    )
    monkeypatch.setattr(
        dispatch._preflight,
        "_capture_frozen_core_controls",
        lambda *_args, **_kwargs: pytest.fail(
            "modality rejection must not capture execution controls"
        ),
    )

    expected = (
        "only supports blocking run"
        if route == "in_process" and operation == "submit"
        else kind.replace("_", " ")
    )
    with pytest.raises(ValueError, match=expected):
        getattr(dispatch, operation)(_unsupported_root(kind, wrapped=wrapped))


def test_sync_wrapper_deferred_wrapped_target_returns_unadvanced() -> None:
    """Keep arbitrary synchronous wrapper code and its result deferred."""

    advanced = False

    def deferred_target():
        """Expose a generator that records only caller-driven advancement."""

        nonlocal advanced
        advanced = True
        yield None

    @functools.wraps(deferred_target)
    def wrapper():
        """Remain a synchronous ordinary root despite copied metadata."""

        return deferred_target()

    result = dispatch.with_options(backend=dispatch.InProcess()).run(wrapper)

    assert not advanced
    result.close()


def test_local_submit_capability_rejects_before_modality_or_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer actionable local-submit guidance over later workload checks."""

    monkeypatch.setattr(
        dispatch._preflight,
        "prepare_probe",
        lambda *_args, **_kwargs: pytest.fail(
            "probe preparation must not occur"
        ),
    )

    async def unsupported() -> None:
        """Supply a root that would otherwise fail modality validation."""

    with pytest.raises(ValueError, match="only supports blocking run"):
        dispatch.with_options(backend=dispatch.InProcess()).submit(unsupported)


def test_missing_choice_is_actionable_before_probe() -> None:
    """Do not silently select a local or registered execution backend."""

    with pytest.raises(ValueError, match="execution choice"):
        dispatch.explain(lambda: None)


def test_workload_controls_remain_untouched_at_the_later_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass control-named workload data unchanged through the U7 seam."""

    dispatch.set_execute_backend_default(SubProcessConfig())
    captured: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def submit_backend(_prepared, fn, args, kwargs):
        """Capture the seam without invoking the workload."""

        captured.append((fn, args, kwargs))
        return "accepted"

    monkeypatch.setattr(dispatch.api, "_submit_backend", submit_backend)

    def workload() -> None:
        """Supply a synchronous source-backed workload root."""

    assert (
        dispatch.submit(
            workload,
            "argument",
            env="data",
            world="data",
            backend="data",
            probe="data",
        )
        == "accepted"
    )
    assert captured == [
        (
            workload,
            ("argument",),
            {
                "env": "data",
                "world": "data",
                "backend": "data",
                "probe": "data",
            },
        )
    ]
