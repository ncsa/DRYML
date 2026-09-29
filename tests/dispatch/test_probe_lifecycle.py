"""Lifecycle ordering tests for independent one-off Dispatch probes."""

from __future__ import annotations

import functools

import pytest

import dryml
from dryml.dispatch import ProbeOptions
from dryml.dispatch._probe import (
    ProbePlacementError,
    execute_prepared_probe,
    prepare_probe,
    run_probe,
)
from dryml.execute import CleanupError
from dryml.execute.subprocess import SubProcessConfig
from dryml.environments import req as environment_req


def _target() -> None:
    """Supply a simple source-backed inspection root."""


class _Future:
    """Minimal accepted Execute Future fixture for result/cleanup ordering."""

    def __init__(
        self,
        result=None,
        error: BaseException | None = None,
        cleanup: BaseException | None = None,
    ):
        self._result = result
        self._error = error
        self._cleanup = cleanup
        self.cleaned = False

    def result(self):
        """Return the configured terminal value or error."""

        if self._error is not None:
            raise self._error
        return self._result

    def cleanup(self):
        """Record cleanup and raise its configured incomplete outcome."""

        self.cleaned = True
        if self._cleanup is not None:
            raise self._cleanup


def test_remote_result_is_not_decoded_until_cleanup_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup failure retains recovery ownership and blocks result use."""

    from dryml.dispatch import _probe

    future = _Future(result=b"not-consumed")
    cleanup = CleanupError("cleanup incomplete", execution=future)
    future._cleanup = cleanup
    monkeypatch.setattr(
        _probe, "execute_submit", lambda *args, **kwargs: future
    )
    decoded = []
    monkeypatch.setattr(
        _probe, "decode_result", lambda *args: decoded.append(True)
    )

    with pytest.raises(CleanupError) as raised:
        run_probe(
            _target,
            options=ProbeOptions(
                placement="execute", backend=SubProcessConfig()
            ),
        )

    assert raised.value.execution is future
    assert future.cleaned
    assert decoded == []


def test_remote_primary_failure_retains_cleanup_failure_as_its_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution failure remains primary when cleanup is also incomplete."""

    from dryml.dispatch import _probe

    primary = TimeoutError("probe timed out")
    future = _Future(error=primary)
    cleanup = CleanupError("cleanup incomplete", execution=future)
    future._cleanup = cleanup
    monkeypatch.setattr(
        _probe, "execute_submit", lambda *args, **kwargs: future
    )

    with pytest.raises(TimeoutError) as raised:
        run_probe(
            _target,
            options=ProbeOptions(
                placement="execute", backend=SubProcessConfig()
            ),
        )

    assert raised.value is primary
    assert raised.value.__cause__ is cleanup
    assert future.cleaned


def test_remote_primary_failure_retains_unexpected_cleanup_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected cleanup failures never erase the primary probe outcome."""

    from dryml.dispatch import _probe

    primary = TimeoutError("probe timed out")
    cleanup = RuntimeError("unexpected cleanup failure")
    future = _Future(error=primary, cleanup=cleanup)
    monkeypatch.setattr(
        _probe, "execute_submit", lambda *args, **kwargs: future
    )

    with pytest.raises(TimeoutError) as raised:
        run_probe(
            _target,
            options=ProbeOptions(
                placement="execute", backend=SubProcessConfig()
            ),
        )

    assert raised.value is primary
    assert raised.value.__cause__ is cleanup
    assert future.cleaned


def test_real_subprocess_probe_handles_a_detached_closure_without_store() -> (
    None
):
    """A real worker receives a projection rather than a closure transport."""

    offset = 3

    def closure() -> int:
        return offset

    result = run_probe(
        closure,
        options=ProbeOptions(placement="execute", backend=SubProcessConfig()),
    )

    assert result.placement == "execute"
    assert result.coverage.complete


def test_core_function_wrapper_retains_its_raw_declaration_layer() -> None:
    """Core's passive seam adds a raw root without replacing a wrapper."""

    @dryml.function
    @environment_req(tags=("raw",), source="raw")
    def wrapped() -> None:
        return None

    result = run_probe(wrapped, options=ProbeOptions(placement="in_process"))

    assert result.environment.value.tags == ("raw",)


@pytest.mark.parametrize("function_first", (True, False))
def test_function_decorator_orders_preserve_declarations(
    function_first: bool,
) -> None:
    """Both decorator orders retain the exact layer carrying declarations."""

    if function_first:

        @dryml.function
        @environment_req(tags=("ordered",), source="ordered")
        def wrapped() -> None:
            return None

    else:

        @environment_req(tags=("ordered",), source="ordered")
        @dryml.function
        def wrapped() -> None:
            return None

    result = run_probe(wrapped, options=ProbeOptions(placement="in_process"))

    assert result.environment.value.tags == ("ordered",)


def test_nested_owner_and_ordinary_wrapper_keep_actual_bodies() -> None:
    """Core links remain opaque while an ordinary outer wrapper is analyzed."""

    @environment_req(tags=("nested",), source="nested")
    def target() -> None:
        return None

    inner = dryml.function(target)
    outer = dryml.function(inner)

    @functools.wraps(outer)
    def ordinary() -> None:
        outer()

    result = run_probe(ordinary, options=ProbeOptions(placement="in_process"))

    assert result.environment.value.tags == ("nested",)


def test_bound_object_method_needs_no_store_or_constructor() -> None:
    """A bound Object method is probed without constructing its owner."""

    calls: list[str] = []

    class BoundObject(dryml.Object):
        """Object fixture whose initializer must not run during inspection."""

        def __init__(self) -> None:
            calls.append("constructor")

        @environment_req(tags=("bound",), source="bound")
        def inspect(self) -> None:
            """Provide a declaration-bearing selected method."""

    receiver = object.__new__(BoundObject)
    result = run_probe(
        receiver.inspect, options=ProbeOptions(placement="in_process")
    )

    assert result.environment.value.tags == ("bound",)
    assert calls == []


def test_prepared_probe_retains_a_guard_after_result() -> None:
    """A later preflight can recheck this capture instead of recapturing it."""

    prepared = prepare_probe(
        _target, options=ProbeOptions(placement="in_process")
    )
    execute_prepared_probe(prepared)
    _target.__dryml_annotations__ = (
        getattr(_target, "__dryml_annotations__", ()) + ()
    )

    prepared.validate()


def test_prepared_probe_rejects_attached_declaration_drift() -> None:
    """Attached declaration changes invalidate retained call facts."""

    from dryml.environments import req as environment_req

    def target() -> None:
        """Supply a local declaration carrier for guard validation."""

    prepared = prepare_probe(
        target, options=ProbeOptions(placement="in_process")
    )
    environment_req(tags=("after-capture",), source="after-capture")(target)

    with pytest.raises(ProbePlacementError, match="declarations changed"):
        prepared.validate()
