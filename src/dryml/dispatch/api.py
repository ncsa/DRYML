"""Public Dispatch configuration, preflight, explanation, and execution.

The module exposes explicit backend and blocking in-process execution routes.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

from dryml.core.execute import (
    _prepare_frozen_core_submission,
    _submit_frozen_core_submission,
)
from dryml.execute import submit as _generic_submit
from dryml.execute.errors import CleanupError
from dryml.runtime import publication

from . import _state
from ._admission import check_current_compatibility
from ._preflight import _Preflight, preflight
from ._probe import ProbePlacementError
from .errors import DispatchError
from .models import (
    DispatchCoverageWarning,
    DispatchReport,
    DispatchView,
    InProcess,
)


def _checked_preflight(
    fn: Any, *, operation: str, view: DispatchView | None
) -> _Preflight:
    """Return eligible preflight facts or raise the public submission error."""

    result = preflight(  # type: ignore[arg-type]
        fn, operation=operation, view=view
    )
    if type(result) is DispatchReport:
        raise DispatchError(result)
    return result


def _warn_coverage(prepared: _Preflight) -> None:
    """Emit the valid-incomplete warning for accepted operations."""

    if prepared.report.coverage == "incomplete":
        warnings.warn(
            "Dispatch static requirement coverage is incomplete",
            DispatchCoverageWarning,
            stacklevel=3,
        )


def _submit_backend(
    prepared: _Preflight,
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Prepare and accept one Dispatch call through core's one-off owner.

    The retained preflight owns call-entry controls, exact selector, combined
    requirements, and target guard. Core owns payload preparation, backend
    acceptance, result adaptation, and cleanup after this handoff.
    """

    config = prepared.options.backend
    if type(config) is InProcess or config is None:
        raise RuntimeError(
            "backend Dispatch submission requires an Execute configuration"
        )
    environment = prepared.report.environment
    world = prepared.report.world
    if environment is None or world is None:
        raise RuntimeError(
            "eligible Dispatch preflight is missing requirement results"
        )
    validators = (lambda: _preflight_validator(prepared),)
    frozen = _prepare_frozen_core_submission(
        config,
        None,
        fn,
        args,
        kwargs=kwargs,
        core=prepared.options.core,
        validators=validators,
        frozen_controls=prepared.core_controls,
    )
    return _submit_frozen_core_submission(
        lambda *call_args, **controls: _generic_submit(
            *call_args, backend=config, **controls
        ),
        config,
        frozen,
        environment=environment.value,
        environment_spec=prepared.selection,
        world=world.value,
        execution_timeout="inherit",
        stream_output=None,
        done_callbacks=(),
        output=None,
        one_off=True,
        validators=validators,
    )


def _preflight_validator(prepared: _Preflight) -> None:
    """Raise the public preflight error when retained target evidence drifts.

    The result never converts a workload exception because it runs before
    invocation.
    """

    try:
        prepared.validate()
    except ProbePlacementError as error:
        diagnostics = tuple(
            dict.fromkeys(
                (*prepared.report.diagnostics, "dispatch.target_changed")
            )
        )[:64]
        raise DispatchError(
            replace(
                prepared.report,
                eligible=False,
                probe_reason=(
                    "probe target changed during workload preparation"
                ),
                diagnostics=diagnostics,
            )
        ) from error


def _run_in_process(
    prepared: _Preflight,
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Admit and invoke one local synchronous workload exactly once.

    Args:
        prepared: Eligible call-entry preflight facts and retained target
            guard.
        fn: Original caller-supplied callable.
        args: Original positional workload arguments.
        kwargs: Original keyword workload arguments.

    Returns:
        The unmodified direct return value from ``fn``.

    Raises:
        DispatchError: If fresh held-generation evidence or the final target
            guard rejects the call before invocation.
        BaseException: Any exception raised by ``fn`` unchanged.

    Side Effects:
        Holds the current runtime publication generation through fresh evidence
        and the final target guard, then invokes once on the caller thread. It
        does not construct Execute/core transport, create a Store, or
        save/refresh values. The lease is released on every exit, including
        interruption.
    """

    environment = prepared.report.environment
    world = prepared.report.world
    if environment is None or world is None:
        raise RuntimeError(
            "eligible Dispatch preflight is missing requirement results"
        )
    with publication.lease() as generation:
        outcome = check_current_compatibility(
            prepared.options.core,
            environment,
            world,
            prepared.selection,
            generation,
        )
        if not outcome.eligible:
            diagnostics = tuple(
                dict.fromkeys(
                    (*prepared.report.diagnostics, *outcome.diagnostics)
                )
            )[:64]
            raise DispatchError(
                replace(
                    prepared.report,
                    eligible=False,
                    probe_reason=(
                        "current process changed before local admission"
                    ),
                    diagnostics=diagnostics,
                )
            )
        _preflight_validator(prepared)
        return fn(*args, **kwargs)


def _explain(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    view: DispatchView | None,
) -> DispatchReport:
    """Implement explanation for root and immutable view facades."""

    del args, kwargs
    result = preflight(fn, operation="explain", view=view)
    return result.report if type(result) is _Preflight else result


def explain(fn: Any, /, *args: Any, **kwargs: Any) -> DispatchReport:
    """Inspect Dispatch configuration and requirements without submitting work.

    Args:
        fn: Supported synchronous workload root.
        *args: Workload positional data, retained only by the caller.
        **kwargs: Workload keyword data. Names such as ``env`` and ``backend``
            are never interpreted as Dispatch controls.

    Returns:
        One immutable bounded non-reserving Dispatch report.

    Raises:
        TypeError, ValueError: For invalid configuration or workload modality.

    Side Effects:
        Runs bounded declaration probing and may create a short-lived selected
        backend for non-reserving discovery. It does not invoke the workload,
        export or publish Store state, or reserve workload resources.
    """

    return _explain(fn, args, kwargs, None)


def _submit(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    view: DispatchView | None,
) -> Any:
    """Implement submission preflight for root and immutable view facades."""

    prepared = _checked_preflight(fn, operation="submit", view=view)
    _warn_coverage(prepared)
    _preflight_validator(prepared)
    return _submit_backend(prepared, fn, args, kwargs)


def submit(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Submit a synchronous workload through the selected Execute backend.

    Args:
        fn: Supported synchronous workload root.
        *args: Workload positional data.
        **kwargs: Workload keyword data, including control-named values.

    Returns:
        The existing core execution future for the accepted backend call.

    Raises:
        DispatchError: If probing, compatibility, or a retained target guard
            rejects the call before backend acceptance.
        BaseException: Existing core/backend admission, cancellation, uncertain
            outcome, and cleanup surfaces after acceptance.

    Side Effects:
        Runs bounded preflight, prepares one core payload, and submits it to
        the explicit backend. It never retries, selects another backend, or
        wraps the returned future.
    """

    return _submit(fn, args, kwargs, None)


def _run(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    view: DispatchView | None,
) -> Any:
    """Implement blocking preflight for root and immutable view facades."""

    prepared = _checked_preflight(fn, operation="run", view=view)
    _warn_coverage(prepared)
    if prepared.report.workload_placement == "in_process":
        return _run_in_process(prepared, fn, args, kwargs)
    _preflight_validator(prepared)
    future = _submit_backend(prepared, fn, args, kwargs)
    try:
        result = future.result()
    except BaseException as error:
        try:
            future.cleanup()
        except CleanupError as cleanup_error:
            raise error from cleanup_error
        raise
    future.cleanup()
    return result


def run(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous workload through its selected Dispatch route.

    Args:
        fn: Supported synchronous workload root.
        *args: Workload positional data.
        **kwargs: Workload keyword data, including control-named values.

    Returns:
        The recovered backend result for an Execute route, or the direct result
        for the explicit in-process route.

    Raises:
        DispatchError: If preflight or retained target validation rejects the
            workload before acceptance.
        BaseException: Existing core/backend result and cleanup failures, with
            the primary execution failure retained when cleanup also fails.

    Side Effects:
        Runs bounded preflight and one backend submission for Execute routes.
        It neither retries work nor changes the selected route.
    """

    return _run(fn, args, kwargs, None)


backends = _state.backends
register_backend = _state.register_backend
set_execute_backend_default = _state.set_execute_backend_default
set_probe_default = _state.set_probe_default
set_worker_environment_default = _state.set_worker_environment_default
set_worker_python_default = _state.set_worker_python_default
set_worker_world_default = _state.set_worker_world_default
unregister_backend = _state.unregister_backend
with_options = _state.with_options


__all__ = [
    "backends",
    "explain",
    "register_backend",
    "run",
    "set_execute_backend_default",
    "set_probe_default",
    "set_worker_environment_default",
    "set_worker_python_default",
    "set_worker_world_default",
    "submit",
    "unregister_backend",
    "with_options",
]
