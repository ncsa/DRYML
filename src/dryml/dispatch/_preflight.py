"""Dispatch configuration capture, modality validation, and report assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from dryml.code.errors import CodeAnalysisError
from dryml.core._callable_inspection import (
    CallableDescription,
    describe_callable,
)
from dryml.core.execute import _capture_frozen_core_controls
from dryml.environments.combination import merge_environment_requirements
from dryml.environments.errors import EnvironmentSpecError
from dryml.environments.selection import resolve_environment_spec
from dryml.requirements import (
    RequirementIssue,
    RequirementReport,
    RequirementResult,
)
from dryml.runtime import publication
from dryml.worlds.combination import merge_world_requirements

from . import _state
from ._admission import (
    check_current_compatibility,
    check_execute_compatibility,
)
from ._probe import (
    ProbePlacementError,
    ProbeResult,
    _PreparedProbe,
    execute_prepared_probe,
    prepare_probe,
)
from ._probe_protocol import ProbeProtocolError
from .models import DispatchReport, DispatchView, InProcess

_COVERAGE_WARNING = "dispatch.coverage_incomplete"


@dataclass(frozen=True, slots=True)
class _Preflight:
    """Retain eligible call-entry state for later execution handoff only.

    The private carrier owns no executor, Store, workload arguments, or
    resource handle. It keeps the probe guard so later preparation can recheck
    observed target/declaration drift before accepting work.
    """

    options: _state._EffectiveOptions
    core_controls: object
    selection: object | None
    probe: _PreparedProbe
    report: DispatchReport

    def validate(self) -> None:
        """Recheck retained target facts before a later execution owner."""

        self.probe.validate()


def _methods(backend: object | None) -> frozenset[Literal["run", "submit"]]:
    """Return capability facts without creating a backend."""

    return (
        frozenset({"run"})
        if type(backend) is InProcess
        else frozenset({"run", "submit"})
    )


def _report(
    options: _state._EffectiveOptions,
    *,
    probe: ProbeResult | None,
    eligible: bool,
    reason: str,
    diagnostics: tuple[str, ...] = (),
    environment: RequirementResult[Any] | None = None,
    world: RequirementResult[Any] | None = None,
) -> DispatchReport:
    """Build one value-free bounded report from completed preflight facts."""

    if type(options.backend) is InProcess:
        workload_placement: Literal["in_process", "execute"] | None = (
            "in_process"
        )
        workload_backend = None
    elif options.backend is None:
        workload_placement = None
        workload_backend = None
    else:
        workload_placement = "execute"
        workload_backend = options.backend_label
    if probe is None:
        return DispatchReport(
            workload_placement,
            workload_backend,
            _methods(options.backend),
            None,
            None,
            reason,
            None,
            environment,
            world,
            eligible,
            diagnostics,
        )
    probe_backend = options.probe_label
    if probe.placement == "execute" and probe_backend is None:
        probe_backend = "SubProcessConfig"
    probe_reason = (
        "current process satisfies probe controls"
        if probe.placement == "in_process"
        else "selected Execute probe configuration"
    )
    warnings = (_COVERAGE_WARNING,) if not probe.coverage.complete else ()
    return DispatchReport(
        workload_placement,
        workload_backend,
        _methods(options.backend),
        probe.placement,
        probe_backend,
        probe_reason,
        "complete" if probe.coverage.complete else "incomplete",
        environment,
        world,
        eligible,
        (diagnostics + probe.coverage.diagnostics)[:64],
        warnings,
    )


def _combine_results(
    configured: object | None,
    discovered: RequirementResult[Any],
    *,
    owner: Literal["environment", "world"],
) -> RequirementResult[Any]:
    """Combine an explicit override with probe output through its owner.

    Existing failed discovery results remain untouched. A fixed configured
    world is intersected, never widened, with discovered requirements.
    """

    if not discovered.ok or configured is None:
        return discovered
    if discovered.value is None:
        return RequirementResult(configured)
    try:
        value = (
            merge_environment_requirements(configured, discovered.value)
            if owner == "environment"
            else merge_world_requirements(configured, discovered.value)
        )
    except Exception:
        return RequirementResult(
            report=RequirementReport(
                (
                    RequirementIssue(
                        "dryml.dispatch." f"{owner}_requirement_conflict",
                        "configured and discovered requirements conflict",
                    ),
                )
            )
        )
    return RequirementResult(value)


def _validate_choice(
    options: _state._EffectiveOptions, operation: str
) -> None:
    """Reject missing or locally unsupported choices before capture."""

    if options.backend is None:
        raise ValueError(
            "Dispatch requires an explicit execution choice "
            "or configured default"
        )
    if operation == "submit" and type(options.backend) is InProcess:
        raise ValueError(
            "InProcess Dispatch only supports blocking run; "
            "select an Execute backend for submit"
        )


def _validate_modality(description: CallableDescription) -> None:
    """Reject non-synchronous or unidentifiable workload invocation roots."""

    if description.native_modality != "sync":
        modality = description.native_modality.replace("_", " ")
        raise ValueError(
            f"Dispatch does not support {modality} workload roots"
        )


def preflight(
    fn: Any,
    *,
    operation: Literal["explain", "run", "submit"],
    view: DispatchView | None,
) -> _Preflight | DispatchReport:
    """Capture Dispatch configuration before execution ownership.

    Args:
        fn: Caller-supplied synchronous workload root.
        operation: Requested public Dispatch operation.
        view: Optional immutable configuration view.

    Returns:
        A retained eligible preflight or an ineligible explanation report for
        a trustworthy probe/setup failure.

    Raises:
        TypeError, ValueError: For malformed configuration, missing execution
        choice, unsupported capability, or unsupported workload modality.
        BaseException: Existing Execute timeout, crash, uncertainty, and
            cleanup failures without a trustworthy report remain unchanged.

    Side Effects:
        Takes one locked configuration snapshot, resolves selectors and core
        controls before declaration probing and runs only declaration analysis.
        It never invokes ``fn``, opens/exports a Store, starts a workload
        backend, or reserves workload resources.
    """

    options = _state._effective_options(view)
    _validate_choice(options, operation)
    description = describe_callable(fn)
    _validate_modality(description)
    # Local execution must not resolve ambient core settings: it only rejects
    # explicit worker controls and leaves the caller's session authoritative.
    core_controls = (
        None
        if type(options.backend) is InProcess
        else _capture_frozen_core_controls(options.core, executor_core=None)
    )
    try:
        selection = (
            None
            if options.python is None
            else resolve_environment_spec(options.python)
        )
    except EnvironmentSpecError:
        return _report(
            options,
            probe=None,
            eligible=False,
            reason="selected environment is unavailable",
            diagnostics=("dispatch.selector_invalid",),
        )
    try:
        try:
            prepared = prepare_probe(fn, options=options.probe)
        except EnvironmentSpecError:
            return _report(
                options,
                probe=None,
                eligible=False,
                reason="selected probe environment is unavailable",
                diagnostics=("dispatch.probe_selector_invalid",),
            )
        result = execute_prepared_probe(prepared)
    except (
        ProbePlacementError,
        ProbeProtocolError,
        CodeAnalysisError,
    ) as error:
        reason = (
            str(error)
            if isinstance(error, ProbePlacementError)
            else "Dispatch probe failed"
        )
        diagnostic = (
            error.code
            if isinstance(error, CodeAnalysisError)
            else "dispatch.probe_failed"
        )
        return _report(
            options,
            probe=None,
            eligible=False,
            reason=reason,
            diagnostics=(diagnostic,),
        )
    environment = _combine_results(
        options.environment, result.environment, owner="environment"
    )
    world = _combine_results(options.world, result.world, owner="world")
    if not environment.ok or not world.ok:
        outcome_diagnostics = ("dispatch.requirements_conflict",)
        eligible = False
    elif type(options.backend) is InProcess:
        # Explanation observes with a short lease; local execution acquires a
        # separate lease after probe cleanup and rechecks before direct call.
        with publication.lease() as generation:
            outcome = check_current_compatibility(
                options.core, environment, world, selection, generation
            )
        eligible = outcome.eligible
        outcome_diagnostics = outcome.diagnostics
    else:
        assert options.backend is not None
        outcome = check_execute_compatibility(
            options.backend, environment, world, selection
        )
        eligible = outcome.eligible
        outcome_diagnostics = outcome.diagnostics
    reason = (
        "requirements are compatible"
        if eligible
        else "configured and discovered requirements are incompatible"
    )
    report = _report(
        options,
        probe=result,
        eligible=eligible,
        reason=reason,
        diagnostics=outcome_diagnostics,
        environment=environment,
        world=world,
    )
    if not eligible:
        return report
    try:
        prepared.validate()
    except ProbePlacementError:
        return _report(
            options,
            probe=result,
            eligible=False,
            reason="probe target changed during compatibility discovery",
            diagnostics=("dispatch.target_changed",),
            environment=environment,
            world=world,
        )
    return _Preflight(
        options,
        core_controls,
        selection,
        prepared,
        report,
    )


__all__ = ["_Preflight", "preflight"]
