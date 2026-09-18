"""Independent placement and lifecycle ownership for projected probes."""

from __future__ import annotations

import contextvars
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from dryml.code.inspection import (
    InspectionCapture,
    InspectionOwnerFacts,
    _capture_inspection,
    capture_inspection,
)
from dryml.code.errors import InvalidTargetError
from dryml.core._callable_inspection import describe_callable
from dryml.environments import inspect_current
from dryml.environments.selection import (
    ResolvedEnvironmentSelection,
    compare_selection,
    resolve_environment_spec,
)
from dryml.environments.kernel import (
    EnvironmentRequirementsKernel,
    _view_to_data as _environment_view_to_data,
)
from dryml.execute.config import BackendConfig
from dryml.execute.executor import submit as execute_submit
from dryml.execute.subprocess import SubProcessConfig
from dryml.requirements import RequirementResult
from dryml.runtime import publication
from dryml.session import snapshot_for_generation
from dryml.worlds import check_selected_process_satisfies_requirement
from dryml.worlds.kernel import (
    WorldRequirementsKernel,
    _view_to_data as _world_view_to_data,
)

from ._probe_protocol import (
    ProbeProtocolError,
    ProbeWireResult,
    build_request,
    decode_result,
    execute_probe,
)
from .models import ProbeOptions


class ProbePlacementError(RuntimeError):
    """Report that an explicitly selected probe placement is unavailable.

    The error contains only a stable placement reason. It does not report
    caller arguments, targets, source paths, or environment data.
    """


@dataclass(frozen=True, slots=True)
class ProbeCoverage:
    """Bounded static-dependency coverage returned by one declaration probe.

    Args:
        complete: Whether static resolution proved reachable supported edges.
        diagnostics: Bounded framework coverage categories without source text.

    Side Effects:
        None. Coverage is analysis evidence, not workload admission or a
        resource reservation.
    """

    complete: bool
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Detached typed declaration result from one completed probe lifecycle.

    Args:
        placement: Actual inline or Execute placement.
        coverage: Bounded static traversal evidence.
        environment: Existing combined environment requirement result.
        world: Existing combined world requirement result.

    Side Effects:
        None. This result contains no callable, Store, backend, future, source,
        or reservation handle.
    """

    placement: Literal["in_process", "execute"]
    coverage: ProbeCoverage
    environment: RequirementResult[Any]
    world: RequirementResult[Any]


@dataclass(frozen=True, slots=True)
class _PreparedProbe:
    """Retain call-entry probe facts and its local drift guard.

    This package-internal carrier lets Dispatch preflight retain one capture
    through later workload preparation. It never crosses Execute, appears in a
    result/report, or owns a workload lifecycle.
    """

    capture: InspectionCapture
    request: object
    environment_view: object
    world_view: object
    options: ProbeOptions
    selection: ResolvedEnvironmentSelection | None
    inline_deadline: float

    def validate(self) -> None:
        """Reject observed code or declaration drift from call-entry facts."""

        try:
            self.capture.validate()
        except InvalidTargetError as error:
            raise ProbePlacementError(
                "probe target changed during inspection") from error
        if _views(self.capture) != (self.environment_view, self.world_view):
            raise ProbePlacementError(
                "probe target declarations changed during inspection"
            )


def _views(capture: object) -> tuple[object, object]:
    """Project typed owner views from the same local inspection capture."""

    environment = EnvironmentRequirementsKernel._from_capture(
        capture
    )  # type: ignore[arg-type]
    world = WorldRequirementsKernel._from_capture(
        capture
    )  # type: ignore[arg-type]
    return _environment_view_to_data(environment._view), _world_view_to_data(
        world._view
    )


def _validate_owner_carriers(
    capture: object, carriers: tuple[object, ...]
) -> None:
    """Reject known core declarations unavailable through the capture root."""

    if type(capture) is not InspectionCapture or any(
            not capture._has_local_carrier(carrier) for carrier in carriers):
        raise ProbePlacementError(
            "probe could not preserve a recognized callable declaration layer"
        )


def _capture_description(description: object) -> InspectionCapture:
    """Capture wrappers and core-described raw roots through generic facts.

    Core's passive description is the only owner relationship Dispatch
    translates.
    When source-free capture cannot prove that relationship itself, the
    bridge supplies only core-proven relationship facts to generic capture. The
    generic owner performs one bounded candidate expansion and never reads
    private association state or guesses arbitrary decorator metadata.
    """

    original = description.original  # type: ignore[attr-defined]
    raw_target = description.raw_target  # type: ignore[attr-defined]
    if description.owner != "function":  # type: ignore[attr-defined]
        return capture_inspection(original)
    carriers = description.declaration_carriers  # type: ignore[attr-defined]
    related = tuple(carriers)
    if not any(item is raw_target for item in related):
        related += (raw_target,)
    # Core function wrappers are invocation plumbing. Their declarations remain
    # visible, while their implementation body cannot create dependency edges.
    opaque = tuple(item for item in related if item is not raw_target)
    return _capture_inspection(
        original,
        owner_facts=InspectionOwnerFacts(original, related, opaque),
    )


def _inline_compatible(
    options: ProbeOptions, selection: ResolvedEnvironmentSelection | None
) -> bool:
    """Check bootstrap constraints against a held current generation.

    The check borrows session/runtime evidence. It never activates a framework,
    changes a Session, allocates a world, or substitutes a selector.
    """

    if options.backend is not None:
        return False
    try:
        with publication.lease() as generation:
            return _inline_generation_compatible(
                options, selection, generation
            )
    except Exception:
        return False


def _inline_generation_compatible(
    options: ProbeOptions,
    selection: ResolvedEnvironmentSelection | None,
    generation: object,
) -> bool:
    """Check inline evidence while a caller holds its generation lease."""

    snapshot = snapshot_for_generation(generation)  # type: ignore[arg-type]
    if snapshot.health != "healthy" or snapshot.mode == "orchestrator":
        return False
    record = None
    if options.environment is not None or selection is not None:
        record = inspect_current()
    if (
        options.environment is not None
        and not options.environment.check(record, policy="strict").admission_ok
    ):
        return False
    if selection is not None:
        selection_report = compare_selection(selection, record)
        # Selection comparison is owner evidence rather than requirement-policy
        # evaluation, so its compatible empty report is affirmative admission.
        if not selection_report.ok or selection_report.issues:
            return False
    allocation = snapshot.allocation
    return check_selected_process_satisfies_requirement(
        None if allocation is None else allocation.role,
        None if allocation is None else allocation.process,
        options.world,
    ).admission_ok


def _placement(
    options: ProbeOptions, selection: ResolvedEnvironmentSelection | None
) -> tuple[Literal["in_process", "execute"], BackendConfig | None]:
    """Choose one probe owner without borrowing workload configuration."""

    backend = options.backend
    if type(backend) is str:
        raise ProbePlacementError(
            "probe backend names require the Dispatch registry"
        )
    if options.placement == "in_process":
        if backend is not None:
            raise ProbePlacementError(
                "inline probe placement cannot use an Execute backend"
            )
        if not _inline_compatible(options, selection):
            raise ProbePlacementError(
                "current process is not compatible with inline probing"
            )
        return "in_process", None
    if backend is not None:
        return "execute", backend
    if options.placement == "execute":
        return "execute", SubProcessConfig()
    if _inline_compatible(options, selection):
        return "in_process", None
    return "execute", SubProcessConfig()


def _check_inline_timeout(deadline: float) -> None:
    """Raise at one cooperative passive-analysis boundary when overdue."""

    if time.monotonic() > deadline:
        raise TimeoutError("inline probe exceeded its cooperative timeout")


def _fresh_inline(request: bytes, deadline: float) -> bytes:
    """Run the fixed worker entry in a new logical Context with bounds."""

    _check_inline_timeout(deadline)
    return contextvars.Context().run(execute_probe, request, deadline=deadline)


def _execute_remote(
    request: bytes,
    options: ProbeOptions,
    backend: BackendConfig,
    selection: ResolvedEnvironmentSelection | None = None,
) -> bytes:
    """Retrieve a one-off Execute result, then require cleanup before use."""

    future = execute_submit(
        execute_probe,
        request,
        backend=backend,
        environment=options.environment,
        environment_spec=selection,
        world=options.world,
        execution_timeout=options.execution_timeout,
    )
    result: bytes | None = None
    failure: BaseException | None = None
    try:
        result = future.result()
    except BaseException as exc:
        failure = exc
    try:
        future.cleanup()
    except BaseException as cleanup:
        if failure is not None:
            raise failure from cleanup
        raise
    if failure is not None:
        raise failure
    if type(result) is not bytes:
        raise ProbeProtocolError("probe Execute result is invalid")
    return result


def prepare_probe(
    target: Callable[..., Any], *, options: ProbeOptions | None = None
) -> _PreparedProbe:
    """Capture one retained probe request and its local validation guard.

    This internal seam resolves an exact selector once at call entry. It checks
    the closed request before an Execute backend can be created.
    """

    effective = ProbeOptions() if options is None else options
    if type(effective) is not ProbeOptions:
        raise TypeError("probe options must be ProbeOptions or None")
    deadline = time.monotonic() + effective.execution_timeout
    selection = (
        None
        if effective.environment_spec is None
        else resolve_environment_spec(effective.environment_spec)
    )
    description = describe_callable(target)
    if effective.backend is None and effective.placement != "execute":
        _check_inline_timeout(deadline)
    capture = _capture_description(description)
    if effective.backend is None and effective.placement != "execute":
        _check_inline_timeout(deadline)
    _validate_owner_carriers(capture, description.declaration_carriers)
    capture.validate()
    environment_view, world_view = _views(capture)
    request = build_request(
        capture.target,
        environment_view=environment_view,
        world_view=world_view,
        max_targets=effective.max_targets,
        max_depth=effective.max_depth,
    )
    if effective.backend is None and effective.placement != "execute":
        _check_inline_timeout(deadline)
    return _PreparedProbe(
        capture,
        request,
        environment_view,
        world_view,
        effective,
        selection,
        deadline,
    )


def execute_prepared_probe(prepared: _PreparedProbe) -> ProbeResult:
    """Run one prepared probe while retaining its guard for later preflight."""

    if type(prepared) is not _PreparedProbe:
        raise TypeError("prepared probe is invalid")
    prepared.validate()
    placement, backend = _placement(prepared.options, prepared.selection)
    if placement == "in_process":
        # Keep the generation healthy and compatible through the fixed DAG.
        with publication.lease() as generation:
            if not _inline_generation_compatible(
                prepared.options, prepared.selection, generation
            ):
                raise ProbePlacementError(
                    "current process is not compatible with inline probing"
                )
            raw = _fresh_inline(
                prepared.request.data, prepared.inline_deadline
            )
    else:
        assert backend is not None
        if prepared.selection is None:
            raw = _execute_remote(
                prepared.request.data, prepared.options, backend
            )
        else:
            raw = _execute_remote(
                prepared.request.data,
                prepared.options,
                backend,
                prepared.selection,
            )
    prepared.validate()
    decoded: ProbeWireResult = decode_result(raw, prepared.request)
    return ProbeResult(
        placement,
        ProbeCoverage(decoded.complete, decoded.diagnostics),
        decoded.environment,
        decoded.world,
    )


def run_probe(
    target: Callable[..., Any], *, options: ProbeOptions | None = None
) -> ProbeResult:
    """Capture and run one declaration probe with independent placement policy.

    Args:
        target: Actual callable invocation root retained by the workload path.
        options: Optional immutable independent controls. Omitting it uses
            the 30-second, 256-target, depth-32 defaults.

    Returns:
        Typed combined requirement results and bounded coverage evidence.

    Raises:
        ProbePlacementError: If inline or a named/explicit backend cannot
            satisfy the selected probe placement.
        ProbeProtocolError: If capture projections, worker transport, result
            association, or ordinary scheduler outcomes are invalid.
        CleanupError: If generic Execute cleanup is incomplete; its retryable
            ``execution`` Future is preserved and no result is consumed.
        BaseException: Existing Execute timeout, crash, or worker error. It is
            not converted to incomplete discovery.

    Side Effects:
        Passively captures source-free static facts and runs only the generic
        analysis kernel DAG. It may create a separately owned one-off Execute
        probe but never invokes ``target``, opens a Store, or creates work.
    """

    return execute_prepared_probe(prepare_probe(target, options=options))


__all__ = [
    "ProbeCoverage",
    "ProbePlacementError",
    "ProbeResult",
    "execute_prepared_probe",
    "prepare_probe",
    "run_probe",
]
