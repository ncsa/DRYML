"""Dispatch compatibility checks over caller-owned current-process evidence.

The current-process checker consumes one publication generation already held by
its caller. This lets explanation observe under a short lease while local
execution retains the same generation through final guards and direct workload
invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from dryml.environments import EnvironmentRequirement, inspect_current
from dryml.environments.selection import (
    ResolvedEnvironmentSelection,
    compare_selection,
)
from dryml.execute.config import BackendConfig
from dryml.execute.errors import CleanupError
from dryml.execute.executor import Executor
from dryml.requirements import RequirementResult
from dryml.runtime.publication import SessionGeneration
from dryml.session import snapshot_for_generation
from dryml.worlds import (
    WorldRequirement,
    check_selected_process_satisfies_requirement,
)


@dataclass(frozen=True, slots=True)
class CompatibilityOutcome:
    """Hold bounded non-reserving Dispatch compatibility evidence.

    Args:
        eligible: Whether the observed selected target satisfies every hard
            requirement.
        diagnostics: Stable owner-safe outcome codes only.

    Side Effects:
        None. This carrier owns no backend, allocation, Store, or lease.
    """

    eligible: bool
    diagnostics: tuple[str, ...] = ()


def check_execute_compatibility(
    backend_config: BackendConfig,
    environment: RequirementResult[EnvironmentRequirement],
    world: RequirementResult[WorldRequirement],
    selection: ResolvedEnvironmentSelection | None,
) -> CompatibilityOutcome:
    """Discover compatibility through exactly one selected Execute backend.

    Args:
        backend_config: Frozen configuration for the selected workload backend.
        environment: Already-combined environment requirement result.
        world: Already-combined world requirement result.
        selection: Already-resolved exact environment pin, if supplied.

    Returns:
        An affirmative outcome only when complete discovery supplies compatible
        owner evidence. Discovery feasibility is never a reservation.

    Raises:
        CleanupError: If the independent discovery backend cannot release its
            owned lifecycle resources.

    Side Effects:
        Creates, starts, discovers through, and closes only the selected
        backend.
        It never submits workload code, creates a Store, or reserves resources.
    """

    if not environment.ok or not world.ok:
        return CompatibilityOutcome(False, ("dispatch.requirements_conflict",))
    executor = None
    try:
        executor = Executor(backend_config)
        snapshot = executor.discover(
            environment=environment.value,
            environment_spec=selection,
            world=world.value,
            timeout=backend_config.discovery_timeout,
        )
    except CleanupError:
        raise
    except Exception:
        return CompatibilityOutcome(
            False, ("dispatch.backend_discovery_unavailable",)
        )
    finally:
        if executor is not None:
            executor.close(
                cancel=False, timeout=backend_config.termination_timeout
            )
    return _check_discovery(
        snapshot, environment.value, world.value, selection
    )


def check_current_compatibility(
    core: object | None,
    environment: RequirementResult[EnvironmentRequirement],
    world: RequirementResult[WorldRequirement],
    selection: ResolvedEnvironmentSelection | None,
    generation: SessionGeneration,
) -> CompatibilityOutcome:
    """Check fresh current-process evidence for one held generation.

    Args:
        core: Dispatch-level core override, if any.
        environment: Already-combined environment requirement result.
        world: Already-combined world requirement result.
        selection: Already-resolved exact current-environment assertion.
        generation: Exact publication generation held by the caller.

    Returns:
        Compatibility evidence derived only from ``generation``.

    Side Effects:
        May inspect current environment evidence. It does not read the current
        publication, acquire a lease, activate runtime controls, allocate
        resources, or construct Execute/core execution objects.
    """

    if not environment.ok or not world.ok:
        return CompatibilityOutcome(False, ("dispatch.requirements_conflict",))
    if _has_core_override(core):
        return CompatibilityOutcome(
            False, ("dispatch.in_process_core_override",)
        )
    try:
        if not isinstance(generation, SessionGeneration):
            raise TypeError("generation must be a held SessionGeneration")
        snapshot = snapshot_for_generation(generation)
        if snapshot.health != "healthy":
            return CompatibilityOutcome(
                False, ("dispatch.in_process_unhealthy",)
            )
        if snapshot.mode == "orchestrator":
            return CompatibilityOutcome(
                False, ("dispatch.in_process_orchestrator",)
            )
        record = None
        if environment.value is not None or selection is not None:
            record = inspect_current()
        if environment.value is not None:
            report = environment.value.check(record, policy="strict")
            if not report.admission_ok:
                return CompatibilityOutcome(
                    False,
                    ("dispatch.in_process_environment_incompatible",),
                )
        if selection is not None:
            report = compare_selection(selection, record)
            if not report.ok or report.issues:
                return CompatibilityOutcome(
                    False, ("dispatch.in_process_selection_mismatch",)
                )
        allocation = snapshot.allocation
        report = check_selected_process_satisfies_requirement(
            None if allocation is None else allocation.role,
            None if allocation is None else allocation.process,
            world.value,
        )
        if not report.admission_ok:
            return CompatibilityOutcome(
                False, ("dispatch.in_process_world_incompatible",)
            )
    except Exception:
        return CompatibilityOutcome(
            False, ("dispatch.in_process_evidence_unavailable",)
        )
    return CompatibilityOutcome(True)


def _check_discovery(
    snapshot: Any,
    environment: EnvironmentRequirement | None,
    world: WorldRequirement | None,
    selection: ResolvedEnvironmentSelection | None,
) -> CompatibilityOutcome:
    """Require complete affirmative owner evidence from discovery."""

    # Discovery still proves that the selected backend can be observed when no
    # hard requirement needs owner evidence. Its optional inventory may be
    # incomplete without making an unconstrained workload incompatible.
    if environment is None and world is None and selection is None:
        return CompatibilityOutcome(True)
    if not getattr(snapshot, "complete", False) or getattr(
        snapshot, "issues", ()
    ):
        return CompatibilityOutcome(
            False, ("dispatch.backend_discovery_incomplete",)
        )
    candidates = tuple(getattr(snapshot, "environments", ()))
    compatible_keys: set[str] = set()
    if environment is not None or selection is not None:
        for candidate in candidates:
            if getattr(candidate, "launchable", None) is not True:
                continue
            record = getattr(candidate, "record", None)
            if record is None:
                continue
            candidate_report = getattr(candidate, "report", None)
            if (
                candidate_report is not None
                and not getattr(candidate_report, "admission_ok", False)
            ):
                continue
            if environment is not None and not environment.check(
                record, policy="strict"
            ).admission_ok:
                continue
            if selection is not None:
                selection_report = compare_selection(selection, record)
                if not selection_report.ok or selection_report.issues:
                    continue
            compatible_keys.add(getattr(candidate, "key", ""))
        if not compatible_keys:
            code = (
                "dispatch.backend_selection_mismatch"
                if selection is not None
                else "dispatch.backend_environment_incompatible"
            )
            return CompatibilityOutcome(False, (code,))
    if world is None:
        return CompatibilityOutcome(True)
    for plan in tuple(getattr(snapshot, "plans", ())):
        report = getattr(plan, "report", None)
        world_report = (
            None if report is None else getattr(report, "world", None)
        )
        environment_report = (
            None if report is None else getattr(report, "environment", None)
        )
        if (
            report is not None
            and not getattr(report, "issues", ())
            and world_report is not None
            and getattr(world_report, "admission_ok", False)
            and (
                environment is None
                or (
                    environment_report is not None
                    and getattr(environment_report, "admission_ok", False)
                )
            )
            and (
                not compatible_keys
                or getattr(plan, "environment_key", None) in compatible_keys
            )
        ):
            return CompatibilityOutcome(True)
    return CompatibilityOutcome(
        False, ("dispatch.backend_world_incompatible",)
    )


def _has_core_override(core: object | None) -> bool:
    """Return whether a local route received any worker-only core field.

    Every present :class:`~dryml.core.execute.CoreOptions` field is inspected
    so a future worker option cannot become silently inert for direct
    execution.
    """

    if core is None:
        return False
    from dryml.core.execute import CoreOptions

    if not isinstance(core, CoreOptions):
        return True
    return any(
        getattr(core, field.name) != "inherit" for field in fields(core)
    )


__all__ = [
    "CompatibilityOutcome",
    "check_current_compatibility",
    "check_execute_compatibility",
]
