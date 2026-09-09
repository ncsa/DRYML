"""Fresh owner-checked feasibility and final admission for Execute."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Event
from time import monotonic

from dryml.environments import CompatibilityReport, EnvironmentRecord, EnvironmentRequirement
from dryml.requirements import RequirementBarrierError, RequirementError, require_admission
from dryml.worlds import LocalResourceInventory, WorldAllocation, WorldCompatibilityReport, WorldRequirement, WorldSpec, assign_local_world, synthesize
from dryml.worlds.compatibility import check_allocation_satisfies_requirement, check_world_spec_satisfies_requirement

from .models import AdmissionReport, ExecutionIssue


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Describe non-authorizing feasibility or final actual-worker admission.

    ``go`` is only true for a fresh, associated actual-worker grant. This module
    never emits a transport GO message; the Future owns its final pre-load gate.
    ``feasible`` describes a side-effect-free plan and is never a reservation.
    """

    go: bool
    report: AdmissionReport
    world: WorldSpec | None
    allocation: WorldAllocation | None
    feasible: bool = False

    @property
    def ok(self) -> bool:
        """Return whether final actual-worker admission is presently authorized."""
        return self.go


def plan_admission(
    *,
    environment: EnvironmentRequirement | None = None,
    world: WorldRequirement | None = None,
    record: EnvironmentRecord | None = None,
    inventory: LocalResourceInventory | None = None,
    deadline: float,
    cancelled: Event | None = None,
) -> AdmissionDecision:
    """Evaluate requirements without reserving or authorizing any worker.

    Args:
        environment: Optional owner-combined environment requirement.
        world: Optional owner-combined world requirement.
        record: Candidate environment evidence used only for feasibility.
        inventory: Candidate resource evidence used only for synthesis.
        deadline: Finite absolute monotonic planning deadline.
        cancelled: Optional cancellation signal.

    Returns:
        A non-authorizing decision. ``feasible`` is true only when all supplied
        owner checks are compatible; ``go`` is always false.

    Side Effects:
        None. This function does not reserve, launch, associate, or signal a
        worker.
    """
    issues = _gate_issues(deadline, cancelled)
    environment_report: CompatibilityReport | None = None
    world_report: WorldCompatibilityReport | None = None
    selected_world: WorldSpec | None = None
    selected_allocation: WorldAllocation | None = None
    if issues:
        return AdmissionDecision(False, AdmissionReport(None, None, tuple(issues)), None, None, False)
    if environment is not None:
        if record is None:
            issues.append(ExecutionIssue("environment_evidence_unavailable", "candidate environment evidence is unavailable"))
        else:
            environment_report, environment_issue = _environment_report(environment, record)
            if environment_issue is not None:
                issues.append(environment_issue)
    if world is not None:
        shape_issue = _shape_issue(world)
        if shape_issue is not None:
            issues.append(shape_issue)
        elif inventory is None:
            issues.append(ExecutionIssue("world_evidence_unavailable", "candidate resource evidence is unavailable"))
        else:
            result = synthesize(world, inventory=inventory)
            selected_world = result.world
            world_report = result.compatibility
            if not result.ok or selected_world is None:
                issues.append(ExecutionIssue("world_incompatible", "world requirement cannot be satisfied by current evidence"))
            else:
                try:
                    selected_allocation = assign_local_world(selected_world, inventory=inventory)
                    world_report = check_allocation_satisfies_requirement(selected_allocation, world)
                    require_admission(world_report, operation="execute feasibility")
                except RequirementBarrierError:
                    issues.append(ExecutionIssue("world_incompatible", "candidate resource assignment does not satisfy the requirement"))
                except Exception:
                    issues.append(ExecutionIssue("world_evidence_unavailable", "candidate resource assignment could not be verified"))
    issues.extend(_gate_issues(deadline, cancelled))
    return AdmissionDecision(False, AdmissionReport(environment_report, world_report, tuple(issues)), selected_world, selected_allocation, not issues)


def admit(
    *,
    environment: EnvironmentRequirement | None = None,
    world: WorldRequirement | None = None,
    record: EnvironmentRecord | None = None,
    inventory: LocalResourceInventory | None = None,
    allocation: WorldAllocation | None = None,
    deadline: float,
    cancelled: Event | None = None,
) -> AdmissionDecision:
    """Check final actual-worker evidence without emitting a transport GO.

    Args:
        environment: Optional owner-combined environment requirement.
        world: Optional owner-combined world requirement.
        record: Fresh actual-worker environment record.
        inventory: Optional feasibility evidence when no grant is available.
        allocation: Fresh owner allocation associated with the actual worker.
        deadline: Finite absolute monotonic final-admission deadline.
        cancelled: Optional cancellation signal.

    Returns:
        A final decision. Inventory-only success is represented as a feasible
        plan with ``go`` false; only an actual, control-proven allocation can set
        ``go`` true.

    Side Effects:
        None. Future transport performs the final authorization fence.
    """
    issues = _gate_issues(deadline, cancelled)
    environment_report: CompatibilityReport | None = None
    world_report: WorldCompatibilityReport | None = None
    selected_world: WorldSpec | None = None
    planned_allocation: WorldAllocation | None = None
    if issues:
        return AdmissionDecision(False, AdmissionReport(None, None, tuple(issues)), None, allocation, False)
    if environment is not None:
        if record is None:
            issues.append(ExecutionIssue("environment_evidence_unavailable", "actual worker environment evidence is unavailable"))
        else:
            environment_report, environment_issue = _environment_report(environment, record)
            if environment_issue is not None:
                issues.append(environment_issue)
    if world is not None:
        shape_issue = _shape_issue(world)
        if shape_issue is not None:
            issues.append(shape_issue)
        elif allocation is None:
            plan = plan_admission(world=world, inventory=inventory, deadline=deadline, cancelled=cancelled)
            world_report = plan.report.world
            selected_world, planned_allocation = plan.world, plan.allocation
            issues.extend(plan.report.issues)
            if plan.feasible:
                issues.append(ExecutionIssue("world_grant_unavailable", "feasibility is not an actual worker grant"))
        else:
            control_issue = _control_issue(world, allocation)
            if control_issue is not None:
                issues.append(control_issue)
            world_report = check_allocation_satisfies_requirement(allocation, world)
            try:
                require_admission(world_report, operation="execute admission")
            except RequirementBarrierError:
                issues.append(ExecutionIssue("world_incompatible", "actual worker allocation does not satisfy the requirement"))
            except RequirementError:
                issues.append(ExecutionIssue("world_evidence_unavailable", "actual worker allocation report is not admissible"))
    if environment is not None and world is not None and record is not None and allocation is not None:
        if allocation.backend.get("environment_id") != record.semantic_id:
            issues.append(ExecutionIssue("worker_environment_unassociated", "worker allocation is not associated with its checked environment"))
    issues.extend(_gate_issues(deadline, cancelled))
    return AdmissionDecision(not issues, AdmissionReport(environment_report, world_report, tuple(issues)), selected_world, allocation if allocation is not None else planned_allocation, False)


def _admit_observed_logical(
    *,
    environment: EnvironmentRequirement | None = None,
    world: WorldRequirement | None = None,
    record: EnvironmentRecord | None = None,
    observed_world: WorldSpec | None = None,
    controls: Mapping[str, str] | None = None,
    deadline: float,
) -> AdmissionDecision:
    """Admit a backend's actual logical grant without inventing exact bindings.

    Ray exposes scheduler-assigned logical CPU, memory, and accelerator amounts,
    but not an exact CPU allocation compatible with :class:`WorldAllocation`.
    This private helper preserves owner requirement/barrier evaluation for that
    evidence while keeping ``allocation`` absent in the returned decision.

    Args:
        environment: Optional actual-worker environment requirement.
        world: Optional single-process world requirement.
        record: Fresh actual-worker environment record.
        observed_world: World shape reconstructed from the backend's raw grant.
        controls: Backend-proven logical controls by resource family.
        deadline: Absolute finite monotonic admission deadline.

    Returns:
        A final admission decision with no fabricated exact allocation.
    """
    issues = _gate_issues(deadline, None)
    environment_report: CompatibilityReport | None = None
    world_report: WorldCompatibilityReport | None = None
    if environment is not None:
        if record is None:
            issues.append(ExecutionIssue("environment_evidence_unavailable", "actual worker environment evidence is unavailable"))
        else:
            environment_report, environment_issue = _environment_report(environment, record)
            if environment_issue is not None:
                issues.append(environment_issue)
    if world is not None:
        shape_issue = _shape_issue(world)
        if shape_issue is not None:
            issues.append(shape_issue)
        elif observed_world is None:
            issues.append(ExecutionIssue("world_grant_unavailable", "actual logical worker grant is unavailable"))
        else:
            world_report = check_world_spec_satisfies_requirement(observed_world, world)
            try:
                require_admission(world_report, operation="execute logical admission")
            except RequirementBarrierError:
                issues.append(ExecutionIssue("world_incompatible", "actual logical worker grant does not satisfy the requirement"))
            except RequirementError:
                issues.append(ExecutionIssue("world_evidence_unavailable", "actual logical worker grant is not admissible"))
            requested = next(iter(world.roles.values())).resources
            logical_controls = controls if isinstance(controls, Mapping) else {}
            if (requested.cpus.min is not None or requested.cpus.max is not None) and logical_controls.get("cpus") != "logical":
                issues.append(ExecutionIssue("world_cpu_control_unsupported", "backend cannot prove its logical CPU grant"))
            if requested.accelerators and logical_controls.get("accelerators") != "logical":
                issues.append(ExecutionIssue("world_accelerator_control_unsupported", "backend cannot prove assigned accelerator identities"))
            if (requested.memory.min is not None or requested.memory.max is not None) and logical_controls.get("memory") != "logical":
                issues.append(ExecutionIssue("world_memory_control_unsupported", "backend cannot prove its logical memory grant"))
    issues.extend(_gate_issues(deadline, None))
    return AdmissionDecision(not issues, AdmissionReport(environment_report, world_report, tuple(issues)), observed_world, None, False)


def _gate_issues(deadline: float, cancelled: Event | None) -> list[ExecutionIssue]:
    """Return final gate failures without allowing malformed deadlines through."""
    issues: list[ExecutionIssue] = []
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        return [ExecutionIssue("admission_deadline_invalid", "admission deadline must be a finite monotonic timestamp")]
    if cancelled is not None and cancelled.is_set():
        issues.append(ExecutionIssue("admission_cancelled", "admission was cancelled before authorization"))
    if monotonic() >= deadline:
        issues.append(ExecutionIssue("admission_expired", "admission deadline elapsed before authorization"))
    return issues


def _environment_report(requirement: EnvironmentRequirement, record: EnvironmentRecord) -> tuple[CompatibilityReport, ExecutionIssue | None]:
    """Run the environment owner's compatibility and barrier semantics exactly."""
    report = requirement.check(record, policy="compatible")
    try:
        require_admission(report, operation="execute admission")
    except RequirementBarrierError:
        return report, ExecutionIssue("environment_incompatible", "actual worker environment does not satisfy the requirement")
    except RequirementError:
        return report, ExecutionIssue("environment_evidence_unavailable", "actual worker environment report is not admissible")
    return report, None


def _shape_issue(requirement: WorldRequirement) -> ExecutionIssue | None:
    """Reject shapes Execute cannot represent before backend-specific controls."""
    if len(requirement.roles) != 1:
        return ExecutionIssue("world_multi_role_unsupported", "one Execute submission must have exactly one role")
    role = next(iter(requirement.roles.values()))
    if not role.replicas.satisfied_by(1):
        return ExecutionIssue("world_replicas_unsupported", "one Execute submission must admit exactly one replica")
    if role.topology:
        return ExecutionIssue("world_topology_unsupported", "topology controls are unsupported for one-process Execute admission")
    resources = role.resources
    if resources.devices or resources.named:
        return ExecutionIssue("world_named_resource_unsupported", "named and device resource controls are unsupported")
    if resources.accelerator_memory:
        return ExecutionIssue("world_accelerator_memory_unsupported", "per-accelerator memory enforcement is unsupported")
    return None


def _control_issue(requirement: WorldRequirement, allocation: WorldAllocation) -> ExecutionIssue | None:
    """Require exact owner-declared applied controls for constrained resources."""
    role = next(iter(requirement.roles.values()))
    resources = role.resources
    controls = allocation.backend.get("execute_controls")
    controls = controls if isinstance(controls, Mapping) else {}
    if (resources.cpus.min is not None or resources.cpus.max is not None) and controls.get("cpus") != "applied":
        return ExecutionIssue("world_cpu_control_unsupported", "backend cannot prove applied CPU controls")
    if resources.accelerators and controls.get("accelerators") != "applied":
        return ExecutionIssue("world_accelerator_control_unsupported", "backend cannot prove applied accelerator controls")
    if resources.memory.min is not None or resources.memory.max is not None:
        if allocation.backend.get("kind") == "subprocess":
            return ExecutionIssue("world_memory_control_unsupported", "subprocess memory hard enforcement is not proven")
        if controls.get("memory") != "logical":
            return ExecutionIssue("world_memory_control_unsupported", "backend cannot prove its logical memory grant")
    return None


__all__ = ["AdmissionDecision", "admit", "plan_admission"]
