"""Immutable contract values for the additive Execute implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from dryml.environments import CompatibilityReport, EnvironmentRecord, EnvironmentRequirement
from dryml.environments.specs import EnvironmentSpec
from dryml.worlds import LocalResourceInventory, WorldAllocation, WorldCompatibilityReport, WorldRequirement, WorldSpec

if TYPE_CHECKING:
    from dryml.execute.output import ExecutionOutput


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ExecutionIssue:
    """Hold one bounded Execute diagnostic.

    Args:
        code: Stable machine-oriented issue identifier.
        message: Bounded safe explanation retained for the coordinator.

    Side Effects:
        None. This immutable value never retains workload payloads or secrets.
    """

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AdmissionReport:
    """Aggregate environment/world admission evidence and Execute diagnostics.

    Args:
        environment: Optional immutable environment compatibility report.
        world: Optional immutable world compatibility report.
        issues: Bounded Execute-owned issues, copied to an immutable tuple.

    Side Effects:
        Construction detaches the issue sequence; it neither admits work nor
        alters owner-domain reports.
    """

    environment: CompatibilityReport | None = None
    world: WorldCompatibilityReport | None = None
    issues: tuple[ExecutionIssue, ...] = ()

    def __post_init__(self) -> None:
        """Detach the issue sequence into an immutable tuple."""
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True)
class PayloadSpool:
    """Describe one immutable validated coordinator-owned invocation snapshot.

    Attributes:
        path: Private spool path, valid only through its submission cleanup.
        size_bytes: Serialized payload size after validation.
        sha256: Coordinator integrity digest.
        serializer: Serializer name and version used for the payload.
        python_implementation: Producing Python implementation name.
        python_version: Producing Python major, minor, and patch tuple.
        pickle_protocol: Serializer protocol number.

    Side Effects:
        None. The descriptor does not grant durable file ownership or transport.
    """

    path: Path
    size_bytes: int
    sha256: str
    serializer: str
    serializer_version: str
    python_implementation: str
    python_version: tuple[int, int, int]
    pickle_protocol: int


@dataclass(frozen=True, slots=True)
class ResultSpool:
    """Describe one immutable validated coordinator-owned result snapshot.

    Attributes:
        path: Private result spool path, valid only through submission cleanup.
        size_bytes: Validated serialized result size.
        sha256: Coordinator integrity digest.
        serializer: Serializer name and version used for the result.
        python_implementation: Producing worker Python implementation name.
        python_version: Producing worker Python major, minor, and patch tuple.
        pickle_protocol: Serializer protocol number.

    Side Effects:
        Result files share a submission-owned spool child with the invocation and
        remain private descriptors, not durable references or caller file rights.
    """

    path: Path
    size_bytes: int
    sha256: str
    serializer: str
    serializer_version: str
    python_implementation: str
    python_version: tuple[int, int, int]
    pickle_protocol: int


@dataclass(frozen=True, slots=True)
class OutputSnapshot:
    """Provide one immutable retained-output observation.

    Attributes:
        stdout: Retained decoded stdout prefix.
        stderr: Retained decoded stderr prefix.
        stdout_truncated: Whether stdout exceeded its configured byte allowance.
        stderr_truncated: Whether stderr exceeded its configured byte allowance.
        complete: Whether both validated final fences arrived in time.
        live_delivery_issue: Optional best-effort live-mirroring failure.

    Side Effects:
        None. Captured workload output remains caller-private coordinator data.
    """

    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    complete: bool
    live_delivery_issue: ExecutionIssue | None


@dataclass(frozen=True, slots=True)
class SubmittedCall(Generic[T]):
    """Hold accepted coordinator metadata without the live callable graph.

    Attributes:
        submission_id: Coordinator-unique accepted submission identifier.
        admission_deadline: Monotonic deadline for backend admission.
        payload: Validated private invocation spool descriptor.
        environment: Optional owner-defined environment requirement.
        world: Optional owner-defined world requirement.
        execution_timeout: Optional post-GO workload deadline.
        stream_output: Whether accepted output mirrors live best-effort.
        output: Bound caller-owned output holder.

    Side Effects:
        None. Backends receive this transport record, not a live callable object.
    """

    submission_id: str
    admission_deadline: float
    payload: PayloadSpool
    environment: EnvironmentRequirement | None
    world: WorldRequirement | None
    execution_timeout: float | None
    stream_output: bool
    output: "ExecutionOutput"


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    """Describe one future outcome and independent cleanup state.

    Attributes:
        submission_id: Coordinator correlation identifier.
        state: Current execution lifecycle state.
        cleanup_state: Independent cleanup/recovery lifecycle state.
        cleanup_scope: Backend-owned worker or owned-group boundary.
        cleanup_issues: Immutable bounded cleanup diagnostics.
        backend_job_id: Optional backend-native diagnostic identifier.
        worker_id: Optional qualified worker identity.
        pid: Optional worker-confirmed process identifier.
        environment: Selected environment candidate, when applicable.
        allocation: Observed allocation, when applicable.
        cancel_requested: Whether running cancellation was accepted by a backend.
        report: Optional admission evidence.

    Side Effects:
        Construction copies cleanup issues and never changes future state.
    """

    submission_id: str
    state: Literal["pending", "admitting", "running", "succeeded", "failed", "cancelled", "uncertain"]
    cleanup_state: Literal["pending", "reconciling", "complete", "incomplete"]
    cleanup_scope: Literal["worker", "owned-group"]
    cleanup_issues: tuple[ExecutionIssue, ...]
    backend_job_id: str | None
    worker_id: str | None
    pid: int | None
    environment: EnvironmentCandidate | None
    allocation: WorldAllocation | None
    cancel_requested: bool
    report: AdmissionReport | None

    def __post_init__(self) -> None:
        """Detach mutable cleanup issue inputs from the snapshot."""
        object.__setattr__(self, "cleanup_issues", tuple(self.cleanup_issues))


@dataclass(frozen=True, slots=True)
class EnvironmentCandidate:
    """Describe one bounded discovered environment candidate.

    Attributes:
        key: Stable local candidate key for this observation.
        spec: Candidate selection specification.
        record: Optional observed environment record.
        report: Optional requirement compatibility evidence.
        launchable: Whether a bounded probe established launchability.
        issues: Immutable candidate probe/discovery diagnostics.

    Side Effects:
        Construction copies issues; discovery does not provision this candidate.
    """

    key: str
    spec: EnvironmentSpec = field(repr=False)
    record: EnvironmentRecord | None = field(repr=False)
    report: CompatibilityReport | None = field(repr=False)
    launchable: bool | None
    issues: tuple[ExecutionIssue, ...]

    def __post_init__(self) -> None:
        """Detach mutable issue inputs from this candidate."""
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True)
class FeasiblePlan:
    """Describe a non-reserving environment/world feasibility observation.

    Attributes:
        environment_key: Optional selected discovery candidate key.
        world: Observed single-process world shape.
        allocation: Optional exact local allocation evidence.
        report: Environment/world admission evidence.

    Side Effects:
        None. This plan does not reserve capacity or invoke work.
    """

    environment_key: str | None
    world: WorldSpec
    allocation: WorldAllocation | None
    report: AdmissionReport


def _freeze_amount_mapping(name: str, values: Mapping[str, float | None]) -> Mapping[str, float | None]:
    """Validate finite nonnegative resource amounts and freeze the mapping."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen: dict[str, float | None] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} values must be finite nonnegative numbers or None")
        frozen[key] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class ResourceAmounts:
    """Summarize observed or charged logical resources without requirement algebra.

    Args:
        cpus: Optional finite nonnegative logical CPU amount.
        memory_bytes: Optional nonnegative integer logical memory amount.
        accelerators: Accelerator-kind amounts, copied into an immutable mapping.
        named: Named-resource amounts, copied into an immutable mapping.

    Raises:
        TypeError: If mappings, keys, or memory type are invalid.
        ValueError: If a supplied amount is negative or non-finite.

    Side Effects:
        Construction detaches mappings; quantities do not reserve physical devices.
    """

    cpus: float | None
    memory_bytes: int | None
    accelerators: Mapping[str, float | None]
    named: Mapping[str, float | None]

    def __post_init__(self) -> None:
        """Validate scalar amounts and detach mutable resource mappings."""
        for name, value in (("cpus", self.cpus), ("memory_bytes", self.memory_bytes)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite nonnegative number or None")
        if self.memory_bytes is not None and not isinstance(self.memory_bytes, int):
            raise TypeError("memory_bytes must be an integer or None")
        object.__setattr__(self, "accelerators", _freeze_amount_mapping("accelerators", self.accelerators))
        object.__setattr__(self, "named", _freeze_amount_mapping("named", self.named))


@dataclass(frozen=True, slots=True)
class ActiveAllocation:
    """Describe one outstanding coordinator/backend resource charge.

    Attributes:
        submission_id: Charged submission correlation identifier.
        backend_job_id: Optional backend-native task identifier.
        worker_id: Optional qualified worker identity for release evidence.
        pid: Optional observed worker PID.
        resources: Charged logical amounts.
        allocation: Optional owner-domain allocation observation.
        state: Reservation, running, release, or unconfirmed charge state.

    Side Effects:
        None. A record is local accounting evidence, not a global resource lease.
    """

    submission_id: str
    backend_job_id: str | None
    worker_id: str | None
    pid: int | None
    resources: ResourceAmounts
    allocation: WorldAllocation | None
    state: Literal["reserved", "running", "releasing", "unconfirmed"]


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Provide one timestamped backend-scoped capacity observation.

    Attributes:
        observed_at: UTC observation timestamp.
        accounting_scope: Fixed coordinator-and-backend scope marker.
        inventory: Optional exact local inventory.
        total: Observed backend logical capacity.
        allocated: Known coordinator/backend charges.
        available: Backend-observable available capacity.
        allocations: Immutable outstanding charge observations.
        complete: Whether all required observation evidence was available.
        issues: Immutable bounded observation diagnostics.

    Side Effects:
        Construction copies collections; it does not reserve or isolate resources.
    """

    observed_at: datetime
    accounting_scope: Literal["coordinator-and-backend"]
    inventory: LocalResourceInventory | None
    total: ResourceAmounts
    allocated: ResourceAmounts
    available: ResourceAmounts
    allocations: tuple[ActiveAllocation, ...]
    complete: bool
    issues: tuple[ExecutionIssue, ...]

    def __post_init__(self) -> None:
        """Detach allocation and issue sequences from this snapshot."""
        object.__setattr__(self, "allocations", tuple(self.allocations))
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True)
class DiscoverySnapshot:
    """Provide one timestamped non-reserving discovery observation.

    Attributes:
        observed_at: UTC observation timestamp.
        environments: Immutable bounded candidate observations.
        resources: Backend-scoped resource observation.
        plans: Immutable non-reserving feasible plans.
        complete: Whether all bounded discovery work completed.
        issues: Immutable bounded discovery diagnostics.

    Side Effects:
        Construction copies collections. Discovery never launches workload code.
    """

    observed_at: datetime
    environments: tuple[EnvironmentCandidate, ...]
    resources: ResourceSnapshot
    plans: tuple[FeasiblePlan, ...]
    complete: bool
    issues: tuple[ExecutionIssue, ...]

    def __post_init__(self) -> None:
        """Detach all collection fields from callers' mutable inputs."""
        object.__setattr__(self, "environments", tuple(self.environments))
        object.__setattr__(self, "plans", tuple(self.plans))
        object.__setattr__(self, "issues", tuple(self.issues))


__all__ = ["ActiveAllocation", "AdmissionReport", "DiscoverySnapshot", "EnvironmentCandidate", "ExecutionIssue", "ExecutionSnapshot", "FeasiblePlan", "OutputSnapshot", "PayloadSpool", "ResourceAmounts", "ResourceSnapshot", "ResultSpool", "SubmittedCall"]
