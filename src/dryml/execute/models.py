"""Immutable contract values for the additive Execute implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from dryml.environments import CompatibilityReport, EnvironmentRecord, EnvironmentRequirement
from dryml.environments.specs import EnvironmentSpec
from dryml.worlds import LocalResourceInventory, WorldAllocation, WorldCompatibilityReport, WorldRequirement, WorldSpec

if TYPE_CHECKING:
    from dryml.execute.output import ExecutionOutput


@dataclass(frozen=True, slots=True)
class ExecutionIssue:
    """Hold one bounded Execute diagnostic code and message."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AdmissionReport:
    """Aggregate optional domain evidence and bounded Execute issues."""

    environment: CompatibilityReport | None = None
    world: WorldCompatibilityReport | None = None
    issues: tuple[ExecutionIssue, ...] = ()

    def __post_init__(self) -> None:
        """Detach the issue sequence into an immutable tuple."""
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True)
class PayloadSpool:
    """Describe one immutable, validated coordinator-owned call snapshot."""

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
    """Describe one immutable, validated coordinator-owned result snapshot.

    Result files share a submission-owned spool child with the invocation and
    remain private implementation descriptors rather than durable references.
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
    """Provide an immutable retained-output observation for one submission."""

    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    complete: bool
    live_delivery_issue: ExecutionIssue | None


@dataclass(frozen=True, slots=True)
class SubmittedCall:
    """Hold accepted coordinator data without retaining the live callable graph."""

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
    """Describe one Future's outcome and independently tracked cleanup state."""

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
    """Describe one environment candidate inside a discovery snapshot."""

    key: str
    spec: EnvironmentSpec
    record: EnvironmentRecord | None
    report: CompatibilityReport | None
    launchable: bool | None
    issues: tuple[ExecutionIssue, ...]

    def __post_init__(self) -> None:
        """Detach mutable issue inputs from this candidate."""
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True)
class FeasiblePlan:
    """Describe a non-reserving single-process environment/world plan."""

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
    """Summarize observed or charged resource quantities without requirement algebra."""

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
    """Describe one outstanding coordinator/backend resource charge."""

    submission_id: str
    backend_job_id: str | None
    worker_id: str | None
    pid: int | None
    resources: ResourceAmounts
    allocation: WorldAllocation | None
    state: Literal["reserved", "running", "releasing", "unconfirmed"]


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Provide one timestamped backend-scoped capacity observation."""

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
    """Provide one timestamped non-reserving discovery observation."""

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
