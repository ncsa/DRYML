"""Generic execution through explicit backend configurations.

Use :class:`Executor` with a configuration from a specialization module, such as
``dryml.execute.subprocess.SubProcessConfig``. Common imports remain independent
of DRYML core/runtime state and do not eagerly import backend specializations.
"""

from .backend import Backend
from .config import BackendConfig
from .errors import (
    AdmissionError,
    BackendUnavailableError,
    CleanupError,
    ExecutionDeadlineExceeded,
    ExecutionError,
    ExecutionUncertainError,
    RemoteExecutionError,
)
from .executor import Executor, ExecutorView, run, submit
from .future import ExecutionFuture
from .models import (
    ActiveAllocation,
    AdmissionReport,
    DiscoverySnapshot,
    EnvironmentCandidate,
    ExecutionIssue,
    ExecutionSnapshot,
    FeasiblePlan,
    OutputSnapshot,
    ResourceAmounts,
    ResourceSnapshot,
)
from .output import ExecutionOutput


__all__ = [
    "ActiveAllocation",
    "AdmissionError",
    "AdmissionReport",
    "Backend",
    "BackendConfig",
    "BackendUnavailableError",
    "CleanupError",
    "DiscoverySnapshot",
    "EnvironmentCandidate",
    "ExecutionDeadlineExceeded",
    "ExecutionError",
    "ExecutionFuture",
    "ExecutionIssue",
    "ExecutionOutput",
    "ExecutionSnapshot",
    "ExecutionUncertainError",
    "Executor",
    "ExecutorView",
    "FeasiblePlan",
    "OutputSnapshot",
    "RemoteExecutionError",
    "ResourceAmounts",
    "ResourceSnapshot",
    "run",
    "submit",
]
