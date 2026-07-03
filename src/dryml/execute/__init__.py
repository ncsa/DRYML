"""Legacy local execution helpers pending dispatch v2.

``dryml.execute`` keeps the existing pickled-callable local/process execution
tests working. New architectural work should target world/runtime primitives and
the future dispatch layer rather than expanding this package.
"""

from dryml.execute.backend import BackendBase, InlineBackend, LocalProcessBackend
from dryml.execute.future import OrchestratedFuture
from dryml.execute.orchestrator import ExecutionOrchestrator, run, submit
from dryml.execute.protocol import (
    ExecutionError,
    ExecutionRequest,
    ExecutionResponse,
    RemoteExecutionError,
    StoreRef,
)


__all__ = [
    "BackendBase",
    "ExecutionError",
    "ExecutionOrchestrator",
    "ExecutionRequest",
    "ExecutionResponse",
    "InlineBackend",
    "LocalProcessBackend",
    "OrchestratedFuture",
    "RemoteExecutionError",
    "StoreRef",
    "run",
    "submit",
]
