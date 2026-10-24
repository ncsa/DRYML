"""Synchronous managed lifecycle, detached results and safe-point authority.

managed_operation declarations return InvocationResult after successful execution
or replay. Its SavedNode/status/identity metadata is detached; value access opens
fresh result authority and may raise ManagedResultError without reexecution.
ManagedConfig selects invocation resources and explicit rerun/start controls.
Imports expose owner-defined types without activating execution backends or
materializing receivers; declaration and invocation validation raise ManagedError
subclasses at their documented boundaries.
"""

from .config import ManagedConfig
from .descriptor import managed_operation
from .errors import (
    ManagedConfigError,
    ManagedConflictError,
    ManagedContextError,
    ManagedControlError,
    ManagedDeclarationError,
    ManagedError,
    ManagedInterrupted,
    ManagedPublicationError,
    ManagedRecoveryError,
    ManagedResultError,
    ManagedRerunRequiredError,
    ManagedStoreError,
)
from .identity import argument_digest, operation_digest
from .model import (InterruptRequestResult, ManagedStatus, ExecutionIdentity,
                    InvocationResult, FailedInvocationResult)
from .results import ResultReference, ResultSource
from .context import ManagedContext

__all__ = [
    "InterruptRequestResult",
    "InvocationResult",
    "FailedInvocationResult",
    "ExecutionIdentity",
    "ResultReference",
    "ResultSource",
    "ManagedResultError",
    "ManagedConfig",
    "ManagedContext",
    "ManagedConfigError",
    "ManagedConflictError",
    "ManagedContextError",
    "ManagedControlError",
    "ManagedDeclarationError",
    "ManagedError",
    "ManagedInterrupted",
    "ManagedPublicationError",
    "ManagedRecoveryError",
    "ManagedRerunRequiredError",
    "ManagedStatus",
    "ManagedStoreError",
    "argument_digest",
    "managed_operation",
    "operation_digest",
]
