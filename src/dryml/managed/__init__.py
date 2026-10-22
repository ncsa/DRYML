"""Synchronous managed-operation declaration, lifecycle, and safe-point authority."""

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
