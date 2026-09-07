"""Checked declarations, policy, and persistence primitives for managed operations.

U4 additionally supplies internal Store resolution and current-control authority.
Invocation lifecycle, status, checkpointing, and interruption remain deferred to
later managed units.
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
    ManagedRerunRequiredError,
    ManagedStoreError,
)
from .identity import argument_digest, operation_digest
from .model import InterruptRequestResult, ManagedStatus

__all__ = [
    "InterruptRequestResult",
    "ManagedConfig",
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
