"""Checked declarations, caller policy, and identities for managed operations.

U3 establishes authoring and input matching only. Invocation lifecycle, status,
checkpointing, and interruption authority are intentionally deferred to later
managed units.
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
    "ManagedRecoveryError",
    "ManagedRerunRequiredError",
    "ManagedStatus",
    "ManagedStoreError",
    "argument_digest",
    "managed_operation",
    "operation_digest",
]
