"""Synchronous managed-operation declaration, lifecycle, and selected authority.

U6 provides start, resume, rerun, status, and cooperative request publication.
U7 owns checkpoint callbacks and interruption safe points.
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
from .context import ManagedContext

__all__ = [
    "InterruptRequestResult",
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
