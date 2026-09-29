"""Typed errors for managed-operation declaration and lifecycle boundaries."""

from __future__ import annotations


class ManagedError(Exception):
    """Base error carrying a stable machine-readable managed reason.

    Args:
        reason: Static reason code describing the failed managed boundary.
        message: Optional human-readable detail that does not replace ``reason``.

    Attributes:
        reason: The static reason code callers can safely inspect.
    """

    default_reason = "managed_error"

    def __init__(self, reason: str | None = None, message: str | None = None):
        """Initialize the error without deriving a reason from user data."""

        selected_reason = self.default_reason if reason is None else reason
        if type(selected_reason) is not str or not selected_reason:
            raise TypeError("managed error reasons must be non-empty strings")
        self.reason = selected_reason
        super().__init__(selected_reason if message is None else f"{selected_reason}: {message}")


class ManagedDeclarationError(ManagedError):
    """Raised when a method cannot be a checked managed declaration."""

    default_reason = "invalid_declaration"


class ManagedConfigError(ManagedError):
    """Raised when caller config or normalized managed input is unsupported."""

    default_reason = "invalid_config"


class ManagedStoreError(ManagedError):
    """Raised when caller-selected storage lacks managed support."""

    default_reason = "unsupported_store"


class ManagedControlError(ManagedError):
    """Raised when lifecycle control authority cannot be used safely."""

    default_reason = "control_error"


class ManagedPublicationError(ManagedControlError):
    """Raised when current-control publication has a classified uncertain outcome.

    Args:
        outcome: ``not_committed``, ``committed``, or ``indeterminate``.
        message: Optional static diagnostic detail.

    Attributes:
        outcome: Publication state callers must preserve rather than guessing.
    """

    def __init__(self, outcome: str, message: str | None = None):
        """Initialize a publication error with a closed outcome classification."""

        if outcome not in {"not_committed", "committed", "indeterminate"}:
            raise TypeError("managed publication outcomes must be closed values")
        self.outcome = outcome
        super().__init__("publication_" + outcome, message)


class ManagedConflictError(ManagedError):
    """Raised when a managed operation conflicts with an active owner."""

    default_reason = "conflict"


class ManagedContextError(ManagedError):
    """Raised when a managed execution context is used outside its invocation."""

    default_reason = "invalid_context"


class ManagedRecoveryError(ManagedError):
    """Raised when retained managed authority cannot be recovered safely."""

    default_reason = "recovery_error"


class ManagedRerunRequiredError(ManagedError):
    """Raised when selected operation authority requires an explicit rerun."""

    default_reason = "already_completed"


class ManagedInterrupted(ManagedError):
    """Raised when a managed safe point commits a cooperative interruption."""

    default_reason = "interrupted"


__all__ = [
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
    "ManagedStoreError",
]
