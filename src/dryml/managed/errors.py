"""Errors raised by managed-operation declarations and logical references."""


class ManagedError(Exception):
    """Base error for managed-operation authoring and lifecycle surfaces."""


class ManagedDeclarationError(ManagedError, ValueError):
    """A managed method or output declaration is invalid."""


class DuplicateOutputError(ManagedDeclarationError):
    """A managed output slot is declared more than once."""


class PrimaryOutputError(ManagedDeclarationError):
    """A managed declaration does not have exactly one primary output."""


class UnknownOutputError(ManagedError, KeyError):
    """A caller requested an output slot absent from the declaration."""


class InvalidSubjectPathError(ManagedDeclarationError):
    """An output subject path does not identify an Object definition."""


class UnstableOutputsError(ManagedDeclarationError):
    """A delegated output provider returned a non-deterministic contract."""


class ManagedLifecycleUnavailableError(ManagedError, NotImplementedError):
    """Realization-backed lifecycle behavior is not installed for a method."""


class ManagedStateError(ManagedError, ValueError):
    """Managed control state is malformed, incompatible, or inconsistent."""


class ManagedStoreUnsupportedError(ManagedError, NotImplementedError):
    """A Store lacks a capability required by managed operation control."""


class AmbiguousManagedStoreError(ManagedError, RuntimeError):
    """Managed Store selection is absent or ambiguous."""


class ManagedLeaseConflictError(ManagedError, RuntimeError):
    """Another process currently owns the operation's OS lock."""


class ManagedTakeoverRequiredError(ManagedLeaseConflictError):
    """Released ownership from a live process requires explicit takeover."""


class StaleManagedLeaseError(ManagedError, RuntimeError):
    """A closed or superseded lease attempted to mutate operation state."""


class ManagedRerunRequiredError(ManagedError, RuntimeError):
    """Pending work cannot resume and requires an explicit rerun."""


class ManagedInputValidationRequiredError(ManagedError, RuntimeError):
    """Completed reuse requires a stable-input validation result."""


class StaleManagedResultError(ManagedError, RuntimeError):
    """A completed result no longer matches the current logical inputs."""


__all__ = [
    "AmbiguousManagedStoreError",
    "DuplicateOutputError",
    "InvalidSubjectPathError",
    "ManagedDeclarationError",
    "ManagedError",
    "ManagedLifecycleUnavailableError",
    "ManagedInputValidationRequiredError",
    "ManagedLeaseConflictError",
    "ManagedRerunRequiredError",
    "ManagedStateError",
    "ManagedStoreUnsupportedError",
    "ManagedTakeoverRequiredError",
    "PrimaryOutputError",
    "StaleManagedLeaseError",
    "StaleManagedResultError",
    "UnknownOutputError",
    "UnstableOutputsError",
]
