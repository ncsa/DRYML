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


__all__ = [
    "DuplicateOutputError",
    "InvalidSubjectPathError",
    "ManagedDeclarationError",
    "ManagedError",
    "ManagedLifecycleUnavailableError",
    "PrimaryOutputError",
    "UnknownOutputError",
    "UnstableOutputsError",
]
