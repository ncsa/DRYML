"""Public authoring and logical-reference API for managed operations."""

from .declarations import (
    DelegatedOutputs,
    ManagedMethodDeclaration,
    ManagedOutput,
    ManagedOutputs,
    OutputDeclaration,
    OutputDeclarations,
    normalize_outputs,
    resolve_definition_path,
)
from .descriptor import BoundManagedMethod, ManagedMethod, managed
from .errors import (
    DuplicateOutputError,
    InvalidSubjectPathError,
    ManagedDeclarationError,
    ManagedError,
    ManagedLifecycleUnavailableError,
    PrimaryOutputError,
    UnknownOutputError,
    UnstableOutputsError,
)
from .refs import ManagedOutputRef

__all__ = [
    "BoundManagedMethod",
    "DelegatedOutputs",
    "DuplicateOutputError",
    "InvalidSubjectPathError",
    "ManagedDeclarationError",
    "ManagedError",
    "ManagedLifecycleUnavailableError",
    "ManagedMethod",
    "ManagedMethodDeclaration",
    "ManagedOutput",
    "ManagedOutputRef",
    "ManagedOutputs",
    "OutputDeclaration",
    "OutputDeclarations",
    "PrimaryOutputError",
    "UnknownOutputError",
    "UnstableOutputsError",
    "managed",
    "normalize_outputs",
    "resolve_definition_path",
]
