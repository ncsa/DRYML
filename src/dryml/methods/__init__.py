"""Logical callable IR, passive traits, selection, and local preparation.

The package owns dependency-light Method declarations, deterministic authored
implementation catalogs, direct eager selection, and exact process-local call
preparation. It does not own dispatch, managed lifecycle, persistence, or code
transformation policy.
"""

from dryml.core.backend import Backend
from dryml.core.tensor_spec import BatchMode

from .errors import (
    ImplementationDeclarationError,
    ImplementationSelectionError,
    MethodError,
    PreparedCallMismatchError,
    SelectionFailureReason,
    SelectionTraitName,
)
from .implementation import MethodImplementation
from .method import Method
from .signature import MethodCallMode, MethodCallNode, MethodCallNodeKind, MethodCallSignature
from .traits import Traits, traits

__all__ = [
    "Backend",
    "BatchMode",
    "Traits",
    "traits",
    "MethodImplementation",
    "MethodCallMode",
    "MethodCallNodeKind",
    "MethodCallNode",
    "MethodCallSignature",
    "Method",
    "MethodError",
    "ImplementationDeclarationError",
    "ImplementationSelectionError",
    "PreparedCallMismatchError",
    "SelectionFailureReason",
    "SelectionTraitName",
]
