"""Logical Method authoring, passive traits, and inspectable implementation catalogs.

This package owns dependency-light Method declarations. U3 provides direct-call
forwarding and catalog construction; U4 adds alternative selection and call
preparation without changing the authored declaration surface.
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
from .traits import Traits, traits

__all__ = [
    "Backend",
    "BatchMode",
    "Traits",
    "traits",
    "MethodImplementation",
    "Method",
    "MethodError",
    "ImplementationDeclarationError",
    "ImplementationSelectionError",
    "PreparedCallMismatchError",
    "SelectionFailureReason",
    "SelectionTraitName",
]
