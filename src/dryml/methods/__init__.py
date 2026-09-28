"""Logical callable IR, passive traits, selection, and local preparation.

The package owns dependency-light Method declarations, deterministic authored
implementation catalogs, direct eager selection, and exact process-local call
preparation. It does not own dispatch, managed lifecycle, persistence, or code
transformation policy.
"""

from .errors import (
    ImplementationDeclarationError,
    ImplementationSelectionError,
    MethodError,
    PreparedCallMismatchError,
    SelectionFailureReason,
    SelectionTraitName,
)
from .accumulator import Accumulator, AccumulatorGroup
from .implementation import MethodImplementation
from .conversion import ConversionEdge
from .ir import MethodGraph, MethodGraphNode, MethodGraphNodeKind, MethodPort, MethodPortKind
from .method import Method
from .stream import IteratorPort, StreamGraphCursor, StreamNode
from .signature import MethodCallMode, MethodCallNode, MethodCallNodeKind, MethodCallSignature
from .traits import Traits, traits

__all__ = [
    "Traits",
    "traits",
    "MethodImplementation",
    "ConversionEdge",
    "MethodGraph",
    "MethodGraphNode",
    "MethodGraphNodeKind",
    "MethodPort",
    "MethodPortKind",
    "IteratorPort",
    "StreamGraphCursor",
    "StreamNode",
    "MethodCallMode",
    "MethodCallNodeKind",
    "MethodCallNode",
    "MethodCallSignature",
    "Method",
    "Accumulator",
    "AccumulatorGroup",
    "MethodError",
    "ImplementationDeclarationError",
    "ImplementationSelectionError",
    "PreparedCallMismatchError",
    "SelectionFailureReason",
    "SelectionTraitName",
]
