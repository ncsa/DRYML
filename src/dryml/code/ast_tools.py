"""Compatibility exports for AST access helper classes and functions."""

from dryml.code.algorithms.ast_access import (
    AccessCollector,
    AttrAccess,
    MethodCall,
    collect_accesses_from_source,
)

__all__ = ["AccessCollector", "AttrAccess", "MethodCall", "collect_accesses_from_source"]
