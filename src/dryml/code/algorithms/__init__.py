"""Reusable dependency-light built-in code-analysis kernels."""

from .lexical_dependencies import (
    LexicalDependencies,
    LexicalDependency,
    LexicalDependencyKernel,
    collect_lexical_dependencies,
)

__all__ = [
    "LexicalDependencies",
    "LexicalDependency",
    "LexicalDependencyKernel",
    "collect_lexical_dependencies",
]
